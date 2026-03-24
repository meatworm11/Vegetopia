"""
training.py — Pool-based NCA training loop.

Strategy (Growing NCA "what persists, exists"):
  - Maintain a pool of 1024 grid states at various growth stages.
  - Each iteration: sample a batch, replace the worst sample with a fresh seed,
    unroll the NCA for 64-96 random steps, compute loss, backprop, write back.
  - Per-parameter gradient normalization prevents exploding gradients through
    long unrolled sequences.
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # non-interactive — safe for file saving, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    BATCH_SIZE, CELL_FIRE_RATE, LEARNING_RATE,
    N_CHANNELS, POOL_SIZE, TRAIN_STEPS_RANGE, DEVICE,
    CH_CELL_TYPE,
)
from model import NCA, make_seed


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _compute_loss(
    output: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (total_loss, shape_loss, type_loss, overflow_loss, bg_loss).

    target shape: (N_CHANNELS, H, W) — broadcast over batch automatically.
    """
    tgt = target.unsqueeze(0)   # (1, C, H, W) — broadcasts to batch

    alive_mask = (output[:, 3:4] > 0.1).float()

    shape_loss = F.mse_loss(output[:, :4], tgt[:, :4].expand_as(output[:, :4]))

    type_loss = F.mse_loss(
        output[:, CH_CELL_TYPE:CH_CELL_TYPE + 1] * alive_mask,
        tgt[:, CH_CELL_TYPE:CH_CELL_TYPE + 1].expand_as(
            output[:, CH_CELL_TYPE:CH_CELL_TYPE + 1]
        ) * alive_mask,
    )

    overflow = (output[:, 12:].abs() - 5).clamp(min=0).mean()

    # Background loss: penalise any alpha outside the target's alive region
    target_alpha_mask = (tgt[:, 3:4] > 0.1).float()
    bg_loss = (output[:, 3:4] * (1.0 - target_alpha_mask)).mean()

    total = shape_loss + 0.1 * type_loss + 0.01 * overflow + 2.0 * bg_loss
    return total, shape_loss, type_loss, overflow, bg_loss


def _per_sample_loss(states: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on RGB+alpha per sample. Returns (batch,) tensor — no grad needed."""
    tgt = target[:4].unsqueeze(0)
    return (states[:, :4] - tgt).pow(2).mean(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def save_snapshot(
    output: torch.Tensor,
    target: torch.Tensor,
    name: str,
    step: int,
    snapshots_dir: Path,
) -> Path:
    """
    Side-by-side PNG: target (left) | batch[0] current state (right).
    RGB channels composited over white background using alpha.
    """
    def _composite(t: np.ndarray) -> np.ndarray:
        rgb   = t[:3].transpose(1, 2, 0).clip(0, 1)
        alpha = t[3].clip(0, 1)
        return (rgb * alpha[:, :, None] + (1.0 - alpha[:, :, None])).clip(0, 1)

    sample_np = output[0].detach().cpu().float().numpy()
    target_np = target.detach().cpu().float().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(_composite(target_np), interpolation='nearest')
    axes[0].set_title('Target', fontsize=10)
    axes[0].axis('off')
    axes[1].imshow(_composite(sample_np), interpolation='nearest')
    axes[1].set_title(f'Step {step:,}', fontsize=10)
    axes[1].axis('off')
    plt.tight_layout()

    out_path = snapshots_dir / f'{name}_step{step:04d}.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: NCA,
    target: torch.Tensor,
    name: str,
    n_steps: int = 5000,
    device: torch.device = DEVICE,
    snapshot_every: int = 500,
    checkpoint_every: int = 1000,
) -> NCA:
    """
    Pool-based NCA training.

    Args:
        model            : NCA model already moved to device
        target           : target tensor (N_CHANNELS, H, W) on device
        name             : species name — used for snapshot/checkpoint filenames
        n_steps          : total training iterations
        device           : torch device
        snapshot_every   : save side-by-side image every N steps
        checkpoint_every : save intermediate .pt checkpoint every N steps

    Returns:
        Trained NCA model.
    """
    species_dir   = Path('species');   species_dir.mkdir(exist_ok=True)
    snapshots_dir = Path('snapshots'); snapshots_dir.mkdir(exist_ok=True)

    # Pool lives on CPU to avoid VRAM pressure; batches are moved to device per step.
    seed_cpu = make_seed(1, device=torch.device('cpu'))            # (1, C, H, W)
    pool     = seed_cpu.expand(POOL_SIZE, -1, -1, -1).clone()     # (POOL_SIZE, C, H, W)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng       = np.random.default_rng()

    header = f"{'Step':>6}  {'Loss':>9}  {'Shape':>9}  {'Type':>7}  {'OFlow':>7}  {'BgLoss':>7}  {'Elapsed':>7}  ETA"
    print(header)
    print("-" * len(header))

    t0           = time.time()
    loss_history = []

    for step in range(1, n_steps + 1):

        # --- Sample batch -------------------------------------------------------
        indices = rng.choice(POOL_SIZE, size=BATCH_SIZE, replace=False)
        batch   = pool[indices].to(device)

        # Replace the highest-loss sample with a fresh seed (prevents degenerate states)
        with torch.no_grad():
            per_sample = _per_sample_loss(batch, target)
        worst        = int(per_sample.argmax().item())
        batch[worst] = make_seed(1, device=device).squeeze(0)

        # --- Unroll NCA ---------------------------------------------------------
        n_nca = int(rng.integers(TRAIN_STEPS_RANGE[0], TRAIN_STEPS_RANGE[1] + 1))
        x     = batch
        for _ in range(n_nca):
            x = model(x, fire_rate=CELL_FIRE_RATE)

        # --- Loss ---------------------------------------------------------------
        total, shape_l, type_l, overflow_l, bg_l = _compute_loss(x, target)

        # --- Backward + gradient normalization ----------------------------------
        optimizer.zero_grad()
        total.backward()
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data /= (param.grad.data.norm() + 1e-8)
        optimizer.step()

        # --- Write outputs back to pool (detached — no grad history kept) -------
        pool[indices] = x.detach().cpu()

        # --- Logging ------------------------------------------------------------
        loss_val = total.item()
        loss_history.append(loss_val)

        if step % 100 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / step * (n_steps - step)
            print(
                f"{step:>6,}  {loss_val:>9.5f}  "
                f"{shape_l.item():>9.5f}  {type_l.item():>7.5f}  "
                f"{overflow_l.item():>7.4f}  "
                f"{bg_l.item():>7.5f}  "
                f"{elapsed:>6.0f}s  {eta:.0f}s"
            )

        # --- Snapshot -----------------------------------------------------------
        if step % snapshot_every == 0:
            path = save_snapshot(x, target, name, step, snapshots_dir)
            print(f"         [snapshot] {path}")

        # --- Intermediate checkpoint --------------------------------------------
        if step % checkpoint_every == 0:
            ckpt = species_dir / f'{name}_step{step:04d}.pt'
            torch.save({
                'step':             step,
                'model_state_dict': model.state_dict(),
                'n_channels':       N_CHANNELS,
                'name':             name,
                'loss':             loss_val,
            }, ckpt)
            print(f"         [checkpoint] {ckpt}")

    # --- Final model ------------------------------------------------------------
    final = species_dir / f'{name}.pt'
    torch.save({
        'step':             n_steps,
        'model_state_dict': model.state_dict(),
        'n_channels':       N_CHANNELS,
        'name':             name,
        'loss':             loss_history[-1] if loss_history else None,
    }, final)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed:.0f}s total  |  final loss {loss_history[-1]:.5f}")
    print(f"Model     -> {final}")
    print(f"Snapshots -> {snapshots_dir}/")

    return model
