"""
training.py — Pool-based NCA training loop.

Strategy (Growing NCA "what persists, exists"):
  - Maintain a pool of 1024 grid states at various growth stages.
  - Each iteration: sample a batch, replace the worst sample with a fresh seed,
    unroll the NCA for 80-128 random steps, compute loss, backprop, write back.
  - Per-parameter gradient normalization prevents exploding gradients through
    long unrolled sequences.

Energy-aware mode (energy=True):
  - Channels 4-5 are NOT clamped to 1.0 — plants must absorb nutrients
  - Each NCA step is followed by environment physics: diffusion, absorption,
    transport, dissipation, and death check
  - A survival bonus rewards keeping cells alive

Curriculum mode (curriculum=True, requires energy=True):
  - Phase A (steps 1-3000): nutrients clamped (phase-1 style), but new loss
    terms active (root masking, stem colour, survival bonus by cell count)
  - Phase B (steps 3001-6000): energy ON with gentle rates
    (dissipation 0.0002, absorption 0.1)
  - Phase C (steps 6001+): full energy rates
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
    CELL_FIRE_RATE, LEARNING_RATE,
    N_CHANNELS, POOL_SIZE, TRAIN_STEPS_RANGE, DEVICE,
    GRID_H, GRID_W,
    CH_ALPHA, CH_EARTH, CH_AIR, CH_CELL_TYPE, ROOT_LOSS_MASK,
    NUTRIENT_TRANSPORT_STEPS,
)
from model import NCA, make_seed

# Training batch size — smaller than simulation for speed
TRAIN_BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

# Target stem colour for colour-consistency loss
_STEM_BROWN = torch.tensor([0.45, 0.30, 0.15]).view(3, 1, 1)


def _compute_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    energy_health: bool = False,
    soil_row: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (total_loss, shape_loss, type_loss, overflow_loss, bg_loss, stem_loss, energy_loss, root_reward).

    target shape: (N_CHANNELS, H, W) — broadcast over batch automatically.
    energy_health: if True, penalize alive cells with nutrients below 0.85.
    soil_row: first soil row index — used for root presence reward.
    """
    tgt = target.unsqueeze(0)   # (1, C, H, W) — broadcasts to batch

    alive_mask = (output[:, 3:4] > 0.1).float()

    # --- Shape loss (with root masking) ---
    if ROOT_LOSS_MASK:
        tgt_ctype = tgt[:, CH_CELL_TYPE:CH_CELL_TYPE + 1]
        root_mask = ((tgt_ctype - 0.25).abs() < 0.15).float()   # 1 where root
        shape_weight = 1.0 - root_mask                           # 0 at root cells
        shape_diff = (output[:, :4] - tgt[:, :4].expand_as(output[:, :4])) ** 2
        shape_loss = (shape_diff * shape_weight).mean()
    else:
        shape_loss = F.mse_loss(output[:, :4], tgt[:, :4].expand_as(output[:, :4]))

    # --- Cell-type loss ---
    type_loss = F.mse_loss(
        output[:, CH_CELL_TYPE:CH_CELL_TYPE + 1] * alive_mask,
        tgt[:, CH_CELL_TYPE:CH_CELL_TYPE + 1].expand_as(
            output[:, CH_CELL_TYPE:CH_CELL_TYPE + 1]
        ) * alive_mask,
    )

    # --- Hidden channel overflow ---
    overflow = (output[:, 12:].abs() - 5).clamp(min=0).mean()

    # --- Background loss ---
    target_alpha_mask = (tgt[:, 3:4] > 0.1).float()
    bg_loss = (output[:, 3:4] * (1.0 - target_alpha_mask)).mean()

    # --- Stem colour consistency loss ---
    out_ctype = output[:, CH_CELL_TYPE:CH_CELL_TYPE + 1]
    is_stem = ((out_ctype - 0.5).abs() < 0.1).float() * alive_mask  # (B,1,H,W)
    stem_target = _STEM_BROWN.to(output.device).unsqueeze(0)         # (1,3,1,1)
    stem_diff = (output[:, :3] - stem_target) ** 2                   # (B,3,H,W)
    stem_count = is_stem.sum().clamp(min=1)
    stem_loss = (stem_diff * is_stem).sum() / (stem_count * 3)

    total = shape_loss + 0.1 * type_loss + 0.01 * overflow + 2.0 * bg_loss + 0.1 * stem_loss

    # --- Energy health loss ---
    # Penalize alive cells with nutrients below 0.85.
    # Forces the model to grow roots (earth supply) and leaves (air supply).
    energy_loss = torch.tensor(0.0, device=output.device)
    if energy_health:
        earth = output[:, CH_EARTH:CH_EARTH + 1]   # (B, 1, H, W)
        air   = output[:, CH_AIR:CH_AIR + 1]       # (B, 1, H, W)
        n_alive = alive_mask.sum().clamp(min=1)
        earth_deficit = (0.85 - earth).clamp(min=0) * alive_mask
        air_deficit   = (0.85 - air).clamp(min=0) * alive_mask
        energy_loss = (earth_deficit.sum() + air_deficit.sum()) / n_alive
        total = total + 0.5 * energy_loss

    # --- Root presence reward ---
    # Direct differentiable signal: alive cells with cell_type ≈ 0.25 in soil
    # rows are valuable.  Uses soft proximity to 0.25 so gradients flow through
    # cell_type channel.
    root_reward = torch.tensor(0.0, device=output.device)
    if energy_health and soil_row > 0:
        EXPECTED_ROOTS = 20.0
        ctype_soil = output[:, CH_CELL_TYPE:CH_CELL_TYPE + 1, soil_row:, :]  # (B,1,soil_H,W)
        alive_soil = alive_mask[:, :, soil_row:, :]                           # (B,1,soil_H,W)
        # Soft root score: 1.0 when cell_type == 0.25, falls off away from it
        root_score = (1.0 - ((ctype_soil - 0.25) / 0.15).pow(2)).clamp(min=0)
        root_count = (root_score * alive_soil).sum() / max(output.shape[0], 1)
        root_reward = -0.5 * (root_count / EXPECTED_ROOTS).clamp(max=1.0)
        total = total + root_reward

    return total, shape_loss, type_loss, overflow, bg_loss, stem_loss, energy_loss, root_reward


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
    energy: bool = False,
    lr: float = LEARNING_RATE,
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
        energy           : if True, run environment physics during unroll
                           with finite nutrient pool (no regeneration)
        lr               : learning rate (default LEARNING_RATE; 5e-4 for fine-tuning)

    Returns:
        Trained NCA model.
    """
    species_dir   = Path('species');   species_dir.mkdir(exist_ok=True)
    snapshots_dir = Path('snapshots'); snapshots_dir.mkdir(exist_ok=True)

    # --- Energy-aware training environment ---
    env_state = None
    soil_row  = 0      # 0 = no soil info → root reward disabled
    if energy:
        from environment import (
            create_environment, EnvironmentState,
            regenerate_sources, diffuse_nutrients,
            plant_absorb_nutrients, transport_nutrients,
            plant_dissipate_energy, plant_death_check,
        )
        env_grid = create_environment(GRID_H, GRID_W, rng=np.random.default_rng(42))
        env_state = EnvironmentState(env_grid, device=device)
        # Prepare a fully-charged snapshot of the nutrient grid.
        # Each rollout resets to this snapshot — no regeneration during unroll,
        # so the plant faces a finite nutrient pool.
        print("Warming up training environment nutrients...")
        for _ in range(500):
            regenerate_sources(env_state)
            diffuse_nutrients(env_state, n_steps=5)
        env_nutrient_snapshot = env_state.nutrient_grid.clone()
        soil_row = GRID_H - GRID_H // 5   # first soil row (matches create_environment)
        print(f"  done.  soil_row={soil_row}")

    # Pool lives on CPU to avoid VRAM pressure; batches are moved to device per step.
    seed_cpu = make_seed(1, device=torch.device('cpu'))            # (1, C, H, W)
    pool     = seed_cpu.expand(POOL_SIZE, -1, -1, -1).clone()     # (POOL_SIZE, C, H, W)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng       = np.random.default_rng()

    header = (
        f"{'Step':>6}  {'Loss':>9}  {'Shape':>9}  {'Type':>7}  "
        f"{'OFlow':>7}  {'BgLoss':>7}  {'Stem':>7}  {'Energy':>7}  {'Root':>7}  {'Alive':>5}  {'Elapsed':>7}  ETA"
    )
    print(header)
    print("-" * len(header))

    if energy:
        print("Sources regeneration: DISABLED (finite nutrient pool per rollout)")
        print("NCA nutrient override: DISABLED (channels 4-5 preserved from input)")
        print("Rollout curriculum: max = min(40 + step//20, 128)")
        print("Death immunity: first 10 NCA steps of each rollout")

    t0           = time.time()
    loss_history = []

    for step in range(1, n_steps + 1):

        # --- Sample batch -------------------------------------------------------
        indices = rng.choice(POOL_SIZE, size=TRAIN_BATCH_SIZE, replace=False)
        batch   = pool[indices].to(device)

        # Replace the highest-loss sample with a fresh seed
        with torch.no_grad():
            per_sample = _per_sample_loss(batch, target)
        worst        = int(per_sample.argmax().item())
        batch[worst] = make_seed(1, device=device).squeeze(0)

        # --- Reset environment nutrients to full for this rollout ---------------
        if energy:
            env_state.nutrient_grid.copy_(env_nutrient_snapshot)

        # --- Unroll NCA ---------------------------------------------------------
        # Rollout curriculum: ramp max rollout from 40 up to 128 over training.
        # Shorter early rollouts reduce nutrient drain per pool visit, giving
        # the model time to learn basic growth before nutrient pressure.
        if energy:
            rollout_max = min(40 + step // 20, TRAIN_STEPS_RANGE[1])
            rollout_min = min(TRAIN_STEPS_RANGE[0], rollout_max)
        else:
            rollout_min, rollout_max = TRAIN_STEPS_RANGE
        n_nca = int(rng.integers(rollout_min, rollout_max + 1))
        x     = batch
        for nca_step in range(n_nca):
            x = model(x, fire_rate=CELL_FIRE_RATE, nutrients_enabled=energy)

            # Energy-aware: run environment physics after each NCA step.
            # No regeneration — finite nutrient pool forces efficient growth.
            if energy:
                with torch.no_grad():
                    diffuse_nutrients(env_state, n_steps=1)
                    for b in range(x.shape[0]):
                        sample = x[b:b+1]  # (1, C, H, W)
                        plant_absorb_nutrients(sample, env_state)
                        transport_nutrients(sample, n_steps=NUTRIENT_TRANSPORT_STEPS)
                        plant_dissipate_energy(sample)
                        # Death immunity for the first 10 NCA steps — let the
                        # plant establish before it can die.
                        if nca_step >= 10:
                            plant_death_check(sample)

        # --- Debug: nutrient levels after unroll -----------------------------------
        if energy and step % 100 == 0:
            with torch.no_grad():
                s = x[0]  # first batch item: (C, H, W)
                alive = s[CH_ALPHA] > 0.1
                if alive.any():
                    earth_alive = s[CH_EARTH][alive]
                    air_alive   = s[CH_AIR][alive]
                    print(f"  [nutrient debug] alive={int(alive.sum())}  "
                          f"earth: min={earth_alive.min():.4f} max={earth_alive.max():.4f} mean={earth_alive.mean():.4f}  "
                          f"air: min={air_alive.min():.4f} max={air_alive.max():.4f} mean={air_alive.mean():.4f}")
                else:
                    print(f"  [nutrient debug] no alive cells")

        # --- Loss ---------------------------------------------------------------
        # Energy health loss active when energy flag is set (all curriculum phases)
        total, shape_l, type_l, overflow_l, bg_l, stem_l, energy_l, root_r = _compute_loss(
            x, target, energy_health=energy, soil_row=soil_row,
        )

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
            n_alive = int((x[:, CH_ALPHA] > 0.1).float().sum().item())
            print(
                f"{step:>6,}  {loss_val:>9.5f}  "
                f"{shape_l.item():>9.5f}  {type_l.item():>7.5f}  "
                f"{overflow_l.item():>7.4f}  "
                f"{bg_l.item():>7.5f}  "
                f"{stem_l.item():>7.5f}  "
                f"{energy_l.item():>7.5f}  "
                f"{root_r.item():>7.4f}  "
                f"{n_alive:>5}  "
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
                'energy':           energy,
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
        'energy':           energy,
    }, final)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed:.0f}s total  |  final loss {loss_history[-1]:.5f}")
    print(f"Model     -> {final}")
    print(f"Snapshots -> {snapshots_dir}/")

    return model
