"""
model.py — NCA update rule.

Architecture (Growing NCA, adapted for 32-channel state):
  Perception  : fixed Sobel filters → 96-channel input per cell
  UpdateRule  : two 1×1 convs (96→128→32), final layer zero-initialized
  NCA.forward : perceive → delta → stochastic mask → living mask → clamp → env restore
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    N_CHANNELS, GRID_H, GRID_W, SEED_POSITIONS,
    CH_ALPHA, CH_RGB, CH_HIDDEN, CH_EARTH, CH_AIR, CH_SPECIES_ID,
    CELL_FIRE_RATE,
)

# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

class Perception(nn.Module):
    """
    Fixed (non-trainable) Sobel perception.

    For each of the N_CHANNELS input channels, compute the Sobel gradient in
    x and y using a depthwise 3×3 convolution with circular (toroidal) padding.
    Concatenate gradients with the raw state:
        raw (32) + grad_x (32) + grad_y (32) = 96 channels.

    Output: (batch, 96, H, W)
    """

    def __init__(self, n_channels: int = N_CHANNELS):
        super().__init__()
        self.n_channels = n_channels

        # Sobel kernels — shape (1, 1, 3, 3), broadcast over all channels
        sobel_x = torch.tensor([
            [-1.,  0.,  1.],
            [-2.,  0.,  2.],
            [-1.,  0.,  1.],
        ]).view(1, 1, 3, 3) / 8.0

        sobel_y = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.],
        ]).view(1, 1, 3, 3) / 8.0

        # Expand to depthwise kernels: (n_channels, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x.expand(n_channels, -1, -1, -1).clone())
        self.register_buffer('sobel_y', sobel_y.expand(n_channels, -1, -1, -1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Circular padding so grid wraps (prevents edge artifacts)
        xp = F.pad(x, (1, 1, 1, 1), mode='circular')

        gx = F.conv2d(xp, self.sobel_x, groups=self.n_channels)
        gy = F.conv2d(xp, self.sobel_y, groups=self.n_channels)

        return torch.cat([x, gx, gy], dim=1)   # (batch, 96, H, W)


# ---------------------------------------------------------------------------
# Update rule
# ---------------------------------------------------------------------------

class UpdateRule(nn.Module):
    """
    Learned per-cell update rule implemented as 1×1 convolutions.

    Input  : 96-channel perception vector
    Output : 32-channel state delta (ds)

    The final layer is zero-initialized so the network starts as a no-op.
    """

    def __init__(self, n_channels: int = N_CHANNELS, hidden: int = 128):
        super().__init__()
        perception_channels = n_channels * 3   # raw + grad_x + grad_y

        self.net = nn.Sequential(
            nn.Conv2d(perception_channels, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, n_channels, kernel_size=1),
        )

        # Zero-initialize final layer: network starts as "do nothing"
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, perception: torch.Tensor) -> torch.Tensor:
        return self.net(perception)


# ---------------------------------------------------------------------------
# Full NCA
# ---------------------------------------------------------------------------

class NCA(nn.Module):
    """
    Neural Cellular Automaton.

    The forward pass runs one discrete timestep on the entire grid in parallel.
    Environment-managed channels (earth, air, species_id) are restored after
    each step so the network cannot learn to modify them.

    Args:
        n_channels : total state channels (default N_CHANNELS = 32)
        hidden     : hidden units in UpdateRule (default 128)
    """

    def __init__(self, n_channels: int = N_CHANNELS, hidden: int = 128):
        super().__init__()
        self.n_channels = n_channels
        self.perceive    = Perception(n_channels)
        self.update_rule = UpdateRule(n_channels, hidden)

    def forward(
        self,
        x: torch.Tensor,
        fire_rate: float = CELL_FIRE_RATE,
        step_size: float = 1.0,
        species_id_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        One NCA step.

        Args:
            x              : state tensor (batch, N_CHANNELS, H, W)
            fire_rate      : probability each cell updates this step
            step_size      : scales the state delta (1.0 during training)
            species_id_map : (batch, 1, H, W) protected species-ID values.
                             If provided, CH_SPECIES_ID is restored after the step.

        Returns:
            Updated state tensor, same shape as x.
        """
        # 1. Perceive
        perception = self.perceive(x)

        # 2. Compute state delta
        ds = self.update_rule(perception)

        # 3. Stochastic per-cell mask — same mask across all channels of a cell
        mask = (
            torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
            < fire_rate
        ).float()
        ds = ds * mask

        # 4. Apply update (out-of-place — keeps autograd graph clean)
        new_x = x + ds * step_size

        # 5. Living cell mask
        #    A cell is alive if any cell in its 3×3 neighbourhood has alpha > 0.1.
        #    Dead cells are zeroed across all channels.
        alive_mask = (
            F.max_pool2d(new_x[:, CH_ALPHA:CH_ALPHA + 1], kernel_size=3, stride=1, padding=1) > 0.1
        ).float()
        new_x = new_x * alive_mask

        # 6. Reconstruct output via torch.cat — avoids all in-place ops on new_x,
        #    which would corrupt the autograd graph when backpropping through unrolled steps.
        #
        #    Channel layout being reconstructed:
        #      0-2  : RGB   (clamped to [0,1])
        #      3    : alpha (clamped to [0,1])
        #      4    : earth (restored to 1.0 — env-managed)
        #      5    : air   (restored to 1.0 — env-managed)
        #      6-9  : water, integrity, age, cell_type (pass through)
        #      10   : species_id (restored from map, or passed through)
        #      11-31: reserved + hidden (pass through)

        sp_id = (
            species_id_map
            if species_id_map is not None
            else new_x[:, CH_SPECIES_ID:CH_SPECIES_ID + 1]
        )

        return torch.cat([
            new_x[:, 0:3].clamp(0.0, 1.0),           # RGB
            new_x[:, 3:4].clamp(0.0, 1.0),           # alpha
            torch.ones_like(new_x[:, 4:5]),           # earth = 1.0
            torch.ones_like(new_x[:, 5:6]),           # air   = 1.0
            new_x[:, 6:10],                           # water, integrity, age, cell_type
            sp_id,                                    # species_id (env-managed)
            new_x[:, 11:],                            # reserved + hidden
        ], dim=1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Seed initialization
# ---------------------------------------------------------------------------

def make_seed(
    batch_size: int,
    device: torch.device,
    n_channels: int = N_CHANNELS,
    grid_h: int = GRID_H,
    grid_w: int = GRID_W,
    seed_positions: list[tuple[int, int]] = SEED_POSITIONS,
    species_id: float = 0.0,
) -> torch.Tensor:
    """
    Create a batch of seed states.

    All channels are 0 except at seed_positions:
        alpha          = 1.0  (cell is alive)
        hidden channels = 1.0  (non-zero initial signal for the network)
        species_id     = species_id (ownership tag)

    RGB stays 0 (black seed, invisible against dark soil).

    Args:
        batch_size   : number of seeds in the batch
        device       : target device
        n_channels   : total channels (must match NCA)
        grid_h/w     : grid dimensions
        seed_positions: list of (row, col) tuples — typically 2 cells
        species_id   : float ID for CH_SPECIES_ID (0.0 for phase-1 single-species)

    Returns:
        Float32 tensor of shape (batch_size, n_channels, grid_h, grid_w)
    """
    x = torch.zeros(batch_size, n_channels, grid_h, grid_w, device=device)

    for row, col in seed_positions:
        x[:, CH_ALPHA, row, col]    = 1.0
        x[:, CH_HIDDEN, row, col]   = 1.0   # slice assignment broadcasts over batch
        x[:, CH_SPECIES_ID, row, col] = species_id

    # Phase-1 env channels
    x[:, CH_EARTH] = 1.0
    x[:, CH_AIR]   = 1.0

    return x


# ---------------------------------------------------------------------------
# Smoke test — run directly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from config import DEVICE

    print(f"Device : {DEVICE}")

    model = NCA().to(DEVICE)
    print(f"Params : {model.parameter_count():,}")

    # Verify Perception output shape
    dummy = torch.zeros(1, N_CHANNELS, GRID_H, GRID_W, device=DEVICE)
    perc  = model.perceive(dummy)
    assert perc.shape == (1, N_CHANNELS * 3, GRID_H, GRID_W), f"Perception shape wrong: {perc.shape}"
    print(f"Perception output : {tuple(perc.shape)}  OK")

    # Verify UpdateRule output shape
    ds = model.update_rule(perc)
    assert ds.shape == (1, N_CHANNELS, GRID_H, GRID_W), f"UpdateRule shape wrong: {ds.shape}"
    print(f"UpdateRule output : {tuple(ds.shape)}  OK")

    # Verify zero initialization — UpdateRule should produce zero delta on any input
    # (env channel restoration happens after the network, so we test ds directly)
    x0 = torch.zeros(1, N_CHANNELS, GRID_H, GRID_W, device=DEVICE)
    with torch.no_grad():
        ds = model.update_rule(model.perceive(x0))
    max_ds = ds.abs().max().item()
    print(f"Max network delta on zero input (should be 0) : {max_ds:.6f}  {'OK' if max_ds < 1e-5 else 'FAIL'}")

    # Run 10 steps from a real seed
    seed = make_seed(batch_size=2, device=DEVICE)
    print(f"\nSeed shape : {tuple(seed.shape)}")
    print(f"Alive cells at step 0 : {(seed[:, CH_ALPHA] > 0.1).sum().item()}")

    x = seed.clone()
    with torch.no_grad():
        for _ in range(10):
            x = model(x, fire_rate=CELL_FIRE_RATE)

    alive_after = (x[:, CH_ALPHA] > 0.1).sum().item()
    print(f"Alive cells after 10 steps : {alive_after}")
    print(f"RGB range  : [{x[:, CH_RGB].min():.3f}, {x[:, CH_RGB].max():.3f}]")
    print(f"Alpha range: [{x[:, CH_ALPHA].min():.3f}, {x[:, CH_ALPHA].max():.3f}]")
    print(f"Earth ch   : {x[:, CH_EARTH].min():.1f} … {x[:, CH_EARTH].max():.1f}  (should be 1.0)")
    print(f"Air ch     : {x[:, CH_AIR].min():.1f} … {x[:, CH_AIR].max():.1f}  (should be 1.0)")
    print("\nAll checks passed.")
