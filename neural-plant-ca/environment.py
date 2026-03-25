"""
environment.py — Environment grid and nutrient diffusion.

The environment has two layers:
  1. env_type_grid (int32, H×W)  — terrain type per cell (soil, rock, air, sun)
  2. nutrient_grid (float32, 2×H×W) — channel 0 = earth nutrients, channel 1 = air nutrients

Nutrient sources regenerate each step, then nutrients diffuse through compatible
cell types, creating natural gradients:
  - Earth nutrients: strongest near bedrock/rock, fade upward through soil only
  - Air nutrients: strongest at sun row, fade downward through air only

Layout (top to bottom):
    row 0           : ENV_SUN
    rows 1 .. 79%   : ENV_EMPTY (sky/air)
    rows 80% .. H-2 : ENV_SOIL (with scattered ENV_ROCK)
    row H-1         : ENV_ROCK (bedrock)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    ENV_EMPTY, ENV_SOIL, ENV_ROCK, ENV_SUN,
    CH_ALPHA, CH_EARTH, CH_AIR, CH_CELL_TYPE, CH_INTEGRITY,
    ABSORPTION_RATE, DISSIPATION_RATE, DEATH_THRESHOLD,
    NUTRIENT_TRANSPORT_RATE,
    INTEGRITY_DECAY_ROOT, INTEGRITY_DECAY_STEM, INTEGRITY_DECAY_LEAF,
    INTEGRITY_SOIL, INTEGRITY_ROCK,
)

# Nutrient channel indices
NUTRIENT_EARTH = 0
NUTRIENT_AIR   = 1

# Diffusion parameters
DIFFUSION_DECAY = 0.9997


# ---------------------------------------------------------------------------
# Terrain grid creation
# ---------------------------------------------------------------------------

def create_environment(
    height: int,
    width: int,
    rock_density: float = 0.06,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Create an environment grid of shape (height, width) with dtype int32.

    Returns:
        int32 ndarray where each cell is one of ENV_EMPTY/SOIL/ROCK/SUN.
    """
    if rng is None:
        rng = np.random.default_rng()

    grid = np.full((height, width), ENV_EMPTY, dtype=np.int32)

    # Top row = sun
    grid[0, :] = ENV_SUN

    # Bottom 20% = soil
    soil_top = height - height // 5
    grid[soil_top:, :] = ENV_SOIL

    # Bottom row = bedrock (rock)
    grid[-1, :] = ENV_ROCK

    # Scatter rocks in the soil layer (excluding bedrock row)
    soil_region = grid[soil_top:-1, :]
    rock_mask = rng.random(soil_region.shape) < rock_density
    soil_region[rock_mask] = ENV_ROCK

    return grid


def soil_surface_row(env_grid: np.ndarray) -> int:
    """
    Return the row index of the first soil/rock row (the soil surface).
    """
    height = env_grid.shape[0]
    for r in range(height):
        if env_grid[r, 0] in (ENV_SOIL, ENV_ROCK):
            return r
    return height - 1


# ---------------------------------------------------------------------------
# EnvironmentState
# ---------------------------------------------------------------------------

class EnvironmentState:
    """
    Holds the terrain grid and nutrient tensors, plus precomputed diffusion masks.

    All tensors live on the specified device (default CPU).
    """

    def __init__(
        self,
        env_type_grid: np.ndarray,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        self.height, self.width = env_type_grid.shape
        self.device = device

        # Keep the numpy grid for environment queries
        self.env_type_grid = env_type_grid

        # Nutrient grid: (2, H, W) — ch0 = earth, ch1 = air
        self.nutrient_grid = init_nutrients(env_type_grid, device)

        # Precompute diffusion masks (float, 1×1×H×W for conv compatibility)
        env_t = torch.from_numpy(env_type_grid).to(device)

        # Earth diffuses through SOIL and ROCK
        self._earth_mask = (
            (env_t == ENV_SOIL) | (env_t == ENV_ROCK)
        ).float().unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

        # Air diffuses through EMPTY and SUN
        self._air_mask = (
            (env_t == ENV_EMPTY) | (env_t == ENV_SUN)
        ).float().unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

        # Source masks for regeneration
        self._rock_mask = (env_t == ENV_ROCK).float()   # (H, W)
        self._sun_mask  = (env_t == ENV_SUN).float()    # (H, W)

        # 4-neighbor + self diffusion kernel (cross pattern, equal weight)
        # Each cell = average of self + 4 cardinal neighbours.
        kernel = torch.tensor([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], device=device, dtype=torch.float32) / 5.0
        self._kernel = kernel.unsqueeze(0).unsqueeze(0)   # (1, 1, 3, 3)


# ---------------------------------------------------------------------------
# Nutrient initialisation
# ---------------------------------------------------------------------------

def init_nutrients(
    env_type_grid: np.ndarray,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Create the initial nutrient grid (2, H, W).

    - ENV_ROCK cells → earth nutrients = 1.0
    - ENV_SUN  cells → air nutrients   = 1.0
    - Everything else → 0.0
    """
    h, w = env_type_grid.shape
    nutrients = torch.zeros(2, h, w, device=device)

    env_t = torch.from_numpy(env_type_grid).to(device)
    nutrients[NUTRIENT_EARTH][env_t == ENV_ROCK] = 1.0
    nutrients[NUTRIENT_AIR][env_t == ENV_SUN]    = 1.0

    return nutrients


# ---------------------------------------------------------------------------
# Source regeneration
# ---------------------------------------------------------------------------

def regenerate_sources(state: EnvironmentState) -> None:
    """
    Reset nutrient levels at source cells every step.

    - ENV_ROCK → earth = 1.0
    - ENV_SUN  → air   = 1.0
    """
    state.nutrient_grid[NUTRIENT_EARTH] = torch.where(
        state._rock_mask > 0,
        torch.ones_like(state.nutrient_grid[NUTRIENT_EARTH]),
        state.nutrient_grid[NUTRIENT_EARTH],
    )
    state.nutrient_grid[NUTRIENT_AIR] = torch.where(
        state._sun_mask > 0,
        torch.ones_like(state.nutrient_grid[NUTRIENT_AIR]),
        state.nutrient_grid[NUTRIENT_AIR],
    )


# ---------------------------------------------------------------------------
# Nutrient diffusion
# ---------------------------------------------------------------------------

def diffuse_nutrients(state: EnvironmentState, n_steps: int = 1) -> None:
    """
    Diffuse nutrients through compatible cell types.

    Each step: 3×3 average (zero-padded) × type mask × decay.
    Run multiple steps per sim tick for faster spreading.
    """
    kernel = state._kernel

    for _ in range(n_steps):
        # --- Earth nutrients: diffuse through soil/rock only ---
        earth = state.nutrient_grid[NUTRIENT_EARTH].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        earth_padded = F.pad(earth, (1, 1, 1, 1), mode='constant', value=0)
        earth_avg = F.conv2d(earth_padded, kernel)
        # Mask: only update cells that are soil/rock; others stay 0
        earth_new = earth_avg * state._earth_mask * DIFFUSION_DECAY
        state.nutrient_grid[NUTRIENT_EARTH] = earth_new.squeeze(0).squeeze(0)

        # --- Air nutrients: diffuse through air/sun only ---
        air = state.nutrient_grid[NUTRIENT_AIR].unsqueeze(0).unsqueeze(0)
        air_padded = F.pad(air, (1, 1, 1, 1), mode='constant', value=0)
        air_avg = F.conv2d(air_padded, kernel)
        air_new = air_avg * state._air_mask * DIFFUSION_DECAY
        state.nutrient_grid[NUTRIENT_AIR] = air_new.squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# Plant ↔ environment nutrient exchange
# ---------------------------------------------------------------------------

def plant_absorb_nutrients(
    plant_grid: torch.Tensor,
    state: EnvironmentState,
    rate: float | None = None,
) -> None:
    """
    Alive root cells absorb earth nutrients; alive leaf cells absorb air nutrients.

    plant_grid shape: (1, C, H, W)  — the NCA state tensor.
    Modifies plant_grid channels 4-5 and state.nutrient_grid in-place.

    Args:
        rate: absorption rate override (default: ABSORPTION_RATE from config)
    """
    abs_rate = rate if rate is not None else ABSORPTION_RATE
    with torch.no_grad():
        alive = plant_grid[0, CH_ALPHA] > 0.1                # (H, W) bool
        ctype = plant_grid[0, CH_CELL_TYPE]                   # (H, W)

        # --- Root absorption (earth nutrients from soil/rock) ---
        is_root = alive & (ctype > 0.10) & (ctype < 0.40)    # (H, W)
        if is_root.any():
            env_earth = state.nutrient_grid[NUTRIENT_EARTH]   # (H, W)
            available = env_earth * is_root.float()
            absorbed  = torch.minimum(
                available,
                torch.full_like(available, abs_rate),
            )
            state.nutrient_grid[NUTRIENT_EARTH] = env_earth - absorbed
            plant_grid[0, CH_EARTH] = (plant_grid[0, CH_EARTH] + absorbed).clamp(0, 1)

        # --- Leaf absorption (air nutrients from air/sun) ---
        is_leaf = alive & (ctype > 0.60) & (ctype < 0.90)    # (H, W)
        if is_leaf.any():
            env_air = state.nutrient_grid[NUTRIENT_AIR]
            available = env_air * is_leaf.float()
            absorbed  = torch.minimum(
                available,
                torch.full_like(available, abs_rate),
            )
            state.nutrient_grid[NUTRIENT_AIR] = env_air - absorbed
            plant_grid[0, CH_AIR] = (plant_grid[0, CH_AIR] + absorbed).clamp(0, 1)


def plant_dissipate_energy(
    plant_grid: torch.Tensor,
    rate: float | None = None,
) -> None:
    """
    Every alive cell loses a small amount of both nutrients per step (maintenance).

    Args:
        rate: dissipation rate override (default: DISSIPATION_RATE from config)
    """
    dissip_rate = rate if rate is not None else DISSIPATION_RATE
    with torch.no_grad():
        alive = (plant_grid[0, CH_ALPHA] > 0.1).float()      # (H, W)
        plant_grid[0, CH_EARTH] = (
            plant_grid[0, CH_EARTH] - dissip_rate * alive
        ).clamp(0)
        plant_grid[0, CH_AIR] = (
            plant_grid[0, CH_AIR] - dissip_rate * alive
        ).clamp(0)


def plant_death_check(plant_grid: torch.Tensor) -> None:
    """
    Cell-type-specific death conditions:
      - Root  (≈0.25): dies if earth < threshold
      - Stem  (≈0.50): dies if earth < threshold
      - Leaf  (≈0.75): dies if earth OR air < threshold
      - Flower(≈1.00): dies if earth OR air < threshold
      - Unspecialized:  dies if BOTH earth AND air < threshold (legacy rule)
    """
    with torch.no_grad():
        alive = plant_grid[0, CH_ALPHA] > 0.1
        ctype = plant_grid[0, CH_CELL_TYPE]
        low_e = plant_grid[0, CH_EARTH] < DEATH_THRESHOLD
        low_a = plant_grid[0, CH_AIR]   < DEATH_THRESHOLD

        is_root   = alive & (ctype > 0.10) & (ctype < 0.40)
        is_stem   = alive & (ctype > 0.40) & (ctype < 0.60)
        is_leaf   = alive & (ctype > 0.60) & (ctype < 0.90)
        is_flower = alive & (ctype > 0.90)
        is_other  = alive & ~is_root & ~is_stem & ~is_leaf & ~is_flower

        dying = (
            (is_root & low_e)
            | (is_stem & low_e)
            | (is_leaf & (low_e | low_a))
            | (is_flower & (low_e | low_a))
            | (is_other & low_e & low_a)
        )
        if dying.any():
            plant_grid[0, :, dying] = 0.0


# ---------------------------------------------------------------------------
# Nutrient transport between plant cells
# ---------------------------------------------------------------------------

def transport_nutrients(plant_grid: torch.Tensor, n_steps: int = 1) -> None:
    """
    Each alive plant cell shares NUTRIENT_TRANSPORT_RATE of its nutrients with
    neighbouring alive cells that have less of that nutrient.

    This creates flow: roots pass earth up through stems to leaves,
    leaves pass air down through stems to roots.

    plant_grid shape: (1, C, H, W)
    """
    with torch.no_grad():
        for _ in range(n_steps):
            alive = (plant_grid[0, CH_ALPHA] > 0.1).float()  # (H, W)

            for ch in (CH_EARTH, CH_AIR):
                vals = plant_grid[0, ch]  # (H, W)

                # Pad with zeros (non-alive boundary)
                padded = F.pad(vals.unsqueeze(0).unsqueeze(0),
                               (1, 1, 1, 1), mode='constant', value=0)
                alive_padded = F.pad(alive.unsqueeze(0).unsqueeze(0),
                                     (1, 1, 1, 1), mode='constant', value=0)

                # 4 cardinal neighbours
                neighbours = [
                    padded[:, :, 0:-2, 1:-1],  # up
                    padded[:, :, 2:,   1:-1],  # down
                    padded[:, :, 1:-1, 0:-2],  # left
                    padded[:, :, 1:-1, 2:],    # right
                ]
                alive_nb = [
                    alive_padded[:, :, 0:-2, 1:-1],
                    alive_padded[:, :, 2:,   1:-1],
                    alive_padded[:, :, 1:-1, 0:-2],
                    alive_padded[:, :, 1:-1, 2:],
                ]

                vals_4d = vals.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

                # Accumulate donation to neighbours and reception from neighbours
                total_give = torch.zeros_like(vals)
                total_recv = torch.zeros_like(vals)

                for nb_val, nb_alive in zip(neighbours, alive_nb):
                    # This cell gives to neighbour if neighbour is alive and has less
                    diff = vals_4d - nb_val  # positive = we have more
                    give_mask = (diff > 0) & (alive.unsqueeze(0).unsqueeze(0) > 0) & (nb_alive > 0)
                    give = diff * NUTRIENT_TRANSPORT_RATE * give_mask.float()
                    total_give += give.squeeze(0).squeeze(0)

                    # This cell receives from neighbour (neighbour has more)
                    recv_mask = (diff < 0) & (alive.unsqueeze(0).unsqueeze(0) > 0) & (nb_alive > 0)
                    recv = (-diff) * NUTRIENT_TRANSPORT_RATE * recv_mask.float()
                    total_recv += recv.squeeze(0).squeeze(0)

                plant_grid[0, ch] = (vals - total_give + total_recv).clamp(0, 1)


# ---------------------------------------------------------------------------
# Structural integrity
# ---------------------------------------------------------------------------

def update_structural_integrity(
    plant_grid: torch.Tensor,
    env_type_grid: np.ndarray,
    n_steps: int = 1,
) -> None:
    """
    Propagate structural integrity through plant cells.

    - ENV_ROCK cells provide integrity=10.0
    - ENV_SOIL cells provide integrity=5.0
    - Each alive plant cell inherits MAX integrity from 3×3 neighbourhood minus
      a cell-type-dependent decay (root=1.0, stem=2.0, leaf=3.0).
    - Run multiple passes per step for faster percolation.

    plant_grid shape: (1, C, H, W)
    """
    with torch.no_grad():
        h, w = plant_grid.shape[2], plant_grid.shape[3]
        device = plant_grid.device
        env_t = torch.from_numpy(env_type_grid).to(device)

        alive = plant_grid[0, CH_ALPHA] > 0.1  # (H, W)
        ctype = plant_grid[0, CH_CELL_TYPE]     # (H, W)

        # Compute per-cell decay based on cell type
        decay = torch.full((h, w), 3.0, device=device)  # default = leaf decay
        is_root = (ctype > 0.10) & (ctype < 0.40)
        is_stem = (ctype > 0.40) & (ctype < 0.60)
        is_leaf = (ctype > 0.60) & (ctype < 0.90)
        decay[is_root] = INTEGRITY_DECAY_ROOT
        decay[is_stem] = INTEGRITY_DECAY_STEM
        decay[is_leaf] = INTEGRITY_DECAY_LEAF

        # Ground integrity sources
        ground_integrity = torch.zeros(h, w, device=device)
        ground_integrity[env_t == ENV_ROCK] = INTEGRITY_ROCK
        ground_integrity[env_t == ENV_SOIL] = INTEGRITY_SOIL

        for _ in range(n_steps):
            integrity = plant_grid[0, CH_INTEGRITY]  # (H, W)

            # Start with ground sources (always reset)
            integrity = torch.where(
                (env_t == ENV_ROCK) | (env_t == ENV_SOIL),
                ground_integrity,
                integrity,
            )

            # Max-pool 3×3 to find max integrity in neighbourhood
            int_4d = integrity.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            max_nb = F.max_pool2d(
                F.pad(int_4d, (1, 1, 1, 1), mode='constant', value=0),
                kernel_size=3, stride=1, padding=0,
            ).squeeze(0).squeeze(0)  # (H, W)

            # Each alive cell gets max_neighbour - decay
            new_integrity = (max_nb - decay).clamp(min=0)

            # Only alive plant cells get integrity; dead cells = 0
            plant_grid[0, CH_INTEGRITY] = new_integrity * alive.float()


# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------

def apply_gravity(
    plant_grid: torch.Tensor,
    env_type_grid: np.ndarray,
) -> None:
    """
    Any alive plant cell with structural integrity <= 0.0 AND with an empty cell
    below it (ENV_EMPTY, no plant) falls one cell down.

    Process bottom-to-top so a column of unsupported cells falls together.
    Cells on soil or on other plant cells with positive integrity don't fall.

    plant_grid shape: (1, C, H, W)
    """
    with torch.no_grad():
        h, w = plant_grid.shape[2], plant_grid.shape[3]
        device = plant_grid.device
        env_t = torch.from_numpy(env_type_grid).to(device)

        # Process from second-to-last row up to row 0 (bottom to top)
        for r in range(h - 2, -1, -1):
            alive_row = plant_grid[0, CH_ALPHA, r] > 0.1         # (W,) bool
            low_integrity = plant_grid[0, CH_INTEGRITY, r] <= 0.0  # (W,) bool

            # Below must be empty terrain AND no plant cell there
            below_empty_terrain = (env_t[r + 1] == ENV_EMPTY) | (env_t[r + 1] == ENV_SUN)
            below_no_plant = plant_grid[0, CH_ALPHA, r + 1] <= 0.1

            can_fall = alive_row & low_integrity & below_empty_terrain & below_no_plant

            if can_fall.any():
                cols = can_fall.nonzero(as_tuple=True)[0]
                # Copy state vector down, zero original
                plant_grid[0, :, r + 1, cols] = plant_grid[0, :, r, cols]
                plant_grid[0, :, r, cols] = 0.0
