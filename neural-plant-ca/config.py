import torch

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
GRID_H = 64
GRID_W = 64

# ---------------------------------------------------------------------------
# State vector — 32 channels per cell
# ---------------------------------------------------------------------------
N_CHANNELS = 32
HIDDEN_CHANNELS = 20  # channels 12-31

# Named channel indices / slices (use these everywhere instead of raw numbers)
CH_RGB        = slice(0, 3)   # R, G, B
CH_ALPHA      = 3             # >0.1 means alive/mature
CH_EARTH      = 4             # earth nutrients (clamped to 1.0 in phase 1)
CH_AIR        = 5             # air nutrients   (clamped to 1.0 in phase 1)
CH_WATER      = 6             # reserved — phase 4
CH_INTEGRITY  = 7             # structural integrity — phase 2
CH_AGE        = 8             # incremented each step
CH_CELL_TYPE  = 9             # 0=unspecialized, 0.25=root, 0.5=stem, 0.75=leaf, 1.0=flower
CH_SPECIES_ID = 10            # environment-managed: 0=empty, 1=sp1, 2=sp2, ...
CH_RESERVED   = 11            # reserved
CH_HIDDEN     = slice(12, 32) # learned hidden state

# Phase-1 environment-managed channels: NCA may read but must not write these.
# They are restored to fixed values after every NCA step.
ENV_CHANNELS_PHASE1 = {
    CH_EARTH:      1.0,   # unlimited nutrients
    CH_AIR:        1.0,
    # CH_SPECIES_ID is restored per-cell from a separate ownership tensor
}

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
CELL_FIRE_RATE     = 0.5        # stochastic update probability per cell per step
TRAIN_STEPS_RANGE  = (80, 128)  # random unroll length per training iteration
POOL_SIZE          = 1024
BATCH_SIZE         = 8
LEARNING_RATE      = 1e-3
N_TRAINING_STEPS   = 10000

# ---------------------------------------------------------------------------
# Seed positions  (row, col) — index 0 is the bottom cell (future root zone)
# ---------------------------------------------------------------------------
SEED_POSITIONS = [
    (GRID_H - GRID_H // 4,     GRID_W // 2),   # (48, 32) — root zone
    (GRID_H - GRID_H // 4 - 1, GRID_W // 2),   # (47, 32) — stem zone
]

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = get_device()
