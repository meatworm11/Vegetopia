import torch

# ---------------------------------------------------------------------------
# Grid — training dimensions (used by model, target, training loop)
# ---------------------------------------------------------------------------
GRID_H = 64
GRID_W = 64

# ---------------------------------------------------------------------------
# Grid — simulation / viewer dimensions (larger landscape for display)
# ---------------------------------------------------------------------------
SIM_GRID_H      = 128
SIM_GRID_W      = 256
CELL_RENDER_SIZE = 4    # pixels per cell on screen → window is 1024×512

# ---------------------------------------------------------------------------
# Environment cell types (background grid, NOT plant cell_type channel 9)
# ---------------------------------------------------------------------------
ENV_EMPTY = 0    # void / air above ground
ENV_SOIL  = 1    # basic soil
ENV_ROCK  = 2    # immovable, future nutrient/integrity source
ENV_SUN   = 3    # top row, future air nutrient source
ENV_WATER = 4    # reserved for future

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
# Energy system (plant nutrient absorption / dissipation / death)
# ---------------------------------------------------------------------------
ABSORPTION_RATE       = 0.05     # nutrients absorbed per step by root/leaf cells
DISSIPATION_RATE      = 0.002    # nutrients lost per step per alive cell (maintenance)
DEATH_THRESHOLD       = 0.01     # cell dies based on cell-type-specific rules
NUTRIENTS_ENABLED     = False    # False = phase-1 (channels 4-5 clamped to 1.0)
ROOT_LOSS_MASK        = True     # exclude root cells from shape loss during training
NUTRIENT_TRANSPORT_RATE = 0.20   # fraction of nutrients shared with poorer neighbours
NUTRIENT_TRANSPORT_STEPS = 2     # transport passes per simulation step

# ---------------------------------------------------------------------------
# Structural integrity and gravity
# ---------------------------------------------------------------------------
INTEGRITY_DECAY_ROOT  = 1.0    # integrity lost per hop for root cells
INTEGRITY_DECAY_STEM  = 2.0    # integrity lost per hop for stem cells
INTEGRITY_DECAY_LEAF  = 3.0    # integrity lost per hop for leaf cells
INTEGRITY_SOIL        = 50.0   # integrity provided by ENV_SOIL
INTEGRITY_ROCK        = 100.0  # integrity provided by ENV_ROCK
INTEGRITY_STEPS       = 3      # propagation passes per simulation step
GRAVITY_ENABLED       = False  # False = phase-1 (plants float freely)

# ---------------------------------------------------------------------------
# Growth cap
# ---------------------------------------------------------------------------
MAX_CELLS             = 800    # max alive cells per species; new cells reverted above this

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
