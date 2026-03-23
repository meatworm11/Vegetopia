"""
target.py — Procedural tree target generation.

Each call to make_target() produces a slightly different tree via small random
perturbations to trunk height, canopy size, and root angles. This is intentional:
different species trained with different --seed values will learn distinct shapes.
Diversity during training comes from the pool, not from per-sample target variation.
"""

import numpy as np
import torch

from config import GRID_H, GRID_W, N_CHANNELS, SEED_POSITIONS, CH_CELL_TYPE

# ---------------------------------------------------------------------------
# Preset definitions
# Geometry units: pixels (for radius/width/length) or fractions of GRID_H.
# canopy_aspect = width / height of the canopy ellipse:
#   1.0 = circle, <1.0 = taller than wide, >1.0 = wider than tall
# canopy_noise = max pixel perturbation applied to the canopy radius per angle bin
#   (higher = more ragged/organic edge)
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    'oak': {
        'trunk_height_frac': 0.35,   # trunk height as fraction of GRID_H
        'trunk_width':       3,
        'canopy_radius':     12.0,
        'canopy_aspect':     1.15,   # slightly wider than tall — spreading crown
        'canopy_color':      (0.20, 0.65, 0.20),
        'trunk_color':       (0.45, 0.30, 0.15),
        'root_color':        (0.35, 0.20, 0.10),
        'n_roots':           3,
        'root_length':       4,
        'canopy_noise':      2.5,
        # small random perturbation ranges applied per run:
        'perturb_height':    0.04,   # ± fraction of GRID_H added to trunk_height_frac
        'perturb_radius':    1.5,    # ± pixels added to canopy_radius
    },
    'pine': {
        'trunk_height_frac': 0.46,
        'trunk_width':       2,
        'canopy_radius':     9.0,
        'canopy_aspect':     0.45,   # tall and narrow — conifer silhouette
        'canopy_color':      (0.08, 0.48, 0.14),
        'trunk_color':       (0.38, 0.22, 0.10),
        'root_color':        (0.28, 0.16, 0.07),
        'n_roots':           2,
        'root_length':       3,
        'canopy_noise':      1.5,
        'perturb_height':    0.04,
        'perturb_radius':    1.0,
    },
    'fern': {
        'trunk_height_frac': 0.18,
        'trunk_width':       2,
        'canopy_radius':     11.0,
        'canopy_aspect':     1.50,   # wide and irregular — spreading fronds
        'canopy_color':      (0.18, 0.72, 0.28),
        'trunk_color':       (0.28, 0.45, 0.14),
        'root_color':        (0.22, 0.34, 0.10),
        'n_roots':           3,
        'root_length':       3,
        'canopy_noise':      3.5,    # ragged edge
        'perturb_height':    0.03,
        'perturb_radius':    2.0,
    },
    'bush': {
        'trunk_height_frac': 0.12,
        'trunk_width':       3,
        'canopy_radius':     14.0,
        'canopy_aspect':     1.60,   # very wide, low dome
        'canopy_color':      (0.22, 0.60, 0.18),
        'trunk_color':       (0.45, 0.30, 0.15),
        'root_color':        (0.35, 0.20, 0.10),
        'n_roots':           3,
        'root_length':       4,
        'canopy_noise':      3.0,
        'perturb_height':    0.02,
        'perturb_radius':    2.0,
    },
}

DEFAULT_PRESET = 'oak'


def make_target(preset_name: str = DEFAULT_PRESET,
                rng: np.random.Generator | None = None) -> torch.Tensor:
    """
    Generate a target state tensor of shape (N_CHANNELS, GRID_H, GRID_W).

    Only channels 0-9 carry meaningful values; hidden channels (12-31) are zero.
    Nutrient channels (4, 5) are set to 1.0 (phase 1: unlimited resources).

    Args:
        preset_name: one of the keys in PRESETS ('oak', 'pine', 'fern', 'bush')
        rng: numpy random Generator for reproducibility.
             If None, an unseeded generator is used (non-reproducible).

    Returns:
        Float32 torch.Tensor of shape (N_CHANNELS, GRID_H, GRID_W)
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Choose from: {list(PRESETS)}")
    if rng is None:
        rng = np.random.default_rng()

    p = PRESETS[preset_name].copy()

    # Apply small random perturbations so each training run produces a distinct shape
    p['trunk_height_frac'] = float(np.clip(
        p['trunk_height_frac'] + rng.uniform(-p['perturb_height'], p['perturb_height']),
        0.08, 0.60,
    ))
    p['canopy_radius'] = float(np.clip(
        p['canopy_radius'] + rng.uniform(-p['perturb_radius'], p['perturb_radius']),
        4.0, 20.0,
    ))

    target = np.zeros((N_CHANNELS, GRID_H, GRID_W), dtype=np.float32)

    # Seed positions (bottom cell = root zone, top cell = stem zone)
    seed_row, seed_col = SEED_POSITIONS[0]

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    trunk_height_px  = max(3, int(GRID_H * p['trunk_height_frac']))
    trunk_half_w     = p['trunk_width'] // 2
    trunk_extra      = p['trunk_width'] % 2          # 1 if odd width, else 0
    trunk_top_row    = seed_row - trunk_height_px
    trunk_col_lo     = seed_col - trunk_half_w
    trunk_col_hi     = seed_col + trunk_half_w + trunk_extra  # exclusive

    # Canopy center: one pixel above trunk top so crown sits on trunk
    canopy_cr = trunk_top_row - 1
    canopy_cc = seed_col

    # ------------------------------------------------------------------
    # Build pixel-coordinate arrays
    # ------------------------------------------------------------------
    rows = np.arange(GRID_H)
    cols = np.arange(GRID_W)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')   # both (H, W)

    # ------------------------------------------------------------------
    # Canopy mask — ellipse with per-angle-bin noise for organic edge
    # ------------------------------------------------------------------
    dr = (rr - canopy_cr).astype(float)
    dc = (cc - canopy_cc).astype(float)

    # Ellipse: semi-axis horizontal = radius * aspect, vertical = radius
    aspect      = p['canopy_aspect']
    ellipse_d   = np.sqrt((dc / aspect) ** 2 + dr ** 2)

    n_bins      = 32
    radius_bump = rng.uniform(-p['canopy_noise'], p['canopy_noise'], n_bins)
    angle       = np.arctan2(dr, dc)                        # -π … π
    bin_idx     = ((angle + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    noisy_r     = p['canopy_radius'] + radius_bump[bin_idx]
    canopy_mask = ellipse_d < noisy_r

    # ------------------------------------------------------------------
    # Trunk mask
    # ------------------------------------------------------------------
    trunk_mask = (
        (rr >= trunk_top_row) & (rr <= seed_row) &
        (cc >= trunk_col_lo)  & (cc <  trunk_col_hi)
    )

    # ------------------------------------------------------------------
    # Root mask — 2-3 short lines below the seed, slightly splayed
    # ------------------------------------------------------------------
    root_mask = np.zeros((GRID_H, GRID_W), dtype=bool)
    n_roots   = p['n_roots']

    # Splay angles relative to straight down (π/2 in image coords, +row = down)
    base_angles = np.linspace(-np.pi / 5, np.pi / 5, n_roots)
    for base_ang in base_angles:
        ang    = np.pi / 2 + base_ang + rng.uniform(-0.08, 0.08)
        length = max(2, p['root_length'] + rng.integers(-1, 2))
        for step in range(1, length + 1):
            r = seed_row + int(round(step * np.sin(ang)))
            c = seed_col + int(round(step * np.cos(ang)))
            if 0 <= r < GRID_H and 0 <= c < GRID_W:
                root_mask[r, c] = True

    # ------------------------------------------------------------------
    # Paint channels — order matters: trunk/roots overwrite canopy at overlap
    # ------------------------------------------------------------------
    def _paint(mask, color_rgb, alpha, cell_type):
        target[0][mask] = color_rgb[0]
        target[1][mask] = color_rgb[1]
        target[2][mask] = color_rgb[2]
        target[3][mask] = alpha
        target[CH_CELL_TYPE][mask] = cell_type

    _paint(canopy_mask, p['canopy_color'], 1.0, 0.75)   # leaf
    _paint(trunk_mask,  p['trunk_color'],  1.0, 0.50)   # stem
    _paint(root_mask,   p['root_color'],   1.0, 0.25)   # root

    # Phase 1: nutrients always full
    target[4] = 1.0   # CH_EARTH
    target[5] = 1.0   # CH_AIR

    # Hidden channels and reserved channels stay 0.0
    return torch.from_numpy(target)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_target(preset_name: str = DEFAULT_PRESET,
                     seed: int | None = None) -> torch.Tensor:
    """
    Display a generated target with matplotlib (RGB, cell-type map, alpha).

    Returns the generated target tensor so you can inspect it further.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rng    = np.random.default_rng(seed)
    target = make_target(preset_name, rng)
    t      = target.numpy()

    seed_row, seed_col = SEED_POSITIONS[0]

    # ---- panel 1: RGB composited over white ----
    rgb   = t[:3].transpose(1, 2, 0)           # (H, W, 3)
    alpha = t[3]                                # (H, W)
    composite = rgb * alpha[:, :, None] + np.ones_like(rgb) * (1 - alpha[:, :, None])
    composite = composite.clip(0, 1)

    # ---- panel 2: cell-type map ----
    ct_img = np.ones((GRID_H, GRID_W, 3))      # white background
    ct = t[CH_CELL_TYPE]
    type_colors = {
        0.25: (0.55, 0.27, 0.07),   # root
        0.50: (0.60, 0.40, 0.12),   # stem
        0.75: (0.18, 0.60, 0.18),   # leaf
        1.00: (1.00, 0.84, 0.00),   # flower
    }
    for val, color in type_colors.items():
        mask = np.abs(ct - val) < 0.01
        ct_img[mask] = color

    # ---- panel 3: alpha ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    ax = axes[0]
    ax.imshow(composite, interpolation='nearest')
    ax.plot(seed_col, seed_row,     'r^', markersize=6, label='seed (root)')
    ax.plot(seed_col, seed_row - 1, 'r^', markersize=4, alpha=0.6)
    ax.set_title(f'RGB  ({preset_name})', fontsize=11)
    ax.legend(fontsize=7, loc='upper right')
    ax.axis('off')

    ax = axes[1]
    ax.imshow(ct_img, interpolation='nearest')
    patches = [
        mpatches.Patch(color=(0.55, 0.27, 0.07), label='Root  (0.25)'),
        mpatches.Patch(color=(0.60, 0.40, 0.12), label='Stem  (0.50)'),
        mpatches.Patch(color=(0.18, 0.60, 0.18), label='Leaf  (0.75)'),
    ]
    ax.legend(handles=patches, fontsize=7, loc='upper right')
    ax.set_title('Cell-type map', fontsize=11)
    ax.axis('off')

    ax = axes[2]
    ax.imshow(alpha, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('Alpha channel', fontsize=11)
    ax.axis('off')

    alive_px = int((alpha > 0.5).sum())
    fig.suptitle(
        f'Target: {preset_name}  |  {alive_px} alive pixels  |  '
        f'seed={"random" if seed is None else seed}',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.show()

    return target


# ---------------------------------------------------------------------------
# Quick sanity check — run directly to preview all presets
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    preset = sys.argv[1] if len(sys.argv) > 1 else 'oak'
    seed   = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    print(f"Visualizing preset '{preset}' with seed={seed}")
    t = visualize_target(preset, seed)
    print(f"Target shape : {tuple(t.shape)}")
    print(f"Alive pixels : {(t[3] > 0.5).sum().item()}")
    print(f"Cell types   : root={( (t[9] - 0.25).abs() < 0.01 ).sum().item()} "
          f"stem={(  (t[9] - 0.50).abs() < 0.01 ).sum().item()} "
          f"leaf={( (t[9] - 0.75).abs() < 0.01 ).sum().item()}")
