"""
run_simulation.py — Multi-species NCA viewer.

Entry points
------------
    # From the command line:
    python run_simulation.py oak
    python run_simulation.py oak pine fern

    # From train_species.py (--preview flag):
    from run_simulation import run_viewer
    run_viewer(['oak'])

Controls
--------
    Left-click   place a new seed at cursor
    Right-click  kill cells in a 3×3 area at cursor
    Space        pause / unpause
    S            step one frame while paused
    R            reset all simulations
    +  /  =      speed up  (more NCA steps per frame)
    -            slow down (fewer NCA steps per frame)
    Q / Escape   quit
"""

from __future__ import annotations

import sys
from pathlib import Path

import pygame
import torch

from config import GRID_H, GRID_W, N_CHANNELS
from model import NCA
from viewer import SpeciesViewer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELL_SIZE        = 8          # pixels per NCA cell
FPS_TARGET       = 30
BG_COLOUR        = (20, 20, 20)
LABEL_COLOUR     = (200, 200, 200)
PANEL_PADDING    = 12         # px between panels
LABEL_HEIGHT     = 22         # px reserved below each panel for the species name
MIN_STEPS_FRAME  = 1
MAX_STEPS_FRAME  = 16


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(name: str, device: torch.device) -> NCA:
    """
    Load a trained NCA from species/{name}.pt.

    Raises FileNotFoundError with a helpful message if the file is missing.
    """
    path = Path('species') / f'{name}.pt'
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Train it first with:\n"
            f"  python train_species.py --name {name} --preset {name} --steps 5000"
        )

    ckpt = torch.load(path, map_location=device, weights_only=True)

    n_ch = ckpt.get('n_channels', N_CHANNELS)
    model = NCA(n_channels=n_ch).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    trained_steps = ckpt.get('step', '?')
    final_loss    = ckpt.get('loss')
    loss_str      = f'{final_loss:.5f}' if final_loss is not None else 'n/a'
    print(f"  Loaded '{name}' — {trained_steps} steps, loss {loss_str}")

    return model


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _layout(n_species: int, cell_size: int) -> tuple[int, int, list[int]]:
    """
    Return (window_width, window_height, list_of_panel_x_offsets).

    Panels are laid out in a single row.
    """
    panel_w = GRID_W * cell_size
    panel_h = GRID_H * cell_size + LABEL_HEIGHT

    total_w = n_species * panel_w + (n_species + 1) * PANEL_PADDING
    total_h = panel_h + 2 * PANEL_PADDING

    xs = [PANEL_PADDING + i * (panel_w + PANEL_PADDING) for i in range(n_species)]
    return total_w, total_h, xs


# ---------------------------------------------------------------------------
# Main viewer loop
# ---------------------------------------------------------------------------

def run_viewer(species_names: list[str]) -> None:
    """
    Open a pygame window and display one NCA per species side-by-side.

    Args:
        species_names : list of trained species names (must have species/{name}.pt)
    """
    if not species_names:
        print("run_viewer: no species specified.")
        return

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Device : {device}")
    print(f"Loading {len(species_names)} species...")

    # Load models
    models = {}
    for name in species_names:
        models[name] = _load_model(name, device)

    # pygame setup
    pygame.init()
    pygame.display.set_caption('Vegetopia — Neural Plant CA')

    win_w, win_h, panel_xs = _layout(len(species_names), CELL_SIZE)
    screen = pygame.display.set_mode((win_w, win_h))
    clock  = pygame.time.Clock()

    try:
        font = pygame.font.SysFont('monospace', 13)
    except Exception:
        font = pygame.font.Font(None, 16)

    # Create viewers
    viewers: list[SpeciesViewer] = [
        SpeciesViewer(
            name=name,
            model=models[name],
            device=device,
            cell_size=CELL_SIZE,
            bg=BG_COLOUR,
        )
        for name in species_names
    ]

    steps_per_frame = 1
    paused          = False
    running         = True

    def _hit_viewer(mx: int, my: int):
        """Return (viewer, local_px, local_py) for the panel under the cursor, or None."""
        panel_top = PANEL_PADDING
        for v, px in zip(viewers, panel_xs):
            lx = mx - px
            ly = my - panel_top
            if 0 <= lx < v.pixel_w and 0 <= ly < v.pixel_h:
                return v, lx, ly
        return None

    while running:
        # --- Events -----------------------------------------------------------
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif ev.key == pygame.K_SPACE:
                    paused = not paused

                elif ev.key == pygame.K_s and paused:
                    for v in viewers:
                        v.step(1)

                elif ev.key == pygame.K_r:
                    for v in viewers:
                        v.reset()

                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    steps_per_frame = min(steps_per_frame * 2, MAX_STEPS_FRAME)

                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    steps_per_frame = max(steps_per_frame // 2, MIN_STEPS_FRAME)

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                hit = _hit_viewer(*ev.pos)
                if hit is not None:
                    v, lx, ly = hit
                    row, col = v.pixel_to_grid(lx, ly)
                    if ev.button == 1:      # left-click → place seed
                        v.place_seed(row, col)
                    elif ev.button == 3:    # right-click → kill 3×3
                        v.kill_area(row, col, radius=1)

        # --- Simulate ---------------------------------------------------------
        if not paused:
            for v in viewers:
                v.step(steps_per_frame)

        # --- Draw -------------------------------------------------------------
        screen.fill(BG_COLOUR)

        panel_top = PANEL_PADDING
        for v, px in zip(viewers, panel_xs):
            v.draw(screen, px, panel_top)

            # Species label below the panel
            label = font.render(
                f'{v.name}  (step {v.step_count:,})', True, LABEL_COLOUR
            )
            label_y = panel_top + GRID_H * CELL_SIZE + 4
            screen.blit(label, (px, label_y))

        # Status bar
        pause_str = 'PAUSED' if paused else 'running'
        spd_label = font.render(
            f'{pause_str}  speed: {steps_per_frame}×  [+/-]  |  '
            f'Space=pause  S=step  R=reset  LMB=seed  RMB=kill  Q=quit',
            True, (120, 120, 120),
        )
        screen.blit(spd_label, (PANEL_PADDING, 4))

        pygame.display.flip()
        clock.tick(FPS_TARGET)

    pygame.quit()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Vegetopia NCA viewer')
    parser.add_argument(
        'species', nargs='*',
        help='Species name(s) to display',
    )
    parser.add_argument(
        '--species', dest='species_flag', nargs='+', metavar='NAME',
        help='Species name(s) to display (alternative to positional args)',
    )
    args = parser.parse_args()

    names = args.species_flag or args.species
    if not names:
        parser.print_help()
        sys.exit(1)

    run_viewer(names)


if __name__ == '__main__':
    main()
