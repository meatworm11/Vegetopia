"""
run_simulation.py — Multi-species NCA viewer with environment and nutrient diffusion.

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
    Left-click   place a new seed at cursor (snapped to soil surface)
    Right-click  kill cells in a 3×3 area at cursor
    Space        pause / unpause
    S            step one frame while paused
    R            reset all simulations
    D            toggle nutrient debug overlay
    E            toggle energy system (nutrients_enabled)
    G            toggle gravity (structural integrity + falling)
    +  /  =      speed up  (more NCA steps per frame)
    -            slow down (fewer NCA steps per frame)
    Q / Escape   quit
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pygame
import torch

from config import (
    SIM_GRID_H, SIM_GRID_W, CELL_RENDER_SIZE, N_CHANNELS,
    NUTRIENTS_ENABLED, GRAVITY_ENABLED,
    NUTRIENT_TRANSPORT_STEPS, INTEGRITY_STEPS,
)
from environment import (
    create_environment, soil_surface_row,
    EnvironmentState, regenerate_sources, diffuse_nutrients,
    plant_absorb_nutrients, plant_dissipate_energy, plant_death_check,
    transport_nutrients, update_structural_integrity, apply_gravity,
)
from model import NCA
from viewer import SpeciesViewer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FPS_TARGET            = 30
BG_COLOUR             = (20, 20, 20)
LABEL_COLOUR          = (200, 200, 200)
PANEL_PADDING         = 12
LABEL_HEIGHT          = 22
MIN_STEPS_FRAME       = 1
MAX_STEPS_FRAME       = 16
DIFFUSION_SUBSTEPS    = 4     # nutrient diffusion steps per sim tick


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(name: str, device: torch.device) -> NCA:
    """
    Load a trained NCA from species/{name}.pt.
    """
    path = Path('species') / f'{name}.pt'
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Train it first with:\n"
            f"  python train_species.py --name {name} --preset {name} --steps 10000"
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

def _layout(cell_size: int) -> tuple[int, int]:
    """Return (window_width, window_height) for a single shared panel."""
    panel_w = SIM_GRID_W * cell_size
    panel_h = SIM_GRID_H * cell_size + LABEL_HEIGHT
    total_w = panel_w + 2 * PANEL_PADDING
    total_h = panel_h + 2 * PANEL_PADDING
    return total_w, total_h


# ---------------------------------------------------------------------------
# Main viewer loop
# ---------------------------------------------------------------------------

def run_viewer(species_names: list[str]) -> None:
    """
    Open a pygame window and display NCA species on a shared environment
    with nutrient diffusion running each frame.
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
    print(f"Grid   : {SIM_GRID_H}×{SIM_GRID_W}  ({CELL_RENDER_SIZE}px/cell)")
    print(f"Loading {len(species_names)} species...")

    # Load models
    models = {}
    for name in species_names:
        models[name] = _load_model(name, device)

    # Create shared environment with nutrient state
    env_grid = create_environment(SIM_GRID_H, SIM_GRID_W, rng=np.random.default_rng(42))
    soil_row = soil_surface_row(env_grid)
    env_state = EnvironmentState(env_grid, device=device)
    print(f"Soil surface at row {soil_row}")

    # Let nutrients reach steady-state before first frame
    # (~10k effective diffusion steps to fill 100+ rows of air)
    print("Initialising nutrient gradients...")
    for _ in range(2000):
        regenerate_sources(env_state)
        diffuse_nutrients(env_state, n_steps=5)
    print("  done.")

    # pygame setup
    pygame.init()
    pygame.display.set_caption('Vegetopia — Neural Plant CA')

    win_w, win_h = _layout(CELL_RENDER_SIZE)
    screen = pygame.display.set_mode((win_w, win_h))
    clock  = pygame.time.Clock()

    try:
        font = pygame.font.SysFont('monospace', 13)
    except Exception:
        font = pygame.font.Font(None, 16)

    # Create viewers — all share the same environment state
    viewers: list[SpeciesViewer] = [
        SpeciesViewer(
            name=name,
            model=models[name],
            env_state=env_state,
            soil_row=soil_row,
            device=device,
            cell_size=CELL_RENDER_SIZE,
        )
        for name in species_names
    ]

    panel_x         = PANEL_PADDING
    panel_top       = PANEL_PADDING
    steps_per_frame    = 1
    paused             = False
    show_nutrients     = False
    nutrients_enabled  = NUTRIENTS_ENABLED
    gravity_enabled    = GRAVITY_ENABLED
    running            = True

    def _hit_viewer(mx: int, my: int):
        """Return (viewer, local_px, local_py) if cursor is inside the panel."""
        for v in viewers:
            lx = mx - panel_x
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
                    # Single step: advance all systems
                    regenerate_sources(env_state)
                    diffuse_nutrients(env_state, n_steps=DIFFUSION_SUBSTEPS)
                    if nutrients_enabled:
                        for v in viewers:
                            plant_absorb_nutrients(v._state, env_state)
                            plant_dissipate_energy(v._state)
                            transport_nutrients(v._state, n_steps=NUTRIENT_TRANSPORT_STEPS)
                    if gravity_enabled:
                        for v in viewers:
                            update_structural_integrity(v._state, env_state.env_type_grid, n_steps=INTEGRITY_STEPS)
                            apply_gravity(v._state, env_state.env_type_grid)
                    for v in viewers:
                        v.step(1, nutrients_enabled=nutrients_enabled)
                    if nutrients_enabled:
                        for v in viewers:
                            plant_death_check(v._state)

                elif ev.key == pygame.K_r:
                    for v in viewers:
                        v.reset()

                elif ev.key == pygame.K_d:
                    show_nutrients = not show_nutrients

                elif ev.key == pygame.K_e:
                    nutrients_enabled = not nutrients_enabled
                    print(f"Energy system: {'ON' if nutrients_enabled else 'OFF'}")

                elif ev.key == pygame.K_g:
                    gravity_enabled = not gravity_enabled
                    print(f"Gravity: {'ON' if gravity_enabled else 'OFF'}")

                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    steps_per_frame = min(steps_per_frame * 2, MAX_STEPS_FRAME)

                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    steps_per_frame = max(steps_per_frame // 2, MIN_STEPS_FRAME)

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                hit = _hit_viewer(*ev.pos)
                if hit is not None:
                    v, lx, ly = hit
                    row, col = v.pixel_to_grid(lx, ly)
                    if ev.button == 1:
                        v.place_seed(row, col)
                    elif ev.button == 3:
                        v.kill_area(row, col, radius=1)

        # --- Simulate ---------------------------------------------------------
        if not paused:
            # Nutrient diffusion (runs every frame regardless of NCA)
            regenerate_sources(env_state)
            diffuse_nutrients(env_state, n_steps=DIFFUSION_SUBSTEPS)

            # Plant absorbs nutrients before NCA step
            if nutrients_enabled:
                for v in viewers:
                    plant_absorb_nutrients(v._state, env_state)
                    plant_dissipate_energy(v._state)
                    transport_nutrients(v._state, n_steps=NUTRIENT_TRANSPORT_STEPS)

            # Structural integrity + gravity
            if gravity_enabled:
                for v in viewers:
                    update_structural_integrity(v._state, env_state.env_type_grid, n_steps=INTEGRITY_STEPS)
                    apply_gravity(v._state, env_state.env_type_grid)

            # NCA step(s)
            for v in viewers:
                v.step(steps_per_frame, nutrients_enabled=nutrients_enabled)

            # Death check after NCA step
            if nutrients_enabled:
                for v in viewers:
                    plant_death_check(v._state)

        # --- Draw -------------------------------------------------------------
        screen.fill(BG_COLOUR)

        for v in viewers:
            v.draw(screen, panel_x, panel_top, show_nutrients=show_nutrients,
                   gravity_enabled=gravity_enabled)

        # Species labels + plant stats
        label_y = panel_top + SIM_GRID_H * CELL_RENDER_SIZE + 4
        stat_parts = []
        for v in viewers:
            s = v.plant_stats()
            base = (
                f'{v.name}:{v.step_count:,}  '
                f'alive={s["alive"]}  '
                f'earth={s["avg_earth"]:.2f}  air={s["avg_air"]:.2f}'
            )
            if gravity_enabled:
                base += f'  integ={s["avg_integrity"]:.1f}'
            stat_parts.append(base)
        label = font.render('  |  '.join(stat_parts), True, LABEL_COLOUR)
        screen.blit(label, (PANEL_PADDING, label_y))

        # Status bar
        pause_str   = 'PAUSED' if paused else 'running'
        nutr_str    = ' [overlay ON]' if show_nutrients else ''
        energy_str  = ' [ENERGY ON]' if nutrients_enabled else ' [energy OFF]'
        grav_str    = ' [GRAVITY ON]' if gravity_enabled else ''
        spd_label = font.render(
            f'{pause_str}  speed: {steps_per_frame}×{energy_str}{grav_str}{nutr_str}  |  '
            f'Space=pause  S=step  R=reset  D=overlay  E=energy  G=gravity  Q=quit',
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
