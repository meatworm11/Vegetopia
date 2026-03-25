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
    Right-click  kill cells in a 3x3 area at cursor
    Space        pause / unpause
    S            step one frame while paused
    R            reset all simulations
    D            toggle nutrient debug overlay
    E            toggle energy system (nutrients_enabled)
    G            toggle gravity (structural integrity + falling)
    F            toggle fullscreen
    1-9          switch active species for seed placement
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
from viewer import SpeciesViewer, build_ghost_preview


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FPS_TARGET            = 30
BG_COLOUR             = (20, 20, 20)
LABEL_COLOUR          = (200, 200, 200)
DIM_COLOUR            = (120, 120, 120)
PANEL_PADDING         = 12
STATUS_BAR_HEIGHT     = 36   # space for bottom status bar
MIN_STEPS_FRAME       = 1
MAX_STEPS_FRAME       = 16
DIFFUSION_SUBSTEPS    = 4    # nutrient diffusion steps per sim tick


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
    panel_h = SIM_GRID_H * cell_size
    total_w = panel_w + 2 * PANEL_PADDING
    total_h = panel_h + 2 * PANEL_PADDING + STATUS_BAR_HEIGHT
    return total_w, total_h


# ---------------------------------------------------------------------------
# Simulation step helper
# ---------------------------------------------------------------------------

_debug_frame = 0

def _sim_step(
    viewers: list[SpeciesViewer],
    env_state: EnvironmentState,
    steps_per_frame: int,
    nutrients_enabled: bool,
    gravity_enabled: bool,
) -> None:
    """Run one frame of simulation: diffusion, absorption, NCA, death."""
    global _debug_frame
    _debug_frame += 1

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
            v.snapshot_alive()
            plant_death_check(v._state)
            v.detect_deaths()

    # Debug: print alive count every 30 frames (~1 second)
    if _debug_frame % 30 == 0:
        for v in viewers:
            alive = int((v._state[0, 3] > 0.1).sum().item())
            cap_str = ' CAP' if alive >= 800 else ''
            print(f'[frame {_debug_frame}] {v.name}: alive={alive}{cap_str}'
                  f'  energy={"ON" if nutrients_enabled else "OFF"}'
                  f'  gravity={"ON" if gravity_enabled else "OFF"}')


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
    print(f"Grid   : {SIM_GRID_H}x{SIM_GRID_W}  ({CELL_RENDER_SIZE}px/cell)")
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
    print("Initialising nutrient gradients...")
    for _ in range(2000):
        regenerate_sources(env_state)
        diffuse_nutrients(env_state, n_steps=5)
    print("  done.")

    # pygame setup
    pygame.init()
    pygame.display.set_caption('Vegetopia — Neural Plant CA')

    cell_size  = CELL_RENDER_SIZE   # mutable — changes on fullscreen toggle
    win_w, win_h = _layout(cell_size)
    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
    clock  = pygame.time.Clock()
    fullscreen = False

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
            cell_size=cell_size,
        )
        for name in species_names
    ]

    panel_x            = PANEL_PADDING
    panel_top          = PANEL_PADDING
    steps_per_frame    = 1
    paused             = False
    show_nutrients     = False
    nutrients_enabled  = NUTRIENTS_ENABLED
    gravity_enabled    = GRAVITY_ENABLED
    active_species_idx = 0   # index into viewers[]
    running            = True

    # Number keys 1-9 map to species indices
    NUM_KEYS = {
        pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2,
        pygame.K_4: 3, pygame.K_5: 4, pygame.K_6: 5,
        pygame.K_7: 6, pygame.K_8: 7, pygame.K_9: 8,
    }

    def _hit_panel(mx: int, my: int) -> tuple[int, int] | None:
        """Return (local_px, local_py) if cursor is inside the grid panel."""
        lx = mx - panel_x
        ly = my - panel_top
        pw = SIM_GRID_W * cell_size
        ph = SIM_GRID_H * cell_size
        if 0 <= lx < pw and 0 <= ly < ph:
            return lx, ly
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
                    _sim_step(viewers, env_state, 1,
                              nutrients_enabled, gravity_enabled)

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

                elif ev.key == pygame.K_f:
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        sw, sh = screen.get_size()
                        # Scale cell size to fill screen, maintaining aspect ratio
                        cell_size = min(sw // SIM_GRID_W, (sh - STATUS_BAR_HEIGHT) // SIM_GRID_H)
                        cell_size = max(cell_size, 1)
                        # Center the grid in the screen
                        panel_x = (sw - SIM_GRID_W * cell_size) // 2
                        panel_top = (sh - STATUS_BAR_HEIGHT - SIM_GRID_H * cell_size) // 2
                    else:
                        cell_size = CELL_RENDER_SIZE
                        panel_x = PANEL_PADDING
                        panel_top = PANEL_PADDING
                        screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
                    for v in viewers:
                        v.set_cell_size(cell_size)

                elif ev.key in NUM_KEYS:
                    idx = NUM_KEYS[ev.key]
                    if idx < len(viewers):
                        active_species_idx = idx
                        print(f"Active species: {viewers[active_species_idx].name}")

                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    steps_per_frame = min(steps_per_frame * 2, MAX_STEPS_FRAME)

                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    steps_per_frame = max(steps_per_frame // 2, MIN_STEPS_FRAME)

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                hit = _hit_panel(*ev.pos)
                if hit is not None:
                    lx, ly = hit
                    v = viewers[active_species_idx]
                    row, col = v.pixel_to_grid(lx, ly)
                    if ev.button == 1:
                        v.place_seed(row, col)
                    elif ev.button == 3:
                        v.kill_area(row, col, radius=1)

        # --- Simulate ---------------------------------------------------------
        if not paused:
            _sim_step(viewers, env_state, steps_per_frame,
                      nutrients_enabled, gravity_enabled)

        # --- Draw -------------------------------------------------------------
        screen.fill(BG_COLOUR)

        for v in viewers:
            v.draw(screen, panel_x, panel_top, show_nutrients=show_nutrients,
                   gravity_enabled=gravity_enabled)

        # Ghost preview at cursor position
        mx, my = pygame.mouse.get_pos()
        hit = _hit_panel(mx, my)
        if hit is not None:
            lx, ly = hit
            col = lx // cell_size
            ghost = build_ghost_preview(
                soil_row, col,
                SIM_GRID_H, SIM_GRID_W, cell_size,
            )
            screen.blit(ghost, (panel_x, panel_top))

        # --- Status bar (bottom) ----------------------------------------------
        fps = clock.get_fps()
        total_alive = sum(v.plant_stats()['alive'] for v in viewers)
        active_name = viewers[active_species_idx].name
        total_steps = viewers[0].step_count if viewers else 0

        pause_str  = 'PAUSED' if paused else 'running'
        energy_str = 'ON' if nutrients_enabled else 'OFF'
        grav_str   = 'ON' if gravity_enabled else 'OFF'
        overlay_str = ' [D]' if show_nutrients else ''

        line1 = (
            f'{pause_str}  step:{total_steps:,}  '
            f'FPS:{fps:.0f}  '
            f'species:[{active_species_idx+1}]{active_name}  '
            f'energy:{energy_str}  gravity:{grav_str}{overlay_str}  '
            f'alive:{total_alive}'
        )

        # Second line: nutrient stats when energy is on
        if nutrients_enabled:
            stat_parts = []
            for v in viewers:
                s = v.plant_stats()
                part = f'{v.name}: E={s["avg_earth"]:.2f} A={s["avg_air"]:.2f}'
                if gravity_enabled:
                    part += f' I={s["avg_integrity"]:.0f}'
                stat_parts.append(part)
            line2 = '  |  '.join(stat_parts)
        else:
            line2 = '  |  '.join(
                f'{v.name}:{v.step_count:,} alive={v.plant_stats()["alive"]}'
                for v in viewers
            )

        bar_y = panel_top + SIM_GRID_H * cell_size + 4
        screen.blit(font.render(line1, True, LABEL_COLOUR), (PANEL_PADDING, bar_y))
        screen.blit(font.render(line2, True, DIM_COLOUR), (PANEL_PADDING, bar_y + 16))

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
