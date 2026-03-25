"""
viewer.py — Pygame-based NCA viewer with environment background and nutrient overlay.

Renders a background terrain layer (sky gradient, soil, rock) then composites
NCA plant cells on top using alpha blending.  Press D to toggle a nutrient
debug overlay (earth = red→yellow in soil, air = blue→cyan in sky).

Public API
----------
    SpeciesViewer(name, model, env_state, soil_row, device, cell_size)
        .reset()                   — reinitialise grid from seed
        .step(n)                   — advance n NCA steps
        .draw(surface, x, y, ...) — blit frame with optional overlays
        .place_seed(row, col)      — place a seed (snapped to soil surface)
        .kill_area(row, col)       — zero cells in a 3×3 area
        .pixel_to_grid(px, py)     — screen coords → grid (row, col)
        .detect_deaths()           — call after death_check to track dying cells
"""

from __future__ import annotations

import numpy as np
import pygame
import torch

from config import (
    SIM_GRID_H, SIM_GRID_W, CELL_RENDER_SIZE, N_CHANNELS,
    CH_RGB, CH_ALPHA, CH_HIDDEN, CH_SPECIES_ID, CH_EARTH, CH_AIR,
    CH_CELL_TYPE, CH_INTEGRITY, CELL_FIRE_RATE, MAX_CELLS,
    ENV_EMPTY, ENV_SOIL, ENV_ROCK, ENV_SUN,
)
from environment import EnvironmentState, NUTRIENT_EARTH, NUTRIENT_AIR
from model import NCA, make_seed


# ---------------------------------------------------------------------------
# Environment background rendering
# ---------------------------------------------------------------------------

def build_env_background(
    env_grid: np.ndarray,
    cell_size: int,
) -> pygame.Surface:
    """
    Render the environment grid to a pygame Surface (built once, cached).

    - ENV_EMPTY / ENV_SUN: sky-blue gradient (lighter at top, darker near ground)
    - ENV_SOIL: brown with per-cell colour variation for texture
    - ENV_ROCK: dark grey-blue, distinct from soil
    """
    h, w = env_grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    sky_top    = np.array([135, 206, 235], dtype=np.float32)
    sky_bottom = np.array([ 90, 150, 200], dtype=np.float32)

    for r in range(h):
        t = r / max(h - 1, 1)
        sky_colour = (sky_top * (1 - t) + sky_bottom * t).astype(np.uint8)

        for c in range(w):
            cell = env_grid[r, c]
            if cell == ENV_SOIL:
                base = np.array([101, 67, 33], dtype=np.int32)
                noise = ((r * 7 + c * 13) % 11) - 5
                img[r, c] = np.clip(base + noise, 0, 255).astype(np.uint8)
            elif cell == ENV_ROCK:
                base = np.array([70, 72, 82], dtype=np.int32)
                noise = ((r * 3 + c * 17) % 7) - 3
                img[r, c] = np.clip(base + noise, 0, 255).astype(np.uint8)
            else:
                img[r, c] = sky_colour

    small = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
    if cell_size == 1:
        return small
    sw, sh = small.get_size()
    return pygame.transform.scale(small, (sw * cell_size, sh * cell_size))


# ---------------------------------------------------------------------------
# Nutrient overlay rendering
# ---------------------------------------------------------------------------

def build_nutrient_overlay(
    env_state: EnvironmentState,
    cell_size: int,
) -> pygame.Surface:
    """
    Build a semi-transparent RGBA surface showing nutrient concentrations.

    - Earth nutrients (in soil/rock cells): black → red → yellow as 0 → 0.5 → 1
    - Air nutrients (in air/sun cells): black → blue → cyan as 0 → 0.5 → 1
    - Overlay alpha is proportional to nutrient level (transparent where 0).
    """
    h, w = env_state.height, env_state.width
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    earth = env_state.nutrient_grid[NUTRIENT_EARTH].cpu().numpy().clip(0, 1)
    air   = env_state.nutrient_grid[NUTRIENT_AIR].cpu().numpy().clip(0, 1)
    env   = env_state.env_type_grid

    # Earth heatmap: red→yellow  (R stays high, G ramps up)
    earth_mask = (env == ENV_SOIL) | (env == ENV_ROCK)
    e = earth[earth_mask]
    overlay[earth_mask, 0] = (np.minimum(e * 2, 1.0) * 255).astype(np.uint8)       # R
    overlay[earth_mask, 1] = (np.maximum(e * 2 - 1, 0.0) * 255).astype(np.uint8)   # G
    overlay[earth_mask, 2] = 0                                                       # B
    overlay[earth_mask, 3] = (e * 180).astype(np.uint8)                              # A

    # Air heatmap: blue→cyan  (B stays high, G ramps up)
    air_mask = (env == ENV_EMPTY) | (env == ENV_SUN)
    a = air[air_mask]
    overlay[air_mask, 0] = 0                                                         # R
    overlay[air_mask, 1] = (np.maximum(a * 2 - 1, 0.0) * 255).astype(np.uint8)     # G
    overlay[air_mask, 2] = (np.minimum(a * 2, 1.0) * 255).astype(np.uint8)         # B
    overlay[air_mask, 3] = (a * 180).astype(np.uint8)                                # A

    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(surf)[:, :, :] = overlay[:, :, :3].transpose(1, 0, 2)
    pygame.surfarray.pixels_alpha(surf)[:, :] = overlay[:, :, 3].transpose(1, 0)

    if cell_size != 1:
        surf = pygame.transform.scale(surf, (w * cell_size, h * cell_size))

    return surf


def build_plant_health_overlay(
    state: torch.Tensor,
    cell_size: int,
    show_integrity: bool = False,
) -> pygame.Surface:
    """
    Build overlay colouring alive plant cells.

    When show_integrity=False (default):
      - Green when both nutrients high, yellow middling, red near death

    When show_integrity=True:
      - Green = high integrity, red = about to fall (integrity <= 0)

    state shape: (N_CHANNELS, H, W)
    """
    t = state.detach().cpu().float()
    h, w = t.shape[1], t.shape[2]

    alpha = t[CH_ALPHA].numpy()
    alive = alpha > 0.1

    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    if show_integrity:
        integrity = t[CH_INTEGRITY].numpy().clip(0, 100)
        norm = integrity / 100.0
        overlay[alive, 0] = (np.maximum(1.0 - norm[alive] * 2, 0.0) * 255).astype(np.uint8)
        overlay[alive, 1] = (np.minimum(norm[alive] * 2, 1.0) * 255).astype(np.uint8)
        overlay[alive, 2] = 0
        overlay[alive, 3] = 140
    else:
        earth = t[CH_EARTH].numpy().clip(0, 1)
        air   = t[CH_AIR].numpy().clip(0, 1)
        health = np.where(alive, np.minimum(earth, air), 0.0)
        overlay[alive, 0] = (np.maximum(1.0 - health[alive] * 2, 0.0) * 255).astype(np.uint8)
        overlay[alive, 1] = (np.minimum(health[alive] * 2, 1.0) * 255).astype(np.uint8)
        overlay[alive, 2] = 0
        overlay[alive, 3] = 140

    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(surf)[:, :, :] = overlay[:, :, :3].transpose(1, 0, 2)
    pygame.surfarray.pixels_alpha(surf)[:, :] = overlay[:, :, 3].transpose(1, 0)

    if cell_size != 1:
        surf = pygame.transform.scale(surf, (w * cell_size, h * cell_size))
    return surf


# ---------------------------------------------------------------------------
# Death flash overlay
# ---------------------------------------------------------------------------

def build_death_flash_overlay(
    dying_mask: np.ndarray,
    cell_size: int,
) -> pygame.Surface | None:
    """
    Build a yellow flash overlay for cells that just died.
    dying_mask: bool (H, W) — True where a cell died this frame.
    Returns None if no dying cells.
    """
    if not dying_mask.any():
        return None
    h, w = dying_mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[dying_mask, 0] = 255   # R
    overlay[dying_mask, 1] = 230   # G
    overlay[dying_mask, 2] = 50    # B
    overlay[dying_mask, 3] = 200   # A

    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(surf)[:, :, :] = overlay[:, :, :3].transpose(1, 0, 2)
    pygame.surfarray.pixels_alpha(surf)[:, :] = overlay[:, :, 3].transpose(1, 0)

    if cell_size != 1:
        surf = pygame.transform.scale(surf, (w * cell_size, h * cell_size))
    return surf


# ---------------------------------------------------------------------------
# Ghost preview overlay
# ---------------------------------------------------------------------------

def build_ghost_preview(
    soil_row: int,
    col: int,
    grid_h: int,
    grid_w: int,
    cell_size: int,
) -> pygame.Surface:
    """
    Build a translucent 2-cell seed preview at (soil_row, col) and (soil_row-1, col).
    """
    surf = pygame.Surface((grid_w * cell_size, grid_h * cell_size), pygame.SRCALPHA)
    colour = (100, 255, 100, 90)   # translucent green
    if 0 <= col < grid_w:
        # Bottom cell (at soil surface)
        pygame.draw.rect(surf, colour,
                         (col * cell_size, soil_row * cell_size, cell_size, cell_size))
        # Top cell (one above)
        if soil_row > 0:
            pygame.draw.rect(surf, colour,
                             (col * cell_size, (soil_row - 1) * cell_size, cell_size, cell_size))
    return surf


# ---------------------------------------------------------------------------
# Plant compositing
# ---------------------------------------------------------------------------

def _composite_plants_onto(
    bg_surface: pygame.Surface,
    state: torch.Tensor,
    cell_size: int,
) -> pygame.Surface:
    """
    Composite NCA plant cells (with alpha) onto a copy of the background.

    state shape: (N_CHANNELS, H, W)
    Returns a new surface (does not mutate bg_surface).
    """
    t = state.detach().cpu().float()
    h, w = t.shape[1], t.shape[2]

    rgb   = t[CH_RGB].permute(1, 2, 0).numpy().clip(0.0, 1.0)
    alpha = t[CH_ALPHA].numpy().clip(0.0, 1.0)

    mask = alpha > 0.01
    if not mask.any():
        return bg_surface.copy()

    plant_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(plant_surf)[:, :, :] = (
        (rgb * 255).astype(np.uint8).transpose(1, 0, 2)
    )
    pygame.surfarray.pixels_alpha(plant_surf)[:, :] = (
        (alpha * 255).astype(np.uint8).transpose(1, 0)
    )

    if cell_size != 1:
        plant_surf = pygame.transform.scale(
            plant_surf, (w * cell_size, h * cell_size),
        )

    result = bg_surface.copy()
    result.blit(plant_surf, (0, 0))
    return result


# ---------------------------------------------------------------------------
# SpeciesViewer
# ---------------------------------------------------------------------------

class SpeciesViewer:
    """
    Manages one NCA species on the simulation grid with environment background.

    The NCA model was trained on GRID_H*GRID_W (64x64) but runs on any size
    since it uses only local (1x1 conv + 3x3 Sobel) operations.
    """

    def __init__(
        self,
        name: str,
        model: NCA,
        env_state: EnvironmentState,
        soil_row: int,
        device: torch.device,
        cell_size: int = CELL_RENDER_SIZE,
    ) -> None:
        self.name       = name
        self.model      = model
        self.env_state  = env_state
        self.soil_row   = soil_row
        self.device     = device
        self.cell_size  = cell_size

        self.grid_h = env_state.height
        self.grid_w = env_state.width
        self.pixel_w = self.grid_w * cell_size
        self.pixel_h = self.grid_h * cell_size

        self._state: torch.Tensor | None = None
        self._bg_surface: pygame.Surface | None = None
        self._surface: pygame.Surface | None = None
        self._step_count: int = 0
        self._dirty: bool = True

        # Death flash: bool mask of cells that died this frame
        self._dying_flash: np.ndarray = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        self._flash_frames: int = 0  # frames remaining for flash
        self._alive_snapshot: torch.Tensor | None = None  # alive mask before death check

        self.reset()

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reinitialise to a fresh seed placed at the soil surface."""
        seed_row = self.soil_row
        seed_col = self.grid_w // 2
        seed_positions = [
            (seed_row, seed_col),
            (seed_row - 1, seed_col),
        ]
        self._state = make_seed(
            batch_size=1,
            device=self.device,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            seed_positions=seed_positions,
        )
        self._step_count = 0
        self._dying_flash[:] = False
        self._flash_frames = 0
        self._dirty = True

    def snapshot_alive(self) -> None:
        """Save alive mask before death check so we can detect deaths."""
        if self._state is not None:
            self._alive_snapshot = (self._state[0, CH_ALPHA] > 0.1).clone()

    def detect_deaths(self) -> None:
        """Compare current alive mask with snapshot to find newly dead cells."""
        if self._alive_snapshot is None or self._state is None:
            return
        now_alive = self._state[0, CH_ALPHA] > 0.1
        just_died = self._alive_snapshot & ~now_alive
        if just_died.any():
            self._dying_flash = just_died.cpu().numpy()
            self._flash_frames = 3  # show for 3 frames
            self._dirty = True
        self._alive_snapshot = None

    def step(self, n: int = 1, nutrients_enabled: bool = False) -> None:
        """Advance the simulation by *n* NCA steps (no grad).

        Growth cap: if alive cells exceed MAX_CELLS after any step,
        revert newly spawned cells (alpha went from 0 to >0) back to zero.
        """
        with torch.no_grad():
            for _ in range(n):
                was_alive = self._state[0, CH_ALPHA] > 0.1
                self._state = self.model(
                    self._state,
                    fire_rate=CELL_FIRE_RATE,
                    nutrients_enabled=nutrients_enabled,
                )
                now_alive = self._state[0, CH_ALPHA] > 0.1
                if int(now_alive.sum().item()) > MAX_CELLS:
                    new_cells = now_alive & ~was_alive
                    if new_cells.any():
                        self._state[0, :, new_cells] = 0.0
        self._step_count += n
        self._dirty = True

    def place_seed(self, row: int, col: int) -> None:
        """Place a seed snapped to the soil surface row."""
        row = self.soil_row
        if not (0 <= col < self.grid_w):
            return
        with torch.no_grad():
            self._state[0, CH_ALPHA, row, col]    = 1.0
            self._state[0, CH_HIDDEN, row, col]   = 1.0
            self._state[0, CH_EARTH, row, col]    = 1.0
            self._state[0, CH_AIR, row, col]      = 1.0
            if row > 0:
                self._state[0, CH_ALPHA, row - 1, col]  = 1.0
                self._state[0, CH_HIDDEN, row - 1, col] = 1.0
                self._state[0, CH_EARTH, row - 1, col]  = 1.0
                self._state[0, CH_AIR, row - 1, col]    = 1.0
        self._dirty = True

    def kill_area(self, row: int, col: int, radius: int = 1) -> None:
        """Zero all channels in a (2*radius+1) square centred on (row, col)."""
        r0 = max(0, row - radius)
        r1 = min(self.grid_h, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(self.grid_w, col + radius + 1)
        with torch.no_grad():
            self._state[0, :, r0:r1, c0:c1] = 0.0
        self._dirty = True

    def pixel_to_grid(self, px: int, py: int) -> tuple[int, int]:
        """Convert pixel offset (relative to panel top-left) to grid (row, col)."""
        col = px // self.cell_size
        row = py // self.cell_size
        return row, col

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _ensure_bg(self) -> None:
        """Build the environment background surface (once)."""
        if self._bg_surface is None:
            self._bg_surface = build_env_background(
                self.env_state.env_type_grid, self.cell_size,
            )

    def _rebuild_surface(self) -> None:
        if self._state is None:
            return
        self._ensure_bg()
        self._surface = _composite_plants_onto(
            self._bg_surface,
            self._state[0],
            self.cell_size,
        )
        self._dirty = False

    def draw(
        self,
        target: pygame.Surface,
        x: int = 0,
        y: int = 0,
        show_nutrients: bool = False,
        gravity_enabled: bool = False,
    ) -> None:
        """Blit the current frame onto *target*, optionally with nutrient/integrity overlay."""
        if self._dirty:
            self._rebuild_surface()
        if self._surface is not None:
            target.blit(self._surface, (x, y))

        # Death flash overlay
        if self._flash_frames > 0:
            flash_surf = build_death_flash_overlay(self._dying_flash, self.cell_size)
            if flash_surf is not None:
                target.blit(flash_surf, (x, y))
            self._flash_frames -= 1
            if self._flash_frames <= 0:
                self._dying_flash[:] = False

        if show_nutrients:
            overlay = build_nutrient_overlay(self.env_state, self.cell_size)
            target.blit(overlay, (x, y))
            # Show integrity overlay when gravity is on, otherwise nutrient health
            if self._state is not None:
                health_ov = build_plant_health_overlay(
                    self._state[0], self.cell_size,
                    show_integrity=gravity_enabled,
                )
                target.blit(health_ov, (x, y))

    # ------------------------------------------------------------------
    # Properties / stats
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        return self._step_count

    def plant_stats(self) -> dict:
        """Return stats about alive plant cells and their nutrient/integrity levels."""
        if self._state is None:
            return {'alive': 0, 'avg_earth': 0.0, 'avg_air': 0.0, 'avg_integrity': 0.0}
        with torch.no_grad():
            t = self._state[0]
            alive = t[CH_ALPHA] > 0.1
            n_alive = int(alive.sum().item())
            if n_alive == 0:
                return {'alive': 0, 'avg_earth': 0.0, 'avg_air': 0.0, 'avg_integrity': 0.0}
            avg_e = float(t[CH_EARTH][alive].mean().item())
            avg_a = float(t[CH_AIR][alive].mean().item())
            avg_i = float(t[CH_INTEGRITY][alive].mean().item())
            return {'alive': n_alive, 'avg_earth': avg_e, 'avg_air': avg_a, 'avg_integrity': avg_i}
