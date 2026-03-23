"""
viewer.py — Pygame-based NCA viewer for a single species.

Renders the NCA grid in real-time, compositing RGB + alpha over a configurable
background colour.  The simulation can be reset by pressing R or SPACE.

Public API
----------
    SpeciesViewer(name, model, seed_fn, device, cell_size, bg_colour)
        .reset()          — reinitialise grid from seed
        .step(n)          — advance n NCA steps
        .draw(surface, x, y) — blit current frame onto a pygame surface
        .handle_event(ev) — process a pygame event; returns True if consumed
"""

from __future__ import annotations

import numpy as np
import pygame
import torch

from config import GRID_H, GRID_W, N_CHANNELS, CH_RGB, CH_ALPHA, CELL_FIRE_RATE
from model import NCA, make_seed


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _composite_to_surface(
    state: torch.Tensor,
    cell_size: int,
    bg: tuple[int, int, int],
) -> pygame.Surface:
    """
    Convert a single NCA state (N_CHANNELS, H, W) to a pygame Surface.

    RGB channels are composited over *bg* using the alpha channel.
    Output pixel size = (W * cell_size, H * cell_size).
    """
    t = state.detach().cpu().float()

    rgb   = t[CH_RGB].permute(1, 2, 0).numpy().clip(0.0, 1.0)   # (H, W, 3)
    alpha = t[CH_ALPHA].numpy().clip(0.0, 1.0)                   # (H, W)

    bg_f  = np.array(bg, dtype=np.float32) / 255.0
    comp  = rgb * alpha[:, :, None] + bg_f * (1.0 - alpha[:, :, None])
    comp  = (comp.clip(0.0, 1.0) * 255).astype(np.uint8)         # (H, W, 3)

    # Scale up by cell_size using nearest-neighbour
    if cell_size == 1:
        surf = pygame.surfarray.make_surface(comp.transpose(1, 0, 2))
    else:
        small = pygame.surfarray.make_surface(comp.transpose(1, 0, 2))
        w, h  = small.get_size()
        surf  = pygame.transform.scale(small, (w * cell_size, h * cell_size))

    return surf


# ---------------------------------------------------------------------------
# SpeciesViewer
# ---------------------------------------------------------------------------

class SpeciesViewer:
    """
    Manages the simulation state and rendering for one species.

    Args:
        name       : display name (shown in title / label)
        model      : trained NCA, already on *device*
        device     : torch device
        cell_size  : pixels per NCA cell (default 8 → 64×64 grid = 512×512 px)
        bg         : RGB background colour
        steps_per_frame : NCA steps to advance each rendered frame
    """

    def __init__(
        self,
        name: str,
        model: NCA,
        device: torch.device,
        cell_size: int = 8,
        bg: tuple[int, int, int] = (30, 30, 30),
        steps_per_frame: int = 1,
    ) -> None:
        self.name            = name
        self.model           = model
        self.device          = device
        self.cell_size       = cell_size
        self.bg              = bg
        self.steps_per_frame = steps_per_frame

        self.pixel_w = GRID_W * cell_size
        self.pixel_h = GRID_H * cell_size

        self._state: torch.Tensor | None = None
        self._surface: pygame.Surface | None = None
        self._step_count: int = 0
        self._dirty: bool = True

        self.reset()

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reinitialise to a fresh single-cell seed."""
        self._state     = make_seed(batch_size=1, device=self.device)  # (1, C, H, W)
        self._step_count = 0
        self._dirty      = True

    def step(self, n: int = 1) -> None:
        """Advance the simulation by *n* NCA steps (no grad)."""
        with torch.no_grad():
            for _ in range(n):
                self._state = self.model(self._state, fire_rate=CELL_FIRE_RATE)
        self._step_count += n
        self._dirty = True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _rebuild_surface(self) -> None:
        if self._state is None:
            return
        self._surface = _composite_to_surface(
            self._state[0],   # drop batch dim → (C, H, W)
            self.cell_size,
            self.bg,
        )
        self._dirty = False

    def draw(self, target: pygame.Surface, x: int = 0, y: int = 0) -> None:
        """Blit the current frame onto *target* at pixel position (x, y)."""
        if self._dirty:
            self._rebuild_surface()
        if self._surface is not None:
            target.blit(self._surface, (x, y))

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def handle_event(self, ev: pygame.event.Event) -> bool:
        """
        Handle a pygame event.  Returns True if consumed.

        R / SPACE  →  reset simulation
        """
        if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_r, pygame.K_SPACE):
            self.reset()
            return True
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        return self._step_count
