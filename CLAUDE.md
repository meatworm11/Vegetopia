# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vegetopia is a Neural Cellular Automaton (NCA) project (PyTorch + pygame) that trains plant-like growth simulations. Each "species" is an NCA model that learns to grow from a two-cell seed into a target plant shape (oak, pine, fern, bush) on a 64x64 training grid, then runs on a larger 128x256 simulation grid with environmental terrain and nutrient diffusion.

All code lives in `neural-plant-ca/`. Run all commands from that directory.

**Current phase: phase 2 — environment physics.** Nutrient transport, tiered death, structural integrity, and gravity are implemented. Energy-aware training with curriculum is available via `--energy --curriculum` flags. Current trained models (phase 1) don't know about energy yet; retraining with curriculum is the next step.

## Commands

```bash
cd neural-plant-ca

# Train a species (outputs to species/ and snapshots/)
python -u train_species.py --name oak --preset oak --steps 10000
python -u train_species.py --name pine --preset pine --steps 10000 --seed 7

# Energy-aware training (phase 2) — finite nutrient pool, no curriculum
python -u train_species.py --name oak_v2 --preset oak --steps 10000 --energy
python -u train_species.py --name oak_v2 --preset oak --steps 10000 --energy --init-from species/oak.pt

# Quick smoke test (fast, CPU, few steps)
python -u train_species.py --name oak --steps 2000 --device cpu

# View a trained species (requires pygame)
python run_simulation.py oak
python run_simulation.py oak pine fern       # multiple species on shared environment

# Train and immediately launch viewer
python -u train_species.py --name oak --preset oak --steps 10000 --preview

# Smoke-test the model module directly
python model.py

# Preview a target shape without training
python target.py oak 42
```

Use `python -u` when piping output (e.g. to `tee`), otherwise Python buffers stdout and the log stays empty until the run ends.

## File Layout

```
neural-plant-ca/
  config.py          — all constants, channel indices, hyperparameters
  model.py           — NCA model (Sobel perception + 1x1 conv update rule)
  target.py          — procedural target shape generation (oak, pine, fern, bush)
  training.py        — pool-based training loop with curriculum support
  train_species.py   — CLI entry point for training
  environment.py     — terrain, nutrient diffusion, energy system, integrity, gravity
  viewer.py          — SpeciesViewer class, rendering, overlays
  run_simulation.py  — pygame main loop, controls, CLI entry point for viewer
  species/           — trained model checkpoints (.pt files)
  snapshots/         — training progress images (.png files)
```

## 32-Channel Cell State Vector

`(batch, 32, H, W)` float tensor. Channel layout defined in `config.py`:

| Channel | Name | Description |
|---------|------|-------------|
| 0-2 | RGB | Cell colour |
| 3 | alpha | `> 0.1` = alive |
| 4 | earth | Earth nutrients (clamped to 1.0 when `NUTRIENTS_ENABLED=False`) |
| 5 | air | Air nutrients (clamped to 1.0 when `NUTRIENTS_ENABLED=False`) |
| 6 | water | Reserved — phase 4 |
| 7 | integrity | Structural integrity — phase 2 |
| 8 | age | Incremented each step |
| 9 | cell_type | 0=unspecialized, 0.25=root, 0.5=stem, 0.75=leaf, 1.0=flower |
| 10 | species_id | **Write-protected** — environment-managed, restored after every NCA step |
| 11 | reserved | — |
| 12-31 | hidden | 20 channels of learned hidden state |

## Architecture

**Training pipeline:** `train_species.py` (CLI) -> `training.py` (loop) -> `model.py` (NCA step) + `target.py` (target generation).

**Viewer pipeline:** `run_simulation.py` (`run_viewer()` + CLI) -> `viewer.py` (`SpeciesViewer`) -> `model.py` (NCA step) + `environment.py` (nutrient diffusion + energy system).

### NCA Forward Pass

`model.py`: fixed Sobel perception (32->96 channels) -> two 1x1 convs (96->128->32) -> stochastic fire mask -> living cell mask -> env channel restoration. Final conv is zero-initialized (training starts as no-op). Accepts `nutrients_enabled` flag to control whether earth/air channels pass through or are clamped to 1.0.

### Training Loop

`training.py`: pool-based strategy (1024 states). Each step: sample batch of 4, replace worst with fresh seed, unroll NCA for 80-128 random steps, backprop with per-parameter gradient normalization. Loss = shape MSE (root-masked) + 0.1x cell-type + 0.01x overflow + 2.0x bg + 0.1x stem-colour. Energy-aware mode (`--energy`): runs environment physics during unroll with a finite nutrient pool (no regeneration — nutrients deplete as the plant grows), adds 0.5x energy health loss (penalizes alive cells with nutrients below 0.85). Fine-tuning: `--init-from` loads pretrained weights at LR 5e-4.

### Environment System

`environment.py`: terrain grid (sky, soil, rock, sun) + nutrient diffusion. Environment and NCA logic are kept as separate functions — the NCA model never calls environment code directly.

- `EnvironmentState`: holds terrain grid + nutrient tensor `(2, H, W)` + precomputed diffusion masks and kernel
- Diffusion: 5-point cross kernel, decay=0.9997, masked by cell type (earth through soil/rock, air through air/sun)
- Sources regenerate each step: rock->earth=1.0, sun->air=1.0
- Plant energy: root cells absorb earth nutrients, leaf cells absorb air; all alive cells dissipate both
- Nutrient transport: 20% sharing with poorer neighbours, 2 passes per step
- Tiered death: root/stem die if earth < 0.01; leaf/flower die if earth OR air < 0.01; unspecialized die if BOTH < 0.01
- Structural integrity: propagates from soil (50) / rock (100) via max-pool minus cell-type decay
- Gravity: unsupported cells (integrity <= 0) fall one row down, processed bottom-to-top

### Viewer

`viewer.py` + `run_simulation.py`: shared environment with terrain background rendering, plant alpha compositing, and optional nutrient/health overlays. Visual features include death flash (yellow wilt), ghost seed preview at cursor, and fullscreen toggle.

Controls: Space=pause, S=step, R=reset, D=overlay, E=energy, G=gravity, F=fullscreen, 1-9=switch species, +/-=speed, LMB=seed, RMB=kill, Q=quit.

## Key Architectural Decisions

- **Roots excluded from shape loss:** Root cells (cell_type ~0.25) are masked out of the shape MSE loss during training (`ROOT_LOSS_MASK=True`). This lets the NCA freely explore root structures without being penalized for underground shape mismatch.
- **Channel 10 (species_id) is write-protected:** Restored to a fixed value after every NCA step. The NCA can read it but cannot modify it.
- **Environment and NCA are separate:** `environment.py` functions operate on state tensors but are never called from inside `model.py`. The training loop and viewer orchestrate the interleaving.
- **Two grid sizes:** Training on 64x64 (`GRID_H/W`), simulation on 128x256 (`SIM_GRID_H/W`). NCA is fully convolutional so it runs at any size.

## Training Workflow

Training runs on **Google Colab** (GPU). Trained model checkpoints are saved to **Google Drive** at `Vegetopia/species/`. Local development and viewer testing happen on the local machine.

- `species/{name}.pt` — final trained model (also `{name}_stepN.pt` checkpoints every 1000 steps)
- `snapshots/{name}_stepN.png` — side-by-side target vs. current state, saved every 500 steps
- Checkpoint format: `{'step', 'model_state_dict', 'n_channels', 'name', 'loss', 'energy', 'curriculum_phase'}`

## Known Issues

- **No real root growth yet:** Current trained models produce root cell types but roots don't meaningfully grow underground. Needs energy-aware retraining so the NCA learns that roots are essential for nutrient absorption.
- **Gravity needs retraining:** Gravity is implemented but current models weren't trained with it. Trees collapse when gravity is toggled on. After energy-aware retraining, models should learn to build structurally sound trunks.
- **Energy health loss untested:** The energy health loss (penalizes alive cells with nutrients below 0.85, weighted 0.5, finite nutrient pool) replaced the old survival bonus but hasn't been validated with a full training run yet.
