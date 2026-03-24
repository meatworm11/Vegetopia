# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vegetopia is a Neural Cellular Automaton (NCA) project that trains plant-like growth simulations. Each "species" is an NCA model that learns to grow from a two-cell seed into a target plant shape (oak, pine, fern, bush) on a 64×64 training grid, then runs on a larger 128×256 simulation grid with environmental terrain and nutrient diffusion.

All code lives in `neural-plant-ca/`. Run all commands from that directory.

## Commands

```bash
cd neural-plant-ca

# Train a species (outputs to species/ and snapshots/)
python -u train_species.py --name oak --preset oak --steps 10000
python -u train_species.py --name pine --preset pine --steps 10000 --seed 7

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

## Architecture

**Training pipeline:** `train_species.py` (CLI) → `training.py` (loop) → `model.py` (NCA step) + `target.py` (target generation).

**Viewer pipeline:** `run_simulation.py` (`run_viewer()` + CLI) → `viewer.py` (`SpeciesViewer`) → `model.py` (NCA step) + `environment.py` (nutrient diffusion + energy system).

### Cell State

`(batch, 32, H, W)` float tensor. Channel layout defined in `config.py`:
- `0-2`: RGB
- `3`: alpha (`> 0.1` = alive)
- `4`: earth nutrients (clamped to 1.0 when `NUTRIENTS_ENABLED=False`)
- `5`: air nutrients (clamped to 1.0 when `NUTRIENTS_ENABLED=False`)
- `6-8`: water (reserved), integrity (reserved), age
- `9`: cell_type (0=unspecialized, 0.25=root, 0.5=stem, 0.75=leaf, 1.0=flower)
- `10`: species_id (env-managed)
- `11`: reserved
- `12-31`: learned hidden state

### NCA Forward Pass

`model.py`: fixed Sobel perception (32→96 channels) → two 1×1 convs (96→128→32) → stochastic fire mask → living cell mask → env channel restoration. Final conv is zero-initialized (training starts as no-op). Accepts `nutrients_enabled` flag to control whether earth/air channels pass through or are clamped to 1.0.

### Training Loop

`training.py`: pool-based strategy (1024 states). Each step: sample batch of 8, replace worst with fresh seed, unroll NCA for 80–128 random steps, backprop with per-parameter gradient normalization. Loss = shape MSE (with optional `ROOT_LOSS_MASK` excluding root cells) + 0.1×cell-type MSE + 0.01×hidden overflow + 2.0×background loss.

### Environment System

`environment.py`: terrain grid (sky, soil, rock, sun) + nutrient diffusion.
- `EnvironmentState`: holds terrain grid + nutrient tensor `(2, H, W)` + precomputed diffusion masks and kernel
- Diffusion: 5-point cross kernel, decay=0.9997, masked by cell type (earth through soil/rock, air through air/sun)
- Sources regenerate each step: rock→earth=1.0, sun→air=1.0
- Plant energy: root cells absorb earth nutrients, leaf cells absorb air; all alive cells dissipate both; cells die when both nutrients drop below threshold

### Viewer

`viewer.py` + `run_simulation.py`: shared environment with terrain background rendering, plant alpha compositing, and optional nutrient/health overlays. Controls: Space=pause, S=step, R=reset, D=nutrient overlay, E=energy toggle, +/-=speed, LMB=seed, RMB=kill, Q=quit.

## Two Grid Sizes

- **Training grid**: 64×64 (`GRID_H`, `GRID_W`) — used by training loop and targets
- **Simulation grid**: 128×256 (`SIM_GRID_H`, `SIM_GRID_W`) — used by viewer; NCA is fully convolutional so it runs at any size

## Outputs

- `species/{name}.pt` — final trained model (also `{name}_stepN.pt` checkpoints every 1000 steps)
- `snapshots/{name}_stepN.png` — side-by-side target vs. current state, saved every 500 steps

Checkpoint format: `{'step', 'model_state_dict', 'n_channels', 'name', 'loss'}`.

## Phase System

Current: **phase 1** — single species, nutrients clamped to 1.0 during training (`NUTRIENTS_ENABLED=False`). The energy system (absorption, dissipation, death) is implemented and can be toggled with E key in the viewer but is off by default. Phase 2+ features (integrity, water, multi-species competition) are reserved in the channel layout.
