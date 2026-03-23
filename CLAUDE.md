# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vegetopia is a Neural Cellular Automaton (NCA) project that trains plant-like growth simulations. Each "species" is an NCA model that learns to grow from a two-cell seed into a target plant shape (oak, pine, fern, bush) on a 64×64 grid.

All code lives in `neural-plant-ca/`. Run all commands from that directory.

## Commands

```bash
cd neural-plant-ca

# Train a species (outputs to species/ and snapshots/)
python -u train_species.py --name oak --preset oak --steps 5000
python -u train_species.py --name pine --preset pine --steps 5000 --seed 7

# Quick smoke test (fast, CPU, few steps)
python -u train_species.py --name oak --steps 2000 --device cpu

# View a trained species (requires pygame)
python run_simulation.py oak
python run_simulation.py oak pine fern       # side-by-side (positional)
python run_simulation.py --species oak pine  # equivalent --species flag

# Train and immediately launch viewer
python -u train_species.py --name oak --preset oak --steps 5000 --preview

# Smoke-test the model module directly
python model.py

# Preview a target shape without training
python target.py oak 42
```

Use `python -u` when piping output (e.g. to `tee`), otherwise Python buffers stdout and the log stays empty until the run ends.

## Architecture

**Training pipeline:** `train_species.py` (CLI) → `training.py` (loop) → `model.py` (NCA step) + `target.py` (target generation).

**Viewer pipeline:** `run_simulation.py` (`run_viewer()` + CLI) → `viewer.py` (`SpeciesViewer`) → `model.py` (NCA step).

**Cell state** is a `(batch, 32, 64, 64)` float tensor. Channel layout defined in `config.py`:
- `0-2`: RGB
- `3`: alpha (`> 0.1` = alive)
- `4`: earth nutrients (env-managed, locked to 1.0 in phase 1)
- `5`: air nutrients (env-managed, locked to 1.0 in phase 1)
- `6-11`: water, integrity, age, cell_type, species_id, reserved
- `12-31`: learned hidden state

**NCA forward pass** (`model.py`): fixed Sobel perception (32→96 channels) → two 1×1 convs (96→128→32) → stochastic fire mask → living cell mask → env channel restoration. The final conv layer is zero-initialized so training starts from a no-op.

**Training loop** (`training.py`): pool-based strategy (pool of 1024 states). Each step: sample a batch of 8, replace the worst sample with a fresh seed, unroll the NCA for 64–96 random steps, backprop with per-parameter gradient normalization (prevents exploding gradients through long unrolls). Loss = MSE on RGBA + 0.1×cell-type MSE + 0.01×hidden channel overflow penalty.

**Targets** (`target.py`): procedurally generated per-preset trees with small random perturbations each run (`perturb_height`, `perturb_radius`). Presets: `oak`, `pine`, `fern`, `bush`.

**Viewer** (`viewer.py`, `run_simulation.py`): `SpeciesViewer` wraps one model + grid state; `run_simulation.run_viewer(names)` lays out panels side-by-side in pygame. Viewer controls: R/Space = reset, +/- = speed (1×–16× steps/frame), Q/Esc = quit.

## Outputs

- `species/{name}.pt` — final trained model (also `{name}_stepN.pt` checkpoints every 1000 steps)
- `snapshots/{name}_stepN.png` — side-by-side target vs. current state, saved every 500 steps

Checkpoint format: `{'step', 'model_state_dict', 'n_channels', 'name', 'loss'}`.

## Phase System

The codebase uses a phased roadmap. Current implementation is **phase 1**: single species, unlimited nutrients (earth=1.0, air=1.0 always), no multi-species competition. Phase 2+ features (integrity, water channel, multi-species) are reserved in the channel layout but not yet active.
