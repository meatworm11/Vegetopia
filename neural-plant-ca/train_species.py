"""
train_species.py — CLI entry point for training a neural plant species.

Usage:
    python train_species.py --name oak --preset oak --steps 5000
    python train_species.py --name pine --preset pine --steps 5000 --seed 7
    python train_species.py --name oak --steps 2000 --device cpu   # smoke test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from config import DEVICE, N_TRAINING_STEPS
from model import NCA
from target import DEFAULT_PRESET, PRESETS, make_target
from training import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train a neural plant species.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--name', required=True,
        help='Species name — used for output filenames (species/{name}.pt, snapshots/{name}_stepN.png)',
    )
    parser.add_argument(
        '--preset', default=DEFAULT_PRESET, choices=list(PRESETS),
        help='Target shape preset',
    )
    parser.add_argument(
        '--steps', type=int, default=N_TRAINING_STEPS,
        help='Number of training iterations',
    )
    parser.add_argument(
        '--device', default=None,
        help='Device override: cuda, mps, or cpu (default: auto-detect)',
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for target generation. Omit for a unique tree each run.',
    )
    parser.add_argument(
        '--preview', action='store_true',
        help='Open pygame viewer on the trained model when done (requires viewer.py)',
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else DEVICE
    rng    = np.random.default_rng(args.seed)

    # ---- Summary -----------------------------------------------------------
    print("=" * 50)
    print(f"  Species : {args.name}")
    print(f"  Preset  : {args.preset}")
    print(f"  Steps   : {args.steps:,}")
    print(f"  Device  : {device}")
    print(f"  Seed    : {args.seed if args.seed is not None else 'random'}")
    print("=" * 50)
    print()

    # ---- Target ------------------------------------------------------------
    target   = make_target(args.preset, rng).to(device)
    alive_px = int((target[3] > 0.5).sum().item())
    print(f"Target generated  —  {alive_px} alive pixels\n")

    # ---- Model -------------------------------------------------------------
    model = NCA().to(device)
    print(f"NCA model  —  {model.parameter_count():,} parameters\n")

    # ---- Train -------------------------------------------------------------
    model = train(
        model,
        target,
        name=args.name,
        n_steps=args.steps,
        device=device,
    )

    # ---- Preview -----------------------------------------------------------
    if args.preview:
        try:
            from run_simulation import run_viewer
            run_viewer([args.name])
        except ImportError:
            print('\nNote: run_simulation.py not yet built — skipping preview.')
        except Exception as e:
            print(f'\nPreview failed: {e}')


if __name__ == '__main__':
    main()
