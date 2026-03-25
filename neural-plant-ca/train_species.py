"""
train_species.py — CLI entry point for training a neural plant species.

Usage:
    python train_species.py --name oak --preset oak --steps 5000
    python train_species.py --name pine --preset pine --steps 5000 --seed 7
    python train_species.py --name oak --steps 2000 --device cpu   # smoke test

    # Energy-aware fine-tuning from a pretrained model:
    python train_species.py --name oak_v2 --preset oak --steps 10000 --energy --init-from species/oak.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from config import DEVICE, N_CHANNELS, N_TRAINING_STEPS, LEARNING_RATE
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
    parser.add_argument(
        '--energy', action='store_true',
        help='Enable energy-aware training: nutrients not clamped, environment physics '
             'run during unroll, survival bonus in loss',
    )
    parser.add_argument(
        '--init-from', dest='init_from', default=None, metavar='PATH',
        help='Load pretrained weights from a checkpoint file as starting point. '
             'Uses LR 5e-4 for fine-tuning unless --lr is specified.',
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Learning rate override (default: 1e-3, or 5e-4 when --init-from is used)',
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else DEVICE
    rng    = np.random.default_rng(args.seed)

    # Determine learning rate
    if args.lr is not None:
        lr = args.lr
    elif args.init_from is not None:
        lr = 5e-4
    else:
        lr = LEARNING_RATE

    # ---- Summary -----------------------------------------------------------
    print("=" * 50)
    print(f"  Species   : {args.name}")
    print(f"  Preset    : {args.preset}")
    print(f"  Steps     : {args.steps:,}")
    print(f"  Device    : {device}")
    print(f"  Seed      : {args.seed if args.seed is not None else 'random'}")
    print(f"  Energy    : {'ON' if args.energy else 'OFF'}")
    print(f"  Init from : {args.init_from or '(scratch)'}")
    print(f"  LR        : {lr}")
    print("=" * 50)
    print()

    # ---- Target ------------------------------------------------------------
    target   = make_target(args.preset, rng).to(device)
    alive_px = int((target[3] > 0.5).sum().item())
    print(f"Target generated  —  {alive_px} alive pixels\n")

    # ---- Model -------------------------------------------------------------
    model = NCA().to(device)

    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.exists():
            print(f"ERROR: --init-from file not found: {init_path}")
            sys.exit(1)
        ckpt = torch.load(init_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        prev_steps = ckpt.get('step', '?')
        prev_loss  = ckpt.get('loss')
        loss_str   = f'{prev_loss:.5f}' if prev_loss is not None else 'n/a'
        print(f"Loaded pretrained weights from {init_path}")
        print(f"  Previous training: {prev_steps} steps, loss {loss_str}\n")

    print(f"NCA model  —  {model.parameter_count():,} parameters\n")

    # ---- Train -------------------------------------------------------------
    model = train(
        model,
        target,
        name=args.name,
        n_steps=args.steps,
        device=device,
        energy=args.energy,
        lr=lr,
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
