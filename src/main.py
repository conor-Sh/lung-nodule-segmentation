from __future__ import annotations

import argparse
from pathlib import Path

import torch

from utils import default_device
from ResUNet import ResUNet
from training import run_training
from inference import infer_single_scan


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a 2D ResUNet on preprocessed .npz volumes."
    )
    parser.add_argument(
        "--data-root",
        default="data/processed",
        help="Local root containing .npz files.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="Hugging Face dataset repo ID with .npz files.",
    )
    parser.add_argument(
        "--repo-prefix",
        default="processed",
        help="Optional path prefix within the HF repo.",
    )
    parser.add_argument(
        "--cache-dir", default=None, help="Optional Hugging Face cache directory."
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="Square size for 2D slices."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--slice-mode", choices=["all", "positive", "mixed"], default="mixed"
    )
    parser.add_argument("--negative-stride", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--checkpoint-path", default="outputs/checkpoints/resunet_2d_best.pt"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the checkpoint at --checkpoint-path",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save example predictions after training to --viz-out",
    )
    parser.add_argument(
        "--viz-count",
        type=int,
        default=4,
        help="Number of example slices to visualise",
    )
    parser.add_argument(
        "--viz-out",
        default="outputs/visuals",
        help="Directory to save visualisations",
    )
    parser.add_argument(
        "--predict-scan",
        default=None,
        help="Path to a single .npz scan to run inference on (uses --checkpoint-path)",
    )
    parser.add_argument(
        "--predict-out",
        default="outputs/predictions",
        help="Directory to save single-scan inference outputs",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # -------------------------
    # PREDICTION MODE
    # -------------------------
    if args.predict_scan:
        device = default_device()
        model = ResUNet(num_classes=1).to(device)
        ckpt_path = Path(args.checkpoint_path)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            print(
                f"Warning: checkpoint {ckpt_path} not found — running with random weights"
            )

        infer_single_scan(
            Path(args.predict_scan),
            model,
            args.image_size,
            device,
            args.predict_out,
        )
        return

    # -------------------------
    # TRAINING MODE (WITH RESUME)
    # -------------------------
    ckpt_path = Path(args.checkpoint_path)

    if args.resume and ckpt_path.exists():
        print(f"Resuming training from checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=default_device())
        args.start_epoch = state.get("epoch", 0) + 1
        args.resume_state = state
    else:
        if args.resume:
            print(f"No checkpoint found at {ckpt_path}, starting fresh.")
        args.start_epoch = 1
        args.resume_state = None

    run_training(args)


if __name__ == "__main__":
    main()
