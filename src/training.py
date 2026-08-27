from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import seed_everything, default_device, list_npz_files
from data import build_loaders
from ResUNet import ResUNet
from losses import BCEDiceLoss, dice_score
from inference import visualize_predictions


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_dice = dice_score(logits.detach(), masks).item()
        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "dice": f"{batch_dice:.4f}"}, refresh=False
        )

        total_loss += loss.item() * images.size(0)
        total_dice += batch_dice * images.size(0)

    count = max(1, len(loader.dataset))
    return total_loss / count, total_dice / count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    pbar = tqdm(loader, desc="val", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        batch_dice = dice_score(logits, masks).item()
        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "dice": f"{batch_dice:.4f}"}, refresh=False
        )

        total_loss += loss.item() * images.size(0)
        total_dice += batch_dice * images.size(0)

    count = max(1, len(loader.dataset))
    return total_loss / count, total_dice / count


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def run_training(args) -> None:
    seed_everything(args.seed)

    file_paths = list_npz_files(
        data_root=args.data_root,
        hf_repo_id=args.hf_repo_id,
        repo_prefix=args.repo_prefix,
        cache_dir=args.cache_dir,
    )

    if not file_paths:
        raise FileNotFoundError("No .npz files found for training.")

    train_loader, val_loader = build_loaders(
        file_paths=file_paths,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        slice_mode=args.slice_mode,
        negative_stride=args.negative_stride,
        num_workers=args.num_workers,
    )

    device = default_device()
    model = ResUNet(num_classes=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    checkpoint_path = Path(args.checkpoint_path)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_loss)
            print(f"Saved checkpoint to {checkpoint_path}")

    if getattr(args, "visualize", False):
        viz_out = Path(getattr(args, "viz_out", "outputs/visuals"))
        viz_count = int(getattr(args, "viz_count", 4))
        visualize_predictions(
            model, file_paths, args.image_size, device, viz_out, viz_count
        )
