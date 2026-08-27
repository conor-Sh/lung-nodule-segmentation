from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data import load_npz_volume, ensure_3d, select_slice


def to_display_slice(slice_array: np.ndarray) -> np.ndarray:
    slice_array = np.asarray(slice_array, dtype=np.float32)
    slice_min = float(slice_array.min())
    slice_max = float(slice_array.max())
    if slice_max > slice_min:
        return (slice_array - slice_min) / (slice_max - slice_min)
    return np.zeros_like(slice_array, dtype=np.float32)


def save_alpha_overlay(
    image_array: np.ndarray,
    output_path: Path,
    title: str,
    pred_mask: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
    pred_prob: np.ndarray | None = None,
) -> None:
    image = to_display_slice(image_array)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")

    if pred_prob is not None:
        ax.imshow(pred_prob, cmap="Reds", alpha=0.35, vmin=0.0, vmax=1.0)

    if gt_mask is not None:
        gt = np.asarray(gt_mask, dtype=np.float32)
        gt_overlay = np.zeros((*gt.shape, 4), dtype=np.float32)
        gt_overlay[..., 1] = gt
        gt_overlay[..., 3] = gt * 0.35
        ax.imshow(gt_overlay)

    if pred_mask is not None:
        pred = np.asarray(pred_mask, dtype=np.float32)
        pred_overlay = np.zeros((*pred.shape, 4), dtype=np.float32)
        pred_overlay[..., 0] = pred
        pred_overlay[..., 3] = pred * 0.35
        ax.imshow(pred_overlay)

    ax.set_title(title)
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def infer_single_scan(
    scan_path: Path,
    model: torch.nn.Module,
    image_size: int,
    device: torch.device,
    out_dir: Path | str = "outputs/predictions",
    threshold: float = 0.5,
) -> None:
    """Run model on a single .npz scan and save probability volume + example PNGs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image, mask, spacing, origin = load_npz_volume(Path(scan_path))
    image = ensure_3d(image)
    mask = ensure_3d(mask)

    depth = image.shape[0]
    probs = np.zeros((depth, image_size, image_size), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for s in range(depth):
            slice_img = select_slice(image, s)
            img_t = (
                torch.from_numpy(slice_img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            )
            img_t = F.interpolate(
                img_t,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )
            img_t = img_t.to(device)
            pred_logits = model(img_t)
            pred = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()
            probs[s] = pred

    stem = Path(scan_path).stem
    out_npz = out_dir / f"{stem}_pred.npz"
    np.savez_compressed(out_npz, probs=probs, spacing=spacing, origin=origin)

    c = depth // 2
    prob_c = probs[c]
    img_c = F.interpolate(
        torch.from_numpy(select_slice(image, c).astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    mask_c = F.interpolate(
        torch.from_numpy((select_slice(mask, c) > 0).astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0),
        size=(image_size, image_size),
        mode="nearest",
    )[0, 0].numpy()
    pred_bin = (prob_c > threshold).astype(np.uint8)

    save_alpha_overlay(
        image_array=img_c,
        output_path=out_dir / f"{stem}_center_overlay.png",
        title="prediction overlay",
        pred_mask=pred_bin,
        pred_prob=prob_c,
    )

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(to_display_slice(img_c), cmap="gray")
    axes[0].set_title("image")
    axes[0].axis("off")
    axes[1].imshow(mask_c, cmap="gray")
    axes[1].set_title("mask")
    axes[1].axis("off")
    axes[2].imshow(pred_bin, cmap="gray")
    axes[2].set_title("pred")
    axes[2].axis("off")
    fig.savefig(out_dir / f"{stem}_center.png", bbox_inches="tight")
    plt.close(fig)


def visualize_predictions(
    model: torch.nn.Module,
    file_paths: Sequence[Path],
    image_size: int,
    device: torch.device,
    out_dir: Path | str = "outputs/visuals",
    n_examples: int = 4,
) -> None:
    """Run the model on a few slices and save validation composites and overlays."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    sample_files = rng.sample(list(file_paths), k=min(n_examples, len(file_paths)))

    model.eval()
    with torch.no_grad():
        for i, fp in enumerate(sample_files):
            image, mask, _, _ = load_npz_volume(fp)
            image = ensure_3d(image)
            mask = ensure_3d(mask)

            s = image.shape[0] // 2
            img_slice = select_slice(image, s)
            mask_slice = select_slice(mask, s) > 0

            img_t = (
                torch.from_numpy(img_slice.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            )
            img_t = F.interpolate(
                img_t,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )
            img_t = img_t.to(device)

            pred_logits = model(img_t)
            pred = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()
            pred_bin = (pred > 0.5).astype(np.uint8)

            img_np = img_t[0, 0].cpu().numpy()
            mask_np = F.interpolate(
                torch.from_numpy(mask_slice.astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0),
                size=(image_size, image_size),
                mode="nearest",
            )[0, 0].numpy()

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("image")
            axes[0].axis("off")

            axes[1].imshow(mask_np, cmap="gray")
            axes[1].set_title("mask")
            axes[1].axis("off")

            axes[2].imshow(pred_bin, cmap="gray")
            axes[2].set_title("pred")
            axes[2].axis("off")

            out_path = out_dir / f"pred_{i:03d}.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

            save_alpha_overlay(
                image_array=img_np,
                output_path=out_dir / f"pred_{i:03d}_overlay.png",
                title="val overlay",
                gt_mask=mask_np > 0.5,
                pred_mask=pred_bin > 0,
                pred_prob=pred,
            )
