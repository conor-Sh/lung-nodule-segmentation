"""Train a 2D ResUNet on preprocessed lung CT .npz volumes.

The script loads preprocessed volumes, converts them into 2D slices, resizes
them to a fixed square size for batching, and trains a lightweight ResUNet.

It supports either local data or Hugging Face dataset repos.
"""

from __future__ import annotations

import argparse
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import HfApi, hf_hub_download
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_npz_files(
    data_root: str | None = None,
    hf_repo_id: str | None = None,
    repo_prefix: str | None = None,
    cache_dir: str | None = None,
) -> list[Path]:
    if data_root:
        root = Path(data_root)
        return sorted(root.rglob("*.npz"))

    if not hf_repo_id:
        raise ValueError("Provide either data_root or hf_repo_id.")

    api = HfApi()
    prefix = repo_prefix.strip("/") if repo_prefix else ""
    file_paths: list[Path] = []

    for file_name in api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset"):
        if not file_name.endswith(".npz"):
            continue
        if prefix and not file_name.startswith(prefix + "/"):
            continue
        local_path = hf_hub_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            filename=file_name,
            cache_dir=cache_dir,
        )
        file_paths.append(Path(local_path))

    return sorted(file_paths)


def split_files(
    file_paths: Sequence[Path],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    file_paths = list(file_paths)
    rng = random.Random(seed)
    rng.shuffle(file_paths)

    val_count = (
        max(1, int(len(file_paths) * val_fraction)) if len(file_paths) > 1 else 0
    )
    val_files = file_paths[:val_count]
    train_files = file_paths[val_count:]

    if not train_files and val_files:
        train_files = val_files[:1]
        val_files = val_files[1:]

    return train_files, val_files


def load_npz_volume(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as sample:
        image = sample["image"]
        mask = sample["mask"]
        spacing = sample["spacing"] if "spacing" in sample.files else None
        origin = sample["origin"] if "origin" in sample.files else None
    return image, mask, spacing, origin


def ensure_3d(volume: np.ndarray) -> np.ndarray:
    if volume.ndim == 2:
        return volume[None, ...]
    if volume.ndim == 3:
        return volume
    raise ValueError(f"Expected a 2D or 3D array, got shape {volume.shape}")


def select_slice(volume: np.ndarray, slice_index: int) -> np.ndarray:
    if volume.ndim == 2:
        return volume
    return volume[slice_index]


def resize_tensor_pair(
    image: torch.Tensor, mask: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    image = F.interpolate(
        image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)
    mask = F.interpolate(mask.unsqueeze(0), size=(size, size), mode="nearest").squeeze(
        0
    )
    return image, mask


def random_flip(
    image: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[1])
    return image, mask


def add_noise(
    image: torch.Tensor, mask: torch.Tensor, sigma: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() < 0.5:
        noise = torch.randn_like(image) * sigma
        image = torch.clamp(image + noise, 0.0, 1.0)
    return image, mask


def apply_transforms(
    image: torch.Tensor,
    mask: torch.Tensor,
    transforms: Sequence,
) -> tuple[torch.Tensor, torch.Tensor]:
    for transform in transforms:
        image, mask = transform(image, mask)
    return image, mask


@dataclass(frozen=True)
class SliceRef:
    file_path: Path
    slice_index: int


class LungSliceDataset(Dataset):
    def __init__(
        self,
        file_paths: Sequence[Path],
        image_size: int = 256,
        slice_mode: str = "mixed",
        negative_stride: int = 8,
        augment: bool = False,
    ):
        self.file_paths = [Path(path) for path in file_paths]
        self.image_size = image_size
        self.slice_mode = slice_mode
        self.negative_stride = max(1, negative_stride)
        self.augment = augment
        self.transforms = [random_flip, add_noise] if augment else []
        self.samples = self._build_index()

    def _build_index(self) -> list[SliceRef]:
        samples: list[SliceRef] = []

        for file_path in self.file_paths:
            image, mask, _, _ = load_npz_volume(file_path)
            image = ensure_3d(image)
            mask = ensure_3d(mask)

            num_slices = image.shape[0]
            positive_slices = np.where(mask.reshape(num_slices, -1).sum(axis=1) > 0)[
                0
            ].tolist()

            if self.slice_mode == "all":
                slice_indices = list(range(num_slices))
            elif self.slice_mode == "positive":
                slice_indices = positive_slices or [num_slices // 2]
            elif self.slice_mode == "mixed":
                negative_slices = [
                    idx for idx in range(num_slices) if idx not in set(positive_slices)
                ]
                slice_indices = (
                    positive_slices + negative_slices[:: self.negative_stride]
                )
                if not slice_indices:
                    slice_indices = [num_slices // 2]
            else:
                raise ValueError("slice_mode must be one of: all, positive, mixed")

            samples.extend(
                SliceRef(file_path=file_path, slice_index=idx) for idx in slice_indices
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ref = self.samples[index]
        image, mask, _, _ = load_npz_volume(ref.file_path)
        image = ensure_3d(image)
        mask = ensure_3d(mask)

        image_slice = select_slice(image, ref.slice_index)
        mask_slice = select_slice(mask, ref.slice_index)

        image_tensor = torch.from_numpy(image_slice.astype(np.float32)).unsqueeze(0)
        mask_tensor = torch.from_numpy((mask_slice > 0).astype(np.float32)).unsqueeze(0)

        image_tensor, mask_tensor = resize_tensor_pair(
            image_tensor, mask_tensor, self.image_size
        )

        if self.transforms:
            image_tensor, mask_tensor = apply_transforms(
                image_tensor, mask_tensor, self.transforms
            )

        return image_tensor, mask_tensor


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.projection is not None:
            identity = self.projection(identity)

        out += identity
        return F.relu(out)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.res_block(x)
        down = self.max_pool(skip)
        return skip, down


class UpBlock(nn.Module):
    def __init__(self, decoder_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up_convolution = nn.ConvTranspose2d(
            decoder_channels, out_channels, kernel_size=2, stride=2
        )
        self.res_block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_convolution(x)
        if x.size() != skip.size():
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.res_block(x)


class ResUNet(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.down1 = Encoder(1, 64)
        self.down2 = Encoder(64, 128)
        self.down3 = Encoder(128, 256)
        self.down4 = Encoder(256, 512)

        self.bottleneck = ResidualBlock(512, 1024)

        self.up1 = UpBlock(1024, 512, 512)
        self.up2 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up4 = UpBlock(128, 64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.bottleneck(x)
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        return self.final_conv(x)


def dice_loss(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice = dice_loss(pred, target)
        return bce_loss + dice


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    pred = (torch.sigmoid(pred) > threshold).float()
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    score = (2.0 * intersection + smooth) / (denom + smooth)
    return score.mean()


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


def build_loaders(
    file_paths: Sequence[Path],
    image_size: int,
    batch_size: int,
    val_fraction: float,
    seed: int,
    slice_mode: str,
    negative_stride: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_files, val_files = split_files(
        file_paths, val_fraction=val_fraction, seed=seed
    )

    train_dataset = LungSliceDataset(
        train_files,
        image_size=image_size,
        slice_mode=slice_mode,
        negative_stride=negative_stride,
        augment=True,
    )
    val_dataset = LungSliceDataset(
        val_files,
        image_size=image_size,
        slice_mode="positive" if slice_mode != "all" else "all",
        negative_stride=negative_stride,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def create_synthetic_npz_dataset(
    root: Path, num_files: int = 4, depth: int = 16, size: int = 96
) -> list[Path]:
    # synthetic dataset generation removed in final script
    raise RuntimeError(
        "create_synthetic_npz_dataset was removed. Use real processed .npz files for training."
    )


def run_training(args: argparse.Namespace) -> None:
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

    # optional visualization after training
    if getattr(args, "visualize", False):
        viz_out = Path(getattr(args, "viz_out", "outputs/visuals"))
        viz_count = int(getattr(args, "viz_count", 4))
        visualize_predictions(
            model, file_paths, args.image_size, device, viz_out, viz_count
        )


def infer_single_scan(
    scan_path: Path,
    model: nn.Module,
    image_size: int,
    device: torch.device,
    out_dir: Path | str = "outputs/predictions",
    threshold: float = 0.5,
) -> None:
    """Run model on a single .npz scan and save probability volume + example PNGs.

    Writes: <scan>_pred.npz (probs volume) and a center-slice PNG showing image/mask/pred.
    """
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

    # save center-slice visual for quick inspection
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

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(img_c, cmap="gray")
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


def run_smoke_test() -> None:
    # smoke-test removed in final script; keep function name reserved
    raise RuntimeError(
        "run_smoke_test is removed. Use real data with --data-root or --hf-repo-id."
    )


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
        "--smoke-test",
        action="store_true",
        help="(deprecated) removed in final script; do not use",
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


def visualize_predictions(
    model: nn.Module,
    file_paths: Sequence[Path],
    image_size: int,
    device: torch.device,
    out_dir: Path | str = "outputs/visuals",
    n_examples: int = 4,
) -> None:
    """Run the model on a few slices and save side-by-side images (input, mask, prediction).

    This helper writes PNGs to out_dir/pred_###.png
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sample files to visualise (deterministic)
    rng = random.Random(42)
    sample_files = rng.sample(list(file_paths), k=min(n_examples, len(file_paths)))

    model.eval()
    with torch.no_grad():
        for i, fp in enumerate(sample_files):
            image, mask, _, _ = load_npz_volume(fp)
            image = ensure_3d(image)
            mask = ensure_3d(mask)

            # pick center slice
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


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # If user requested single-scan inference, load model checkpoint and run inference
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
            args.predict_scan, model, args.image_size, device, args.predict_out
        )
        return

    run_training(args)


if __name__ == "__main__":
    main()
