from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import split_files


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
