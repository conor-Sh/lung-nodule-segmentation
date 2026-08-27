from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download


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
    if data_root and Path(data_root).exists():
        root = Path(data_root)
        files = sorted(root.rglob("*.npz"))
        if files:
            return files

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
