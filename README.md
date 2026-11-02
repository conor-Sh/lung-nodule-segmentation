# Lung Nodule Segmentation

A work-in-progress pipeline for 2D lung nodule segmentation on LUNA16 CT scans, covering preprocessing, ResUNet training, and single-scan inference.

> [!IMPORTANT]
> **Project status: work in progress.** The preprocessing pipeline, 2D ResUNet trainer, validation loop, checkpointing, and single-scan inference with visualisations are implemented and runnable. This repository does not yet include a published checkpoint, a held-out evaluation, or recorded metrics, and the model is 2D rather than volumetric. No results are claimed.

## Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Method](#method)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Outputs](#outputs)
- [Limitations and roadmap](#limitations-and-roadmap)
- [Acknowledgements](#acknowledgements)

## Overview

The project turns LUNA16 CT volumes and nodule annotations into compressed `.npz` samples containing a normalised HU image and a binary nodule mask. A 2D ResUNet is then trained on axial slices with a combined BCE and Dice loss, and a trained checkpoint can be applied to a single preprocessed scan to produce a probability volume and visual overlays.

## Dataset

[LUNA16](https://luna16.grand-challenge.org/) is a lung nodule detection dataset derived from the LIDC-IDRI collection. The scans are CT volumes distributed as `.mhd` headers with companion `.raw` files.

Nodule annotations live in `data/raw/annotations.csv`, which contains 1,186 annotation rows. Each row describes one nodule:

| Column | Meaning |
| --- | --- |
| `seriesuid` | DICOM series identifier for the scan |
| `coordX`, `coordY`, `coordZ` | Nodule centre in world coordinates (mm) |
| `diameter_mm` | Nodule diameter in millimetres |

Preprocessing produces one `.npz` file per scan. Each file contains:

| Key | Contents |
| --- | --- |
| `image` | HU volume clipped to [-1000, 400] HU and scaled to [0, 1] |
| `mask` | Binary (`uint8`) nodule mask aligned with the image volume |
| `spacing` | Voxel spacing |
| `origin` | Volume origin |

The scan data itself is not committed to this repository. `src/preprocess.py` downloads the configured LUNA16 subsets from Hugging Face dataset repositories and uploads processed scans to a Hugging Face dataset repo.

## Method

### Preprocessing

- HU values are clipped to the lung window of [-1000, 400] HU and mapped to [0, 1] as `(value + 1000) / 1400`.
- Each annotation's world coordinate is converted to a voxel index, and a sphere is rasterised into the mask using a radius derived from `diameter_mm`.
- The normalised image, binary mask, spacing, and origin are written to a compressed `.npz` file.

### Slice sampling

Training operates on 2D axial slices. `--slice-mode` controls which slices are used:

- `all`: every slice in every volume.
- `positive`: only slices that contain part of a nodule mask (falling back to the middle slice when a volume has none).
- `mixed` (default): all positive slices plus every `--negative-stride`-th negative slice (default stride 8).

Validation uses positive slices by default, or all slices when `--slice-mode all` is selected, and never applies augmentation.

### Model

`src/ResUNet.py` implements a residual U-Net with four encoder stages (64, 128, 256, and 512 channels), each built from residual blocks with batch normalisation and a projection shortcut. A 1024-channel residual bottleneck sits between the encoder and decoder. The decoder uses transposed convolutions, skip concatenation, and residual blocks, followed by a 1x1 convolution to a single-channel logit map.

### Loss, metric, and augmentation

- Loss: binary cross-entropy with logits plus soft Dice loss (`src/losses.py`).
- Metric: Dice coefficient with a 0.5 probability threshold, reported per epoch on the validation split.
- Augmentation: random horizontal and vertical flips, plus Gaussian noise (`sigma = 0.05`) applied with 50% probability and clamped to [0, 1] (`src/data.py`).

Training uses Adam with a default learning rate of `1e-3`, batch size 8, image size 256, up to 10 epochs, and a 20% validation split. The split is a random file-level split seeded by `--seed` (default 42); the placeholder files in `data/splits/` are not currently used. The checkpoint with the best validation loss is saved.

## Repository structure

```text
.
├── data/
│   ├── raw/annotations.csv        # LUNA16 nodule annotations (1,186 rows)
│   └── splits/                    # empty placeholder split files (unused)
├── notebooks/
│   ├── preprocessing.ipynb
│   └── file visualisation.ipynb
├── src/
│   ├── main.py                    # CLI entry point for training and prediction
│   ├── preprocess.py              # LUNA16 → .npz preprocessing and upload
│   ├── data.py                    # dataset, slice sampling, augmentation
│   ├── training.py                # training and validation loops
│   ├── inference.py               # single-scan inference and visualisation
│   ├── losses.py                  # BCE + Dice loss and Dice metric
│   ├── ResUNet.py                 # residual U-Net model
│   ├── utils.py                   # device, seeding, data discovery, splits
│   └── implementation.py          # earlier self-contained variant (not the entry point)
├── requirements.txt
└── README.md
```

`src/main.py` is the working entry point. `src/implementation.py` is an earlier self-contained variant kept for reference and is not used by the training or inference commands below.

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists the third-party packages imported by the code without version pins. If the default PyTorch wheel does not match your CUDA or driver setup, install PyTorch first using the command from the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install the remaining requirements.

## Usage

Run all commands from the repository root.

### 1. Preprocess LUNA16 scans

```bash
python src/preprocess.py
```

This step requires network access and Hugging Face authentication. It reads `data/raw/annotations.csv`, downloads the LUNA16 subsets configured in `SOURCE_REPOS` (in `src/preprocess.py`), builds the nodule masks, and uploads compressed `.npz` files to the dataset repository named by `PROCESSED_REPO_ID` (`37voy/Luna16processed` by default). Scans already present in the target repository are skipped.

Authenticate with `huggingface-cli login` or by setting `HF_TOKEN`. The token needs read access to the source repositories and write access to the target repository. To use your own repositories, update `SOURCE_REPOS` and `PROCESSED_REPO_ID` in `src/preprocess.py`.

The script uploads its output to the Hub; it does not create a local `data/processed` directory.

### 2. Train

From locally available `.npz` files:

```bash
python src/main.py --data-root data/processed
```

Directly from a Hugging Face dataset repo:

```bash
python src/main.py --hf-repo-id 37voy/Luna16processed --repo-prefix processed
```

Common options:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--slice-mode` | `mixed` | `all`, `positive`, or `mixed` slice sampling |
| `--negative-stride` | `8` | Keep every Nth negative slice in `mixed` mode |
| `--image-size` | `256` | Square slice size fed to the model |
| `--batch-size` | `8` | Training batch size |
| `--epochs` | `10` | Maximum number of epochs |
| `--checkpoint-path` | `outputs/checkpoints/resunet_2d_best.pt` | Where the best checkpoint is written |
| `--visualize` | off | Save example predictions after training |

To resume from the configured checkpoint path:

```bash
python src/main.py --hf-repo-id 37voy/Luna16processed --resume
```

`--resume` reads the checkpoint file if it exists, but the current training loop still starts at epoch 1; restoring model and optimizer state is not implemented yet.

### 3. Predict on a single scan

```bash
python src/main.py \
  --predict-scan /path/to/scan.npz \
  --checkpoint-path outputs/checkpoints/resunet_2d_best.pt
```

The scan must be a preprocessed `.npz` file with the same keys produced by `src/preprocess.py`. If the checkpoint does not exist, the command warns and runs with randomly initialised weights.

## Outputs

- **Checkpoints:** the best validation-loss checkpoint is written to `outputs/checkpoints/resunet_2d_best.pt` by default. It contains the epoch, model state, optimizer state, and best validation loss.
- **Training log:** each epoch prints training and validation loss plus Dice.
- **Visualisations:** with `--visualize`, validation composites and overlays are written to `outputs/visuals/` as `pred_NNN.png` and `pred_NNN_overlay.png`.
- **Single-scan prediction:** `--predict-scan` writes to `outputs/predictions/`:
  - `<scan>_pred.npz` with per-slice probabilities (resized to `--image-size`), spacing, and origin.
  - `<scan>_center_overlay.png` with the predicted probability and binary mask over the centre slice.
  - `<scan>_center.png` with image, ground-truth mask, and prediction side by side.

## Limitations and roadmap

Current limitations:

- No held-out evaluation or reported metrics. Validation Dice is printed during training but is not a substitute for an independent test set.
- The model operates on 2D axial slices and has no 3D context across neighbouring slices.
- Splits are random at the file level rather than patient-level, so leakage between training and validation is possible.
- Mask generation uses axis-aligned world-to-voxel conversion and isotropic sphere radii derived from a single spacing value; direction cosines and anisotropic spacing are not fully modelled.
- This project is for research and learning only. It is not a medical device and must not be used for clinical decision-making.

Planned next steps:

- Add a held-out test evaluation with recorded metrics.
- Move to patient-level splits.
- Explore volumetric or slice-context models.
- Improve mask geometry for anisotropic spacing and image direction.
- Publish a trained checkpoint and reproducible evaluation commands.

## Acknowledgements

- [LUNA16](https://luna16.grand-challenge.org/) and the LIDC-IDRI collection for the CT data and annotations.
- [Hugging Face](https://huggingface.co/) for dataset hosting and the `huggingface_hub` client used by the preprocessing and data-loading code.
