import os
from pathlib import Path
from io import BytesIO
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download


# ----------------------------
# CONFIG / PATHS
# ----------------------------
def find_project_root(start: Path = Path.cwd()):
    cur = start.resolve()
    for _ in range(10):
        if (cur / "data").exists() and (cur / "data" / "raw").exists():
            return cur
        if (cur / "requirements.txt").exists() or (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


PROJECT_ROOT = find_project_root()
ANNOTATIONS = PROJECT_ROOT / "data" / "raw" / "annotations.csv"

SOURCE_REPOS = {
    "37voy/Luna16s0456": ["subset0", "subset4", "subset5", "subset6"],
    "37voy/Luna16s1278": ["subset1", "subset2", "subset7", "subset8"],
    "37voy/Luna16s39": ["subset3", "subset9"],
}

PROCESSED_REPO_ID = "37voy/Luna16processed"


# ----------------------------
# CORE HELPERS
# ----------------------------
def sample_to_npz_bytes(img, mask, spacing, origin):
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        image=img,
        mask=mask,
        spacing=np.asarray(spacing, dtype=np.float32),
        origin=np.asarray(origin, dtype=np.float32),
    )
    buffer.seek(0)
    return buffer.getvalue()


def get_source_scan_path(repo_id, mhd_path):
    mhd_local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=mhd_path)
    raw_path = mhd_path.replace(".mhd", ".raw")
    hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=raw_path)
    return mhd_local


def load_mhd(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing = img.GetSpacing()  # (x, y, z)
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return arr, spacing, origin, direction


def normalize_hu(arr):
    arr = np.clip(arr, -1000, 400)
    arr = (arr + 1000) / 1400
    return arr.astype(np.float32)


# ----------------------------
# ANNOTATIONS / MASKS
# ----------------------------
def load_annotations(path: Path):
    df = pd.read_csv(str(path))
    return df


def make_nodule_mask(scan_uid, arr, spacing, origin, df):
    mask = np.zeros_like(arr, dtype=np.uint8)
    nods = df[df["seriesuid"] == scan_uid]

    for _, row in nods.iterrows():
        world = np.array([row["coordZ"], row["coordY"], row["coordX"]])
        diameter = row["diameter_mm"]

        # world → voxel
        voxel = np.abs((world - origin[::-1]) / spacing[::-1]).astype(int)
        radius = int(diameter / 2 / spacing[0])

        z, y, x = voxel
        zz, yy, xx = np.ogrid[: arr.shape[0], : arr.shape[1], : arr.shape[2]]
        dist = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2
        mask[dist <= radius**2] = 1

    return mask


# ----------------------------
# SCAN COLLECTION
# ----------------------------
def collect_source_scans(source_repos):
    api = HfApi()
    scan_items = []

    all_allowed = {subset for subsets in source_repos.values() for subset in subsets}
    print("Allowed subsets:", all_allowed)

    for repo_id, subset_names in source_repos.items():
        print(f"\nScanning repo: {repo_id}")
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

        for file_name in files:
            if not file_name.endswith(".mhd"):
                continue

            parts = Path(file_name).parts
            subset_name = next((p for p in parts if p in all_allowed), None)
            if subset_name is None:
                print(f"Skipping file with no valid subset: {file_name}")
                continue

            scan_uid = Path(file_name).stem
            scan_items.append((repo_id, subset_name, file_name))
            print(f"✓ Found scan: subset={subset_name}, uid={scan_uid}")

    print(f"\nTotal scans found: {len(scan_items)}")
    return scan_items


def ensure_local_cache(scan_items):
    print("Pre-downloading all scans into HF cache...")
    for source_repo_id, subset_name, file_name in tqdm(
        scan_items, desc="Caching scans"
    ):
        mhd = file_name
        raw = file_name.replace(".mhd", ".raw")
        hf_hub_download(repo_id=source_repo_id, repo_type="dataset", filename=mhd)
        hf_hub_download(repo_id=source_repo_id, repo_type="dataset", filename=raw)
    print("All scans cached locally.")


# ----------------------------
# HF HELPERS
# ----------------------------
def file_exists_on_hf(api, repo_id, path_in_repo):
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    return path_in_repo in files


# ----------------------------
# WORKER FUNCTION
# ----------------------------
def preprocess_single_scan(item, df):
    source_repo_id, subset_name, file_name = item

    # Download & load
    mhd_path = get_source_scan_path(source_repo_id, file_name)
    arr, spacing, origin, _direction = load_mhd(mhd_path)

    # Preprocess
    arr_norm = normalize_hu(arr)
    scan_uid = Path(file_name).stem
    mask = make_nodule_mask(scan_uid, arr, spacing, origin, df)

    # Pack
    npz_bytes = sample_to_npz_bytes(arr_norm, mask, spacing, origin)
    return subset_name, scan_uid, npz_bytes


# ----------------------------
# MAIN
# ----------------------------
def main():
    print("Using PROJECT_ROOT:", PROJECT_ROOT)
    print("ANNOTATIONS:", ANNOTATIONS, "exists:", ANNOTATIONS.exists())
    print("SOURCE_REPOS:", SOURCE_REPOS)
    print("PROCESSED_REPO_ID:", PROCESSED_REPO_ID)

    df = load_annotations(ANNOTATIONS)
    api = HfApi()
    api.create_repo(repo_id=PROCESSED_REPO_ID, repo_type="dataset", exist_ok=True)

    scan_items = collect_source_scans(SOURCE_REPOS)
    ensure_local_cache(scan_items)

    # Filter out already uploaded scans
    remaining = []
    for source_repo_id, subset_name, file_name in scan_items:
        scan_uid = Path(file_name).stem
        path_in_repo = f"processed/{subset_name}/{scan_uid}.npz"
        if not file_exists_on_hf(api, PROCESSED_REPO_ID, path_in_repo):
            remaining.append((source_repo_id, subset_name, file_name))

    print(f"Already uploaded: {len(scan_items) - len(remaining)}")
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("Nothing left to process.")
        return

    workers = max(1, cpu_count() - 1)
    print(f"Using {workers} parallel workers")

    # Parallel preprocessing with progress bar
    results = []
    with Pool(workers) as pool:
        process_fn = partial(preprocess_single_scan, df=df)
        for result in tqdm(
            pool.imap_unordered(process_fn, remaining),
            total=len(remaining),
            desc="Preprocessing scans",
            unit="scan",
            smoothing=0.1,
        ):
            results.append(result)

    # Upload with progress bar
    for subset_name, scan_uid, npz_bytes in tqdm(
        results,
        desc="Uploading to HF",
        unit="file",
        smoothing=0.1,
    ):
        api.upload_file(
            path_or_fileobj=BytesIO(npz_bytes),
            path_in_repo=f"processed/{subset_name}/{scan_uid}.npz",
            repo_id=PROCESSED_REPO_ID,
            repo_type="dataset",
            commit_message=f"Add processed scan {scan_uid}",
        )

    print("Done.")


if __name__ == "__main__":
    main()
