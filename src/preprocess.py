# src/preprocess.py
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import uniform_filter, median_filter
from pathlib import Path
import os

# ── Constants (computed from typical Sentinel-1 statistics) ──────────────────
VV_MEAN_DB, VV_STD_DB   = -12.0, 4.5
VH_MEAN_DB, VH_STD_DB   = -18.5, 4.0
RATIO_MEAN, RATIO_STD   =   6.5, 2.5


def load_band(filepath):
    """Load a single-band GeoTIFF, return array + metadata."""
    with rasterio.open(filepath) as src:
        band = src.read(1).astype(np.float64)
        meta = src.meta.copy()
    return band, meta


def to_db(band, eps=1e-10):
    """Convert linear sigma0 → decibel scale."""
    safe = np.where(band > 0, band, eps)
    return 10.0 * np.log10(safe)


def lee_speckle_filter(band, size=7):
    """
    Adaptive Lee filter — reduces speckle while preserving edges.
    Works on linear or dB scale (apply before dB conversion for best results).
    """
    mean    = uniform_filter(band, size=size)
    sq_mean = uniform_filter(band ** 2, size=size)
    variance = np.maximum(sq_mean - mean ** 2, 0)
    noise_var = np.mean(variance) / (1 + 1e-6)
    weight   = variance / (variance + noise_var + 1e-10)
    return mean + weight * (band - mean)


def clip_normalize(band_db, mean, std, n_std=3):
    """Clip outliers then min-max normalize to [0, 1]."""
    lo = mean - n_std * std
    hi = mean + n_std * std
    clipped = np.clip(band_db, lo, hi)
    return (clipped - lo) / (hi - lo + 1e-10)


def build_3channel(vv_path, vh_path):
    """
    Full preprocessing pipeline.
    Returns: normalized [H, W, 3] float32 array + rasterio metadata.
    Channels: [VV_norm, VH_norm, cross-pol ratio norm]
    """
    vv_raw, meta = load_band(vv_path)
    vh_raw, _    = load_band(vh_path)

    # 1. Speckle filter (in linear domain)
    vv_filt = lee_speckle_filter(vv_raw)
    vh_filt = lee_speckle_filter(vh_raw)

    # 2. Convert to dB
    vv_db = to_db(vv_filt)
    vh_db = to_db(vh_filt)

    # 3. Cross-polarization ratio (sensitive to vegetation + soil moisture)
    ratio_db = vv_db - vh_db

    # 4. Normalize each channel
    vv_norm    = clip_normalize(vv_db,    VV_MEAN_DB,  VV_STD_DB)
    vh_norm    = clip_normalize(vh_db,    VH_MEAN_DB,  VH_STD_DB)
    ratio_norm = clip_normalize(ratio_db, RATIO_MEAN,  RATIO_STD)

    stack = np.stack([vv_norm, vh_norm, ratio_norm], axis=-1).astype(np.float32)
    return stack, vv_db, meta


def extract_patches(image, mask=None, patch_size=256, stride=128):
    """
    Sliding-window patch extractor.
    stride < patch_size → overlapping patches → more training data.
    """
    H, W = image.shape[:2]
    img_patches, mask_patches, positions = [], [], []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            img_patches.append(image[y:y+patch_size, x:x+patch_size])
            positions.append((y, x))
            if mask is not None:
                mask_patches.append(mask[y:y+patch_size, x:x+patch_size])

    img_arr  = np.array(img_patches, dtype=np.float32)
    mask_arr = np.array(mask_patches, dtype=np.uint8) if mask is not None else None
    return img_arr, mask_arr, positions


def run_preprocessing(vv_path, vh_path, mask_path=None,
                      out_dir="data/patches", patch_size=256, stride=128):
    """
    End-to-end preprocessing: load → filter → dB → normalize → patch → save.
    Saves .npy files for images and masks.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("Building 3-channel stack...")
    stack, vv_db, meta = build_3channel(vv_path, vh_path)

    mask = None
    if mask_path:
        mask, _ = load_band(mask_path)
        mask = (mask > 0).astype(np.uint8)

    print(f"Image shape: {stack.shape}")
    print("Extracting patches...")
    img_patches, mask_patches, positions = extract_patches(
        stack, mask, patch_size, stride
    )

    np.save(os.path.join(out_dir, "images.npy"), img_patches)
    print(f"Saved {len(img_patches)} image patches → {out_dir}/images.npy")

    if mask_patches is not None:
        np.save(os.path.join(out_dir, "masks.npy"), mask_patches)
        print(f"Saved mask patches → {out_dir}/masks.npy")

    # Also save the full dB image for 3D visualization
    np.save(os.path.join(out_dir, "vv_db_full.npy"), vv_db.astype(np.float32))
    print("Saved full dB image for visualization.")

    return img_patches, mask_patches


if __name__ == "__main__":
    run_preprocessing(
        vv_path="data/raw/VV.tif",
        vh_path="data/raw/VH.tif",
        mask_path=None,  # no labels yet
        out_dir="data/patches",
        patch_size=256,
        stride=128
    )