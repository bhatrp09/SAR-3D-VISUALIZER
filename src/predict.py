# src/predict.py
import numpy as np
import torch
import rasterio
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from model import UNet
from preprocess import build_3channel


def load_model(weights_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet(in_channels=3, num_classes=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device


def predict_full_image(vv_path, vh_path, weights_path="models/unet_sar.pth",
                        patch_size=256, stride=128):
    """
    Run inference on a full-size SAR image using sliding window.
    Aggregates overlapping patch predictions by averaging logits.
    Returns: binary mask [H, W] and vv_db [H, W] for 3D viz.
    """
    model, device = load_model(weights_path)
    stack, vv_db, meta = build_3channel(vv_path, vh_path)
    H, W, _ = stack.shape

    # Accumulate logit sums + counts for overlapping patches
    logit_sum = np.zeros((2, H, W), dtype=np.float32)
    count_map = np.zeros((H, W),    dtype=np.float32)

    print("Running inference...")
    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch = stack[y:y+patch_size, x:x+patch_size]
                tensor = (torch.from_numpy(patch)
                          .permute(2, 0, 1)
                          .unsqueeze(0)
                          .to(device))
                logits = model(tensor).squeeze(0).cpu().numpy()
                logit_sum[:, y:y+patch_size, x:x+patch_size] += logits
                count_map[y:y+patch_size, x:x+patch_size]    += 1

    # Average overlapping regions
    count_map = np.maximum(count_map, 1)
    avg_logits = logit_sum / count_map[np.newaxis]
    binary_mask = np.argmax(avg_logits, axis=0).astype(np.uint8)

    print(f"Flood pixels detected: {binary_mask.sum()} "
          f"({100*binary_mask.mean():.1f}% of scene)")

    return binary_mask, vv_db, meta


def save_prediction_geotiff(mask, meta, out_path="data/processed/prediction.tif"):
    """Save the prediction mask as a GeoTIFF with proper georeference."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_meta = meta.copy()
    out_meta.update({"count": 1, "dtype": "uint8"})
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(mask[np.newaxis])
    print(f"Saved prediction GeoTIFF → {out_path}")


if __name__ == "__main__":
    mask, vv_db, meta = predict_full_image(
        vv_path="data/raw/VV.tif",
        vh_path="data/raw/VH.tif",
        weights_path="models/unet_sar.pth"
    )
    save_prediction_geotiff(mask, meta)

    # Save arrays for visualization
    np.save("data/processed/mask_pred.npy", mask)
    np.save("data/processed/vv_db.npy", vv_db.astype(np.float32))
    print("Arrays saved for 3D visualization.")