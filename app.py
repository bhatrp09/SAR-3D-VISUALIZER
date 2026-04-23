# app.py
import streamlit as st
import numpy as np
import torch
import tempfile, os, sys
from pathlib import Path

sys.path.insert(0, "src")
from preprocess import build_3channel
from model import UNet
from predict import predict_full_image
from visualize import make_3d_surface

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAR Flood Detector",
    page_icon="🛰",
    layout="wide"
)

st.title("🛰 SAR Deep Learning — Flood Detection + 3D Visualization")
st.markdown(
    "Upload Sentinel-1 VV and VH GeoTIFF bands. "
    "The model segments flood water and renders the result as an interactive 3D surface."
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    patch_size  = st.select_slider("Patch size",  [128, 256, 512], value=256)
    stride      = st.select_slider("Stride",      [64, 128, 256],  value=128)
    downsample  = st.select_slider("3D resolution (downsample)", [2, 4, 8], value=4)
    weights_path = st.text_input("Model weights path", value="models/unet_sar.pth")

    st.markdown("---")
    st.markdown("**About**")
    st.markdown("U-Net trained on Sentinel-1 GRD imagery (VV + VH + cross-pol ratio)")

# ── File upload ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    vv_file = st.file_uploader("VV band GeoTIFF", type=["tif", "tiff"])
with col2:
    vh_file = st.file_uploader("VH band GeoTIFF", type=["tif", "tiff"])

# ── Run pipeline when both files uploaded ─────────────────────────────────────
if vv_file and vh_file:

    with tempfile.TemporaryDirectory() as tmpdir:
        vv_path = os.path.join(tmpdir, "VV.tif")
        vh_path = os.path.join(tmpdir, "VH.tif")

        with open(vv_path, "wb") as f: f.write(vv_file.read())
        with open(vh_path, "wb") as f: f.write(vh_file.read())

        # ── Preprocessing ─────────────────────────────────────────────────
        with st.spinner("Preprocessing SAR bands..."):
            stack, vv_db, meta = build_3channel(vv_path, vh_path)

        st.success(f"Preprocessed — image shape: {stack.shape}")

        col_prev1, col_prev2, col_prev3 = st.columns(3)
        with col_prev1:
            st.image(stack[:,:,0], caption="VV (normalized)", clamp=True)
        with col_prev2:
            st.image(stack[:,:,1], caption="VH (normalized)", clamp=True)
        with col_prev3:
            st.image(stack[:,:,2], caption="Cross-pol ratio", clamp=True)

        # ── Inference ─────────────────────────────────────────────────────
        if not Path(weights_path).exists():
            st.warning(
                "No model weights found. Using random weights for demo. "
                "Train first with: python src/train.py"
            )
            device = torch.device("cpu")
            model  = UNet(in_channels=3, num_classes=2).to(device)
            model.eval()

            with torch.no_grad():
                tensor = (torch.from_numpy(stack)
                          .permute(2, 0, 1)
                          .unsqueeze(0)
                          .to(device))
                # Resize if too large for memory
                import torch.nn.functional as F
                tensor = F.interpolate(tensor, size=(256, 256), mode="bilinear")
                logits = model(tensor)
                mask   = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
                # Resize back
                from PIL import Image
                mask = np.array(
                    Image.fromarray(mask).resize(
                        (stack.shape[1], stack.shape[0]), Image.NEAREST
                    )
                )
        else:
            with st.spinner("Running U-Net inference..."):
                mask, vv_db, meta = predict_full_image(
                    vv_path, vh_path, weights_path, patch_size, stride
                )

        flood_pct = 100 * mask.mean()
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Flood pixels", f"{mask.sum():,}")
        col_m2.metric("Flood coverage", f"{flood_pct:.1f}%")
        col_m3.metric("Image size", f"{mask.shape[0]}×{mask.shape[1]}")

        st.image(mask.astype(np.uint8) * 255,
                 caption="Segmentation mask (white = flood water)", clamp=True)

        # ── 3D Visualization ──────────────────────────────────────────────
        with st.spinner("Rendering 3D surface..."):
            fig = make_3d_surface(vv_db, mask, downsample=downsample)

        st.subheader("Interactive 3D flood map")
        st.plotly_chart(fig, use_container_width=True)

        # ── Downloads ─────────────────────────────────────────────────────
        html_str = fig.to_html(include_plotlyjs="cdn")
        st.download_button(
            label="Download 3D map (HTML)",
            data=html_str,
            file_name="sar_flood_3d.html",
            mime="text/html"
        )

        mask_bytes = mask.tobytes()
        st.download_button(
            label="Download mask (raw bytes)",
            data=mask_bytes,
            file_name="flood_mask.bin"
        )
else:
    st.info("Upload both VV and VH GeoTIFF files to begin.")

    with st.expander("Don't have SAR data? Here's how to get it"):
        st.code("""
# Option 1: Google Earth Engine (fastest)
pip install earthengine-api
earthengine authenticate
python src/download_gee.py

# Option 2: Copernicus Open Hub
# Register at scihub.copernicus.eu then:
python src/download.py

# Option 3: Use the Sen1Floods11 dataset (pre-labeled)
# Download from: github.com/cloudtostreet/Sen1Floods11
""", language="bash")