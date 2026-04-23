---
title: SAR Flood Detector
emoji: 🛰
colorFrom: blue
colorTo: teal
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# SAR Deep Learning — Flood Detection + 3D Visualization

Upload Sentinel-1 VV and VH GeoTIFF bands to detect flood water using a U-Net
and render the result as an interactive 3D surface.

## How to use
1. Download SAR data from Copernicus Hub or Google Earth Engine
2. Upload VV and VH bands
3. View the segmentation mask and 3D map
4. Download the interactive HTML

## Model
U-Net trained on Sentinel-1 GRD imagery with VV, VH, and cross-polarization ratio channels.