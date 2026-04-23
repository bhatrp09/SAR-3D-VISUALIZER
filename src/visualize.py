# src/visualize.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def normalize_for_viz(arr):
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-10), 0, 1)


def make_3d_surface(vv_db, mask, downsample=4, title="SAR Flood Detection — 3D"):
    """
    Render SAR backscatter as a 3D terrain surface,
    colored by the segmentation mask.
    """
    z = vv_db[::downsample, ::downsample]
    c = mask[::downsample, ::downsample].astype(float)
    H, W = z.shape
    x = np.linspace(0, vv_db.shape[1], W)
    y = np.linspace(0, vv_db.shape[0], H)

    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=z,
        surfacecolor=c,
        colorscale=[[0, "royalblue"], [1, "orangered"]],
        cmin=0, cmax=1,
        colorbar=dict(
            title="Class",
            tickvals=[0.25, 0.75],
            ticktext=["Dry land", "Flood water"],
            len=0.5
        ),
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
        lightposition=dict(x=200, y=200, z=500)
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis=dict(title="Longitude (px)", showgrid=False),
            yaxis=dict(title="Latitude (px)",  showgrid=False),
            zaxis=dict(title="Backscatter (dB)"),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.3)
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=600
    )
    return fig


def make_before_after(vv_before, vv_after, mask, downsample=4):
    """Side-by-side 3D subplots for change detection."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("Before event", "After event — flood detected")
    )

    for col, (vv, title) in enumerate(
            [(vv_before, "Before"), (vv_after, "After")], start=1):
        z = vv[::downsample, ::downsample]
        c = mask[::downsample, ::downsample].astype(float) if col == 2 else np.zeros_like(z)
        H, W = z.shape
        x = np.linspace(0, vv.shape[1], W)
        y = np.linspace(0, vv.shape[0], H)

        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            surfacecolor=c,
            colorscale=[[0, "steelblue"], [1, "orangered"]],
            showscale=(col == 2),
            cmin=0, cmax=1
        ), row=1, col=col)

    fig.update_layout(height=500, title="Change detection — before vs after")
    return fig


if __name__ == "__main__":
    vv_db = np.load("data/processed/vv_db.npy")
    mask  = np.load("data/processed/mask_pred.npy")

    fig = make_3d_surface(vv_db, mask, downsample=4)
    fig.write_html("data/processed/sar_3d.html")
    print("Saved interactive 3D → data/processed/sar_3d.html")
    fig.show()