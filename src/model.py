# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv → BN → ReLU blocks."""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """MaxPool → DoubleConv (encoder block)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Bilinear upsample + skip connection + DoubleConv (decoder block)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd-sized inputs
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for SAR semantic segmentation.
    in_channels=3  : VV, VH, cross-pol ratio
    num_classes=2  : background, flood water (or whatever your task is)
    """
    def __init__(self, in_channels=3, num_classes=2, base_features=64):
        super().__init__()
        f = base_features

        self.inc   = DoubleConv(in_channels, f)
        self.down1 = Down(f,     f*2)
        self.down2 = Down(f*2,   f*4)
        self.down3 = Down(f*4,   f*8)
        self.down4 = Down(f*8,   f*16)

        self.up1   = Up(f*16 + f*8,  f*8)
        self.up2   = Up(f*8  + f*4,  f*4)
        self.up3   = Up(f*4  + f*2,  f*2)
        self.up4   = Up(f*2  + f,    f)

        self.outc  = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


def get_model(num_classes=2, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet(in_channels=3, num_classes=num_classes).to(device)
    total  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net ready on {device} | {total/1e6:.1f}M parameters")
    return model, device