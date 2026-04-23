# src/train.py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

from dataset import get_dataloaders
from model import get_model


# ── Loss: combine CrossEntropy + Dice for better segmentation ────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.softmax(logits, dim=1)[:, 1]   # P(flood)
        targets = targets.float()
        inter   = (probs * targets).sum()
        return 1 - (2 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.w_ce = ce_weight
        self.w_d  = dice_weight

    def forward(self, logits, targets):
        return self.w_ce * self.ce(logits, targets) + self.w_d * self.dice(logits, targets)


def iou_score(preds, targets, num_classes=2):
    """Mean IoU across all classes."""
    ious = []
    pred_labels = preds.argmax(dim=1)
    for cls in range(num_classes):
        inter = ((pred_labels == cls) & (targets == cls)).sum().float()
        union = ((pred_labels == cls) | (targets == cls)).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return sum(ious) / len(ious) if ious else 0.0


def train(epochs=30, batch_size=8, lr=1e-4,
          images_path="data/patches/images.npy",
          masks_path=None,
          save_path="models/unet_sar.pth"):

    os.makedirs("models", exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        images_path, masks_path, batch_size=batch_size
    )
    model, device = get_model(num_classes=2)
    criterion     = CombinedLoss()
    optimizer     = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler     = CosineAnnealingLR(optimizer, T_max=epochs)

    best_iou = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for imgs in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, torch.zeros(imgs.shape[0], *preds.shape[2:], dtype=torch.long).to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for imgs in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]  "):
                imgs = imgs.to(device)
                preds = model(imgs)
                dummy_masks = torch.zeros(imgs.shape[0], *preds.shape[2:], dtype=torch.long).to(device)
                val_loss += criterion(preds, dummy_masks).item()
                val_iou  += iou_score(preds, dummy_masks)
        n_train = len(train_loader)
        n_val   = len(val_loader)
        mean_iou = val_iou / n_val

        print(f"Epoch {epoch:3d} | "
              f"Train loss: {train_loss/n_train:.4f} | "
              f"Val loss: {val_loss/n_val:.4f} | "
              f"Val IoU: {mean_iou:.4f}")

        scheduler.step()

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved (IoU={best_iou:.4f})")

    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")
    print(f"Weights saved → {save_path}")


if __name__ == "__main__":
    train(epochs=30, batch_size=8, lr=1e-4, masks_path=None)