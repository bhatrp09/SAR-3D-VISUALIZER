# src/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import random


class SARDataset(Dataset):
    """
    Loads preprocessed .npy patch files.
    Applies random augmentations during training.
    """

    def __init__(self, images_path, masks_path=None, augment=False):
        self.images  = np.load(images_path)   # [N, H, W, 3]
        self.masks   = np.load(masks_path) if masks_path else None
        self.augment = augment

        print(f"Loaded {len(self.images)} patches | "
              f"shape: {self.images[0].shape} | "
              f"masks: {'yes' if self.masks is not None else 'no'}")

    def __len__(self):
        return len(self.images)

    def _augment(self, img_tensor, mask_tensor):
        """Random horizontal flip + vertical flip + 90° rotations."""
        if random.random() > 0.5:
            img_tensor  = TF.hflip(img_tensor)
            if mask_tensor is not None:
                mask_tensor = TF.hflip(mask_tensor.unsqueeze(0)).squeeze(0)

        if random.random() > 0.5:
            img_tensor  = TF.vflip(img_tensor)
            if mask_tensor is not None:
                mask_tensor = TF.vflip(mask_tensor.unsqueeze(0)).squeeze(0)

        k = random.choice([0, 1, 2, 3])
        if k > 0:
            img_tensor  = torch.rot90(img_tensor,  k, dims=[1, 2])
            if mask_tensor is not None:
                mask_tensor = torch.rot90(mask_tensor, k, dims=[0, 1])

        return img_tensor, mask_tensor

    def __getitem__(self, idx):
        # [H, W, 3] → [3, H, W]
        img  = torch.from_numpy(self.images[idx]).permute(2, 0, 1)
        mask = None
        if self.masks is not None:
            mask = torch.from_numpy(self.masks[idx].astype(np.int64))

        if self.augment:
            img, mask = self._augment(img, mask)

        return (img, mask) if mask is not None else img


def get_dataloaders(images_path, masks_path, batch_size=8,
                    val_split=0.2, num_workers=0):
    """Split dataset into train/val and return DataLoaders."""
    full_ds = SARDataset(images_path, masks_path, augment=False)

    val_len   = int(len(full_ds) * val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    # Enable augmentation only on training set
    train_ds.dataset.augment = True

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    print(f"Train: {train_len} | Val: {val_len} | Batch size: {batch_size}")
    return train_loader, val_loader