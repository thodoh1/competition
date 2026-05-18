"""
dataset.py – PyTorch Dataset for infrared LR/HR image pairs.

Directory layout expected (matches competition structure):
    data_root/
        input_320/   ← LR images for Stage 1 (or input_160/ for Stage 2)
        target_640/  ← HR ground-truth images

Both folders must contain grayscale PNG files with matching filenames.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_gray(path: str) -> np.ndarray:
    """Load a grayscale PNG and return a float32 array in [0, 1]."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H, W) → (1, H, W) tensor."""
    return torch.from_numpy(arr).unsqueeze(0)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class IRSuperResDataset(Dataset):
    """
    Paired LR / HR dataset for infrared super-resolution.

    Args:
        lr_dir      : folder with low-resolution images
        hr_dir      : folder with high-resolution ground truth
        scale       : upscale factor (2 or 4) – used for validation only
        patch_size  : HR patch size for training crops (0 = full image)
        augment     : apply random flip/rotation during training
        filenames   : optional list of filenames to use (subset)
    """

    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        scale: int = 2,
        patch_size: int = 128,
        augment: bool = True,
        filenames: Optional[list] = None,
    ):
        self.lr_dir     = Path(lr_dir)
        self.hr_dir     = Path(hr_dir)
        self.scale      = scale
        self.patch_size = patch_size  # HR patch size (0 = no crop)
        self.augment    = augment

        # Discover paired files
        all_files = sorted(f.name for f in self.hr_dir.glob("*.png"))
        if filenames:
            all_files = [f for f in all_files if f in filenames]

        # Keep only pairs that exist in both directories
        self.filenames = [f for f in all_files
                          if (self.lr_dir / f).exists() and (self.hr_dir / f).exists()]

        if len(self.filenames) == 0:
            raise RuntimeError(
                f"No paired files found.\n  LR: {lr_dir}\n  HR: {hr_dir}"
            )

        print(f"[Dataset] {len(self.filenames)} pairs | "
              f"scale={scale} | patch_size={patch_size} | augment={augment}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]
        lr = load_gray(str(self.lr_dir / fname))
        hr = load_gray(str(self.hr_dir / fname))

        # Random crop (training)
        if self.patch_size > 0:
            lr, hr = self._random_crop(lr, hr)

        # Augmentation
        if self.augment:
            lr, hr = self._augment(lr, hr)

        return to_tensor(lr), to_tensor(hr)

    # ── private ─────────────────────────────────

    def _random_crop(
        self, lr: np.ndarray, hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crop a matching patch from LR and HR."""
        lr_ps = self.patch_size // self.scale
        hr_ps = self.patch_size

        h_lr, w_lr = lr.shape
        if h_lr < lr_ps or w_lr < lr_ps:
            # Image smaller than patch – resize LR/HR to minimum
            lr = np.array(Image.fromarray((lr * 255).astype(np.uint8))
                          .resize((max(w_lr, lr_ps), max(h_lr, lr_ps)), Image.BICUBIC),
                          dtype=np.float32) / 255.0
            hr = np.array(Image.fromarray((hr * 255).astype(np.uint8))
                          .resize((max(w_lr, lr_ps) * self.scale,
                                   max(h_lr, lr_ps) * self.scale), Image.BICUBIC),
                          dtype=np.float32) / 255.0
            h_lr, w_lr = lr.shape

        top_lr  = random.randint(0, h_lr - lr_ps)
        left_lr = random.randint(0, w_lr - lr_ps)

        lr_crop = lr[top_lr:top_lr + lr_ps, left_lr:left_lr + lr_ps]
        hr_crop = hr[top_lr * self.scale:(top_lr + lr_ps) * self.scale,
                     left_lr * self.scale:(left_lr + lr_ps) * self.scale]
        return lr_crop, hr_crop

    @staticmethod
    def _augment(
        lr: np.ndarray, hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random horizontal / vertical flip and 90° rotations."""
        if random.random() < 0.5:
            lr = np.fliplr(lr).copy()
            hr = np.fliplr(hr).copy()
        if random.random() < 0.5:
            lr = np.flipud(lr).copy()
            hr = np.flipud(hr).copy()
        k = random.randint(0, 3)
        if k > 0:
            lr = np.rot90(lr, k).copy()
            hr = np.rot90(hr, k).copy()
        return lr, hr


# ──────────────────────────────────────────────
# Inference-only dataset (LR only, no HR needed)
# ──────────────────────────────────────────────

class IRInferDataset(Dataset):
    """
    Single-directory dataset for inference (no HR ground truth needed).
    Returns (tensor, filename) pairs.
    """

    def __init__(self, lr_dir: str):
        self.lr_dir   = Path(lr_dir)
        self.filenames = sorted(f.name for f in self.lr_dir.glob("*.png"))
        print(f"[InferDataset] {len(self.filenames)} images in {lr_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        fname = self.filenames[idx]
        lr    = load_gray(str(self.lr_dir / fname))
        return to_tensor(lr), fname


# ──────────────────────────────────────────────
# Train / val split utility
# ──────────────────────────────────────────────

def make_train_val_split(
    lr_dir: str,
    hr_dir: str,
    scale: int = 2,
    val_ratio: float = 0.1,
    patch_size: int = 128,
    seed: int = 42,
) -> Tuple[IRSuperResDataset, IRSuperResDataset]:
    """
    Split a paired directory into train and validation datasets.

    Returns:
        (train_dataset, val_dataset)
    """
    all_files = sorted(f.name for f in Path(hr_dir).glob("*.png")
                       if (Path(lr_dir) / f.name).exists())

    rng = random.Random(seed)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * val_ratio))

    val_files   = all_files[:n_val]
    train_files = all_files[n_val:]

    train_ds = IRSuperResDataset(
        lr_dir, hr_dir, scale=scale, patch_size=patch_size,
        augment=True, filenames=train_files,
    )
    val_ds = IRSuperResDataset(
        lr_dir, hr_dir, scale=scale, patch_size=0,
        augment=False, filenames=val_files,
    )
    return train_ds, val_ds
