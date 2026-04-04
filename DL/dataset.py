"""
module: DL/dataset.py

PyTorch Dataset for binary semantic segmentation tiles.

Reads pairs of GeoTIFF tiles (image + mask) from the directory structure
produced by the Prepare tab:

    <dataset_dir>/augmented/v<N>/<split>/images/*.tif
    <dataset_dir>/augmented/v<N>/<split>/masks/*.tif

Each image band is normalised independently to [0, 1] using per-tile
min/max.  Mask pixels are binarised (any value > 0 → 1.0, else 0.0).

Public API
----------
SegmentationDataset(images_dir, masks_dir)  — torch Dataset
build_dataloaders(config)  → (train_loader, val_loader)
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Internal GeoTIFF reader
# ---------------------------------------------------------------------------

def _read_geotiff(path: str) -> np.ndarray:
    """
    Reads a GeoTIFF and returns a (bands, H, W) float32 numpy array.
    Uses GDAL (available via QGIS system-site-packages).
    """
    from osgeo import gdal
    gdal.UseExceptions()

    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise IOError(f"Cannot open raster: {path}")

    bands = ds.RasterCount
    data = np.stack(
        [ds.GetRasterBand(b + 1).ReadAsArray() for b in range(bands)],
        axis=0,
    ).astype(np.float32)
    ds = None  # close file handle
    return data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """
    Paired image/mask GeoTIFF tile dataset.

    Parameters
    ----------
    images_dir : str
        Directory containing image .tif files.
    masks_dir : str
        Directory containing mask .tif files (filenames must match images).
    """

    def __init__(self, images_dir: str, masks_dir: str):
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        image_names = {f for f in os.listdir(
            images_dir) if f.lower().endswith(".tif")}
        mask_names = {f for f in os.listdir(
            masks_dir) if f.lower().endswith(".tif")}
        common = sorted(image_names & mask_names)

        if not common:
            raise FileNotFoundError(
                f"No matching .tif pairs found.\n"
                f"  images: {images_dir}\n"
                f"  masks:  {masks_dir}"
            )

        self.filenames = common

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.filenames[idx]

        # --- Image -----------------------------------------------------------
        img = _read_geotiff(os.path.join(self.images_dir, name))  # (C, H, W)

        # Normalise each band independently to [0, 1]
        for b in range(img.shape[0]):
            bmin = float(img[b].min())
            bmax = float(img[b].max())
            if bmax > bmin:
                img[b] = (img[b] - bmin) / (bmax - bmin)
            else:
                img[b] = 0.0  # flat band → zero

        image_tensor = torch.from_numpy(img)  # (C, H, W) float32

        # --- Mask ------------------------------------------------------------
        mask = _read_geotiff(os.path.join(self.masks_dir, name))  # (1, H, W)
        mask_bin = (mask > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_bin)  # (1, H, W) float32

        return image_tensor, mask_tensor


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(config: dict):
    """
    Builds DataLoaders for the train and val splits.

    Parameters
    ----------
    config : dict
        dataset_dir  str   top-level *_dataset folder
        aug_version  int   augmented version number
        batch_size   int

    Returns
    -------
    train_loader, val_loader
    """
    aug_dir = os.path.join(
        config["dataset_dir"], "augmented", f"v{config['aug_version']}"
    )

    def _make(split: str) -> DataLoader:
        images_dir = os.path.join(aug_dir, split, "images")
        masks_dir = os.path.join(aug_dir, split, "masks")
        ds = SegmentationDataset(images_dir, masks_dir)
        return DataLoader(
            ds,
            batch_size=config["batch_size"],
            shuffle=(split == "train"),
            num_workers=0,       # must be 0 inside QGIS — worker processes
            pin_memory=False,    # crash because sys.executable is qgis.exe
            drop_last=False,
        )

    return _make("train"), _make("valid")
