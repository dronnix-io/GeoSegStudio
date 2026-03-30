"""
module: augmenter.py

Stage 3 — Augmentation.

Reads tile pairs from a chosen splitting version and writes geometrically
transformed copies for train / valid / test subsets.
Transforms are applied identically to both image and mask.

Parallelised with ThreadPoolExecutor — each thread handles all augmentation
methods for one tile pair independently.

Uses osgeo.gdal (bundled with QGIS) for geocoded GeoTIFF I/O.

Output folder structure:
    <output_dir>/<prefix>_dataset/augmented/v<N>/
        train/  images/ masks/
        valid/  images/ masks/
        test/   images/ masks/
        augmentation_info.json

Geocoding notes:
    Flip H / Flip V : geotransform updated correctly.
    Rotate / Mirror : source geotransform kept (pixel values correct,
                      origin approximate — acceptable for DL training tiles).
"""

import os
import json
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .clipper import _next_version, _scan_versions, _write_json, _read_json


# ---------------------------------------------------------------------------
# Augmentation method registry
# ---------------------------------------------------------------------------

def _rot90(arr):   return np.rot90(arr, k=1, axes=(1, 2))
def _rot180(arr):  return np.rot90(arr, k=2, axes=(1, 2))
def _rot270(arr):  return np.rot90(arr, k=3, axes=(1, 2))
def _mirror(arr):  return np.transpose(arr, axes=(0, 2, 1))
def _fliph(arr):   return np.flip(arr, axis=2)
def _flipv(arr):   return np.flip(arr, axis=1)

METHOD_REGISTRY = {
    "Original":   ("original", lambda arr: arr.copy()),
    "Rotate 90":  ("rot90",    _rot90),
    "Rotate 180": ("rot180",   _rot180),
    "Rotate 270": ("rot270",   _rot270),
    "Mirror":     ("mirror",   _mirror),
    "Flip H":     ("fliph",    _fliph),
    "Flip V":     ("flipv",    _flipv),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_augmentation(config: dict,
                     progress_callback=None,
                     cancelled_callback=None) -> dict:
    """
    Applies selected augmentation methods to all three split subsets.

    Parameters
    ----------
    config : dict
        Required keys:
            output_dir, prefix, splitting_version,
            augmentations (list[str]), cpu_count (int)
    """
    prefix            = config.get("prefix", "dataset").strip() or "dataset"
    splitting_version = config["splitting_version"]
    methods           = config["augmentations"]
    output_dir        = config["output_dir"]
    cpu_count         = config.get("cpu_count", 1)

    dataset_dir = os.path.join(output_dir, f"{prefix}_dataset")
    split_dir   = os.path.join(dataset_dir, "splitting", f"v{splitting_version}")

    # --- Versioned output directories ----------------------------------------
    version     = _next_version(dataset_dir, "augmented")
    version_dir = os.path.join(dataset_dir, "augmented", f"v{version}")

    subsets = ("train", "valid", "test")
    for subset in subsets:
        os.makedirs(os.path.join(version_dir, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(version_dir, subset, "masks"),  exist_ok=True)

    # --- Build per-tile task list --------------------------------------------
    tasks = []
    for subset in subsets:
        src_images = os.path.join(split_dir, subset, "images")
        src_masks  = os.path.join(split_dir, subset, "masks")
        dst_images = os.path.join(version_dir, subset, "images")
        dst_masks  = os.path.join(version_dir, subset, "masks")

        if not os.path.isdir(src_images):
            continue

        for name in sorted(f for f in os.listdir(src_images) if f.endswith(".tif")):
            tasks.append({
                "img_src":  os.path.join(src_images, name),
                "mask_src": os.path.join(src_masks,  name),
                "dst_images": dst_images,
                "dst_masks":  dst_masks,
                "stem":     os.path.splitext(name)[0],
                "methods":  methods,
                "subset":   subset,
            })

    total     = len(tasks)
    completed = 0
    counts    = {s: 0 for s in subsets}

    # --- Parallel augmentation -----------------------------------------------
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        futures = {}
        for task in tasks:
            if cancelled_callback and cancelled_callback():
                break
            futures[executor.submit(_augment_tile_worker, task)] = task["subset"]

        for future in as_completed(futures):
            result = future.result()
            counts[result["subset"]] += result["count"]
            completed += 1
            if progress_callback:
                progress_callback(int(completed / total * 100))

    # --- Persist metadata ----------------------------------------------------
    split_info = _read_json(os.path.join(split_dir, "splitting_info.json"))

    info = {
        "version":                    version,
        "created":                    datetime.now().isoformat(),
        "based_on_splitting_version": splitting_version,
        "based_on_clipping_version":  split_info.get("based_on_clipping_version"),
        "methods":                    methods,
        "train_count":                counts["train"],
        "valid_count":                counts["valid"],
        "test_count":                 counts["test"],
    }
    _write_json(os.path.join(version_dir, "augmentation_info.json"), info)

    return {
        "version":     version,
        "counts":      counts,
        "version_dir": version_dir,
    }


def get_available_versions(dataset_dir: str) -> list:
    """Returns sorted list of existing augmented version dicts."""
    return _scan_versions(os.path.join(dataset_dir, "augmented"), "augmentation_info.json")


# ---------------------------------------------------------------------------
# Thread worker
# ---------------------------------------------------------------------------

def _augment_tile_worker(task: dict) -> dict:
    """
    Applies all selected augmentation methods to one tile pair.
    Returns dict with subset name and number of augmented files written.
    """
    img_arr,  img_gt,  img_proj  = _read_geotiff(task["img_src"])
    mask_arr, mask_gt, mask_proj = _read_geotiff(task["mask_src"])

    written = 0
    for method in task["methods"]:
        suffix, transform = METHOD_REGISTRY.get(method, (None, None))
        if suffix is None:
            continue

        aug_img  = transform(img_arr)
        aug_mask = transform(mask_arr)

        img_gt_out  = _adjust_geotransform(img_gt,  aug_img.shape,  method)
        mask_gt_out = _adjust_geotransform(mask_gt, aug_mask.shape, method)

        out_name = f"{task['stem']}_{suffix}.tif"
        _write_geotiff(os.path.join(task["dst_images"], out_name),
                       aug_img,  img_gt_out,  img_proj)
        _write_geotiff(os.path.join(task["dst_masks"],  out_name),
                       aug_mask, mask_gt_out, mask_proj)
        written += 1

    return {"subset": task["subset"], "count": written}


# ---------------------------------------------------------------------------
# GeoTIFF I/O helpers
# ---------------------------------------------------------------------------

def _read_geotiff(path: str):
    from osgeo import gdal
    gdal.UseExceptions()
    ds  = gdal.Open(path, gdal.GA_ReadOnly)
    arr = ds.ReadAsArray()
    if arr.ndim == 2:
        arr = arr[np.newaxis, :]
    gt   = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds   = None
    return arr, gt, proj


def _write_geotiff(path: str, arr: np.ndarray, geotransform: tuple, projection: str):
    from osgeo import gdal
    gdal.UseExceptions()

    if arr.ndim == 2:
        arr = arr[np.newaxis, :]

    bands, height, width = arr.shape

    _DTYPE_MAP = {
        np.uint8:   gdal.GDT_Byte,
        np.uint16:  gdal.GDT_UInt16,
        np.int16:   gdal.GDT_Int16,
        np.int32:   gdal.GDT_Int32,
        np.float32: gdal.GDT_Float32,
        np.float64: gdal.GDT_Float64,
    }
    gdal_dtype = _DTYPE_MAP.get(arr.dtype.type, gdal.GDT_Float32)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        path, width, height, bands, gdal_dtype,
        options=["COMPRESS=LZW", "TILED=YES"]
    )
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)
    for i in range(bands):
        ds.GetRasterBand(i + 1).WriteArray(arr[i])
    ds.FlushCache()
    ds = None


def _adjust_geotransform(gt: tuple, arr_shape: tuple, method: str) -> tuple:
    """Updates the geotransform for flip operations; keeps source for others."""
    x0, pw, _, y0, __, ph = gt
    _, height, width = arr_shape

    if method == "Flip H":
        return (x0 + width * pw, -pw, 0.0, y0, 0.0, ph)
    if method == "Flip V":
        return (x0, pw, 0.0, y0 + height * ph, 0.0, -ph)
    return gt
