"""
module: splitter.py

Stage 2 — Splitting.

Reads tile pairs from a chosen clipping version and copies them into
versioned train / valid / test subsets using a reproducible random shuffle.
File copying is parallelised with ThreadPoolExecutor (I/O bound).

Output folder structure:
    <output_dir>/<prefix>_dataset/splitting/v<N>/
        train/  images/ masks/
        valid/  images/ masks/
        test/   images/ masks/
        splitting_info.json
"""

import os
import json
import shutil
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .clipper import _next_version, _scan_versions, _write_json, _read_json


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_splitting(config: dict,
                  progress_callback=None,
                  cancelled_callback=None) -> dict:
    """
    Splits clipped tile pairs into train / valid / test subsets.

    Parameters
    ----------
    config : dict
        Required keys:
            output_dir, prefix, clipping_version,
            split_percentages (dict: train/valid/test),
            cpu_count (int)
    """
    prefix           = config.get("prefix", "dataset").strip() or "dataset"
    clipping_version = config["clipping_version"]
    split_pct        = config["split_percentages"]
    output_dir       = config["output_dir"]
    cpu_count        = config.get("cpu_count", 1)

    dataset_dir = os.path.join(output_dir, f"{prefix}_dataset")
    clip_dir    = os.path.join(dataset_dir, "clipping", f"v{clipping_version}")
    src_images  = os.path.join(clip_dir, "images")
    src_masks   = os.path.join(clip_dir, "masks")

    # --- Collect and shuffle tile names --------------------------------------
    tile_names = sorted(f for f in os.listdir(src_images) if f.endswith(".tif"))
    random.seed(42)
    random.shuffle(tile_names)
    total = len(tile_names)

    n_train = round(total * split_pct["train"] / 100)
    n_valid = round(total * split_pct["valid"] / 100)

    subsets = {
        "train": tile_names[:n_train],
        "valid": tile_names[n_train: n_train + n_valid],
        "test":  tile_names[n_train + n_valid:],
    }

    # --- Versioned output directories ----------------------------------------
    version     = _next_version(dataset_dir, "splitting")
    version_dir = os.path.join(dataset_dir, "splitting", f"v{version}")

    for subset in ("train", "valid", "test"):
        os.makedirs(os.path.join(version_dir, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(version_dir, subset, "masks"),  exist_ok=True)

    # --- Build copy task list ------------------------------------------------
    copy_tasks = []
    for subset, names in subsets.items():
        dst_images = os.path.join(version_dir, subset, "images")
        dst_masks  = os.path.join(version_dir, subset, "masks")
        for name in names:
            copy_tasks.append({
                "src_img":  os.path.join(src_images, name),
                "src_mask": os.path.join(src_masks,  name),
                "dst_img":  os.path.join(dst_images, name),
                "dst_mask": os.path.join(dst_masks,  name),
                "subset":   subset,
                "name":     name,
            })

    # --- Parallel copy -------------------------------------------------------
    counts    = {"train": 0, "valid": 0, "test": 0}
    completed = 0

    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        futures = {}
        for task in copy_tasks:
            if cancelled_callback and cancelled_callback():
                break
            futures[executor.submit(_copy_tile_pair, task)] = task["subset"]

        for future in as_completed(futures):
            subset = future.result()
            counts[subset] += 1
            completed += 1
            if progress_callback:
                progress_callback(int(completed / total * 100))

    # --- Persist metadata ----------------------------------------------------
    clipping_info = _read_json(os.path.join(clip_dir, "clipping_info.json"))

    info = {
        "version":                   version,
        "created":                   datetime.now().isoformat(),
        "based_on_clipping_version": clipping_version,
        "clipping_raster_layer":     clipping_info.get("raster_layer", ""),
        "clipping_vector_layer":     clipping_info.get("vector_layer", ""),
        "train_pct":                 split_pct["train"],
        "valid_pct":                 split_pct["valid"],
        "test_pct":                  split_pct["test"],
        "train_count":               counts["train"],
        "valid_count":               counts["valid"],
        "test_count":                counts["test"],
    }
    _write_json(os.path.join(version_dir, "splitting_info.json"), info)

    return {
        "version":     version,
        "counts":      counts,
        "version_dir": version_dir,
    }


def get_available_versions(dataset_dir: str) -> list:
    """Returns sorted list of existing splitting version dicts."""
    return _scan_versions(os.path.join(dataset_dir, "splitting"), "splitting_info.json")


# ---------------------------------------------------------------------------
# Thread worker
# ---------------------------------------------------------------------------

def _copy_tile_pair(task: dict) -> str:
    """Copies one image+mask pair. Returns the subset name."""
    if os.path.isfile(task["src_img"]):
        shutil.copy2(task["src_img"],  task["dst_img"])
    if os.path.isfile(task["src_mask"]):
        shutil.copy2(task["src_mask"], task["dst_mask"])
    return task["subset"]
