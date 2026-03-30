"""
module: pipeline.py

Orchestrates the three data-preparation stages and exposes utility functions
for scanning existing versions — used by both the backend and the UI.

Public functions:
    run_clipping(config, ...)       → Stage 1
    run_splitting(config, ...)      → Stage 2
    run_augmentation(config, ...)   → Stage 3
    run_all(config, ...)            → Stages 1 → 2 → 3 in sequence

    get_dataset_dir(output_dir, prefix) → str
    get_clipping_versions(dataset_dir)  → list[dict]
    get_splitting_versions(dataset_dir) → list[dict]
    get_augmented_versions(dataset_dir) → list[dict]
    version_label(v_dict)               → str  (human-readable combo box label)
"""

import os

from .validator import (
    validate_for_clipping,
    validate_for_splitting,
    validate_for_augmentation,
)
from .clipper   import run_clipping   as _clip
from .clipper   import get_available_versions as _clip_versions
from .splitter  import run_splitting  as _split
from .splitter  import get_available_versions as _split_versions
from .augmenter import run_augmentation as _augment
from .augmenter import get_available_versions as _aug_versions


# ---------------------------------------------------------------------------
# Stage runners (validate → execute)
# ---------------------------------------------------------------------------

def run_clipping(config: dict,
                 progress_callback=None,
                 cancelled_callback=None) -> dict:
    """Validates and runs Stage 1 (Clipping). Returns result dict."""
    validated, warnings = validate_for_clipping(config)
    result = _clip(validated,
                   progress_callback=progress_callback,
                   cancelled_callback=cancelled_callback)
    result["warnings"] = warnings
    return result


def run_splitting(config: dict,
                  progress_callback=None,
                  cancelled_callback=None) -> dict:
    """Validates and runs Stage 2 (Splitting). Returns result dict."""
    validated, warnings = validate_for_splitting(config)
    result = _split(validated,
                    progress_callback=progress_callback,
                    cancelled_callback=cancelled_callback)
    result["warnings"] = warnings
    return result


def run_augmentation(config: dict,
                     progress_callback=None,
                     cancelled_callback=None) -> dict:
    """Validates and runs Stage 3 (Augmentation). Returns result dict."""
    validated, warnings = validate_for_augmentation(config)
    result = _augment(validated,
                      progress_callback=progress_callback,
                      cancelled_callback=cancelled_callback)
    result["warnings"] = warnings
    return result


def run_all(config: dict,
            clipping_progress=None,
            splitting_progress=None,
            augmentation_progress=None,
            cancelled_callback=None) -> dict:
    """
    Runs all three stages sequentially.

    The splitting stage automatically uses the clipping version just produced.
    The augmentation stage automatically uses the splitting version just produced.

    Returns a combined result dict with keys:
        clipping, splitting, augmentation  (each a stage result dict)
    """
    clip_result = run_clipping(
        config,
        progress_callback=clipping_progress,
        cancelled_callback=cancelled_callback,
    )

    # Wire the freshly produced clipping version into the splitting config.
    split_config = config.copy()
    split_config["clipping_version"] = clip_result["version"]
    split_config["prefix"] = (
        config.get("clip_params", {}).get("prefix", "").strip() or "dataset"
    )

    split_result = run_splitting(
        split_config,
        progress_callback=splitting_progress,
        cancelled_callback=cancelled_callback,
    )

    # Wire the freshly produced splitting version into the augmentation config.
    aug_config = config.copy()
    aug_config["splitting_version"] = split_result["version"]
    aug_config["prefix"] = split_config["prefix"]

    aug_result = run_augmentation(
        aug_config,
        progress_callback=augmentation_progress,
        cancelled_callback=cancelled_callback,
    )

    return {
        "clipping":     clip_result,
        "splitting":    split_result,
        "augmentation": aug_result,
    }


# ---------------------------------------------------------------------------
# Version scanning utilities  (used by UI combo boxes)
# ---------------------------------------------------------------------------

def get_dataset_dir(output_dir: str, prefix: str) -> str:
    """Returns the root dataset directory path."""
    prefix = prefix.strip() or "dataset"
    return os.path.join(output_dir, f"{prefix}_dataset")


def get_clipping_versions(dataset_dir: str) -> list:
    """Returns sorted list of existing clipping version dicts."""
    return _clip_versions(dataset_dir)


def get_splitting_versions(dataset_dir: str) -> list:
    """Returns sorted list of existing splitting version dicts."""
    return _split_versions(dataset_dir)


def get_augmented_versions(dataset_dir: str) -> list:
    """Returns sorted list of existing augmented version dicts."""
    return _aug_versions(dataset_dir)


def version_label(v_dict: dict) -> str:
    """
    Produces a human-readable label for a version combo box entry.

    Example:
        "v1 — 512 tiles  (win=256, stride=128, px=0.5)"     for clipping
        "v2 — train 80% / valid 10% / test 10%"             for splitting
        "v1 — Original, Rotate 90, Flip H"                  for augmentation
    """
    vnum = v_dict.get("version", "?")
    info = v_dict.get("info", {})

    if "window_size" in info:          # clipping version
        return (
            f"v{vnum}  —  {info.get('total_tiles', '?')} tiles  "
            f"(win={info['window_size']}, "
            f"stride={info.get('stride', '?')}, "
            f"px={info.get('native_pixel_size', '?')})"
        )

    if "train_pct" in info:            # splitting version
        return (
            f"v{vnum}  —  "
            f"train {info.get('train_pct', '?')}% / "
            f"valid {info.get('valid_pct', '?')}% / "
            f"test {info.get('test_pct', '?')}%  "
            f"({info.get('train_count', 0) + info.get('valid_count', 0) + info.get('test_count', 0)} tiles)"
        )

    if "methods" in info:              # augmentation version
        methods_str = ", ".join(info.get("methods", []))
        total = (
            info.get("train_count", 0)
            + info.get("valid_count", 0)
            + info.get("test_count", 0)
        )
        return f"v{vnum}  —  {methods_str}  ({total} tiles)"

    return f"v{vnum}"
