"""
module: DL/evaluator.py

Background evaluation worker for Tab 3.

Loads a trained .pth checkpoint, runs inference on a chosen dataset split
(test / valid / train), computes per-tile and aggregate metrics, collects
representative sample tiles for visualisation, and optionally exports a
per-tile CSV and predicted mask GeoTIFFs.

Signals
-------
phase_update(message)
    Short human-readable description of the current step.

tile_done(current, total)
    Emitted after each tile is evaluated.

evaluation_finished(success, results, message)
    Emitted when the worker exits.  results is a dict (see below) on
    success, or an empty dict on failure.

Results dict keys
-----------------
metrics     dict   IoU, F1/Dice, Precision, Recall, Pixel Accuracy (float)
confusion   dict   TP, FP, TN, FN (int counts)
per_tile    list   one dict per tile: filename, iou, f1, precision,
                   recall, pixel_accuracy
samples     list   up to n_samples dicts: filename, image (C,H,W ndarray),
                   gt_mask (1,H,W ndarray), pred_mask (1,H,W ndarray), iou
total_tiles int
csv_path    str    path to saved CSV (only present when output_dir given)

Config dict keys
----------------
checkpoint_path  str   path to .pth checkpoint file
dataset_dir      str   top-level *_dataset folder
aug_version      int   augmented version number
split            str   "test" | "valid" | "train"
device           str   "cpu" | "cuda:0" | …
output_dir       str   folder for CSV / predicted masks (empty = skip)
save_masks       bool  write predicted masks as GeoTIFFs
n_samples        int   number of sample tiles to collect (default 8)
"""
from __future__ import annotations

import os
import csv

import numpy as np
import torch

from qgis.PyQt.QtCore import QThread, pyqtSignal


class EvaluationWorker(QThread):

    phase_update        = pyqtSignal(str)
    tile_done           = pyqtSignal(int, int)          # current, total
    evaluation_finished = pyqtSignal(bool, object, str) # success, results, message

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config    = config
        self._cancelled = False

    def stop(self):
        self._cancelled = True

    # -------------------------------------------------------------------------

    def run(self):
        try:
            self._evaluate()
        except Exception as exc:
            self.evaluation_finished.emit(False, {}, str(exc))

    # -------------------------------------------------------------------------

    def _evaluate(self):
        cfg    = self._config
        device = torch.device(cfg["device"])

        # --- Load checkpoint -------------------------------------------------
        self.phase_update.emit("Loading model from checkpoint...")
        ckpt = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)  # nosec B614

        saved = ckpt.get("config", {})
        architecture = ckpt.get("architecture") or saved.get("architecture")
        in_channels  = saved.get("in_channels")
        img_size     = saved.get("img_size")

        if not architecture or in_channels is None or img_size is None:
            raise ValueError(
                "Checkpoint is missing model metadata (architecture / "
                "in_channels / img_size). It may have been saved by an "
                "older version of the plugin."
            )

        from .architectures import build_model
        model = build_model(architecture, in_channels, img_size).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # --- Build dataset ---------------------------------------------------
        self.phase_update.emit("Loading dataset...")
        aug_dir    = os.path.join(
            cfg["dataset_dir"], "augmented", f"v{cfg['aug_version']}"
        )
        split      = cfg["split"]
        images_dir = os.path.join(aug_dir, split, "images")
        masks_dir  = os.path.join(aug_dir, split, "masks")

        from .dataset import SegmentationDataset
        dataset = SegmentationDataset(images_dir, masks_dir)
        total   = len(dataset)

        # Decide sample indices: evenly spaced across the tile list
        n_samples    = cfg.get("n_samples", 8)
        step         = max(1, total // n_samples)
        sample_idxs  = set(range(0, total, step))

        # --- Evaluation loop -------------------------------------------------
        per_tile  = []
        samples   = []
        agg_tp = agg_fp = agg_tn = agg_fn = 0.0

        # Set up mask output directory if requested
        mask_out_dir = None
        if cfg.get("save_masks") and cfg.get("output_dir"):
            mask_out_dir = os.path.join(cfg["output_dir"], "predicted_masks", split)
            os.makedirs(mask_out_dir, exist_ok=True)

        with torch.no_grad():
            for i in range(total):
                if self._cancelled:
                    self.evaluation_finished.emit(
                        False, {}, "Evaluation stopped by user."
                    )
                    return

                self.phase_update.emit(
                    f"Evaluating {split} tiles — {i + 1} / {total}"
                )

                image, mask = dataset[i]                          # (C,H,W), (1,H,W)
                logits = model(image.unsqueeze(0).to(device))
                if isinstance(logits, (list, tuple)):
                    logits = logits[-1]

                pred = (torch.sigmoid(logits.cpu()) > 0.5).float()  # (1,1,H,W)
                pred = pred.squeeze(0)                               # (1,H,W)

                tp, fp, tn, fn = _confusion(pred, mask)
                agg_tp += tp; agg_fp += fp
                agg_tn += tn; agg_fn += fn

                tile_iou = tp / (tp + fp + fn + 1e-7)
                tile_f1  = 2 * tp / (2 * tp + fp + fn + 1e-7)
                tile_pre = tp / (tp + fp + 1e-7)
                tile_rec = tp / (tp + fn + 1e-7)
                tile_acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)

                per_tile.append({
                    "filename":       dataset.filenames[i],
                    "iou":            tile_iou,
                    "f1":             tile_f1,
                    "precision":      tile_pre,
                    "recall":         tile_rec,
                    "pixel_accuracy": tile_acc,
                })

                if i in sample_idxs:
                    samples.append({
                        "filename":  dataset.filenames[i],
                        "image":     image.numpy(),        # (C,H,W) float32
                        "gt_mask":   mask.numpy(),         # (1,H,W) float32
                        "pred_mask": pred.numpy(),         # (1,H,W) float32
                        "iou":       tile_iou,
                    })

                # Optionally save predicted mask as GeoTIFF
                if mask_out_dir:
                    _save_mask_geotiff(
                        pred.squeeze(0).numpy(),
                        os.path.join(images_dir, dataset.filenames[i]),
                        os.path.join(mask_out_dir, dataset.filenames[i]),
                    )

                self.tile_done.emit(i + 1, total)

        # --- Aggregate metrics -----------------------------------------------
        results = {
            "metrics": {
                "IoU":            agg_tp / (agg_tp + agg_fp + agg_fn + 1e-7),
                "F1 / Dice":      2 * agg_tp / (2 * agg_tp + agg_fp + agg_fn + 1e-7),
                "Precision":      agg_tp / (agg_tp + agg_fp + 1e-7),
                "Recall":         agg_tp / (agg_tp + agg_fn + 1e-7),
                "Pixel Accuracy": (agg_tp + agg_tn) / (
                    agg_tp + agg_tn + agg_fp + agg_fn + 1e-7
                ),
            },
            "confusion": {
                "TP": int(agg_tp),
                "FP": int(agg_fp),
                "TN": int(agg_tn),
                "FN": int(agg_fn),
            },
            "per_tile":    per_tile,
            "samples":     samples,
            "total_tiles": total,
            "split":       split,
        }

        # --- CSV export ------------------------------------------------------
        if cfg.get("output_dir"):
            results["csv_path"] = _save_csv(results, cfg)

        self.evaluation_finished.emit(True, results, "Evaluation complete.")

    # -------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confusion(pred: torch.Tensor, target: torch.Tensor):
    """Returns (TP, FP, TN, FN) as floats from flattened binary tensors."""
    p = pred.view(-1).float()
    t = target.view(-1).float()
    tp = (p * t).sum().item()
    fp = (p * (1 - t)).sum().item()
    tn = ((1 - p) * (1 - t)).sum().item()
    fn = ((1 - p) * t).sum().item()
    return tp, fp, tn, fn


def _save_csv(results: dict, cfg: dict) -> str:
    os.makedirs(cfg["output_dir"], exist_ok=True)
    split    = results.get("split", "eval")
    csv_path = os.path.join(
        cfg["output_dir"], f"evaluation_{split}_per_tile.csv"
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "iou", "f1", "precision",
                        "recall", "pixel_accuracy"],
        )
        writer.writeheader()
        for row in results["per_tile"]:
            writer.writerow({
                k: (f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in row.items()
            })
    return csv_path


def _save_mask_geotiff(pred_arr: np.ndarray, ref_image_path: str, out_path: str):
    """
    Saves a binary prediction (H, W) uint8 GeoTIFF with the same
    geotransform and CRS as the reference image tile.
    """
    try:
        from osgeo import gdal, osr
        gdal.UseExceptions()

        ref = gdal.Open(ref_image_path, gdal.GA_ReadOnly)
        if ref is None:
            return

        gt  = ref.GetGeoTransform()
        wkt = ref.GetProjection()
        h, w = pred_arr.shape

        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(out_path, w, h, 1, gdal.GDT_Byte)
        ds.SetGeoTransform(gt)
        if wkt:
            ds.SetProjection(wkt)
        ds.GetRasterBand(1).WriteArray((pred_arr * 255).astype(np.uint8))
        ds.FlushCache()
        ds = None
        ref = None
    except Exception:
        pass  # mask saving is optional — never crash the evaluation
