"""
module: DL/predictor.py

Background prediction worker for Tab 4.

Loads a trained .pth checkpoint, runs sliding-window inference on a raw
full-resolution GeoTIFF, stitches the per-tile probability maps back into
a full-resolution prediction, and saves the result as:
  - Raster   : single-band uint8 GeoTIFF  (0 = background, 255 = positive)
  - Vector   : GeoPackage or Shapefile    (positive-class polygons only)
  - Both     : raster + vector side-by-side

Overlap blending
----------------
When overlap_pct > 0 tiles are spaced by stride < img_size.  A float32
accumulator array and a count array are maintained over the full raster.
Each tile adds its sigmoid probabilities; the final mask is
  (accumulator / count) > threshold
which smoothly blends overlapping regions and removes seam artefacts.

Edge tiles
----------
Tiles that extend beyond the raster boundary are zero-padded to img_size,
inference runs on the full padded tile, and only the valid (non-padded)
region is written back to the accumulator.

Signals
-------
phase_update(message)
    Short human-readable description of the current step.

tile_done(current, total)
    Emitted after each tile is processed.

prediction_finished(success, results, message)
    Emitted when the worker exits.
    results keys on success:
      output_paths  list[str]   paths of saved file(s)
      tile_count    int
      raster_size   (width, height)

Config dict keys
----------------
checkpoint_path  str    path to .pth checkpoint
input_raster     str    path to input GeoTIFF
overlap_pct      float  0.0–0.5  (fraction of tile size used as overlap)
threshold        float  sigmoid threshold, default 0.5
device           str    "cpu" | "cuda:0" | …
output_format    str    "raster" | "vector" | "both"
output_path      str    full path including extension (.tif / .gpkg / .shp)
                         For "both": base path without extension; worker
                         appends .tif and .gpkg automatically.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from qgis.PyQt.QtCore import QThread, pyqtSignal


class PredictionWorker(QThread):

    phase_update        = pyqtSignal(str)
    tile_done           = pyqtSignal(int, int)
    prediction_finished = pyqtSignal(bool, object, str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config    = config
        self._cancelled = False

    def stop(self):
        self._cancelled = True

    # -------------------------------------------------------------------------

    def run(self):
        try:
            self._predict()
        except Exception as exc:
            self.prediction_finished.emit(False, {}, str(exc))

    # -------------------------------------------------------------------------

    def _predict(self):
        cfg    = self._config
        device = torch.device(cfg["device"])

        # --- Load checkpoint -------------------------------------------------
        self.phase_update.emit("Loading model from checkpoint…")
        ckpt = torch.load(cfg["checkpoint_path"], map_location=device)

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

        # --- Open input raster -----------------------------------------------
        self.phase_update.emit("Opening input raster…")
        from osgeo import gdal
        gdal.UseExceptions()

        src_ds = gdal.Open(cfg["input_raster"], gdal.GA_ReadOnly)
        if src_ds is None:
            raise IOError(f"Cannot open raster: {cfg['input_raster']}")

        raster_w  = src_ds.RasterXSize
        raster_h  = src_ds.RasterYSize
        geo_trans = src_ds.GetGeoTransform()
        projection = src_ds.GetProjection()
        n_bands   = src_ds.RasterCount

        if n_bands != in_channels:
            raise ValueError(
                f"Band count mismatch: input raster has {n_bands} band(s) "
                f"but the model expects {in_channels}."
            )

        # --- Build tile grid -------------------------------------------------
        overlap_pct = float(cfg.get("overlap_pct", 0.0))
        stride      = max(1, int(img_size * (1.0 - overlap_pct)))
        threshold   = float(cfg.get("threshold", 0.5))

        xs = list(range(0, raster_w, stride))
        ys = list(range(0, raster_h, stride))
        total_tiles = len(xs) * len(ys)

        # --- Accumulator arrays (full raster size) ---------------------------
        accumulator = np.zeros((raster_h, raster_w), dtype=np.float32)
        count_arr   = np.zeros((raster_h, raster_w), dtype=np.float32)

        # --- Sliding-window inference ----------------------------------------
        self.phase_update.emit(
            f"Running inference — {total_tiles} tile(s)…"
        )
        tile_idx = 0

        with torch.no_grad():
            for y0 in ys:
                for x0 in xs:
                    if self._cancelled:
                        self.prediction_finished.emit(
                            False, {}, "Prediction stopped by user."
                        )
                        src_ds = None
                        return

                    # Actual read window (clamped to raster bounds)
                    read_w = min(img_size, raster_w - x0)
                    read_h = min(img_size, raster_h - y0)

                    # Read all bands for this tile
                    tile = np.zeros(
                        (n_bands, img_size, img_size), dtype=np.float32
                    )
                    for b in range(n_bands):
                        band_data = src_ds.GetRasterBand(b + 1).ReadAsArray(
                            x0, y0, read_w, read_h
                        ).astype(np.float32)
                        tile[b, :read_h, :read_w] = band_data

                    # Per-band min-max normalisation (matching dataset.py)
                    for b in range(n_bands):
                        valid = tile[b, :read_h, :read_w]
                        bmin, bmax = float(valid.min()), float(valid.max())
                        if bmax > bmin:
                            tile[b] = (tile[b] - bmin) / (bmax - bmin)
                        else:
                            tile[b] = 0.0

                    tensor = torch.from_numpy(tile).unsqueeze(0).to(device)
                    logits = model(tensor)
                    if isinstance(logits, (list, tuple)):
                        logits = logits[-1]

                    prob = torch.sigmoid(logits).cpu().numpy()  # (1,1,H,W)
                    prob = prob.squeeze()                        # (img_size, img_size)

                    # Write only the valid region back to accumulators
                    accumulator[y0:y0 + read_h, x0:x0 + read_w] += \
                        prob[:read_h, :read_w]
                    count_arr[y0:y0 + read_h, x0:x0 + read_w] += 1.0

                    tile_idx += 1
                    self.tile_done.emit(tile_idx, total_tiles)

        src_ds = None  # close GDAL dataset

        # --- Build final binary mask -----------------------------------------
        self.phase_update.emit("Stitching tiles…")
        count_arr   = np.maximum(count_arr, 1.0)   # avoid divide-by-zero
        prob_map    = accumulator / count_arr
        binary_mask = (prob_map > threshold).astype(np.uint8)  # 0 or 1

        # --- Save outputs ----------------------------------------------------
        output_format = cfg.get("output_format", "vector")
        output_path   = cfg["output_path"]
        output_paths  = []

        if output_format in ("raster", "both"):
            raster_path = output_path + ".tif"
            self.phase_update.emit("Saving raster…")
            _save_raster(binary_mask, raster_path, geo_trans, projection)
            output_paths.append(raster_path)

        if output_format in ("vector", "both"):
            vector_path = output_path + ".gpkg"
            self.phase_update.emit("Polygonizing prediction…")

            # Polygonize from a temp in-memory raster
            tmp_path = _save_raster_temp(binary_mask, geo_trans, projection)
            try:
                _polygonize(tmp_path, vector_path, projection)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            output_paths.append(vector_path)

        results = {
            "output_paths": output_paths,
            "output_format": output_format,
            "tile_count": total_tiles,
            "raster_size": (raster_w, raster_h),
        }
        self.prediction_finished.emit(True, results, "Prediction complete.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_raster(
    mask: np.ndarray,
    out_path: str,
    geo_trans: tuple,
    projection: str,
):
    """Saves a (H, W) uint8 binary array as a GeoTIFF (0/255)."""
    from osgeo import gdal
    h, w = mask.shape
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_path, w, h, 1, gdal.GDT_Byte,
                       ["COMPRESS=LZW", "TILED=YES"])
    ds.SetGeoTransform(geo_trans)
    if projection:
        ds.SetProjection(projection)
    ds.GetRasterBand(1).WriteArray((mask * 255).astype(np.uint8))
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds.FlushCache()
    ds = None


def _save_raster_temp(
    mask: np.ndarray,
    geo_trans: tuple,
    projection: str,
) -> str:
    """Writes a binary mask to a temp GeoTIFF and returns its path."""
    from osgeo import gdal
    h, w = mask.shape
    fd, tmp_path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(tmp_path, w, h, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(geo_trans)
    if projection:
        ds.SetProjection(projection)
    ds.GetRasterBand(1).WriteArray(mask.astype(np.uint8))
    ds.FlushCache()
    ds = None
    return tmp_path


def _polygonize(raster_path: str, out_path: str, projection: str):
    """
    Polygonizes the positive-class pixels (value == 1) from a binary raster
    and saves them as a GeoPackage or Shapefile.

    Only pixels with value == 1 are included (the raster band is used as
    its own mask, so zero pixels are excluded automatically).
    """
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()

    src_ds   = gdal.Open(raster_path, gdal.GA_ReadOnly)
    src_band = src_ds.GetRasterBand(1)

    ext = os.path.splitext(out_path)[1].lower()
    drv_name = "ESRI Shapefile" if ext == ".shp" else "GPKG"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Remove existing output so the driver does not complain
    if os.path.exists(out_path):
        ogr.GetDriverByName(drv_name).DeleteDataSource(out_path)

    drv    = ogr.GetDriverByName(drv_name)
    out_ds = drv.CreateDataSource(out_path)

    srs = osr.SpatialReference()
    if projection:
        srs.ImportFromWkt(projection)

    layer = out_ds.CreateLayer("prediction", srs=srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("class", ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn("area_crs_units2", ogr.OFTReal))

    # Pass src_band as the mask: only non-zero pixels are polygonized
    gdal.Polygonize(src_band, src_band, layer, 0, [], callback=None)

    # Fill the area field so users can see the exact value the postprocessor uses
    layer.ResetReading()
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom:
            from shapely import wkb as _wkb
            try:
                shp = _wkb.loads(bytes(geom.ExportToWkb()))
                feat.SetField("area_crs_units2", shp.area)
                layer.SetFeature(feat)
            except Exception:
                pass

    out_ds.FlushCache()
    out_ds  = None
    src_ds  = None
