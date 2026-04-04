"""
module: clipper.py

Stage 1 — Clipping.

Slides a window over the raster extent, clips each tile using direct GDAL
calls, and rasterizes the vector layer into a matching binary mask tile.
All outputs are fully geocoded GeoTIFFs loadable in QGIS.

Parallelised with ThreadPoolExecutor (GDAL releases the GIL; each thread
opens its own file handles so no data races occur).

Output folder structure:
    <output_dir>/<prefix>_dataset/clipping/v<N>/
        images/   *.tif   (multi-band raster tiles)
        masks/    *.tif   (single-band uint8 binary mask tiles)
        clipping_info.json
"""

import os
import json
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from qgis.core import QgsRasterLayer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_clipping(config: dict,
                 progress_callback=None,
                 cancelled_callback=None) -> dict:
    """
    Clips the raster and rasterizes the vector mask into versioned tile pairs.

    Parameters
    ----------
    config : dict
        Must contain keys produced by validate_for_clipping():
            raster_layer, vector_layer, output_dir, clip_params
        clip_params keys: window_size, stride, burn_value,
                          output_format, cpu_count
    progress_callback : callable(int) | None
        Called after each completed tile with percentage 0–100.
    cancelled_callback : callable() -> bool | None
        Checked before submitting each tile; stops early if True.

    Returns
    -------
    dict with keys: version (int), tile_count (int), skipped_count (int),
                    version_dir (str)
    """
    from osgeo import gdal, ogr
    gdal.UseExceptions()

    raster_layer = config["raster_layer"]
    vector_layer = config["vector_layer"]
    clip = config["clip_params"]

    window_size = clip["window_size"]
    stride = clip["stride"]
    pixel_size = raster_layer.rasterUnitsPerPixelX()
    burn_value = clip["burn_value"]
    cpu_count = clip.get("cpu_count", 1)
    crs_wkt = raster_layer.crs().toWkt()

    # Derive dataset folder name from raster layer name
    prefix = "".join(
        c if c.isalnum() or c in "_-" else "_"
        for c in raster_layer.name()
    ).strip("_") or "dataset"

    # Source file paths (passed to worker threads instead of QGIS objects)
    raster_path = raster_layer.source()
    vector_path, vector_layer_name = _parse_vector_source(
        vector_layer.source())

    # --- Versioned output directories ----------------------------------------
    dataset_dir = os.path.join(config["output_dir"], f"{prefix}_dataset")
    version = _next_version(dataset_dir, "clipping")
    version_dir = os.path.join(dataset_dir, "clipping", f"v{version}")
    images_dir = os.path.join(version_dir, "images")
    masks_dir = os.path.join(version_dir, "masks")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # --- Compute tile grid ---------------------------------------------------
    tile_geo = window_size * pixel_size
    stride_geo = stride * pixel_size
    extent = raster_layer.extent()
    origins = _tile_origins(extent, tile_geo, stride_geo)
    total = len(origins)

    # Build per-tile argument dicts
    tile_args = []
    for idx, (x_origin, y_origin) in enumerate(origins):
        tile_name = f"{prefix}_{idx + 1:05d}"
        tile_args.append({
            "raster_path": raster_path,
            "vector_path": vector_path,
            "vector_layer_name": vector_layer_name,
            "x_min": x_origin,
            "x_max": x_origin + tile_geo,
            "y_max": y_origin,
            "y_min": y_origin - tile_geo,
            "window_size": window_size,
            "burn_value": burn_value,
            "crs_wkt": crs_wkt,
            "image_path": os.path.join(images_dir, f"{tile_name}.tif"),
            "mask_path": os.path.join(masks_dir, f"{tile_name}.tif"),
            "tile_name": tile_name,
        })

    # --- Parallel execution --------------------------------------------------
    tile_count = 0
    skipped_count = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        futures = {}
        for args in tile_args:
            if cancelled_callback and cancelled_callback():
                break
            futures[executor.submit(_clip_tile_worker, args)
                    ] = args["tile_name"]

        for future in as_completed(futures):
            result = future.result()      # raises if worker raised
            if result["has_positive"]:
                tile_count += 1
            else:
                skipped_count += 1
                _remove_if_exists(result["image_path"])
                _remove_if_exists(result["mask_path"])

            completed += 1
            if progress_callback:
                progress_callback(int(completed / total * 100))

    # --- Persist metadata ----------------------------------------------------
    info = {
        "version": version,
        "created": datetime.now().isoformat(),
        "raster_layer": raster_layer.name(),
        "vector_layer": vector_layer.name(),
        "crs": raster_layer.crs().authid(),
        "window_size": window_size,
        "stride": stride,
        "native_pixel_size": pixel_size,
        "burn_value": burn_value,
        "output_format": clip["output_format"],
        "band_count": raster_layer.bandCount(),
        "total_tiles": tile_count,
        "skipped_tiles": skipped_count,
    }
    _write_json(os.path.join(version_dir, "clipping_info.json"), info)

    return {
        "version": version,
        "tile_count": tile_count,
        "skipped_count": skipped_count,
        "version_dir": version_dir,
    }


def get_available_versions(dataset_dir: str) -> list:
    """Returns sorted list of existing clipping version dicts."""
    return _scan_versions(
        os.path.join(
            dataset_dir,
            "clipping"),
        "clipping_info.json")


# ---------------------------------------------------------------------------
# Thread worker
# ---------------------------------------------------------------------------

def _clip_tile_worker(args: dict) -> dict:
    """
    Clips one raster tile and rasterizes the vector mask.
    Runs in a worker thread — opens its own GDAL/OGR handles.
    Returns dict with has_positive (bool), image_path, mask_path.
    """
    from osgeo import gdal, ogr
    gdal.UseExceptions()

    x_min = args["x_min"]
    x_max = args["x_max"]
    y_min = args["y_min"]
    y_max = args["y_max"]
    window_size = args["window_size"]
    burn_value = args["burn_value"]
    image_path = args["image_path"]
    mask_path = args["mask_path"]
    pixel_size = (x_max - x_min) / window_size

    # --- Clip raster tile ----------------------------------------------------
    gdal.Translate(
        image_path,
        args["raster_path"],
        projWin=[x_min, y_max, x_max, y_min],
        width=window_size,
        height=window_size,
        resampleAlg=gdal.GRA_Bilinear,
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
    )

    # --- Create blank mask raster --------------------------------------------
    driver = gdal.GetDriverByName("GTiff")
    mask_ds = driver.Create(
        mask_path, window_size, window_size, 1, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    mask_ds.SetGeoTransform([x_min, pixel_size, 0, y_max, 0, -pixel_size])
    mask_ds.SetProjection(args["crs_wkt"])
    band = mask_ds.GetRasterBand(1)
    band.Fill(0)                            # background = 0

    # --- Open vector and apply spatial filter --------------------------------
    vector_ds = ogr.Open(args["vector_path"])
    ogr_layer = (
        vector_ds.GetLayerByName(args["vector_layer_name"])
        if args["vector_layer_name"]
        else vector_ds.GetLayer(0)
    )
    ogr_layer.SetSpatialFilterRect(x_min, y_min, x_max, y_max)

    # --- Rasterize polygons into mask ----------------------------------------
    gdal.RasterizeLayer(
        mask_ds, [1], ogr_layer,
        burn_values=[burn_value],
    )

    # Check if any foreground pixels were burned
    arr = band.ReadAsArray()
    has_positive = bool(np.any(arr > 0))

    mask_ds.FlushCache()
    mask_ds = None
    vector_ds = None

    return {
        "has_positive": has_positive,
        "image_path": image_path,
        "mask_path": mask_path,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_vector_source(source_uri: str) -> tuple:
    """
    Splits a QGIS vector source URI into (file_path, layer_name).
    Handles plain paths and GeoPackage-style 'path|layername=name' URIs.
    """
    if "|" in source_uri:
        path, params = source_uri.split("|", 1)
        layer_name = None
        for param in params.split("&"):
            if param.startswith("layername="):
                layer_name = param[len("layername="):]
        return path, layer_name
    return source_uri, None


def _tile_origins(extent, tile_geo: float, stride_geo: float) -> list:
    """Returns (x_origin, y_origin) top-left corners for all full tiles."""
    origins = []
    x = extent.xMinimum()
    while x + tile_geo <= extent.xMaximum() + 1e-9:
        y = extent.yMaximum()
        while y - tile_geo >= extent.yMinimum() - 1e-9:
            origins.append((x, y))
            y -= stride_geo
        x += stride_geo
    return origins


def _next_version(dataset_dir: str, stage: str) -> int:
    stage_dir = os.path.join(dataset_dir, stage)
    if not os.path.isdir(stage_dir):
        return 1
    versions = []
    for entry in os.listdir(stage_dir):
        if entry.startswith("v"):
            try:
                versions.append(int(entry[1:]))
            except ValueError:
                pass
    return max(versions) + 1 if versions else 1


def _scan_versions(stage_dir: str, info_filename: str) -> list:
    if not os.path.isdir(stage_dir):
        return []
    results = []
    for entry in sorted(os.listdir(stage_dir)):
        if not entry.startswith("v"):
            continue
        try:
            vnum = int(entry[1:])
        except ValueError:
            continue
        info_path = os.path.join(stage_dir, entry, info_filename)
        info = _read_json(info_path) if os.path.isfile(info_path) else {}
        results.append({"version": vnum, "info": info})
    return sorted(results, key=lambda x: x["version"])


def _write_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _remove_if_exists(path: str):
    if os.path.isfile(path):
        os.remove(path)
