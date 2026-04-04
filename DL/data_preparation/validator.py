"""
module: validator.py

Validates all user inputs before each stage of the data preparation pipeline.

Hard errors  → raises ValidationError (pipeline stops).
Soft warnings → collected and returned (pipeline continues).

Public entry points:
    validate_for_clipping(config)    → checks layers, geometry, CRS, disk space
    validate_for_splitting(config)   → checks split %, clipping prerequisite
    validate_for_augmentation(config)→ checks methods selected, splitting prerequisite
"""

import os
import shutil

from qgis.core import QgsProject, QgsMapLayer, QgsWkbTypes


class ValidationError(Exception):
    """Raised when a pipeline input fails a hard validation check."""
    pass


# ---------------------------------------------------------------------------
# Public entry points — one per pipeline stage
# ---------------------------------------------------------------------------

def validate_for_clipping(config: dict) -> tuple:
    """
    Validates all inputs required by the Clipping stage.

    Checks:
      - Raster and vector layers resolve from the QGIS project
      - Vector geometry is Polygon or MultiPolygon
      - Vector has at least one feature
      - Raster and vector share the same CRS
      - Output directory exists and is writable
      - Vector extent overlaps raster extent
      - Estimated disk space is sufficient
      - Geometry validity (soft warning)

    Returns (validated_config, warnings).
    validated_config enriches config with resolved layer objects.
    """
    warnings = []

    raster_layer = _resolve_layer(
        config["raster_id"],
        QgsMapLayer.RasterLayer,
        "Raster")
    vector_layer = _resolve_layer(
        config["vector_id"],
        QgsMapLayer.VectorLayer,
        "Vector")

    _check_geometry_type(vector_layer)
    _check_has_features(vector_layer)
    _check_crs_match(raster_layer, vector_layer)
    _check_output_dir(config["output_dir"])
    _check_spatial_overlap(raster_layer, vector_layer)
    _check_disk_space(config, raster_layer)

    invalid_count = _count_invalid_geometries(vector_layer)
    if invalid_count > 0:
        warnings.append(
            f"{invalid_count} invalid geometry/geometries detected in the vector layer. "
            "They will be skipped during clipping. Consider fixing them with "
            "Vector → Geometry Tools → Fix Geometries in QGIS."
        )

    validated = config.copy()
    validated["raster_layer"] = raster_layer
    validated["vector_layer"] = vector_layer
    return validated, warnings


def validate_for_splitting(config: dict) -> tuple:
    """
    Validates inputs required by the Splitting stage.

    Checks:
      - Output directory exists and is writable
      - Split percentages sum to 100
      - The requested clipping version exists and has tiles

    Returns (config_copy, warnings).
    """
    _check_output_dir(config["output_dir"])
    _check_split_percentages(config["split_percentages"])
    _check_clipping_prerequisite(config)
    return config.copy(), []


def validate_for_augmentation(config: dict) -> tuple:
    """
    Validates inputs required by the Augmentation stage.

    Checks:
      - Output directory exists and is writable
      - At least one augmentation method is selected
      - The requested splitting version exists and has tiles

    Returns (config_copy, warnings).
    """
    _check_output_dir(config["output_dir"])

    if not config.get("augmentations"):
        raise ValidationError(
            "No augmentation methods selected. "
            "Check at least one method in the Augmentation section."
        )

    _check_splitting_prerequisite(config)
    return config.copy(), []


# ---------------------------------------------------------------------------
# Layer resolution
# ---------------------------------------------------------------------------

def _resolve_layer(layer_id: str, expected_type, label: str):
    """
    Looks up a layer by ID in the current QGIS project.

    Raises ValidationError if the ID is empty, the layer is not found,
    or the layer type does not match expected_type.
    """
    if not layer_id:
        raise ValidationError(
            f"{label} layer is not selected. Please choose a {
                label.lower()} layer " "from the 'Ins & Outs' section.")

    layer = QgsProject.instance().mapLayer(layer_id)

    if layer is None:
        raise ValidationError(
            f"{label} layer with ID '{layer_id}' was not found in the current QGIS project. "
            "It may have been removed after being selected."
        )

    if layer.type() != expected_type:
        type_name = "raster" if expected_type == QgsMapLayer.RasterLayer else "vector"
        raise ValidationError(
            f"The selected {label} layer '{
                layer.name()}' is not a {type_name} layer.")

    return layer


# ---------------------------------------------------------------------------
# Clipping checks
# ---------------------------------------------------------------------------

def _check_geometry_type(vector_layer):
    """
    Ensures the vector layer contains only Polygon or MultiPolygon geometries.
    Raises ValidationError for any other type.
    """
    geom_type = vector_layer.geometryType()

    if geom_type != QgsWkbTypes.PolygonGeometry:
        type_names = {
            QgsWkbTypes.PointGeometry: "Point",
            QgsWkbTypes.LineGeometry: "Line / LineString",
            QgsWkbTypes.UnknownGeometry: "Unknown",
            QgsWkbTypes.NullGeometry: "Null",
        }
        human_type = type_names.get(
            geom_type, f"unsupported (code {geom_type})")
        raise ValidationError(
            f"The vector layer '{vector_layer.name()}' has geometry type '{human_type}'. "
            "Only Polygon and MultiPolygon layers are supported. "
            "Each polygon defines a region of interest that will be burned into the mask."
        )


def _check_has_features(vector_layer):
    """Ensures the vector layer contains at least one feature."""
    if vector_layer.featureCount() == 0:
        raise ValidationError(
            f"The vector layer '{vector_layer.name()}' contains no features. "
            "Add at least one polygon that marks a region of interest."
        )


def _check_crs_match(raster_layer, vector_layer):
    """
    Ensures both layers share the same CRS.
    A mismatch causes polygons to be burned into wrong pixel locations.
    """
    raster_crs = raster_layer.crs()
    vector_crs = vector_layer.crs()

    if raster_crs != vector_crs:
        raise ValidationError(
            f"CRS mismatch detected:\n"
            f"  Raster layer '{raster_layer.name()}': "
            f"{raster_crs.authid()} – {raster_crs.description()}\n"
            f"  Vector layer '{vector_layer.name()}': "
            f"{vector_crs.authid()} – {vector_crs.description()}\n"
            "Both layers must share the same CRS. Reproject one of them using "
            "Layer → Save As or the 'Reproject Layer' processing tool."
        )


def _check_output_dir(output_dir: str):
    """Ensures the output directory path is provided, exists, and is writable."""
    if not output_dir or not output_dir.strip():
        raise ValidationError(
            "No output directory selected. Please choose an output folder "
            "in the 'Ins & Outs' section."
        )

    if not os.path.isdir(output_dir):
        raise ValidationError(
            f"The output directory does not exist:\n  {output_dir}\n"
            "Please select an existing directory."
        )

    if not os.access(output_dir, os.W_OK):
        raise ValidationError(
            f"The output directory is not writable:\n  {output_dir}\n"
            "Check folder permissions or choose a different output location."
        )


def _check_spatial_overlap(raster_layer, vector_layer):
    """
    Ensures the vector extent overlaps the raster extent.
    A fully non-overlapping vector produces no useful tiles.
    """
    raster_extent = raster_layer.extent()
    vector_extent = vector_layer.extent()

    if not raster_extent.intersects(vector_extent):
        raise ValidationError(
            f"The vector layer '{vector_layer.name()}' does not overlap "
            f"the raster layer '{raster_layer.name()}' spatially.\n"
            "All polygons fall outside the raster extent — no tiles can be generated. "
            "Verify that both layers cover the same geographic area and share the same CRS."
        )


def _check_disk_space(config: dict, raster_layer):
    """
    Estimates the disk space required for the clipping output and raises
    ValidationError if the output drive has insufficient free space.

    Estimation: window_size² × band_count × 4 bytes (image, float32 upper bound)
              + window_size² × 1 byte (mask, uint8) per tile,
              multiplied by estimated tile count with a 30% safety margin.
    """
    clip = config["clip_params"]
    window_size = clip["window_size"]
    stride = clip["stride"]
    pixel_size = raster_layer.rasterUnitsPerPixelX()   # native raster resolution
    band_count = raster_layer.bandCount()

    extent = raster_layer.extent()
    tile_geo = window_size * pixel_size
    stride_geo = stride * pixel_size

    nx = max(1, int((extent.width() - tile_geo) / stride_geo) + 1)
    ny = max(1, int((extent.height() - tile_geo) / stride_geo) + 1)
    tile_count = nx * ny

    image_bytes = window_size * window_size * band_count * 4  # float32
    mask_bytes = window_size * window_size                   # uint8
    required = int(tile_count * (image_bytes + mask_bytes) * 1.3)

    free = shutil.disk_usage(config["output_dir"]).free

    if free < required:
        def _fmt(b):
            return f"{b /
                      1024**3:.2f} GB" if b >= 1024**3 else f"{b /
                                                               1024**2:.0f} MB"
        raise ValidationError(
            f"Insufficient disk space for the clipping output.\n"
            f"  Estimated required : {_fmt(required)}  (~{tile_count} tiles)\n"
            f"  Available on disk  : {_fmt(free)}\n"
            "Free up space or choose a different output directory."
        )


def _count_invalid_geometries(vector_layer) -> int:
    """Counts features with null, empty, or geometrically invalid polygons."""
    invalid = 0
    for feature in vector_layer.getFeatures():
        geom = feature.geometry()
        if geom is None or geom.isEmpty() or not geom.isGeosValid():
            invalid += 1
    return invalid


# ---------------------------------------------------------------------------
# Splitting prerequisite check
# ---------------------------------------------------------------------------

def _check_split_percentages(split: dict):
    """Ensures train + valid + test percentages sum to exactly 100."""
    total = split["train"] + split["valid"] + split["test"]
    if total != 100:
        raise ValidationError(
            f"Split percentages must sum to 100%, but currently sum to {total}%.\n"
            f"  Training:   {split['train']}%\n"
            f"  Validation: {split['valid']}%\n"
            f"  Testing:    {split['test']}%\n"
            "Adjust the values in the 'Splitting' section."
        )


def _check_clipping_prerequisite(config: dict):
    """
    Checks that the selected clipping version exists and contains image tiles.
    Raises ValidationError if not found.
    """
    prefix = config.get("prefix", "dataset")
    version = config.get("clipping_version")
    output_dir = config["output_dir"]

    if version is None:
        raise ValidationError(
            "No clipping version selected. Run 'Apply Clipping' first, "
            "then select a version in the Splitting section."
        )

    images_dir = os.path.join(
        output_dir, f"{prefix}_dataset", "clipping", f"v{version}", "images"
    )

    if not os.path.isdir(images_dir) or not os.listdir(images_dir):
        raise ValidationError(
            f"Clipping v{version} has no tiles in:\n  {images_dir}\n"
            "Run 'Apply Clipping' to generate tiles first."
        )


# ---------------------------------------------------------------------------
# Augmentation prerequisite check
# ---------------------------------------------------------------------------

def _check_splitting_prerequisite(config: dict):
    """
    Checks that the selected splitting version exists and contains tiles
    in at least one of the train / valid / test folders.
    Raises ValidationError if not found.
    """
    prefix = config.get("prefix", "dataset")
    version = config.get("splitting_version")
    output_dir = config["output_dir"]

    if version is None:
        raise ValidationError(
            "No splitting version selected. Run 'Apply Splitting' first, "
            "then select a version in the Augmentation section."
        )

    split_dir = os.path.join(
        output_dir, f"{prefix}_dataset", "splitting", f"v{version}"
    )

    for subset in ("train", "valid", "test"):
        images_dir = os.path.join(split_dir, subset, "images")
        if os.path.isdir(images_dir) and os.listdir(images_dir):
            return  # At least one subset has tiles — good to go

    raise ValidationError(
        f"Splitting v{version} has no tiles in:\n  {split_dir}\n"
        "Run 'Apply Splitting' to generate split tiles first."
    )
