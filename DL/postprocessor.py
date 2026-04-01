"""
module: DL/postprocessor.py

Background post-processing worker for Tab 4.

Reads a predicted vector file (GeoPackage / Shapefile), applies a
configurable sequence of geometry operations using Shapely, and writes
a new cleaned GeoPackage.

Operations (applied in this order)
------------------------------------
1. Merge touching / overlapping polygons  (unary_union)
2. Fill holes                             (remove interior rings below threshold)
3. Remove small polygons                  (area < min_area)
4. Remove large polygons                  (area > max_area)
5. Simplify                               (Douglas-Peucker)
6. Smooth edges                           (Chaikin corner-cutting)

Each operation is individually enabled/disabled via the config dict.
Area values are in the CRS map units² (typically m² for projected CRS).

Signals
-------
phase_update(message)
    Human-readable description of the current step.

feature_done(current, total)
    Emitted per-feature during the smooth step (the only per-feature op).

postprocess_finished(success, results, message)
    results keys on success:
      output_path   str   path to saved GeoPackage
      input_count   int   number of input features
      output_count  int   number of output features

Config dict keys
----------------
input_path          str    source vector file
output_path         str    destination GeoPackage path
merge_touching      bool
fill_holes          bool
min_hole_area       float  holes smaller than this are filled (0 = fill all)
filter_min_area     bool
min_area            float
filter_max_area     bool
max_area            float
simplify            bool
simplify_tolerance  float  Douglas-Peucker tolerance in map units
smooth              bool
smooth_iterations   int    Chaikin iterations (2–5 recommended)
"""
from __future__ import annotations

import os

from qgis.PyQt.QtCore import QThread, pyqtSignal


class PostProcessWorker(QThread):

    phase_update         = pyqtSignal(str)
    feature_done         = pyqtSignal(int, int)
    postprocess_finished = pyqtSignal(bool, object, str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config    = config
        self._cancelled = False

    def stop(self):
        self._cancelled = True

    def run(self):
        try:
            self._process()
        except Exception as exc:
            self.postprocess_finished.emit(False, {}, str(exc))

    # -------------------------------------------------------------------------

    def _process(self):
        cfg = self._config

        # --- Read input vector -----------------------------------------------
        self.phase_update.emit("Reading input vector…")

        try:
            from shapely import wkb as shapely_wkb
            from shapely.ops import unary_union
            from shapely.geometry import Polygon, MultiPolygon
        except ImportError:
            raise ImportError(
                "Shapely is required for post-processing.\n"
                "It should be available in the QGIS Python environment.\n"
                "If missing, install it with:  pip install shapely"
            )

        from osgeo import ogr
        src_ds = ogr.Open(cfg["input_path"])
        if src_ds is None:
            raise IOError(f"Cannot open vector file: {cfg['input_path']}")

        src_layer = src_ds.GetLayer(0)
        srs       = src_layer.GetSpatialRef()

        geoms = []
        for feature in src_layer:
            geom_ref = feature.GetGeometryRef()
            if geom_ref is None:
                continue
            try:
                geom = shapely_wkb.loads(bytes(geom_ref.ExportToWkb()))
                if geom and not geom.is_empty:
                    geoms.append(geom)
            except Exception:
                pass

        src_ds = None
        input_count = len(geoms)

        if not geoms:
            raise ValueError("Input vector contains no valid features.")

        # --- 1. Merge touching / overlapping ---------------------------------
        if cfg.get("merge_touching"):
            self.phase_update.emit("Merging touching polygons…")
            merged = unary_union(geoms)
            if merged.is_empty:
                geoms = []
            elif merged.geom_type == "MultiPolygon":
                geoms = list(merged.geoms)
            elif merged.geom_type == "Polygon":
                geoms = [merged]
            else:
                geoms = [g for g in merged.geoms
                         if g.geom_type in ("Polygon", "MultiPolygon")]

        if self._cancelled:
            self.postprocess_finished.emit(False, {}, "Stopped by user.")
            return

        # --- 2. Fill holes ---------------------------------------------------
        if cfg.get("fill_holes"):
            self.phase_update.emit("Filling holes…")
            min_hole = float(cfg.get("min_hole_area", 0.0))
            geoms = [_fill_holes(g, min_hole) for g in geoms]
            geoms = [g for g in geoms if g and not g.is_empty]

        if self._cancelled:
            self.postprocess_finished.emit(False, {}, "Stopped by user.")
            return

        # --- 3 & 4. Area filtering -------------------------------------------
        do_min = cfg.get("filter_min_area") and float(cfg.get("min_area", 0.0)) > 0
        do_max = cfg.get("filter_max_area") and cfg.get("max_area") is not None

        if do_min or do_max:
            self.phase_update.emit("Filtering polygons by area…")
            min_area = float(cfg.get("min_area", 0.0)) if do_min else 0.0
            max_area = float(cfg.get("max_area", 0.0)) if do_max else None
            filtered = []
            for g in geoms:
                area = g.area
                if do_min and area < min_area:
                    continue
                if do_max and area > max_area:
                    continue
                filtered.append(g)
            geoms = filtered

        if self._cancelled:
            self.postprocess_finished.emit(False, {}, "Stopped by user.")
            return

        # --- 5. Simplify -----------------------------------------------------
        if cfg.get("simplify") and geoms:
            self.phase_update.emit("Simplifying geometries…")
            tol   = float(cfg.get("simplify_tolerance", 0.5))
            geoms = [g.simplify(tol, preserve_topology=True) for g in geoms]
            geoms = [g for g in geoms if g and not g.is_empty]

        if self._cancelled:
            self.postprocess_finished.emit(False, {}, "Stopped by user.")
            return

        # --- 6. Smooth edges (per-feature — emits progress) ------------------
        if cfg.get("smooth") and geoms:
            iters   = int(cfg.get("smooth_iterations", 3))
            total   = len(geoms)
            smoothed = []
            for i, g in enumerate(geoms):
                if self._cancelled:
                    self.postprocess_finished.emit(False, {}, "Stopped by user.")
                    return
                self.phase_update.emit(
                    f"Smoothing edges — {i + 1} / {total}…"
                )
                smoothed.append(_smooth_geom(g, iters))
                self.feature_done.emit(i + 1, total)
            geoms = [g for g in smoothed if g and not g.is_empty]

        output_count = len(geoms)

        # --- Write output ----------------------------------------------------
        self.phase_update.emit("Saving output vector…")
        _write_gpkg(geoms, srs, cfg["output_path"])

        results = {
            "output_path":  cfg["output_path"],
            "input_count":  input_count,
            "output_count": output_count,
        }
        self.postprocess_finished.emit(True, results, "Post-processing complete.")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _fill_holes(geom, min_hole_area: float):
    """Remove interior rings smaller than min_hole_area (0 = remove all)."""
    from shapely.geometry import Polygon, MultiPolygon

    if geom.geom_type == "Polygon":
        if min_hole_area <= 0:
            return Polygon(geom.exterior)
        kept = [ring for ring in geom.interiors
                if Polygon(ring).area >= min_hole_area]
        return Polygon(geom.exterior, kept)

    if geom.geom_type == "MultiPolygon":
        parts = [_fill_holes(p, min_hole_area) for p in geom.geoms]
        return MultiPolygon([p for p in parts if p and not p.is_empty])

    return geom


def _chaikin(coords: list, iterations: int) -> list:
    """
    Chaikin's corner-cutting algorithm on a coordinate list.
    Works on both open and closed rings.
    """
    pts = list(coords)
    closed = (pts[0][0] == pts[-1][0] and pts[0][1] == pts[-1][1])
    if closed:
        pts = pts[:-1]

    for _ in range(iterations):
        new_pts = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            new_pts.append((
                0.75 * p0[0] + 0.25 * p1[0],
                0.75 * p0[1] + 0.25 * p1[1],
            ))
            new_pts.append((
                0.25 * p0[0] + 0.75 * p1[0],
                0.25 * p0[1] + 0.75 * p1[1],
            ))
        pts = new_pts

    if closed:
        pts.append(pts[0])
    return pts


def _smooth_geom(geom, iterations: int):
    """Apply Chaikin smoothing to all rings of a Polygon or MultiPolygon."""
    from shapely.geometry import Polygon, MultiPolygon

    if geom.geom_type == "Polygon":
        try:
            ext  = _chaikin(list(geom.exterior.coords), iterations)
            ints = [_chaikin(list(r.coords), iterations) for r in geom.interiors]
            return Polygon(ext, ints)
        except Exception:
            return geom

    if geom.geom_type == "MultiPolygon":
        parts = [_smooth_geom(p, iterations) for p in geom.geoms]
        valid = [p for p in parts if p and not p.is_empty]
        return MultiPolygon(valid) if valid else geom

    return geom


def _write_gpkg(geoms: list, srs, out_path: str):
    """Writes a list of Shapely geometries to a GeoPackage."""
    from osgeo import ogr
    from shapely import wkb as shapely_wkb
    from shapely.geometry import Polygon, MultiPolygon

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    drv = ogr.GetDriverByName("GPKG")
    if os.path.exists(out_path):
        drv.DeleteDataSource(out_path)

    out_ds    = drv.CreateDataSource(out_path)
    out_layer = out_ds.CreateLayer(
        "prediction", srs=srs, geom_type=ogr.wkbMultiPolygon
    )
    out_layer.CreateField(ogr.FieldDefn("area_m2", ogr.OFTReal))

    defn = out_layer.GetLayerDefn()
    for geom in geoms:
        # Normalise to MultiPolygon for a consistent output schema
        if geom.geom_type == "Polygon":
            geom = MultiPolygon([geom])

        feat = ogr.Feature(defn)
        feat.SetGeometry(ogr.CreateGeometryFromWkb(shapely_wkb.dumps(geom)))
        feat.SetField("area_m2", geom.area)
        out_layer.CreateFeature(feat)

    out_ds.FlushCache()
    out_ds = None
