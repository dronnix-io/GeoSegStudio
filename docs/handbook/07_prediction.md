# Chapter 7 — From Prediction to GIS Product

---

### 7.1 The Gap Between a Model and a Deliverable

Evaluation tells you that your model performs well on test tiles. Prediction is where you apply it to a new full-resolution raster — an orthomosaic you have never labeled, perhaps covering an area ten or a hundred times larger than your training region.

The output of this step is not a tile of pixel probabilities. It is a clean vector layer: geographic polygons representing detected objects, in your coordinate system, ready to load into any GIS, measure, query, or deliver to a client.

Getting from a trained model to that deliverable involves sliding-window inference, probability map assembly, thresholding, vectorization, and several post-processing steps that can dramatically affect the visual quality of the output. This chapter covers all of them.

---

### 7.2 Sliding-Window Inference

A trained model processes images at a fixed patch size — the same size used during training. A full-resolution orthomosaic may be 50,000 × 50,000 pixels. No GPU can load that into memory at once.

The solution is **sliding-window inference**: the model processes one patch at a time, moving a window across the raster systematically until every part of the image has been covered. The predictions from each window are assembled into a full-resolution probability map.

**Overlap**

Adjacent windows can overlap. When they do, the overlapping region is predicted twice (or more), and the predictions are averaged. This **overlap blending** reduces the tiling artifact that would otherwise appear at patch boundaries — sharp seams where the model's predictions at the edge of one window do not smoothly match its predictions at the edge of the next.

Higher overlap produces smoother outputs but takes longer to compute (a 50% overlap processes each pixel approximately twice, doubling inference time).

| Overlap | Effect |
|---------|--------|
| 0% | Fastest. Visible seam artifacts likely. |
| 25% | Good for large objects. Minor seam reduction. |
| 50% | Default. Good balance of quality and speed. Recommended for most use cases. |
| 75% | Best quality. Slowest. Use for final deliverables or complex scenes. |

---

### 7.3 Sigmoid Threshold

The model's raw output for each pixel is a probability value between 0 and 1. The sigmoid threshold converts this continuous map into a binary decision: foreground or background.

The default is 0.5. Adjustments here trade off between precision and recall:

- **Threshold 0.3–0.4:** More aggressive detection. Finds more objects including ambiguous cases. More false positives.
- **Threshold 0.5:** Balanced default.
- **Threshold 0.6–0.7:** Conservative detection. Fewer false positives, but more missed objects.

For operational deliverables, tune the threshold based on the specific requirements of the use case. For a solar panel inspection report, a lower threshold that catches more panels (with some false alarms that a human reviewer can discard) is often preferable to a higher threshold that misses damaged panels.

---

### 7.4 Output Formats

GeoSeg Studio supports three output options:

**Raster only (GeoTIFF)**
Saves the binary prediction mask as a single-band GeoTIFF with the same extent, resolution, and coordinate system as the input raster. Foreground pixels are 1, background pixels are 0.

Use this when: you want to do further raster-based analysis (zonal statistics, map algebra), or you want to vectorize the output yourself using a different tool.

**Vector only (GeoPackage or Shapefile)**
Vectorizes the binary prediction mask into polygon features and saves them as a GeoPackage (recommended) or Shapefile. GeoPackage is preferred because it supports longer field names, larger files, and avoids the multiple-file fragmentation of Shapefile format.

Use this when: you need a clean polygon layer ready for GIS analysis or client delivery.

**Both**
Saves both the raster and vector outputs. Useful when you want to inspect the raw prediction alongside the processed polygons, or when both formats are needed downstream.

---

### 7.5 Loading Results into QGIS

If the **Load result into QGIS** option is enabled, the output is automatically added to the QGIS project as a layer when prediction completes. You can see the results immediately without manually browsing to the output file.

For raster outputs, the prediction map is loaded as a styled single-band raster (foreground pixels highlighted).
For vector outputs, the polygon layer is loaded with a default style.

---

### 7.6 Post-Processing: Why It Matters

A raw vectorized prediction often looks rough — pixelated edges, tiny fragments, holes inside objects, and occasional merged polygons where two adjacent objects are detected as one. Post-processing cleans these artifacts and produces a professional-quality deliverable.

GeoSeg Studio applies post-processing in a fixed sequence:

```
Raw prediction mask
    → Merge
    → Fill holes
    → Area filter
    → Simplify
    → Smooth
    → Final vector output
```

Each step is optional and configurable. The following sections explain each one.

---

### 7.7 Merge

The merge step combines polygons that are touching or overlapping into single polygons. This corrects for cases where the sliding-window inference produced small gaps between adjacent predictions of the same object.

**When to use:** almost always. A building whose roof spans a tile boundary may be detected as two separate polygons with a 1–2 pixel gap between them. Merging produces the correct single polygon.

**Tolerance:** The merge tolerance controls how large a gap between polygons is bridged by the merge operation. A tolerance of 0 merges only touching or overlapping polygons. A small positive tolerance (e.g. 0.5–2.0 in the raster's units — meters for most projected coordinate systems) bridges small gaps.

**Caution:** Setting the tolerance too high will merge polygons that represent distinct objects. In a dense parking lot where cars are close together, an aggressive merge tolerance will fuse separate vehicles into one blob. Match the tolerance to the typical spacing between objects in your scene.

---

### 7.8 Fill Holes

The fill holes step removes holes inside polygons — interior regions that were predicted as background surrounded by foreground.

Holes occur for several reasons: shadow inside a building rooftop, a skylight, a car parked inside a large building footprint that appears in the orthomosaic. Depending on your use case, these holes may or may not be meaningful.

**Area threshold:** Fills only holes smaller than the specified area (in the raster's area units — typically square meters). Holes larger than the threshold are preserved.

For building extraction: set the threshold to the area of the largest skylight or internal feature you want to ignore. A value of 5–20 m² covers most skylights and HVAC units without filling large courtyards inside commercial buildings.

For vehicle detection: fill holes is usually not needed — vehicles are small enough that interior holes are rare.

---

### 7.9 Area Filter

The area filter removes polygons that are smaller or larger than specified size limits.

**Minimum area:** Removes polygons smaller than this value. This is the most commonly used filter. It eliminates:
- Isolated noise pixels that the model predicted as foreground
- Tiny fragments from tile boundaries
- Objects that are too small to be the target class (if you are detecting buildings, a 1 m² polygon is almost certainly noise)

**Maximum area:** Removes polygons larger than this value. This is useful if you know your target objects have a maximum plausible size. For vehicle detection, a polygon larger than 25–30 m² is probably two merged vehicles or a false alarm from a large vehicle shadow.

**Setting appropriate thresholds:**
The right thresholds depend on your target object and your imagery resolution. A general approach:
1. Run prediction without any area filtering first
2. Examine the output — note the area of the smallest valid detection and the largest noise fragment
3. Set the minimum area to a value that eliminates the noise while keeping the smallest valid detection

---

### 7.10 Simplify

The simplify step reduces the number of vertices in each polygon using the **Douglas-Peucker algorithm**. It removes vertices that are within a specified tolerance distance of the simplified line, producing smoother and less data-heavy polygons.

For rasterized predictions, raw polygons follow the pixel grid exactly — producing staircase edges with one vertex every pixel. A simplification tolerance of 0.5–1.0 meters (for drone imagery at a few cm GSD) removes most of the staircase while preserving the overall shape accurately.

**Effect on area and shape:** Simplification can slightly reduce the area of polygons by cutting corners. For use cases where precise area measurement matters (solar panel coverage, building footprint area), apply a conservative tolerance or skip simplification.

---

### 7.11 Smooth

The smooth step applies **Chaikin's corner-cutting algorithm**, which repeatedly replaces each vertex with two new vertices at 25% and 75% of each edge segment. The result is a smooth, rounded polygon that looks more natural than the raw pixel-following outline.

Smooth is applied after simplify. The number of iterations controls how smooth the result is — more iterations produce rounder shapes at the cost of slightly less accurate boundary placement.

For most use cases, 2–4 iterations produces visually clean results without significant shape distortion.

**When to skip smoothing:** If your target objects have truly angular boundaries (rectangular solar panels, grid-aligned buildings), heavy smoothing may produce curved edges that look wrong. Use conservative smoothing or skip it for geometrically regular objects.

---

### 7.12 Recommended Post-Processing Settings by Use Case

| Parameter | Buildings | Solar panels | Vehicles |
|-----------|-----------|-------------|---------|
| Merge tolerance | 1.0 m | 0.3 m | 0.1 m |
| Fill holes threshold | 10 m² | 0.5 m² | Off |
| Min area | 10 m² | 1 m² | 3 m² |
| Max area | Off | 100 m² | 30 m² |
| Simplify tolerance | 0.5 m | 0.2 m | 0.3 m |
| Smooth iterations | 3 | 2 | 2 |

These are starting points. Your specific imagery, resolution, and object characteristics will require adjustments. Always visually inspect the first prediction output before committing to a post-processing configuration.

---

### 7.13 Working with Large Rasters

For orthomosaics covering large areas — tens of square kilometers at high resolution — inference can take a significant amount of time, primarily determined by:

- Total pixel count of the input raster
- Overlap percentage (higher overlap = more windows = longer inference)
- Hardware (GPU inference is typically 10–30x faster than CPU)

**Practical guidance:**
- For exploratory runs, use a clipped subset of the raster to verify output quality before running on the full extent
- For final deliverables, use 50% overlap and GPU inference where available
- On CPU, a 10,000 × 10,000 pixel raster with 50% overlap at 256px patch size takes approximately 20–40 minutes. The same on a mid-range GPU takes 2–5 minutes.

There is no hard limit on input raster size — GeoSeg Studio processes one patch at a time, so memory usage is bounded by the patch size, not the raster size.

---

*Next: Chapter 8 — Building Extraction from Drone Imagery*
