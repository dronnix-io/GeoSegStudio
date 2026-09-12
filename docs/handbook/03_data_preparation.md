# Chapter 3 — Your Data, Your Rules: Preparing a Training Dataset

---

### 3.1 What the Model Actually Learns From

A deep learning segmentation model learns by example. You show it thousands of small image patches paired with binary masks — images where the pixels belonging to your target object are white and everything else is black. The model adjusts its internal parameters until it can reliably reproduce those masks from image patches it has never seen before.

This means the quality of your training data determines the ceiling of your model's performance. A model trained on precise, consistent labels over diverse imagery will outperform a model trained on rushed labels over repetitive imagery, regardless of how sophisticated the architecture is. Garbage in, garbage out is not a cliche in deep learning — it is a law.

The Prepare tab in GeoSeg Studio handles the mechanical work of turning your raster and vector layers into a training-ready dataset. Understanding what it does — and why — will help you make better decisions about your data before you even open the plugin.

---

### 3.2 What You Need Before You Start

**A raster layer (your imagery)**
This is your drone orthomosaic, satellite image, or any georeferenced raster. It can be RGB (3 bands), RGBI (4 bands including near-infrared), multispectral (5+ bands), or single-band (grayscale or thermal). GeoSeg Studio supports 1 to 40 input bands.

Key requirements:
- Must be a GeoTIFF or any format QGIS can load as a raster layer
- Must be georeferenced (has a coordinate system)
- Should be at a resolution appropriate for your target object (see Section 3.3)
- No required minimum size, but more coverage means more diverse training tiles

**A vector layer (your labels)**
This is a polygon layer where you have drawn the outlines of the objects you want to detect. Every polygon represents one instance of your target class.

Key requirements:
- Must be a polygon layer (not points or lines)
- Must be in the same coordinate system as your raster, or QGIS must be able to reproject it on the fly
- Polygons should be accurate but do not need to be perfect — see Section 3.4 on labeling strategy

**An output directory**
A folder on your disk where the prepared dataset will be saved. GeoSeg Studio creates a structured folder hierarchy inside this directory automatically.

---

### 3.3 Resolution and Patch Size: Getting the Balance Right

The most important decision in data preparation is the combination of your raster resolution and your chosen patch size. Together they determine how much of the scene each training tile covers and how much detail is visible to the model.

**Ground Sampling Distance (GSD)**
GSD is the distance on the ground represented by one pixel in your imagery. A GSD of 3 cm means each pixel covers a 3 cm x 3 cm area on the ground. Drone imagery at 30–100 m altitude typically has a GSD of 1–5 cm. Satellite imagery is typically 30 cm to several meters.

**Patch size**
The patch size (set in the Clip step) determines the pixel dimensions of each training tile. GeoSeg Studio supports 64, 128, 256, 512, and 1024 pixels.

**The key relationship**
The area covered by one patch = patch size × GSD × patch size × GSD.

At 3 cm GSD with a 256-pixel patch, each tile covers a 7.7 m × 7.7 m area on the ground. If you are detecting buildings that are typically 10–20 m wide, a 256-pixel patch at 3 cm GSD is a good starting point — the building fits comfortably inside the tile with context around it.

At 30 cm GSD (satellite) with a 256-pixel patch, each tile covers 76.8 m × 76.8 m. A 10 m building would occupy roughly 33 × 33 pixels inside that tile — still detectable, but with less detail.

**Practical guidelines by target object:**

| Target | Recommended minimum GSD | Recommended patch size |
|--------|------------------------|----------------------|
| Buildings (large) | 5–10 cm | 256 or 512 px |
| Vehicles / cars | 2–4 cm | 256 px |
| Solar panels | 2–5 cm | 256 or 512 px |
| Vegetation / tree canopy | 5–15 cm | 256 or 512 px |
| Small objects (< 1 m) | 1–2 cm | 256 px |

If your objects are very small relative to your GSD, consider collecting higher-resolution imagery before labeling. No amount of model sophistication compensates for insufficient resolution.

---

### 3.4 Labeling Strategy: When to Be Precise and When to Be Consistent

The most common mistake in training data preparation is inconsistent labeling. A model can learn from imperfect labels — slightly larger or smaller than the actual object boundary — as long as the labels are consistently imperfect in the same way. What confuses a model is labeling some instances one way and other instances another way.

**Precision vs. consistency**
For most practical use cases, consistency matters more than precision. If you decide to label the rooftop outline of each building (excluding overhanging eaves), do that for every building. If you decide to include the shadow, include it for every building. The model will learn whatever boundary convention you apply, as long as you apply it uniformly.

**How many labels do you need?**
There is no universal answer, but here are practical starting points:

| Use case | Minimum labeled polygons | Recommended |
|----------|------------------------|-------------|
| Large objects (buildings, fields) | 200–300 | 500+ |
| Medium objects (vehicles, solar panels) | 300–500 | 800+ |
| Small or densely packed objects | 500–1000 | 1500+ |

These numbers refer to individual polygon instances, not tiles. More is always better, but diminishing returns set in quickly past a few thousand instances for binary segmentation.

**Negative space matters**
Your raster almost certainly contains large areas with no target objects. These "negative" areas are important training signal — they teach the model what background looks like. The Clip step automatically includes tiles that contain no foreground pixels. Do not manually exclude them.

**Handling edge cases**
Decide in advance how to label:
- Partially visible objects at image edges (label them or exclude them)
- Occluded objects (partially blocked by shadows, vegetation, or other objects)
- Damaged or unusual instances

Document your decisions. When you write up your methodology, you will need to describe your labeling protocol precisely.

---

### 3.5 The Clip Step

The Clip step is the first stage of the Prepare pipeline. It slides a window over your raster, clips square tiles at the specified patch size, and simultaneously rasterizes your vector labels into matching binary masks.

**Parameters:**

**Patch size** — The pixel dimensions of each output tile. Tiles are always square. See Section 3.3 for guidance on choosing the right size.

**Overlap** — The fraction by which consecutive tiles overlap. An overlap of 0.5 means each tile shares 50% of its area with its neighbors. Higher overlap produces more tiles and more variety in how objects appear within tiles, at the cost of more disk space and longer preparation time. For most use cases, 0.25–0.5 is appropriate.

**Min mask coverage** — Tiles where the mask contains less than this fraction of foreground pixels are skipped. The default is 0 (keep all tiles). Setting this to a small positive value (e.g. 0.01) can help remove near-empty tiles where only a tiny fragment of an object appears at the tile boundary, but use it cautiously — tiles with small foreground fractions are valid negative-context examples.

**CPU cores** — The number of parallel workers for tile generation. Set this to the number of physical CPU cores on your machine for fastest processing. The default uses all available cores.

**Output versioning**
Each Clip run produces a versioned output: `clip_v1`, `clip_v2`, etc. This means you can experiment with different patch sizes or overlap settings without overwriting previous results. The version number increments automatically.

**What the output looks like**
After clipping, your output directory will contain a `clips/` subfolder with the versioned clip run inside it. Each tile is saved as a GeoTIFF preserving the original coordinate system and band count. Each mask is saved as a single-band binary GeoTIFF (0 = background, 255 = foreground).

---

### 3.6 The Split Step

The Split step divides your clipped tiles into three subsets: **train**, **validation**, and **test**.

- **Train** — the subset the model sees during training and learns from
- **Validation** — the subset used to measure performance after each epoch during training, without the model ever learning from it
- **Test** — the subset held out entirely until final evaluation, providing an unbiased performance estimate

The split is performed by randomly shuffling the tile list and assigning tiles to each subset according to the percentages you specify.

**Recommended split ratios:**

| Dataset size | Train | Validation | Test |
|-------------|-------|-----------|------|
| Small (< 500 tiles) | 70% | 15% | 15% |
| Medium (500–2000 tiles) | 75% | 15% | 10% |
| Large (> 2000 tiles) | 80% | 10% | 10% |

**A note on spatial autocorrelation**
Aerial imagery has a property that random tile splits do not fully address: tiles from nearby areas are more similar to each other than tiles from distant areas. A random split can result in tiles from the same field, the same block, or the same roof appearing in both train and test sets — making your test metrics overoptimistic.

For rigorous research applications, consider using a spatial split: designate specific geographic regions for train, validation, and test rather than random tile assignment. GeoSeg Studio's random split is appropriate for most operational use cases, but if you are publishing research results, document your split strategy carefully.

**Output versioning**
Like the Clip step, each Split run is versioned: `split_v1`, `split_v2`, etc. The Split step reads from a specific clip version, which you select in the version selector. This means you can apply different split ratios to the same clipped tiles without re-running clipping.

---

### 3.7 The Augmentation Step

Data augmentation artificially increases your dataset size by generating modified copies of your training tiles. Each modification preserves the semantic content — a building is still a building after rotation — but presents it in a way the model has not seen before.

GeoSeg Studio supports the following augmentation transforms, applied identically to both the image tile and its mask:

| Transform | What it does |
|-----------|-------------|
| Original | Keeps the tile as-is (always included) |
| Rotate 90 | Rotates 90 degrees clockwise |
| Rotate 180 | Rotates 180 degrees |
| Rotate 270 | Rotates 270 degrees clockwise |
| Mirror | Flips horizontally (left-right) |
| Flip | Flips vertically (top-bottom) |

Applying all 7 transforms multiplies your training set size by 7 at no additional data collection cost.

**When augmentation helps**
Augmentation is most valuable when your dataset is small or when your target objects appear in a limited range of orientations in the training data. Buildings photographed from directly above are nearly rotationally symmetric — augmentation adds minimal information. Vehicles in a parking lot that are all parked at the same angle benefit significantly from rotation augmentation.

**Augmentation is applied to train only**
The validation and test sets are never augmented. This ensures that your evaluation metrics reflect performance on natural, unmodified imagery — the condition that matches real-world inference.

**Output versioning**
Augmentation runs are versioned: `aug_v1`, `aug_v2`, etc. The Augmentation step reads from a specific split version, which you select in the version selector.

---

### 3.8 Dataset Versioning: Why It Matters

Every Clip, Split, and Augmentation run produces a new versioned output. This means you can:

- Experiment with different patch sizes without losing earlier results
- Compare models trained on different split ratios
- Return to an earlier dataset version if a new configuration performs worse

For research applications, versioning is essential for reproducibility. When you write up your methodology, you can report the exact version identifiers and parameters used, and anyone with the same raster and vector data can reproduce your dataset exactly.

Get into the habit of noting which dataset version (clip + split + augmentation) produced each of your trained models. The Train tab records this information in the model's metadata file automatically.

---

### 3.9 The Run All Button

The **Run All** button in the Prepare tab executes Clip, Split, and Augmentation in sequence using the current settings for all three steps. It is convenient for first runs and for re-running the full pipeline after changing source data.

For iterative experimentation — where you want to change only the split ratio, for example — run the individual steps separately to avoid re-running the more expensive Clip step unnecessarily.

---

### 3.10 Worked Example: The Calgary Building Dataset

The Calgary building extraction dataset (available on Hugging Face at `dronnix-io/drone-building-extraction`) is the reference example for this handbook.

**Dataset properties:**
- Imagery: DJI Matrice 4E orthomosaic, RGB + Alpha (4 bands), approximately 3–4 cm GSD
- Coverage: residential area in Calgary, Alberta, Canada
- Labels: manually annotated building footprint polygons
- License: CC BY 4.0

**Recommended Prepare settings for this dataset:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| Patch size | 256 px | Buildings fit well at this resolution and patch size |
| Overlap | 0.25 | Sufficient diversity without excessive redundancy |
| Min mask coverage | 0.0 | Keep all tiles including negative examples |
| Split ratio | 75 / 15 / 10 | Standard ratio for a medium-sized dataset |
| Augmentations | All 7 | Buildings benefit from full rotation augmentation |

After running the full pipeline with these settings, you will have a versioned dataset ready for the Train tab. The next chapter covers model selection and training configuration.

---

*Next: Chapter 4 — Choosing and Configuring Your Model*
