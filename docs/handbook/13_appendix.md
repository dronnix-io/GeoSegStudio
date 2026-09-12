# Appendix

---

## A — Complete Parameter Reference

### Prepare Tab

#### Clip

| Parameter | Description | Valid values | Default |
|-----------|-------------|-------------|---------|
| Raster layer | Input raster to clip | Any loaded QGIS raster layer | — |
| Vector layer | Label polygon layer | Any loaded QGIS polygon layer | — |
| Output directory | Where to save the dataset | Any writable folder path | — |
| Patch size | Tile size in pixels (square) | 64, 128, 256, 512, 1024 | 256 |
| Overlap | Fraction of overlap between adjacent tiles | 0.0–0.75 | 0.25 |
| Min mask coverage | Minimum foreground fraction to keep a tile | 0.0–1.0 | 0.0 |
| CPU cores | Parallel workers for tile generation | 1 to number of physical cores | All cores |

#### Split

| Parameter | Description | Valid values | Default |
|-----------|-------------|-------------|---------|
| Clipping version | Which clip run to split | Version selector | Latest |
| Train % | Fraction assigned to training set | 0–100 (must sum to 100 with val and test) | 75 |
| Validation % | Fraction assigned to validation set | 0–100 | 15 |
| Test % | Fraction assigned to test set | 0–100 | 10 |

#### Augmentation

| Parameter | Description | Valid values | Default |
|-----------|-------------|-------------|---------|
| Splitting version | Which split run to augment | Version selector | Latest |
| Transforms | Which augmentation transforms to apply | Original, Rotate 90/180/270, Mirror, Flip | Original only |

---

### Train Tab

| Parameter | Description | Valid values | Default |
|-----------|-------------|-------------|---------|
| Dataset | Augmentation version to train on | Version selector | Latest |
| Architecture | Model architecture | UNet, Attention UNet, UNet++, LinkNet, DeepLabV3+, SwinUNet, SegFormer | UNet |
| Input channels | Number of input bands | 1–40 | Auto-detected from dataset |
| Base channels | Feature channel width at first encoder level | 16, 32, 64 | 32 |
| Loss function | Training loss | BCE, Dice, BCE+Dice | BCE+Dice |
| Optimizer | Parameter update algorithm | Adam, AdamW, SGD | AdamW |
| Learning rate | Initial learning rate | Any positive float | 0.001 |
| LR scheduler | Learning rate schedule | StepLR, Cosine Annealing, ReduceLROnPlateau | Cosine Annealing |
| Batch size | Tiles per training step | 1–128 | 16 |
| Epochs | Number of complete training passes | 1–1000 | 50 |
| Checkpoint | Save strategy | Best only, Every N epochs | Best only |
| Device | Hardware for training | CPU, CUDA (if available) | Best available |

---

### Evaluate Tab

| Parameter | Description | Valid values | Default |
|-----------|-------------|-------------|---------|
| Model checkpoint | Trained model to evaluate | Any saved checkpoint | — |
| Split | Which data split to evaluate on | Train, Validation, Test | Test |
| Sigmoid threshold | Threshold for binary conversion | 0.0–1.0 | 0.5 |
| Save visualizations | Whether to save prediction visualization tiles | Yes/No | Yes |

---

### Predict Tab

| Parameter | Description | Valid values | Default |
|-----------|-------------|-------------|---------|
| Model checkpoint | Trained model to use for inference | Any saved checkpoint | — |
| Input raster | Raster to run inference on | Any loaded QGIS raster layer | — |
| Output directory | Where to save prediction outputs | Any writable folder path | — |
| Overlap | Sliding window overlap fraction | 0.0–0.75 | 0.5 |
| Sigmoid threshold | Threshold for binary conversion | 0.0–1.0 | 0.5 |
| Output format | What to save | Raster, Vector, Both | Vector |
| Vector format | Format for vector output | GeoPackage, Shapefile | GeoPackage |
| Load into QGIS | Auto-load result as QGIS layer | Yes/No | Yes |
| Merge tolerance | Distance for polygon merging | 0.0+ (map units) | 0.5 |
| Fill holes threshold | Max area of holes to fill | 0.0+ (map area units) | 0 (off) |
| Min area filter | Remove polygons smaller than this | 0.0+ (map area units) | 0 (off) |
| Max area filter | Remove polygons larger than this | 0.0+ (map area units) | 0 (off) |
| Simplify tolerance | Douglas-Peucker tolerance | 0.0+ (map units) | 0 (off) |
| Smooth iterations | Chaikin smoothing iterations | 0–10 | 0 (off) |

---

## B — Troubleshooting Guide

### Installation Issues

**"GeoSeg Studio" does not appear in the QGIS Plugin Manager search results**
The plugin may not yet be published to the official QGIS repository. Install from ZIP using the latest release from GitHub: `github.com/dronnix-io/GeoSegStudio/releases`

**PyTorch installation progress bar freezes**
Network timeout. Delete the `env/` folder inside the plugin directory and try again on a stable connection. Use Ethernet if possible.

**"Installation failed" during PyTorch setup**
Check available disk space (minimum 5 GB required). Check that QGIS is not running as Administrator.

**QGIS crashes after plugin installation**
Disable the plugin, delete `env/`, re-enable and let setup run again. If crash persists, reinstall the plugin from scratch via QGIS Plugin Manager.

---

### Prepare Tab Issues

**Clipping produces zero tiles**
- Check that the raster and vector layers are in the same coordinate reference system, or that QGIS can reproject one to match the other
- Check that the vector layer actually overlaps with the raster extent
- Check that the vector layer contains polygon features (not points or lines)

**Clipping is extremely slow**
- Reduce the number of CPU cores if your machine has other running processes
- Check available RAM — clipping a very large raster with many workers can exhaust memory

**Version selector for Splitting is empty**
Click the refresh button next to the version selector. If still empty, the Clip step has not been run yet for the selected output directory, or the output directory does not match between the Clip and Split steps.

---

### Train Tab Issues

**"Input channels mismatch" error at training start**
The input channels setting does not match the band count of your training tiles. Check the band count of your source raster and update the input channels setting.

**Training loss is NaN (Not a Number) from the first epoch**
Learning rate is too high. Reduce by 10x and try again. Also check that your training tiles are not corrupted (all-zero or all-NaN values).

**GPU not visible in device selector**
- Verify that PyTorch was installed with the correct CUDA version (matching your `nvidia-smi` output)
- Update NVIDIA drivers
- Reinstall the PyTorch environment by deleting `env/` and rerunning setup

**Training loss does not decrease after several epochs**
- Try a higher learning rate (10x current value)
- Check that augmented training tiles are valid (not black or corrupt)
- Try a different architecture

**"Out of memory" error during training**
Reduce batch size. Halve it until the error disappears.

---

### Predict Tab Issues

**Prediction output is all background (all zeros)**
- Sigmoid threshold may be too high. Try 0.3 or lower.
- The input raster may have a different band order or normalization than the training data
- Verify that the model checkpoint was trained on data from a similar sensor and scene type

**Prediction output is all foreground**
- Sigmoid threshold may be too low. Try 0.7 or higher.
- The model may not have converged during training. Check training loss curves.

**Very slow inference on large raster**
Expected on CPU. Reduce overlap to 0.25 for a faster (but lower quality) run. Use GPU for production inference.

**Vector output has many tiny polygon fragments**
Enable the minimum area filter. Set it to the smallest plausible area for your target object class.

**Polygon boundaries look pixelated**
Enable simplify and smooth. Start with a simplify tolerance of 0.5 map units and 2 smooth iterations.

---

## C — Glossary

**Augmentation**
The process of generating modified copies of training data (rotations, flips) to artificially increase dataset size and diversity.

**Backpropagation**
The algorithm that computes how each parameter in a neural network contributed to the prediction error, used to update parameters during training.

**Batch size**
The number of training samples processed in a single forward-backward pass before updating model parameters.

**Binary segmentation**
Segmentation where each pixel is classified into one of two classes: foreground (target object) or background.

**Checkpoint**
A saved copy of a model's parameters at a specific point during training.

**Confusion matrix**
A table showing the counts of true positives, false positives, true negatives, and false negatives for a set of predictions.

**Dice coefficient (F1 score)**
A metric measuring the overlap between predicted and ground truth masks. Equal to 2×TP / (2×TP + FP + FN).

**Encoder-decoder**
A neural network architecture pattern where the input is progressively compressed (encoder) then expanded back to the original size (decoder).

**Epoch**
One complete pass through the entire training dataset during model training.

**False negative (FN)**
A foreground pixel that the model incorrectly classified as background (missed detection).

**False positive (FP)**
A background pixel that the model incorrectly classified as foreground (false alarm).

**Fine-tuning**
Starting training from an existing model checkpoint rather than random initialization, typically to adapt a model trained on one dataset to a new dataset.

**GeoPackage**
An open, standards-based geospatial data format that stores vector and raster data in a single file. Preferred over Shapefile for most use cases.

**GeoTIFF**
A georeferenced raster image format that stores geographic metadata (coordinate system, extent) alongside pixel values.

**Ground sampling distance (GSD)**
The distance on the ground represented by one pixel in aerial or satellite imagery.

**IoU (Intersection over Union)**
A metric measuring the overlap between predicted and ground truth masks as a fraction of their union. Equal to TP / (TP + FP + FN).

**Learning rate**
A hyperparameter controlling the size of parameter update steps during training.

**Loss function**
A mathematical function that measures the difference between model predictions and ground truth, used as the training signal.

**Orthomosaic**
A georeferenced aerial image composed of many overlapping individual frames stitched together to create a seamless, uniform-scale map.

**Overfitting**
When a model performs well on training data but poorly on new data, because it has memorized training patterns rather than learning generalizable features.

**Patch size**
The pixel dimensions of training tiles. Tiles are always square in GeoSeg Studio.

**Precision**
The fraction of foreground predictions that are correct. Equal to TP / (TP + FP).

**Recall**
The fraction of actual foreground pixels that the model detected. Equal to TP / (TP + FN).

**Semantic segmentation**
The task of assigning a class label to every pixel in an image.

**Sigmoid threshold**
The probability cutoff used to convert a continuous prediction map into a binary mask. Pixels above the threshold are labeled foreground; below are labeled background.

**Skip connection**
A direct connection between an encoder layer and the corresponding decoder layer, allowing the decoder to access fine spatial detail from early encoder features.

**Sliding window inference**
Running inference on a large raster by processing one fixed-size patch at a time and assembling the results into a full-size output.

**Transfer learning**
Using a model trained on one task or dataset as a starting point for training on a different (but related) task or dataset.

**True negative (TN)**
A background pixel that the model correctly classified as background.

**True positive (TP)**
A foreground pixel that the model correctly classified as foreground.

**Validation set**
A subset of the data held out from training, used to measure model performance after each epoch without the model ever learning from it.

---

## D — About Dronnix

Dronnix is a drone mapping and geospatial AI company based in Calgary, Alberta, Canada.

We collect and analyze aerial data for solar panel inspection, construction site monitoring, agricultural analysis, and large-scale mapping missions. GeoSeg Studio is our open-source contribution to the geospatial community — we believe that deep learning tools for remote sensing should be accessible to every GIS professional, not only those with a machine learning background.

**Website:** dronnix.com
**GitHub:** github.com/dronnix-io
**Contact:** salar@dronnix.com

---

*GeoSeg Studio is open-source software licensed under the GNU General Public License v2.0.*
*This handbook is copyright 2025 Dronnix Inc. All rights reserved.*
*Unauthorized reproduction or redistribution of this handbook is prohibited.*
