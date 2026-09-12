# GeoSeg Studio Handbook

### A Practical Guide to Deep Learning Segmentation for Geospatial Professionals

**By Salar Ghaffarian, CEO and Lead AI Engineer, [Dronnix Inc.](https://www.dronnix.com)**

---

This handbook is the companion to the [GeoSeg Studio](../../README.md) QGIS plugin. It covers the whole workflow — why geospatial segmentation is hard, how to build a training dataset that works, how to train a model that converges, how to evaluate it honestly, and how to turn predictions into a GIS deliverable you can hand to a client.

It is written for people who need results, not a machine learning course. You do not need to write code, and no chapter assumes you have trained a neural network before. Where a concept matters — what IoU actually measures, why validation loss climbing is not always overfitting — it is explained when it first matters and not before.

Read it start to finish for a complete grounding, or jump to the chapter matching the step you are stuck on.

---

## Contents

**Start here**

| | Chapter | What it covers |
|---|---------|----------------|
| | [Introduction](00_free_intro.md) | What GeoSeg Studio does, who it is for, and a first result in five minutes |

**Part 1 — Foundation**

| | Chapter | What it covers |
|---|---------|----------------|
| 1 | [Remote Sensing Segmentation in 2025](01_remote_sensing_segmentation_2025.md) | The geospatial AI landscape and where this tool fits |
| 2 | [Installing GeoSeg Studio the Right Way](02_installation.md) | Installation, the PyTorch environment, and the common failure modes |

**Part 2 — Your First Real Project**

| | Chapter | What it covers |
|---|---------|----------------|
| 3 | [Your Data, Your Rules](03_data_preparation.md) | Labelling, clipping, splitting and augmenting a training dataset |
| 4 | [Choosing and Configuring Your Model](04_model_configuration.md) | The seven architectures, and how to pick one |
| 5 | [Training That Actually Converges](05_training.md) | Loss functions, optimizers, schedulers, and reading a training curve |
| 6 | [Evaluating Your Model Honestly](06_evaluation.md) | IoU, F1, precision, recall, and what each one hides |
| 7 | [From Prediction to GIS Product](07_prediction.md) | Sliding-window inference, vectorisation and post-processing |

**Part 3 — Real-World Use Cases**

| | Chapter | What it covers |
|---|---------|----------------|
| 8 | [Building Extraction from Drone Imagery](08_building_extraction.md) | A complete project at 3 cm GSD, with every real parameter and metric |
| 9 | Solar Panel Detection from Thermal and RGB Imagery | *In preparation* |
| 10 | Vehicle Detection from Aerial Surveys | *In preparation* |

**Part 4 — For Researchers**

| | Chapter | What it covers |
|---|---------|----------------|
| 11 | [Reproducibility and Citation](11_reproducibility.md) | Writing a methods section, the BibTeX entry, reproducibility guidelines |
| 12 | [What's Coming in Version 2.0](12_version2_roadmap.md) | The roadmap, including pre-trained models |

**Reference**

| | Chapter | What it covers |
|---|---------|----------------|
| — | [Appendix](13_appendix.md) | Full parameter reference, troubleshooting, and glossary |

---

## Sample Dataset

Chapters 3 through 8 can be followed along with a free dataset published for this purpose:

**Drone Building Extraction Dataset — Calgary**
[`huggingface.co/datasets/dronnix-io/drone-building-extraction`](https://huggingface.co/datasets/dronnix-io/drone-building-extraction)

High-resolution drone imagery over a residential area in Calgary, Alberta, with manually annotated building polygons. It loads directly into the Prepare tab with no preprocessing.

---

## Questions and Corrections

Found an error, or something that did not work as described? Open an issue on the [tracker](https://github.com/dronnix-io/GeoSegStudio/issues) — corrections to the handbook are as welcome as bug reports against the plugin.

**Contact:** salar@dronnix.com

---

*The GeoSeg Studio plugin is free and open-source software under the GNU General Public License v2.0. This handbook is © 2025 Dronnix Inc.*
