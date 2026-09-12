# GeoSeg Studio
## A Practical Guide to Deep Learning Segmentation for Geospatial Professionals

### Introduction

**By Salar Ghaffarian, CEO and Lead AI Engineer, Dronnix Inc.**

---

## The Problem Nobody Solved Until Now

You have a drone. You fly a mission. You come back with a beautiful high-resolution orthomosaic — crisp, georeferenced, ready to analyze. Then you open QGIS and the real work begins.

If you need to extract buildings, count vehicles, map solar panels, or identify vegetation — you are about to spend hours digitizing polygons by hand. Or you will write a Python script that requires installing PyTorch, configuring CUDA drivers, structuring datasets, implementing a training loop, handling sliding window inference, and converting raster outputs back to vectors that QGIS can actually use.

Most GIS professionals are not machine learning engineers. Most machine learning engineers are not GIS professionals. That gap has existed for years and almost nobody has tried to close it — until now.

**GeoSeg Studio** is an open-source QGIS plugin that puts a complete deep learning segmentation pipeline inside QGIS. No terminal. No Python scripts. No environment management. You prepare your data, train your model, evaluate it, and run predictions — all from a dock panel inside the software you already use every day.

This handbook is the guide we wish had existed when we started building it. It is published in full, free, in this repository.

---

## What GeoSeg Studio Actually Does

Here is the simplest way to describe it:

You draw polygons over the things you want to detect. GeoSeg Studio learns what those things look like from your imagery. Then it finds them everywhere — across an entire orthomosaic, at full resolution, automatically.

The output is a vector layer: real polygons, in your coordinate system, ready to measure, query, export, or share.

More specifically, GeoSeg Studio gives you:

- **Data preparation** — a tile clipping, splitting, and augmentation pipeline that turns your raster and vector layers into a training-ready dataset
- **Model training** — 7 deep learning architectures, configurable loss functions, optimizers, and schedulers, with live loss curves and per-epoch metrics
- **Evaluation** — IoU, F1, Precision, Recall, Pixel Accuracy, confusion matrix, and per-tile visualization
- **Prediction** — sliding-window inference on full-resolution rasters with overlap blending and configurable thresholds
- **Post-processing** — merge polygons, fill holes, filter by area, simplify edges, smooth boundaries

Everything runs locally on your machine. Your data never leaves your computer.

---

## Who This Is For

This handbook is written for four kinds of people:

**Remote sensing researchers** who need a reproducible, citable ML pipeline for their segmentation experiments without spending weeks on infrastructure.

**GIS analysts and drone mapping professionals** who want to automate object extraction from their imagery without writing a single line of code.

**Graduate students** working on semantic segmentation projects who need a working baseline fast — and a tool they can actually hand to a supervisor or collaborator without a lengthy setup guide.

**Drone mapping companies** who want to offer AI-powered deliverables to clients — building extraction, vehicle counts, solar panel detection — without hiring a dedicated ML team.

If any of these describe you, this tool was built for you.

---

## A 5-Minute First Result

The fastest way to understand what GeoSeg Studio can do is to run it.

We have published a free sample dataset on Hugging Face specifically for this purpose:

**Drone Building Extraction Dataset — Calgary**
`huggingface.co/datasets/dronnix-io/drone-building-extraction`

It contains high-resolution drone imagery (DJI Matrice 4E) over a residential area in Calgary, Alberta, along with manually annotated building polygons. It is ready to use directly in the Prepare tab — no preprocessing required.

Here is what a first session looks like:

**Step 1 — Install**
Install GeoSeg Studio from the QGIS Plugin Manager. On first launch, select your hardware (CPU or CUDA GPU) and let the plugin install PyTorch into an isolated environment. This takes 5–10 minutes once.

**Step 2 — Prepare**
Load the Calgary raster and vector layers into QGIS. In the Prepare tab, set your output directory, configure a patch size of 256, and click Run All. The plugin clips tiles, splits them into train/validation/test sets, and applies augmentation automatically.

**Step 3 — Train**
Switch to the Train tab. Select UNet, set 20 epochs, leave everything else at defaults, and click Start. Watch the loss curve drop in real time.

**Step 4 — Predict**
Switch to the Predict tab. Load a raster you did not train on. Click Run. The plugin runs sliding-window inference and produces a vector layer of extracted building polygons.

From install to first prediction: under an hour on CPU, under 20 minutes on a mid-range GPU.

---

## What You Will Find in the Handbook

This introduction covers the concept, the tool, and a first result. The chapters that follow go much further.

**Part 1 — Foundation**
The landscape of geospatial AI in 2025, where GeoSeg Studio fits, and a complete installation and environment setup guide including the most common failure modes and how to fix them.

**Part 2 — Your First Real Project**
A hands-on walkthrough of the complete pipeline: dataset preparation, model selection and configuration, training a model that actually converges, honest evaluation of your results, and exporting prediction outputs to GIS-ready formats.

**Part 3 — Real-World Use Cases**
Three complete worked examples with real data, real results, and real parameters:
- Building extraction from drone RGB imagery
- Solar panel detection from thermal and RGB imagery
- Vehicle detection from aerial parking lot surveys

**Part 4 — For Researchers**
How to structure your methods section when using GeoSeg Studio, the BibTeX citation entry, reproducibility guidelines, and what is coming in version 2.0.

**Appendix**
Full parameter reference, troubleshooting guide for the 15 most common errors, glossary, and how to reach the Dronnix team.

---

## About Dronnix

Dronnix is a drone mapping and geospatial AI company based in Calgary, Alberta. We collect and analyze aerial data for solar panel inspection, construction monitoring, agricultural analysis, and large-scale mapping missions.

GeoSeg Studio is our open-source contribution to the geospatial community. We believe that deep learning tools for remote sensing should be accessible to every GIS professional, not just those with a machine learning background.

Version 2.0 will introduce pre-trained models for common object extraction tasks — buildings, vehicles, solar panels, and vegetation — available through the Dronnix platform. Early access for handbook readers will be announced by email.

**Read the handbook:** [start with Chapter 1](01_remote_sensing_segmentation_2025.md), or see the [full contents](README.md)

**Contact:** salar@dronnix.com

**GitHub:** github.com/dronnix-io/GeoSegStudio

---

*GeoSeg Studio is open-source software licensed under the GNU General Public License v2.0.*

*Copyright 2025 Dronnix Inc. All rights reserved for this handbook.*
