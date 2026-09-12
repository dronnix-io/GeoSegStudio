# Chapter 11 — Reproducibility and Citation

---

### 11.1 Why Reproducibility Matters in Geospatial AI

A segmentation result is only as trustworthy as the pipeline that produced it. In remote sensing research, results that cannot be reproduced — by you, six months later, or by a peer reviewer trying to validate your work — have limited scientific value.

GeoSeg Studio is designed with reproducibility as a first-class concern. Every stage of the pipeline is versioned. Every model checkpoint records the configuration used to produce it. This chapter explains how to use these features to make your results fully reproducible, and how to document your methodology in a way that satisfies journal and conference standards.

---

### 11.2 What GeoSeg Studio Records Automatically

Every trained model checkpoint produced by GeoSeg Studio includes a metadata file that records:

- Plugin version
- Architecture and configuration (base channels, input channels, patch size)
- Loss function, optimizer, learning rate, scheduler
- Batch size, number of epochs trained
- Dataset version identifiers (clip version, split version, augmentation version)
- Device used for training
- Training and validation metrics at each epoch
- Timestamp of the training run

This metadata is saved alongside the checkpoint in the `models/` subfolder of your output directory. You do not need to manually record any of these values.

---

### 11.3 What You Need to Record Yourself

Automatic metadata covers the training configuration, but reproducibility also requires documenting your input data and labeling process.

**Input raster documentation:**
- Acquisition date and conditions (if relevant to your results)
- Sensor and platform (e.g. DJI Matrice 4E, RGB 3-band)
- Ground sampling distance (GSD)
- Area covered

**Labeling documentation:**
- Who labeled the data
- Labeling protocol (what boundaries were used, how edge cases were handled)
- Total number of labeled polygon instances
- Any quality control steps applied

**Dataset version documentation:**
- Clip version used (patch size, overlap, min mask coverage)
- Split version used (train/validation/test percentages, random seed if recorded)
- Augmentation version used (which transforms were applied)

---

### 11.4 Writing Your Methods Section

When publishing work that uses GeoSeg Studio, your methods section should include enough information for a reader to reproduce your pipeline from scratch. Here is a template:

---

*"Semantic segmentation was performed using GeoSeg Studio [CITATION], an open-source QGIS plugin for deep learning geospatial segmentation. Training data was prepared by clipping the input orthomosaic into [PATCH SIZE] × [PATCH SIZE] pixel tiles with [OVERLAP]% overlap, yielding [N] tiles. Tiles were split into train/validation/test sets at [TRAIN]%/[VAL]%/[TEST]% ratios. Training augmentation included [LIST TRANSFORMS].  
A [ARCHITECTURE] architecture with [BASE CHANNELS] base channels and [N BANDS] input channels was trained for [N EPOCHS] epochs using [LOSS FUNCTION] loss, the [OPTIMIZER] optimizer with an initial learning rate of [LR], and a [SCHEDULER] learning rate schedule. Batch size was [BATCH SIZE]. Training was performed on [DEVICE].  
Final model selection used the epoch with the lowest validation loss. Evaluation was performed on the held-out test split. Inference used a sliding window with [OVERLAP]% overlap and a sigmoid threshold of [THRESHOLD]. Vector post-processing included [LIST POST-PROCESSING STEPS]."*

---

Fill in the bracketed values from your model metadata file. This single paragraph contains everything a reviewer needs to reproduce your results.

---

### 11.5 Citing GeoSeg Studio

If you use GeoSeg Studio in published research, please cite it. This helps justify continued development and gives other researchers a way to find the tool.

**BibTeX entry:**

```bibtex
@software{ghaffarian2025geosegstudio,
  author    = {Ghaffarian, Salar},
  title     = {{GeoSeg Studio}: A {QGIS} Plugin for Deep Learning 
               Semantic Segmentation of Geospatial Raster Data},
  year      = {2025},
  publisher = {Dronnix Inc.},
  url       = {https://github.com/dronnix-io/GeoSegStudio},
  note      = {Open-source software licensed under GPL v2.0}
}
```

**In-text citation format (APA):**
Ghaffarian, S. (2025). *GeoSeg Studio: A QGIS plugin for deep learning semantic segmentation of geospatial raster data* [Software]. Dronnix Inc. https://github.com/dronnix-io/GeoSegStudio

---

### 11.6 Sharing Your Trained Model with a Paper

If you are publishing a paper and want to share your trained model weights so others can reproduce your predictions exactly, you have two options:

**Option A: Archive with your data**
Include the checkpoint file (`*.pt`) as supplementary material with your paper submission, or deposit it in a data repository (Zenodo, Figshare, OSF) and cite the DOI.

**Option B: Hugging Face Model Hub**
Hugging Face provides free hosting for model weights. Upload your checkpoint file and the associated metadata, and share the model URL in your paper. This is the most accessible option for readers who want to reproduce your results quickly.

When sharing a model, always include:
- The model metadata file (records the exact training configuration)
- The dataset description and, if possible, the dataset itself
- The version of GeoSeg Studio used

A model checkpoint without its associated dataset and configuration is not reproducible.

---

### 11.7 Sharing Your Dataset

The Calgary building extraction dataset used throughout this handbook is available on Hugging Face:

`huggingface.co/datasets/dronnix-io/drone-building-extraction`

If you have collected and labeled your own dataset and are willing to share it, consider publishing it on Hugging Face or Zenodo. Open datasets are one of the most valuable contributions you can make to the remote sensing research community.

When publishing a dataset, include:
- The raw raster files (or a clear description of how to obtain them)
- The label vector files
- A description of the labeling protocol
- The coordinate system and resolution
- The license (CC BY 4.0 is the most permissive open license appropriate for research datasets)

---

### 11.8 Version Compatibility

GeoSeg Studio's checkpoint format may change between major versions. A checkpoint trained with version 1.x may not load correctly in version 2.x. When archiving checkpoints for long-term reproducibility, record the exact plugin version alongside the checkpoint. This information is stored in the metadata file automatically.

---

*Next: Chapter 12 — What's Coming in Version 2.0*
