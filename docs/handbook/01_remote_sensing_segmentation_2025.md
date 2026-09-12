# Chapter 1 — Remote Sensing Segmentation in 2025

## Where the Field Stands, and Where the Gap Is

---

### 1.1 The Explosion of Aerial Imagery

Drone hardware has gone through a remarkable transformation in the past decade. What once required a manned aircraft and a specialized crew can now be flown by a single operator with a consumer-grade quadcopter. Resolution that was once measured in meters is now measured in centimeters. A single flight over a construction site, solar farm, or agricultural field can produce a georeferenced orthomosaic of hundreds of millions of pixels — accurate to within a few centimeters on the ground.

The result is that the geospatial industry is no longer limited by data collection. It is limited by analysis.

The bottleneck has shifted. The question is no longer "can we collect this imagery?" The question is "can we extract meaningful information from it at the scale and speed that operational use cases demand?"

Manual digitizing — drawing polygons by hand over satellite or drone imagery — is still the dominant method in many organizations. It is accurate, controllable, and requires no special infrastructure. It is also slow, expensive, and impossible to scale. A trained analyst digitizing buildings from a 10 km² orthomosaic can spend days on a task that should take minutes.

Deep learning changed the theoretical answer to this problem years ago. The practical answer — accessible to the average GIS professional — has taken much longer to arrive.

---

### 1.2 The Machine Learning Ecosystem and Its Blind Spot

The machine learning ecosystem has produced powerful tools for image segmentation: PyTorch, TensorFlow, Hugging Face, Segment Anything Model, and dozens of specialized geospatial AI libraries. These tools are genuinely impressive. They are also genuinely inaccessible to anyone who is not a software engineer.

To train a semantic segmentation model on your own drone imagery using existing tools, you typically need to:

- Set up a Python virtual environment and manage dependency conflicts
- Restructure your GeoTIFF data into a format the framework expects (usually folders of PNG tiles)
- Write or adapt a training script that handles data loading, augmentation, model initialization, loss computation, backpropagation, checkpointing, and validation
- Handle coordinate system transformations when going from raster pixels back to geographic coordinates
- Write inference code that handles large rasters via sliding-window processing
- Convert raster predictions back to vector polygons usable in GIS software

Each of these steps has its own failure modes, its own documentation, and its own community forum threads. A GIS analyst who is excellent at spatial analysis but not a Python developer faces a steep and discouraging learning curve.

This is not a criticism of the machine learning community. These tools were built for machine learning researchers, and they serve that audience well. The gap is simply that nobody built a tool specifically for the GIS professional who wants to use deep learning without becoming a deep learning engineer.

---

### 1.3 The Commercial Landscape

Several commercial platforms have entered this space. ESRI's ArcGIS Pro includes deep learning tools. Google Earth Engine offers pixel classification. Various cloud platforms offer API-based object detection.

These solutions share a common limitation: they are platform-locked, subscription-dependent, and designed for workflows that may not match yours. If you work with proprietary drone data, operate in a regulated environment, or simply need your data to stay on your own infrastructure, cloud-based solutions are often not viable.

There is also a cost dimension. Enterprise geospatial AI platforms are priced for large organizations. A small drone mapping company, a university research group, or a municipal GIS department often cannot justify the cost — especially for exploratory or project-specific work.

The open-source ecosystem has produced research code and proof-of-concept repositories. What it has not produced, until now, is a complete, maintained, user-facing tool that integrates directly into the software that GIS professionals already use.

---

### 1.4 Where GeoSeg Studio Fits

GeoSeg Studio occupies a specific position in this landscape:

- **Open source** — no licensing fees, no subscriptions, no vendor lock-in
- **Local execution** — your data stays on your machine
- **QGIS-native** — built as a plugin for the most widely used open-source GIS platform in the world
- **End-to-end** — data preparation, training, evaluation, and prediction in a single interface
- **No code required** — every parameter is configurable through a UI

It is not the most powerful tool for a machine learning researcher who wants to implement a custom architecture, modify the training loop, or run distributed training across a GPU cluster. For those users, writing PyTorch directly remains the right approach.

It is the right tool for the GIS professional, the drone mapping company, and the researcher who needs a working segmentation pipeline that integrates with their existing QGIS workflow — and who does not want to spend weeks on infrastructure before doing their actual work.

---

### 1.5 Binary Segmentation: What It Means and Why It Is Enough

GeoSeg Studio currently performs **binary semantic segmentation**: for every pixel in your raster, the model outputs one of two classes — foreground or background.

You define what "foreground" means by providing a vector layer of labeled polygons. Buildings, vehicles, solar panels, vegetation — any single object class you can label becomes the target of the model.

This constraint is more liberating than it sounds. The majority of practical object extraction tasks from drone imagery are single-class problems:

- Extract buildings (foreground) from everything else (background)
- Detect solar panels (foreground) from rooftops and ground (background)
- Count vehicles (foreground) from pavement (background)
- Map water bodies (foreground) from land (background)

Binary segmentation handles all of these cleanly. The model architecture, loss functions, and post-processing pipeline are all optimized for this formulation.

Multi-class segmentation — where you simultaneously detect buildings, roads, vegetation, and water in a single pass — is a more complex problem that requires a different labeling workflow, a different output head, and different evaluation metrics. It is on the roadmap for version 2.0.

---

### 1.6 How This Handbook Is Organized

This handbook follows the same structure as GeoSeg Studio itself: prepare, train, evaluate, predict.

**Part 1 (Chapters 1–2)** covers the landscape and setup. You are in Chapter 1. Chapter 2 walks through installation in detail, including every failure mode we have encountered and how to resolve it.

**Part 2 (Chapters 3–7)** is a complete hands-on walkthrough of the pipeline. Each chapter corresponds to one stage: data preparation, model configuration, training, evaluation, and prediction. By the end of Part 2 you will have trained and deployed a working segmentation model.

**Part 3 (Chapters 8–10)** covers three real-world use cases with real data, real parameters, and real results. Each chapter is self-contained and can be read independently.

**Part 4 (Chapters 11–12)** is written for researchers: how to cite GeoSeg Studio correctly, how to document your experimental setup for reproducibility, and what is coming in version 2.0.

The appendix contains a complete parameter reference, a troubleshooting guide, and a glossary for readers who are newer to either the GIS or the machine learning side of the work.

---

*Next: Chapter 2 — Installing GeoSeg Studio the Right Way*
