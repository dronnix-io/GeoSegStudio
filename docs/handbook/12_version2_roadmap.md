# Chapter 12 — What's Coming in Version 2.0

---

### 12.1 From Open-Source Tool to Platform

GeoSeg Studio version 1.x is a complete, functional deep learning pipeline for anyone who wants to train their own segmentation model from scratch. It solves the problem of infrastructure — you no longer need to write PyTorch code or manage Python environments to train and deploy a segmentation model inside QGIS.

Version 2.0 addresses the next barrier: **data collection and labeling**.

Training a model from scratch requires labeled training data. For a new user who has never collected drone imagery for a machine learning project, building a high-quality labeled dataset can take weeks of flights, hours of annotation, and multiple training iterations before the model reaches useful performance.

Version 2.0 introduces **pre-trained models**: model weights trained by Dronnix on large, high-quality datasets for specific object classes that drone mapping professionals encounter most frequently. Users can load a pre-trained model and run inference on their own imagery immediately — no training required — or use it as a starting point for fine-tuning on their specific conditions.

---

### 12.2 Pre-Trained Model Classes

The initial release of pre-trained models will target the following object classes:

**Building extraction**
Trained on drone RGB orthomosaics across multiple geographic regions, building types, and roof materials. Covers residential, commercial, and industrial structures.

**Vehicle detection**
Trained on high-resolution drone imagery of parking lots, roads, and urban areas. Suitable for occupancy monitoring, traffic studies, and construction site management.

**Solar panel detection**
Trained on both RGB and thermal drone imagery. Covers rooftop installations and ground-mounted utility-scale arrays. Particularly relevant for inspection workflows where damaged or underperforming panels need to be identified.

**Vegetation and tree canopy**
Trained on RGB and near-infrared (NDVI-capable) drone imagery. Covers urban tree inventory, agricultural field boundary detection, and reforestation monitoring.

**Water bodies**
Trained on aerial and satellite imagery. Covers lakes, rivers, irrigation channels, and seasonal water extent mapping.

---

### 12.3 The Dronnix Model API

Pre-trained models will be available through the **Dronnix Model API**, accessed directly from within GeoSeg Studio. The workflow:

1. Open GeoSeg Studio and navigate to a new Models tab (added in v2.0)
2. Browse the available pre-trained model library
3. Select a model and download it to your local machine
4. Use it for inference immediately, or load it as a starting checkpoint for fine-tuning

Model downloads are one-time: once downloaded, models work fully offline with no internet connection required.

---

### 12.4 Subscription Tiers

Access to the pre-trained model library will be available through the following tiers:

**Free tier**
- Access to the GeoSeg Studio plugin (same as today — always free and open source)
- Access to community-contributed open models
- One pre-trained model from the standard library per month

**Professional tier**
- Full access to all standard pre-trained models
- Model update notifications when improved versions are released
- Priority email support from the Dronnix team

**Enterprise tier**
- Full professional library access
- Custom model training: Dronnix trains a model on your data and delivers it ready to use
- API access for integration with external workflows and platforms
- Dedicated support

Pricing for Professional and Enterprise tiers will be announced before the v2.0 release. Handbook readers will be offered early-access pricing.

---

### 12.5 Technical Improvements in v2.0

Beyond the pre-trained model library, version 2.0 includes several technical improvements to the core pipeline:

**Multi-class segmentation**
Version 1.x performs binary segmentation: foreground vs. background. Version 2.0 will support multi-class segmentation, where a single model simultaneously detects multiple object classes (buildings, roads, and vegetation in a single pass, for example). This requires a redesigned labeling workflow and a different output representation, both of which will be covered in a v2.0 update to this handbook.

**Linux and macOS support**
The current release is Windows-only due to the complexity of managing PyTorch environments across platforms. Version 2.0 will extend support to Linux and macOS, significantly expanding the accessible user base.

**Batch prediction**
Version 1.x runs inference on one raster at a time. Version 2.0 will support batch prediction: processing a folder of rasters automatically with a single configuration, producing one output per input.

**Improved post-processing**
Additional post-processing options including instance separation (splitting merged objects that should be distinct) and confidence filtering (keeping only predictions above a configurable confidence threshold, not just the binary threshold).

---

### 12.6 How to Stay Updated

**GitHub**
Watch the GeoSeg Studio repository for release announcements:
`github.com/dronnix-io/GeoSegStudio`

**Dronnix website**
Version 2.0 release news and pre-trained model library details:
`dronnix.com`

**Email**
Handbook readers are automatically added to the Dronnix update list. To subscribe if you received this handbook from a third-party source, send a message to:
`salar@dronnix.com`

---

### 12.7 How to Contribute

GeoSeg Studio is open source under the GPL v2.0 license. Contributions are welcome:

**Code contributions**
Fork the repository, create a feature branch, and open a pull request against `main`. The most valuable contributions are:
- New model architectures
- Additional augmentation transforms
- Performance improvements to the data preparation pipeline
- Platform compatibility (Linux, macOS)

**Dataset contributions**
High-quality labeled datasets for underrepresented object classes — mining sites, coastal features, agricultural infrastructure, industrial facilities — are particularly valuable for training the pre-trained model library. Contact `salar@dronnix.com` to discuss dataset partnerships.

**Bug reports and feature requests**
Use the GitHub issue tracker:
`github.com/dronnix-io/GeoSegStudio/issues`

---

*"The best version of GeoSeg Studio is the one that solves your specific problem. If it does not do that yet, tell us why."*

— Salar Ghaffarian, Dronnix Inc.

---

*Next: Appendix — Parameter Reference, Troubleshooting, and Glossary*
