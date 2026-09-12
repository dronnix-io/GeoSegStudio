# Chapter 4 — Choosing and Configuring Your Model

---

### 4.1 What a Segmentation Model Actually Is

A semantic segmentation model is a neural network that takes an image as input and produces a mask as output — an image of the same spatial dimensions where each pixel has a predicted class value. For binary segmentation, every output pixel is a number between 0 and 1: the model's confidence that the pixel belongs to the foreground class.

The model is organized as an **encoder-decoder** pair:

- The **encoder** progressively reduces the spatial dimensions of the input while increasing the number of feature channels. It learns to recognize what is in the image: edges, textures, shapes, high-level structures.
- The **decoder** progressively restores the spatial dimensions back to the original input size while using the features learned by the encoder. It learns to localize where those structures are, at pixel level.

Different architectures connect the encoder and decoder in different ways, which affects what kinds of patterns they learn best and how computationally expensive they are.

GeoSeg Studio includes seven architectures. You do not need to understand every architectural detail to use them well — but understanding the key differences will help you make better choices for your specific use case.

---

### 4.2 The Seven Architectures

---

**UNet**

UNet is the original and most widely used architecture for medical and remote sensing image segmentation. Its defining feature is **skip connections**: direct connections from each encoder level to the corresponding decoder level, so the decoder has access to both high-level semantic features and low-level spatial detail simultaneously.

- Best for: general-purpose use, medium-sized objects, datasets of any size
- Training speed: fast
- Memory usage: moderate
- When to choose it: your first experiment on any new dataset. If UNet performs well, there is little reason to switch.

---

**Attention UNet**

Attention UNet adds **attention gates** to the skip connections of standard UNet. These gates learn to suppress irrelevant regions and focus the decoder's attention on the parts of the feature map that are most likely to contain the target object.

- Best for: cluttered scenes where the background is visually similar to the foreground; datasets where false positives are a problem
- Training speed: slightly slower than UNet
- Memory usage: slightly higher than UNet
- When to choose it: if standard UNet produces too many false positives in complex background regions

---

**UNet++**

UNet++ replaces the simple skip connections of UNet with **dense nested connections**: intermediate nodes between the encoder and decoder that aggregate features at multiple scales. This allows the decoder to receive richer multi-scale representations.

- Best for: fine-grained boundaries, small objects embedded in complex backgrounds
- Training speed: slower than UNet (more parameters)
- Memory usage: higher than UNet
- When to choose it: when boundary accuracy matters more than training speed; when objects have complex or irregular shapes

---

**LinkNet**

LinkNet is a lightweight encoder-decoder architecture that uses **residual connections** similar to ResNet. It achieves competitive performance with significantly fewer parameters than UNet, making it faster to train and requiring less memory.

- Best for: large datasets, real-time or near-real-time inference requirements, limited GPU memory
- Training speed: fast (the fastest of the seven)
- Memory usage: low
- When to choose it: when you need fast training or are working with a GPU with limited VRAM (4–6 GB)

---

**DeepLabV3+**

DeepLabV3+ uses **dilated (atrous) convolutions** and an **Atrous Spatial Pyramid Pooling (ASPP)** module to capture context at multiple scales without reducing spatial resolution. The encoder operates at multiple dilation rates simultaneously, giving the model a wide field of view while preserving detail.

- Best for: objects with significant scale variation within the same image; scenes with complex spatial context
- Training speed: moderate
- Memory usage: moderate to high
- When to choose it: when your target objects vary significantly in size across the dataset, or when spatial context (what surrounds the object) is important for correct classification

---

**SwinUNet**

SwinUNet replaces the convolutional encoder with a **Swin Transformer** — a pure transformer architecture that processes the image as a sequence of non-overlapping patches and models long-range spatial dependencies through self-attention mechanisms.

- Best for: large datasets where global context matters; imagery with repetitive local patterns where context helps disambiguation
- Training speed: slow (transformers require more data and more epochs to converge)
- Memory usage: high
- When to choose it: when you have a large dataset (2000+ training tiles) and a GPU with sufficient VRAM; not recommended for small datasets

---

**SegFormer**

SegFormer uses a **Mix Transformer (MiT)** encoder — a hierarchical transformer that operates at multiple scales — combined with a lightweight MLP decoder. It is more efficient than SwinUNet while retaining the long-range context modeling of transformer architectures.

- Best for: large datasets, multi-scale objects, scenes where both local texture and global structure matter
- Training speed: moderate (faster than SwinUNet)
- Memory usage: moderate to high
- When to choose it: a strong alternative to SwinUNet when you want transformer-based performance without the full training cost

---

### 4.3 Architecture Decision Table

| Situation | Recommended architecture |
|-----------|-------------------------|
| First experiment on a new dataset | UNet |
| Too many false positives in training | Attention UNet |
| Need precise boundaries | UNet++ |
| Limited GPU memory (< 6 GB VRAM) | LinkNet |
| Objects vary a lot in size | DeepLabV3+ |
| Large dataset, global context matters | SegFormer |
| Large dataset, maximum accuracy target | SwinUNet |

In practice, UNet is the right starting point for nearly every new use case. Switch to a more specialized architecture only when you have a specific problem that UNet does not handle well.

---

### 4.4 Input Channels

The **input channels** setting tells the model how many bands your training tiles have. This must match the band count of the rasters you clipped during data preparation.

| Imagery type | Typical band count |
|-------------|-------------------|
| RGB | 3 |
| RGB + Alpha | 4 (or 3 if you excluded the alpha band during clipping) |
| RGBI (RGB + Near-infrared) | 4 |
| Multispectral (e.g. MicaSense RedEdge) | 5 |
| Thermal (single band) | 1 |
| Thermal + RGB | 4 |
| Hyperspectral | 6–40 |

**Important:** The input channels value must match the band count of your training tiles exactly. If there is a mismatch, the model will fail to initialize and training will not start. GeoSeg Studio reads the band count from your training tiles automatically and warns you if the setting does not match — but it is good practice to verify this before starting a long training run.

---

### 4.5 Base Channel Width

The **base channel width** (sometimes called the number of filters) controls how many feature channels the model uses at its first encoder level. Subsequent levels double this number at each downsampling step.

| Base channels | Effect |
|--------------|--------|
| 16 | Smallest model — fast training, lower capacity. Good for simple objects or limited hardware. |
| 32 | Default — good balance of capacity and speed for most use cases |
| 64 | Larger model — more capacity for complex scenes, slower training, higher memory usage |

For most binary segmentation tasks on drone imagery, **32** is the right choice. Increase to 64 if your validation metrics plateau early and you have GPU memory to spare. Decrease to 16 if you are running on CPU or a low-memory GPU.

---

### 4.6 Patch Size Consistency

The patch size used during training must match the patch size used during data preparation. GeoSeg Studio enforces this automatically — the training configuration reads the patch size from the dataset metadata.

However, there is one important flexibility: **inference patch size can differ from training patch size**. During prediction, the sliding window can use a larger or smaller patch than the one used for training, within limits. This is discussed in Chapter 7.

---

### 4.7 A Note on Model Complexity

It can be tempting to choose the most sophisticated architecture available, on the assumption that more complexity equals better results. In practice, for binary segmentation on drone imagery with datasets of a few hundred to a few thousand tiles, the performance differences between architectures are often smaller than the impact of dataset quality and training configuration.

A well-labeled dataset with 500 tiles trained on UNet will typically outperform a poorly labeled dataset with 2000 tiles trained on SwinUNet.

Invest in your data first. Experiment with architectures once you have a solid baseline.

---

*Next: Chapter 5 — Training That Actually Converges*
