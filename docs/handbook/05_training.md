# Chapter 5 — Training That Actually Converges

---

### 5.1 What Happens During Training

Training a neural network is an iterative optimization process. At each step, the model makes a prediction on a batch of training tiles, compares that prediction to the ground truth mask using a loss function, and adjusts its internal parameters slightly in the direction that reduces the loss. This adjustment is called **backpropagation**.

One complete pass through the entire training dataset is called an **epoch**. After each epoch, the model's performance is measured on the validation set — tiles it has never learned from. The validation metrics tell you whether the model is genuinely improving or just memorizing the training data.

GeoSeg Studio's Train tab handles this entire process. Your job is to configure the hyperparameters correctly before clicking Start, and to interpret the training curves during the run.

---

### 5.2 Loss Functions

The loss function measures the difference between the model's prediction and the ground truth. It is the signal the model uses to improve. Choosing the right loss function for your data matters more than most other hyperparameter choices.

---

**Binary Cross-Entropy (BCE)**

BCE treats each pixel independently and measures the prediction error as a log-probability. It works well when your foreground and background classes are roughly balanced — when foreground pixels make up a significant fraction of your training tiles.

When to use: your objects are large relative to the tile size (buildings, large vegetation areas, fields). Foreground pixels are not rare.

When to avoid: your objects are small or sparse relative to the tile size. With sparse foreground, BCE can lead the model to predict mostly background (which achieves low loss trivially) without learning to detect the target.

---

**Dice Loss**

Dice loss measures the overlap between the predicted mask and the ground truth mask as a fraction of the total predicted and ground truth areas. It is inherently balanced — it penalizes the model equally for missing foreground pixels and for incorrectly labeling background as foreground, regardless of class frequency.

When to use: your objects are small, sparse, or thin (vehicles in large parking lots, narrow roads, individual solar panels). Dice loss keeps the model from collapsing to a "predict all background" solution.

When to avoid: datasets where nearly all tiles contain a large amount of foreground. Dice loss can be numerically unstable at very high foreground fractions.

---

**BCE + Dice (Combined)**

Combines both losses into a single objective, typically as a simple average. This is the most robust choice for most use cases because it captures the pixel-wise accuracy of BCE and the overlap-based balance of Dice simultaneously.

When to use: your default choice when you are unsure. Works well across a wide range of object sizes and class frequencies.

**Recommendation:** Start with BCE + Dice for any new dataset. Switch to Dice alone if your objects are very small and sparse, or BCE alone if training is unstable with the combined loss.

---

### 5.3 Optimizers

The optimizer determines how the model's parameters are updated at each training step.

---

**Adam**

Adam (Adaptive Moment Estimation) maintains a separate adaptive learning rate for each parameter. It converges quickly and is robust to learning rate choice, making it the default for most deep learning tasks.

When to use: your default choice. Works well in almost all cases.

---

**AdamW**

AdamW is a variant of Adam with **weight decay** applied correctly (decoupled from the gradient update). This regularizes the model — it prevents parameters from growing too large — and often improves generalization on smaller datasets.

When to use: if your model overfits (training metrics improve but validation metrics plateau or worsen), try AdamW over Adam. For small to medium datasets (under 1000 training tiles), AdamW is often the better choice.

---

**SGD**

SGD (Stochastic Gradient Descent) with momentum is the classic optimizer. It is less adaptive than Adam, meaning it is more sensitive to learning rate choice and takes longer to converge. However, it sometimes finds better final solutions on large datasets when tuned carefully.

When to use: large datasets (5000+ tiles), or when you want to replicate specific results from a published paper that used SGD. Not recommended as a first choice.

---

### 5.4 Learning Rate

The learning rate controls how large each parameter update step is. Too high, and training oscillates or diverges. Too low, and training converges very slowly or gets stuck.

**Recommended starting values:**

| Optimizer | Starting learning rate |
|-----------|----------------------|
| Adam | 1e-3 (0.001) |
| AdamW | 1e-3 or 1e-4 |
| SGD | 1e-2 (0.01) |

These are starting points, not rules. If your training loss oscillates wildly in the first few epochs, halve the learning rate. If loss decreases extremely slowly, try doubling it.

---

### 5.5 Learning Rate Schedulers

A scheduler adjusts the learning rate during training according to a predefined schedule. Starting with a higher learning rate and reducing it as training progresses almost always improves final performance.

---

**StepLR**

Reduces the learning rate by a fixed factor (e.g. 0.1) every N epochs.

When to use: when you have a good sense of how many epochs you need. A common setting: reduce by 0.1 at epoch 50% and 75% of your total epoch count.

---

**Cosine Annealing**

Smoothly reduces the learning rate following a cosine curve, from the initial value down to near zero by the end of training.

When to use: your default scheduler choice. It requires no tuning and works well across a wide range of dataset sizes and architectures.

---

**ReduceLROnPlateau**

Monitors the validation loss and reduces the learning rate by a factor when the loss has not improved for a specified number of epochs (patience).

When to use: when you do not know in advance how many epochs are needed, or when your training loss decreases but validation loss plateaus unexpectedly. More adaptive than the other schedulers, but adds two parameters to tune (factor and patience).

---

### 5.6 Batch Size

The batch size determines how many training tiles are processed in a single forward-backward pass before updating the model parameters.

**Effect of batch size:**
- Larger batches: smoother gradient estimates, faster training per epoch (on GPU), but higher memory requirements
- Smaller batches: noisier gradients (which can actually act as regularization), lower memory requirements

**Practical guidelines:**

| Hardware | Recommended batch size (256px tiles) |
|----------|--------------------------------------|
| CPU | 4–8 |
| GPU, 4 GB VRAM | 8–16 |
| GPU, 8 GB VRAM | 16–32 |
| GPU, 16+ GB VRAM | 32–64 |

If you receive an out-of-memory error, halve the batch size. If training is very slow on GPU, try increasing the batch size.

---

### 5.7 Number of Epochs

An epoch is one complete pass through the training dataset. The number of epochs you need depends on your dataset size, architecture, and learning rate.

**Practical starting points:**

| Dataset size (training tiles) | Starting epoch count |
|------------------------------|---------------------|
| < 500 | 50–100 |
| 500–2000 | 30–60 |
| 2000–5000 | 20–40 |
| > 5000 | 15–30 |

These are starting points. Watch the validation loss curve (covered in Section 5.9) to determine when to stop. Running extra epochs beyond convergence wastes time but does not usually harm performance if you are saving the best checkpoint.

---

### 5.8 Device Selection

GeoSeg Studio automatically detects available hardware. In the device selector, you will see:

- **CPU** — always available. Significantly slower for training than GPU. For a 256px patch dataset with 1000 training tiles, expect 2–5 minutes per epoch on a modern CPU.
- **CUDA — [GPU name]** — available if you installed a CUDA-enabled PyTorch version and your GPU is correctly detected. Expect 10–30x speedup over CPU for training.

Training on CPU is viable for small datasets and short experiments. For anything beyond a quick test, a GPU reduces training time from hours to minutes.

**Realistic training time estimates (256px tiles, UNet, batch size 16):**

| Dataset size | CPU (estimate) | Mid-range GPU (RTX 3060) |
|-------------|---------------|--------------------------|
| 500 tiles, 50 epochs | 2–4 hours | 8–15 minutes |
| 2000 tiles, 40 epochs | 8–16 hours | 30–60 minutes |
| 5000 tiles, 30 epochs | 20–40 hours | 90–180 minutes |

---

### 5.9 Reading the Training Curves

GeoSeg Studio displays live loss curves during training: a separate line for training loss and validation loss, updated after each epoch.

**What good training looks like:**
- Both training loss and validation loss decrease over the first several epochs
- Validation loss eventually flattens and stops improving
- Training loss continues to decrease slightly after validation loss flattens (small amount of overfitting is normal)
- The gap between training and validation loss is small and stable

**Warning signs:**

| Pattern | What it means | What to do |
|---------|--------------|-----------|
| Training loss oscillates or increases | Learning rate too high | Reduce learning rate by 5-10x |
| Both losses decrease very slowly | Learning rate too low, or batch size too small | Increase learning rate |
| Training loss decreases, validation loss increases | Overfitting | Reduce model size (fewer base channels), switch to AdamW, add augmentation |
| Both losses plateau at a high value | Model not converging | Check data quality, try different architecture or loss function |
| Validation loss lower than training loss | Very small dataset with heavy augmentation | Normal in this case, not a problem |

---

### 5.10 Checkpointing

GeoSeg Studio saves model checkpoints during training. Two strategies are available:

**Best model only** — saves the checkpoint from the epoch with the best validation loss. This is the default and is appropriate for most use cases. At the end of training, you have exactly one checkpoint: the best one.

**Every N epochs** — saves a checkpoint every N epochs regardless of performance. Useful if you want to compare models at different stages of training, or if you are doing very long training runs and want recovery points.

Checkpoints are saved to the output directory you specified, under a `models/` subfolder with a timestamped name that includes the architecture, dataset version, and training parameters.

---

### 5.11 Resuming and Fine-Tuning

**Resuming training** — if training was interrupted (power outage, system crash, you stopped it manually), you can resume from the last saved checkpoint by selecting it in the checkpoint selector and clicking Resume. Training continues from the epoch where it stopped.

**Fine-tuning from a checkpoint** — if you have a model trained on one dataset and want to adapt it to a new dataset, load the existing checkpoint and start training on the new dataset. The model starts from learned weights rather than random initialization, which typically leads to faster convergence and better final performance — especially when the new dataset is small.

Fine-tuning is particularly valuable in the GeoSeg Studio context: once GeoSeg Studio v2.0 introduces pre-trained models for common classes (buildings, vehicles, solar panels), you will be able to load those weights and fine-tune on your own local imagery with much less training data than training from scratch.

---

### 5.12 Recommended Starting Configuration

If this is your first training run on a new dataset, use this configuration as a starting point:

| Parameter | Value |
|-----------|-------|
| Architecture | UNet |
| Base channels | 32 |
| Loss function | BCE + Dice |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| LR scheduler | Cosine Annealing |
| Batch size | 16 (adjust for your GPU) |
| Epochs | 50 |
| Checkpointing | Best model only |

Train with these settings, examine the loss curves and validation metrics, then make targeted adjustments based on what you observe. Changing one parameter at a time makes it much easier to understand what is actually improving performance.

---

*Next: Chapter 6 — Evaluating Your Model Honestly*
