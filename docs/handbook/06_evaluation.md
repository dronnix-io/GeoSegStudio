# Chapter 6 — Evaluating Your Model Honestly

---

### 6.1 Why Evaluation Is Not an Afterthought

A model that performs well on training data but poorly on new imagery is useless in practice. Evaluation is the step where you find out whether your model has actually learned to generalize — or whether it has memorized patterns that only exist in your training tiles.

The Evaluate tab in GeoSeg Studio measures your model's performance on a chosen data split (train, validation, or test) and reports a set of standard metrics. Understanding what these metrics mean, and what they do not mean, is essential for making good decisions about your model.

This chapter also covers something most documentation skips: how to diagnose where and why your model fails, which is often more actionable than knowing that it fails.

---

### 6.2 The Confusion Matrix: The Foundation of Everything

Before looking at any metric, it helps to understand the confusion matrix. Every pixel in every prediction falls into one of four categories:

| | Predicted: Foreground | Predicted: Background |
|--|----------------------|-----------------------|
| **Actual: Foreground** | True Positive (TP) | False Negative (FN) |
| **Actual: Background** | False Positive (FP) | True Negative (TN) |

- **True Positive (TP):** The model correctly identified a foreground pixel.
- **True Negative (TN):** The model correctly identified a background pixel.
- **False Positive (FP):** The model said "foreground" but the pixel is actually background. Also called a commission error or false alarm.
- **False Negative (FN):** The model said "background" but the pixel is actually foreground. Also called an omission error or missed detection.

Every metric reported by GeoSeg Studio is derived from these four numbers. Understanding the confusion matrix makes every other metric intuitive.

---

### 6.3 The Metrics

---

**Pixel Accuracy**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

The fraction of all pixels that the model classified correctly.

**The problem with accuracy:** If your foreground class is rare — for example, vehicles occupy 5% of pixels in a parking lot image — a model that predicts "background" for every single pixel achieves 95% accuracy. This metric is almost meaningless for imbalanced datasets.

Use pixel accuracy as a sanity check, not as your primary evaluation metric.

---

**Precision**

```
Precision = TP / (TP + FP)
```

Of all the pixels the model labeled as foreground, what fraction actually were foreground?

High precision means the model rarely cries wolf — when it says "foreground," it is usually right. Low precision means many false alarms.

---

**Recall**

```
Recall = TP / (TP + FN)
```

Of all the actual foreground pixels in the image, what fraction did the model find?

High recall means the model finds most of the objects. Low recall means the model misses many objects.

**The precision-recall tradeoff:** These two metrics are inversely related through the sigmoid threshold. Lowering the threshold (predicting foreground more aggressively) increases recall but decreases precision. Raising the threshold decreases recall but increases precision. The right balance depends on your use case:

- Solar panel inspection: high recall matters more. Missing a damaged panel is worse than a false alarm.
- Counting vehicles for billing: high precision matters more. Overcounting is a billing error.

---

**F1 Score (Dice Coefficient)**

```
F1 = 2 × TP / (2 × TP + FP + FN)
```

The harmonic mean of precision and recall. F1 gives you a single number that balances both. It is the most commonly reported metric for binary segmentation in research literature.

An F1 of 1.0 is perfect. An F1 of 0.0 is worst possible. In practice:

| F1 range | Interpretation |
|----------|---------------|
| > 0.90 | Excellent — suitable for most operational use cases |
| 0.80–0.90 | Good — minor errors, review post-processing settings |
| 0.70–0.80 | Moderate — usable with manual review, consider more training data |
| < 0.70 | Poor — dataset or training configuration needs attention |

These thresholds are guidelines, not rules. What counts as "acceptable" depends entirely on your use case.

---

**IoU (Intersection over Union)**

```
IoU = TP / (TP + FP + FN)
```

Also called the Jaccard Index. IoU measures the overlap between the predicted mask and the ground truth mask as a fraction of their union.

IoU is generally more conservative than F1 — the same model will have a lower IoU than F1 score. IoU is the standard metric in most computer vision benchmarks and is what most published papers report. Use IoU when comparing your results to published literature.

The relationship between F1 and IoU:
```
IoU = F1 / (2 - F1)
```

An F1 of 0.85 corresponds to an IoU of approximately 0.74.

---

### 6.4 Which Metric to Use as Your Primary Metric

For most remote sensing segmentation tasks, use **IoU** as your primary metric when comparing models or reporting results. It is the most widely reported metric in the literature and the hardest to inflate.

For use cases where false positives and false negatives have different costs (inspection, counting, monitoring), use **Precision and Recall separately** to understand the model's error profile, then tune the sigmoid threshold to hit your target operating point.

---

### 6.5 The Sigmoid Threshold

GeoSeg Studio's model outputs a probability map: each pixel has a value between 0 (definitely background) and 1 (definitely foreground). The **sigmoid threshold** is the cutoff that converts this probability map into a binary mask.

The default threshold is 0.5: pixels with a predicted probability above 0.5 are labeled foreground, below 0.5 are labeled background.

Adjusting this threshold is one of the most powerful tools for tuning your model's behavior without retraining:

- **Lower threshold (e.g. 0.3):** More foreground predictions. Higher recall, lower precision. Use when missing objects is costly.
- **Higher threshold (e.g. 0.7):** Fewer foreground predictions. Higher precision, lower recall. Use when false alarms are costly.

The Evaluate tab reports metrics at the default threshold (0.5). When running final predictions in the Predict tab, you can adjust the threshold to match your operational requirements.

---

### 6.6 The Per-Tile CSV

After evaluation, GeoSeg Studio generates a CSV file with per-tile metrics: IoU, F1, Precision, Recall, and Pixel Accuracy for every tile in the evaluated split.

This file is more useful than the aggregate metrics for diagnosing model failures. Sort by IoU ascending and look at the worst-performing tiles:

- Do the worst tiles share a geographic location? (The model may have been under-represented in that area during training.)
- Do they share a visual characteristic? (Shadow, unusual lighting, seasonal variation.)
- Do they contain unusual object instances? (Damaged buildings, partially occluded vehicles.)

These patterns tell you what to add to your training data for the next iteration.

---

### 6.7 Visualization Tiles

The Evaluate tab also saves a set of sample visualization tiles — side-by-side images showing the original imagery, the ground truth mask, and the model's prediction for the same patch.

Examining these visually is irreplaceable. Numbers can tell you that your IoU is 0.78, but only looking at the actual predictions tells you whether the errors are:

- **Shape errors:** The model finds the right objects but gets the boundaries wrong
- **Commission errors:** The model detects foreground where there is none (false alarms)
- **Omission errors:** The model misses entire objects (missed detections)
- **Fragmentation:** The model partially detects objects, finding some parts but not others

Each of these patterns points to a different fix. Shape errors suggest post-processing adjustments. Commission errors suggest raising the threshold or using Attention UNet. Omission errors suggest more diverse training data or a lower threshold. Fragmentation often indicates that the objects are at the edge of what the current resolution and patch size can resolve.

---

### 6.8 Evaluating on Train vs. Validation vs. Test

**Train split evaluation** tells you how well the model fit the training data. Very high train metrics with lower validation metrics indicates overfitting.

**Validation split evaluation** tells you how well the model generalizes to unseen data drawn from the same distribution as the training data. This is what you monitor during training.

**Test split evaluation** gives you an unbiased estimate of real-world performance. You should evaluate on the test split only once — after all model selection and hyperparameter tuning is complete. If you evaluate on the test set repeatedly and use the results to guide decisions, the test set becomes effectively a second validation set and its metrics are no longer unbiased.

For research publications: always report test set metrics, and describe your split strategy precisely.

---

### 6.9 When Your Metrics Look Good But Your Polygons Look Wrong

It is possible to have an F1 of 0.85 on the tile-level evaluation and still find that the polygon outputs look messy — fragmented, over-merged, or geometrically rough.

This is because tile-level metrics measure pixel-level accuracy, while the quality of vector polygons depends on post-processing settings as well. The fix is usually in the Predict tab's post-processing options, not in the model itself.

Common mismatches between good metrics and poor polygon quality:

| Problem in polygons | Likely cause | Post-processing fix |
|--------------------|--------------|--------------------|
| Fragmented polygons (many small pieces where there should be one) | Prediction gaps between adjacent tiles | Increase merge tolerance |
| Holes inside predicted polygons | Local prediction uncertainty | Enable hole filling with appropriate area threshold |
| Jagged, pixelated boundaries | High-frequency boundary noise | Enable smoothing (Chaikin) |
| Many tiny polygon fragments | Sparse noise predictions | Set minimum area filter |
| Over-merged (adjacent objects merged into one) | Merge tolerance too aggressive | Reduce merge tolerance |

Post-processing is covered in detail in Chapter 7.

---

### 6.10 A Practical Evaluation Workflow

1. Train your first model with the recommended starting configuration (Chapter 5)
2. Evaluate on the validation split. Record your IoU and F1.
3. Examine the visualization tiles. Identify the dominant failure mode.
4. Make one targeted change (more data, different architecture, threshold adjustment)
5. Retrain and re-evaluate on the validation split
6. Repeat until validation performance is acceptable
7. Evaluate on the test split exactly once. This is your final reported performance.

Resist the temptation to make multiple simultaneous changes between training runs. One change at a time is slower but gives you clear insight into what actually improves performance.

---

*Next: Chapter 7 — From Prediction to GIS Product*
