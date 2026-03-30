# Train Tab — Known Improvements & Future Work

Items identified during development that are non-blocking but should be
revisited once the Evaluate and Predict tabs are complete.

---

## 1. DataLoader Workers (High Priority)

**Current state:** Workers spinbox is hidden from the UI. `num_workers` is
hardcoded to `0` in `DL/dataset.py`. The underlying widget still exists in
`ui/tab2_hardware.py` but is not shown.

**Problem:** Inside QGIS, `sys.executable` points to `qgis.exe`, not
`python.exe`. Any value of `num_workers > 0` causes PyTorch's DataLoader to
spawn worker processes using `qgis.exe`, which opens new QGIS instances
instead of Python workers.

**Planned fix:** Run inference/training in a fully separate subprocess using
`env/Scripts/python.exe` explicitly, communicating progress back via a
pipe or temp file. This removes the `sys.executable` constraint entirely and
allows `num_workers > 0` for faster data loading on multi-core machines.

**Files to change:** `DL/trainer.py`, `DL/dataset.py`, `ui/tab2_hardware.py`

---

## 2. torch.load FutureWarning (Low Priority)

**Current state:** `torch.load(path, map_location=...)` is called without
`weights_only=True` in two places:
- `ui/tab2_checkpoints.py` — `_load_checkpoint_metadata()`
- `DL/trainer.py` — `_load_checkpoint()`

**Problem:** PyTorch 2.x emits a `FutureWarning` about unpickling arbitrary
objects. In a future PyTorch version the default will change to
`weights_only=True`, which will break loading of checkpoints that contain
non-tensor objects.

**Planned fix:** Add `weights_only=False` explicitly (to silence the warning
and be future-explicit), or restructure checkpoint payloads to contain only
tensors and primitives so `weights_only=True` works cleanly.

**Files to change:** `ui/tab2_checkpoints.py`, `DL/trainer.py`

---

## 3. Early Stopping (Medium Priority)

**Current state:** Training always runs for the full configured number of
epochs with no automatic stop condition.

**Planned addition:** Add an optional "Early Stopping" toggle in the Training
Configuration section with a patience parameter (e.g. stop if Val IoU does
not improve for N consecutive epochs). This prevents wasted compute and
reduces overfitting risk.

**Files to change:** `ui/tab2_training_config.py`, `DL/trainer.py`

---

## 4. Learning Rate Monitor (Low Priority)

**Current state:** The current learning rate is never shown to the user
during training. When a scheduler is active (StepLR, Cosine, ReduceLROnPlateau)
the LR changes during training but the user has no visibility into this.

**Planned addition:** Add a "Current LR" column to the metrics table, or
emit the current LR via a signal and display it in the phase label.

**Files to change:** `DL/trainer.py`, `ui/tab2_run_monitor.py`

---

## 5. Train IoU / F1 (Low Priority)

**Current state:** IoU and F1 are only computed on the validation set.
Train Loss is the only training-phase metric shown.

**Rationale for current design:** Computing metrics per batch during training
adds overhead and Train IoU is less informative than Val IoU for detecting
overfitting. The train/val loss gap is usually sufficient.

**If added later:** Emit train IoU/F1 per epoch and add two extra columns
to the metrics table. Guard with a checkbox ("Compute Train Metrics") so the
user opts in to the extra compute cost.

**Files to change:** `DL/trainer.py`, `ui/tab2_run_monitor.py`
