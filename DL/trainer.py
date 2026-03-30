"""
module: DL/trainer.py

Background training worker for Tab 2.

Runs the full train → validate loop inside a QThread so the QGIS UI
remains responsive throughout.  Progress is communicated to the UI
exclusively via Qt signals.

Signals
-------
phase_update(message)
    Emitted at each major step so the UI can show what is happening
    (e.g. "Building model...", "Loading dataset...", "Epoch 1/5 — Training").

epoch_done(epoch, train_loss, val_loss, val_iou, val_f1)
    Emitted once per epoch after both train and val phases complete.

batch_progress(current, total, phase)
    Emitted after every batch.  phase is "Train" or "Val".

training_finished(success, message)
    Emitted when the worker exits — whether normally, on error, or after
    a user-requested stop.
    NOTE: named 'training_finished' (not 'finished') to avoid shadowing
    QThread's own built-in finished() signal which has no parameters.

Config dict keys (assembled by Tab2Widget._build_config)
---------------------------------------------------------
dataset_dir     str      top-level *_dataset folder
aug_version     int      augmented version number

architecture    str      model key  (e.g. "unet")
in_channels     int      number of raster bands
img_size        int      tile size in pixels (must be in SUPPORTED_SIZES)
base_channels   int      base feature width for CNN models

loss            str      "bce" | "dice" | "bce_dice"
optimizer       str      "adam" | "adamw" | "sgd"
lr              float    initial learning rate
batch_size      int
epochs          int
scheduler       str|None "StepLR" | "CosineAnnealing" | "ReduceLROnPlateau" | None

output_dir      str      folder for saved .pth files
model_name      str      base filename without extension
save_strategy   str      "best" | "every_n"
every_n         int      (only when save_strategy == "every_n")
resume_path     str|None path to a .pth file to resume / fine-tune from

device          str      "cpu" | "cuda:0" | "cuda:1" | …
"""
from __future__ import annotations

import os

import torch

from qgis.PyQt.QtCore import QThread, pyqtSignal


class TrainingWorker(QThread):
    """
    QThread that owns the entire training loop.

    Instantiate, connect signals, then call start().
    Call stop() to request a clean early exit.
    """

    # Signals -----------------------------------------------------------------
    phase_update      = pyqtSignal(str)
    # Short human-readable message describing the current step.

    epoch_done        = pyqtSignal(int, float, float, float, float)
    # epoch (1-based relative), train_loss, val_loss, val_iou, val_f1

    batch_progress    = pyqtSignal(int, int, str)
    # batches_done, total_batches, phase ("Train" | "Val")

    training_finished = pyqtSignal(bool, str)
    # success, human-readable message
    # NOTE: named 'training_finished' (not 'finished') to avoid shadowing
    # QThread's own built-in finished() signal, which has no parameters.

    # -------------------------------------------------------------------------

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config    = config
        self._cancelled = False

    def stop(self):
        """Request cancellation.  The loop checks this flag after each batch."""
        self._cancelled = True

    # -------------------------------------------------------------------------
    # QThread entry point
    # -------------------------------------------------------------------------

    def run(self):
        try:
            self._train()
        except Exception as exc:
            self.training_finished.emit(False, str(exc))

    # -------------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------------

    def _train(self):
        cfg    = self._config
        device = torch.device(cfg["device"])

        # --- Model -----------------------------------------------------------
        self.phase_update.emit("Building model...")
        from .architectures import build_model
        model = build_model(
            cfg["architecture"],
            cfg["in_channels"],
            cfg["img_size"],
        ).to(device)

        # --- Loss ------------------------------------------------------------
        from .losses import build_loss
        criterion = build_loss(cfg["loss"])

        # --- Optimizer -------------------------------------------------------
        optimizer = self._build_optimizer(model, cfg)

        # --- LR Scheduler ----------------------------------------------------
        scheduler = self._build_scheduler(optimizer, cfg)

        # --- DataLoaders -----------------------------------------------------
        self.phase_update.emit("Loading dataset...")
        from .dataset import build_dataloaders
        train_loader, val_loader = build_dataloaders({
            "dataset_dir": cfg["dataset_dir"],
            "aug_version": cfg["aug_version"],
            "batch_size":  cfg["batch_size"],
        })

        # --- Resume from checkpoint ------------------------------------------
        start_epoch = 1
        best_iou    = -1.0

        if cfg.get("resume_path"):
            self.phase_update.emit("Resuming from checkpoint...")
            start_epoch, best_iou = self._load_checkpoint(
                cfg["resume_path"], model, optimizer, device
            )

        # --- Epoch loop ------------------------------------------------------
        total_epochs = cfg["epochs"]

        for epoch in range(start_epoch, start_epoch + total_epochs):
            if self._cancelled:
                self.training_finished.emit(False, "Training stopped by user.")
                return

            rel_epoch = epoch - start_epoch + 1  # 1-based for display

            # Train phase
            self.phase_update.emit(f"Epoch {rel_epoch}/{total_epochs} — Training...")
            train_loss = self._run_phase(
                model, train_loader, criterion, optimizer, device,
                phase="Train", train=True,
            )

            if self._cancelled:
                self.training_finished.emit(False, "Training stopped by user.")
                return

            # Validation phase
            self.phase_update.emit(f"Epoch {rel_epoch}/{total_epochs} — Validating...")
            val_loss, val_iou, val_f1 = self._run_phase(
                model, val_loader, criterion, None, device,
                phase="Val", train=False,
            )

            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Emit epoch metrics to UI
            self.epoch_done.emit(rel_epoch, train_loss, val_loss, val_iou, val_f1)

            # Checkpoint saving
            is_best = val_iou > best_iou
            if is_best:
                best_iou = val_iou

            if cfg.get("output_dir"):
                self.phase_update.emit(f"Epoch {rel_epoch}/{total_epochs} — Saving checkpoint...")
                self._maybe_save(
                    model, optimizer, cfg, epoch, rel_epoch, val_iou, is_best
                )

        self.training_finished.emit(
            True, f"Training complete.  Best Val IoU: {best_iou:.4f}"
        )

    # -------------------------------------------------------------------------
    # Single phase (train or val)
    # -------------------------------------------------------------------------

    def _run_phase(
        self,
        model,
        loader,
        criterion,
        optimizer,        # None during val
        device,
        phase: str,
        train: bool,
    ):
        """
        Runs one epoch phase.

        Returns
        -------
        train phase : float                      (avg loss)
        val   phase : (float, float, float)      (avg loss, avg iou, avg f1)
        """
        n_batches  = len(loader)
        total_loss = 0.0
        total_iou  = 0.0
        total_f1   = 0.0

        if train:
            model.train()
            for batch_idx, (images, masks) in enumerate(loader):
                if self._cancelled:
                    break

                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(images)

                # UNet++ with deep supervision returns a list — use last output
                if isinstance(logits, (list, tuple)):
                    logits = logits[-1]

                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                self.batch_progress.emit(batch_idx + 1, n_batches, phase)
        else:
            model.eval()
            with torch.no_grad():
                for batch_idx, (images, masks) in enumerate(loader):
                    if self._cancelled:
                        break

                    images = images.to(device, non_blocking=True)
                    masks  = masks.to(device, non_blocking=True)

                    logits = model(images)
                    if isinstance(logits, (list, tuple)):
                        logits = logits[-1]

                    loss = criterion(logits, masks)
                    total_loss += loss.item()

                    iou, f1 = _compute_metrics(logits, masks)
                    total_iou += iou
                    total_f1  += f1

                    self.batch_progress.emit(batch_idx + 1, n_batches, phase)

        n = max(n_batches, 1)
        if train:
            return total_loss / n
        else:
            return total_loss / n, total_iou / n, total_f1 / n

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_optimizer(model, cfg):
        lr = cfg["lr"]
        name = cfg["optimizer"]
        if name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr)
        if name == "sgd":
            return torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
            )
        raise ValueError(f"Unknown optimizer: {name!r}")

    @staticmethod
    def _build_scheduler(optimizer, cfg):
        name = cfg.get("scheduler")
        if name is None:
            return None
        if name == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5
            )
        if name == "CosineAnnealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["epochs"]
            )
        if name == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5
            )
        return None

    @staticmethod
    def _load_checkpoint(path: str, model, optimizer, device):
        """Loads state from a .pth checkpoint. Returns (start_epoch, best_iou)."""
        data = torch.load(path, map_location=device)
        model.load_state_dict(data["model_state_dict"])
        if "optimizer_state_dict" in data:
            try:
                optimizer.load_state_dict(data["optimizer_state_dict"])
            except Exception:
                pass  # ignore if optimizer shape changed (fine-tune scenario)
        start_epoch = data.get("epoch", 0) + 1
        best_iou    = data.get("val_iou", -1.0) or -1.0
        return start_epoch, best_iou

    @staticmethod
    def _maybe_save(model, optimizer, cfg, epoch, rel_epoch, val_iou, is_best):
        """Saves a checkpoint according to the configured strategy."""
        output_dir    = cfg.get("output_dir", "").strip()
        model_name    = cfg.get("model_name", "model").strip() or "model"
        save_strategy = cfg.get("save_strategy", "best")

        if not output_dir:
            return

        os.makedirs(output_dir, exist_ok=True)

        payload = {
            "architecture":         cfg["architecture"],
            "epoch":                epoch,
            "val_iou":              val_iou,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                k: cfg[k]
                for k in (
                    "architecture", "in_channels", "img_size",
                    "base_channels", "loss", "optimizer", "lr",
                )
                if k in cfg
            },
        }

        if save_strategy == "best" and is_best:
            torch.save(payload, os.path.join(output_dir, f"{model_name}.pth"))

        elif save_strategy == "every_n":
            every_n = cfg.get("every_n", 10)
            if rel_epoch % every_n == 0:
                fname = f"{model_name}_ep{epoch:04d}.pth"
                torch.save(payload, os.path.join(output_dir, fname))


# ---------------------------------------------------------------------------
# Metric helpers (module-level, no self needed)
# ---------------------------------------------------------------------------

def _compute_metrics(logits: torch.Tensor, targets: torch.Tensor):
    """
    Batch-averaged IoU and F1 at sigmoid threshold 0.5.

    Returns
    -------
    iou, f1 : float, float
    """
    with torch.no_grad():
        preds   = (torch.sigmoid(logits) > 0.5).float().view(-1)
        targets = targets.view(-1)

        tp = (preds * targets).sum().item()
        fp = (preds * (1 - targets)).sum().item()
        fn = ((1 - preds) * targets).sum().item()

        iou = tp / (tp + fp + fn + 1e-7)
        f1  = 2 * tp / (2 * tp + fp + fn + 1e-7)

    return iou, f1
