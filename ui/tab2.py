"""
module: tab2.py

Train tab — orchestrates all training-related sections and wires the
background TrainingWorker to the Run & Monitor widget.
"""
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QSizePolicy, QMessageBox,
)

from .tab2_dataset         import DatasetWidget
from .tab2_model           import ModelWidget
from .tab2_training_config import TrainingConfigWidget
from .tab2_checkpoints     import CheckpointsWidget
from .tab2_hardware        import HardwareWidget
from .tab2_run_monitor     import RunMonitorWidget
from .tab2_plots           import TrainingPlotWidget


class Tab2Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.dataset         = DatasetWidget()
        self.model           = ModelWidget()
        self.training_config = TrainingConfigWidget()
        self.checkpoints     = CheckpointsWidget()
        self.hardware        = HardwareWidget()
        self.run_monitor     = RunMonitorWidget()
        self.plots           = TrainingPlotWidget()

        self._worker = None  # TrainingWorker instance while training

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)

        content_layout.addWidget(self.dataset)
        content_layout.addWidget(self.model)
        content_layout.addWidget(self.training_config)
        content_layout.addWidget(self.checkpoints)
        content_layout.addWidget(self.hardware)
        content_layout.addWidget(self.run_monitor)
        content_layout.addWidget(self.plots)
        content_layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # --- Connections -----------------------------------------------------
        self.run_monitor.start_btn.clicked.connect(self._start_training)
        self.run_monitor.stop_btn.clicked.connect(self._stop_training)

    # -------------------------------------------------------------------------
    # Training control
    # -------------------------------------------------------------------------

    def _start_training(self):
        config, error = self._build_config()
        if error:
            QMessageBox.warning(self, "Cannot Start Training", error)
            return

        from ..DL.trainer import TrainingWorker

        self._worker = TrainingWorker(config, parent=self)
        self._worker.phase_update.connect(self.run_monitor.update_phase)
        self._worker.epoch_done.connect(self.run_monitor.add_epoch_row)
        self._worker.epoch_done.connect(self.plots.add_epoch)
        self._worker.batch_progress.connect(self.run_monitor.update_batch_progress)
        self._worker.training_finished.connect(self._on_training_finished)

        self.run_monitor.reset_monitor()
        self.run_monitor.set_output_paths(config["output_dir"], config["model_name"])
        self.plots.reset()
        self.run_monitor.set_running(True, config["epochs"])
        self._worker.start()

    def _stop_training(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()

    def _on_training_finished(self, success: bool, message: str):
        self.run_monitor.set_running(False)
        self.run_monitor.set_status(message, error=not success)
        self._worker = None

    # -------------------------------------------------------------------------
    # Config assembly & validation
    # -------------------------------------------------------------------------

    def _build_config(self) -> tuple:
        """
        Collects settings from all section widgets into one config dict.

        Returns
        -------
        (config, None)   on success
        (None, message)  when validation fails
        """
        # Dataset
        dataset_dir  = self.dataset.get_dataset_dir()
        aug_version  = self.dataset.get_selected_version()
        in_channels  = self.dataset.get_band_count_int()
        img_size     = self.dataset.get_tile_size_int()

        if not dataset_dir:
            return None, "Please select a dataset folder."
        if aug_version is None:
            return None, "Please select an augmented version."
        if in_channels is None:
            return None, "Could not determine the number of input bands.\nCheck the selected dataset version."
        if img_size is None:
            return None, "Could not determine the tile size.\nCheck the selected dataset version."

        # Model
        model_cfg = self.model.get_model_config()

        # Training config
        train_cfg = self.training_config.get_training_config()

        # Checkpoints
        ckpt_cfg = self.checkpoints.get_checkpoint_config()
        if not ckpt_cfg.get("output_dir"):
            return None, "Please select an output directory for checkpoints."

        # Hardware
        hw_cfg = self.hardware.get_hardware_config()

        config = {
            # Dataset
            "dataset_dir": dataset_dir,
            "aug_version": aug_version,
            # Model
            "architecture":  model_cfg["architecture"],
            "in_channels":   in_channels,
            "img_size":      img_size,
            "base_channels": model_cfg["base_channels"],
            # Training
            "loss":       train_cfg["loss"],
            "optimizer":  train_cfg["optimizer"],
            "lr":         train_cfg["lr"],
            "batch_size": train_cfg["batch_size"],
            "epochs":     train_cfg["epochs"],
            "scheduler":  train_cfg["scheduler"],
            # Checkpoints
            "output_dir":    ckpt_cfg["output_dir"],
            "model_name":    ckpt_cfg["model_name"],
            "save_strategy": ckpt_cfg["save_strategy"],
            "every_n":       ckpt_cfg.get("every_n", 10),
            "resume_path":   ckpt_cfg.get("resume_path"),
            # Hardware
            "device":      hw_cfg["device"],
            "num_workers": hw_cfg["num_workers"],
        }

        # Checkpoint compatibility — must be the last check so config is fully built
        if config.get("resume_path"):
            compat_error = self._check_checkpoint_compat(config)
            if compat_error:
                return None, compat_error

        return config, None

    # -------------------------------------------------------------------------

    @staticmethod
    def _check_checkpoint_compat(config: dict):
        """
        Loads the resume checkpoint and verifies that its saved architecture,
        input band count, and tile size all match the current UI settings.

        Returns None on success, or a detailed human-readable error string
        that tells the user exactly what is wrong and how to fix it.
        """
        path = config["resume_path"]

        try:
            import torch
            data = torch.load(path, map_location="cpu", weights_only=False)  # nosec B614
        except Exception as exc:
            return (
                f"Could not read the checkpoint file:\n  {exc}\n\n"
                "Make sure the file is a valid .pth checkpoint saved by this plugin."
            )

        saved_cfg = data.get("config", {})

        # Collect all mismatches before reporting so the user sees everything at once
        issues = []

        # --- Architecture ----------------------------------------------------
        saved_arch = data.get("architecture") or saved_cfg.get("architecture")
        if saved_arch and saved_arch != config["architecture"]:
            issues.append(
                f"Architecture mismatch\n"
                f"  Checkpoint was trained with : {saved_arch}\n"
                f"  You currently have selected : {config['architecture']}\n"
                f"  Fix: Change the Architecture dropdown to '{saved_arch}'."
            )

        # --- Input bands (in_channels) ---------------------------------------
        saved_bands = saved_cfg.get("in_channels")
        if saved_bands is not None and saved_bands != config["in_channels"]:
            issues.append(
                f"Input band count mismatch\n"
                f"  Checkpoint expects : {saved_bands} band(s)\n"
                f"  Current dataset has: {config['in_channels']} band(s)\n"
                f"  Fix: Select a dataset version that has {saved_bands} band(s), "
                f"or choose a checkpoint trained on {config['in_channels']}-band data."
            )

        # --- Tile size (img_size) --------------------------------------------
        saved_size = saved_cfg.get("img_size")
        if saved_size is not None and saved_size != config["img_size"]:
            issues.append(
                f"Tile size mismatch\n"
                f"  Checkpoint expects : {saved_size} × {saved_size} px tiles\n"
                f"  Current dataset has: {config['img_size']} × {config['img_size']} px tiles\n"
                f"  Fix: Select a dataset version clipped to {saved_size} × {saved_size} px, "
                f"or choose a checkpoint trained on {config['img_size']} × {config['img_size']} px tiles."
            )

        if not issues:
            return None

        header = (
            f"The selected checkpoint is not compatible with the current settings.\n"
            f"Found {len(issues)} problem(s):\n"
            f"{'─' * 52}\n\n"
        )
        return header + "\n\n".join(
            f"[{i + 1}] {issue}" for i, issue in enumerate(issues)
        )
