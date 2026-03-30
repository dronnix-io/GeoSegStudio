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


class Tab2Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.dataset         = DatasetWidget()
        self.model           = ModelWidget()
        self.training_config = TrainingConfigWidget()
        self.checkpoints     = CheckpointsWidget()
        self.hardware        = HardwareWidget()
        self.run_monitor     = RunMonitorWidget()

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
        self._worker.batch_progress.connect(self.run_monitor.update_batch_progress)
        self._worker.training_finished.connect(self._on_training_finished)

        self.run_monitor.reset_monitor()
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

        return config, None
