"""
module: ui/tab3.py

Evaluate tab — orchestrates all evaluation-related sections and wires the
background EvaluationWorker to the Run & Monitor widget.

Sections (top to bottom)
------------------------
EvalModelWidget    — checkpoint file picker + metadata display
EvalDatasetWidget  — dataset folder, augmented version, split selector
EvalRunWidget      — device, output dir, run/stop, progress
EvalResultsWidget  — aggregate metrics + confusion matrix  (hidden until run)
EvalSamplesWidget  — sample prediction grid                (hidden until run)
"""
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QSizePolicy, QMessageBox,
)

from .tab3_model import EvalModelWidget
from .tab3_dataset import EvalDatasetWidget
from .tab3_run import EvalRunWidget
from .tab3_results import EvalResultsWidget
from .tab3_samples import EvalSamplesWidget


class Tab3Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.model = EvalModelWidget()
        self.dataset = EvalDatasetWidget()
        self.run = EvalRunWidget()
        self.results = EvalResultsWidget()
        self.samples = EvalSamplesWidget()

        self._worker = None   # EvaluationWorker instance while running

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)

        content_layout.addWidget(self.model)
        content_layout.addWidget(self.dataset)
        content_layout.addWidget(self.run)
        content_layout.addWidget(self.results)
        content_layout.addWidget(self.samples)
        content_layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # --- Connections -----------------------------------------------------
        self.run.run_btn.clicked.connect(self._start_evaluation)
        self.run.stop_btn.clicked.connect(self._stop_evaluation)

    # -------------------------------------------------------------------------
    # Evaluation control
    # -------------------------------------------------------------------------

    def _start_evaluation(self):
        config, error = self._build_config()
        if error:
            QMessageBox.warning(self, "Cannot Start Evaluation", error)
            return

        from ..DL.evaluator import EvaluationWorker

        self.results.reset()
        self.samples.reset()

        self._worker = EvaluationWorker(config, parent=self)
        self._worker.phase_update.connect(self.run.update_phase)
        self._worker.tile_done.connect(
            lambda cur, tot: self.run.update_tile_progress(cur, tot)
        )
        self._worker.evaluation_finished.connect(self._on_evaluation_finished)

        self.run.set_running(True, total_tiles=0)
        self._worker.start()

    def _stop_evaluation(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()

    def _on_evaluation_finished(
            self,
            success: bool,
            results: object,
            message: str):
        self.run.set_running(False)

        if success:
            self.run.set_status(message, error=False)
            self.results.show_results(results)
            samples = results.get("samples", [])
            if samples:
                self.samples.show_samples(samples)
        else:
            self.run.set_status(message, error=True)

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
        # Model
        checkpoint_path = self.model.get_checkpoint_path()
        meta = self.model.get_metadata()

        if not checkpoint_path:
            return None, "Please select a checkpoint file."
        if not meta:
            return None, (
                "No checkpoint metadata found.\n"
                "Select a valid .pth checkpoint saved by this plugin."
            )

        # Dataset
        dataset_dir = self.dataset.get_dataset_dir()
        aug_version = self.dataset.get_selected_version()
        split = self.dataset.get_split()

        if not dataset_dir:
            return None, "Please select a dataset folder."
        if aug_version is None:
            return None, "Please select an augmented version."

        # Verify that the split directory actually exists
        import os
        split_images = os.path.join(
            dataset_dir, "augmented", f"v{aug_version}", split, "images"
        )
        if not os.path.isdir(split_images):
            return None, (
                f"Could not find the '{split}' split in the selected dataset version.\n"
                f"Expected: {split_images}\n\n"
                "Run the Prepare tab to generate this split, or select a different split."
            )

        # Run / hardware
        run_cfg = self.run.get_run_config()

        if run_cfg["save_masks"] and not run_cfg["output_dir"]:
            return None, (
                "'Save Masks' is checked but no Output Directory is set.\n"
                "Either set an output directory or uncheck 'Save Masks'."
            )

        config = {
            "checkpoint_path": checkpoint_path,
            "dataset_dir": dataset_dir,
            "aug_version": aug_version,
            "split": split,
            "device": run_cfg["device"],
            "output_dir": run_cfg["output_dir"],
            "save_masks": run_cfg["save_masks"],
            "n_samples": run_cfg["n_samples"],
        }

        return config, None
