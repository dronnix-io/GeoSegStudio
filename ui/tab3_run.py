"""
module: ui/tab3_run.py

Run & Monitor section for the Evaluate tab.

Always visible (non-collapsible).  Contains:
  - Device selector (CPU / CUDA GPU) with refresh button
  - Optional output directory for CSV export and predicted mask GeoTIFFs
  - Save Masks checkbox and N Samples spinbox
  - Run / Stop buttons
  - Tile-level progress bar
  - Phase status label
  - Final status label (success / error)
"""
import os

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QProgressBar, QLabel, QComboBox, QLineEdit,
    QFileDialog, QSpinBox, QCheckBox, QSizePolicy,
)
from qgis.PyQt.QtCore import Qt

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget


def _detect_devices():
    """Returns (list of (label, key), (message, is_error)) — mirrors tab2_hardware."""
    devices = [("CPU", "cpu")]
    try:
        import torch
    except ImportError:
        return devices, ("PyTorch not installed.", True)

    if not torch.cuda.is_available():
        return devices, ("CUDA not available.", True)

    count = torch.cuda.device_count()
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        devices.append((f"CUDA:{i}  ({name})", f"cuda:{i}"))

    msg = (
        f"{count} GPU{'s' if count > 1 else ''} detected. "
        "Select one from the dropdown above."
    )
    return devices, (msg, False)


class EvalRunWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Non-collapsible section
        self.section = ExpandableGroupBox("Run & Monitor")
        self.section.toggle_button.setChecked(True)
        self.section.toggle_button.setEnabled(False)

        self.content = SectionContentWidget()
        self.form    = self.content.layout()

        # --- Device ----------------------------------------------------------
        device_row = QHBoxLayout()
        device_row.setContentsMargins(0, 0, 0, 0)
        device_row.setSpacing(4)

        self.device_combo = QComboBox()
        self.device_combo.setToolTip(
            "Compute device used for inference.\n"
            "GPU is faster; CPU always works."
        )
        device_row.addWidget(self.device_combo)

        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(28)
        self.refresh_btn.setToolTip("Re-scan for available compute devices.")
        device_row.addWidget(self.refresh_btn)

        self.form.addRow("Device", device_row)

        self.device_hint = QLabel("")
        self.device_hint.setWordWrap(True)
        self.device_hint.setVisible(False)
        self.form.addRow("", self.device_hint)

        # --- Output directory (optional) -------------------------------------
        out_row = QHBoxLayout()
        out_row.setContentsMargins(0, 0, 0, 0)
        out_row.setSpacing(4)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText(
            "Optional — folder for CSV and predicted masks"
        )
        self.output_dir_edit.setReadOnly(True)
        out_row.addWidget(self.output_dir_edit)

        self.output_dir_btn = QPushButton("…")
        self.output_dir_btn.setFixedWidth(30)
        self.output_dir_btn.setToolTip(
            "Choose a folder to save:\n"
            "  • per-tile CSV (evaluation_<split>_per_tile.csv)\n"
            "  • predicted mask GeoTIFFs (when 'Save Masks' is checked)\n\n"
            "Leave empty to skip all file output."
        )
        out_row.addWidget(self.output_dir_btn)

        self.form.addRow("Output Dir", out_row)

        # --- Save Masks checkbox ---------------------------------------------
        self.save_masks_check = QCheckBox("Save predicted masks as GeoTIFF")
        self.save_masks_check.setChecked(False)
        self.save_masks_check.setToolTip(
            "Write one binary GeoTIFF per tile with the same\n"
            "geotransform and CRS as the original image tile.\n"
            "Requires an Output Dir to be set."
        )
        self.form.addRow("", self.save_masks_check)

        # --- N Samples -------------------------------------------------------
        self.n_samples_spin = QSpinBox()
        self.n_samples_spin.setRange(1, 64)
        self.n_samples_spin.setValue(8)
        self.n_samples_spin.setToolTip(
            "Number of evenly-spaced sample tiles to collect for\n"
            "visual inspection in the Sample Predictions section."
        )
        self.form.addRow("Sample Tiles", self.n_samples_spin)

        # --- Assemble inner --------------------------------------------------
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(10, 10, 10, 10)
        inner_layout.setSpacing(8)
        inner_layout.addWidget(self.content)

        # --- Run / Stop buttons ----------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.run_btn = QPushButton("Run Evaluation")
        self.run_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.stop_btn.setEnabled(False)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        inner_layout.addLayout(btn_row)

        # --- Phase label -----------------------------------------------------
        self.phase_label = QLabel("")
        self.phase_label.setAlignment(Qt.AlignCenter)
        self.phase_label.setVisible(False)
        inner_layout.addWidget(self.phase_label)

        # --- Tile progress bar -----------------------------------------------
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Tile %v / %m")
        self.progress_bar.setVisible(False)
        inner_layout.addWidget(self.progress_bar)

        # --- Status label ----------------------------------------------------
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        inner_layout.addWidget(self.status_label)

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(inner)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

        # --- Connections -----------------------------------------------------
        self.refresh_btn.clicked.connect(self._refresh_devices)
        self.output_dir_btn.clicked.connect(self._browse_output_dir)

        # Populate on startup
        self._refresh_devices()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _refresh_devices(self):
        devices, (message, is_error) = _detect_devices()

        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for label, key in devices:
            self.device_combo.addItem(label, key)
        if len(devices) > 1:
            self.device_combo.setCurrentIndex(1)
        self.device_combo.blockSignals(False)

        color = "red" if is_error else "green"
        self.device_hint.setText(f"<span style='color:{color}'>{message}</span>")
        self.device_hint.setVisible(True)

    def _browse_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir_edit.setText(folder)

    # -------------------------------------------------------------------------
    # Public API — config collection
    # -------------------------------------------------------------------------

    def get_run_config(self) -> dict:
        """Returns the hardware and output configuration for EvaluationWorker."""
        return {
            "device":      self.device_combo.currentData(),
            "output_dir":  self.output_dir_edit.text().strip(),
            "save_masks":  self.save_masks_check.isChecked(),
            "n_samples":   self.n_samples_spin.value(),
        }

    # -------------------------------------------------------------------------
    # Public API — UI state control
    # -------------------------------------------------------------------------

    def set_running(self, running: bool, total_tiles: int = 0):
        """Switches the UI between idle and running states."""
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setRange(0, total_tiles)
            self.progress_bar.setValue(0)
            self.phase_label.setText("Starting evaluation…")
            self.phase_label.setVisible(True)
            self.status_label.setVisible(False)
        else:
            self.phase_label.setVisible(False)

    def update_phase(self, message: str):
        """Updates the phase status label."""
        self.phase_label.setText(message)
        self.phase_label.setVisible(True)

    def update_tile_progress(self, current: int, total: int):
        """Updates the tile progress bar."""
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)

    def set_status(self, message: str, error: bool = False):
        """Shows a final status message and hides the progress bar."""
        color = "red" if error else "green"
        self.status_label.setText(
            f"<span style='color:{color}'>{message}</span>"
        )
        self.status_label.setVisible(True)
        self.progress_bar.setVisible(False)
        self.phase_label.setVisible(False)

    def reset(self):
        """Resets all monitor widgets to idle state."""
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.phase_label.setText("")
        self.phase_label.setVisible(False)
        self.status_label.setVisible(False)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
