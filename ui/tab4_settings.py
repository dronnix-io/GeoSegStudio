"""
module: ui/tab4_settings.py

Settings section for the Predict tab.

Contains:
  - Device selector  (CPU / CUDA GPU)
  - Overlap %        (0–50 %, default 0)
  - Threshold        (0.01–0.99, default 0.50)
  - Estimated tile count (read-only, updated when raster size or overlap changes)

Tab4Widget calls update_tile_estimate() whenever the model img_size or
raster size changes so the estimate stays current.
"""
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
)

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .styles import style_icon_btn


def _detect_devices():
    devices = [("CPU", "cpu")]
    try:
        import torch
    except ImportError:
        return devices, ("PyTorch not installed.", True)
    if not torch.cuda.is_available():
        return devices, ("CUDA not available.", True)
    count = torch.cuda.device_count()
    for i in range(count):
        devices.append(
            (f"CUDA:{i}  ({torch.cuda.get_device_name(i)})", f"cuda:{i}"))
    msg = f"{count} GPU{'s' if count > 1 else ''} detected."
    return devices, (msg, False)


class PredictSettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Settings")
        self.content = SectionContentWidget()
        self.form = self.content.layout()

        # --- Device ----------------------------------------------------------
        device_row = QHBoxLayout()
        device_row.setContentsMargins(0, 0, 0, 0)
        device_row.setSpacing(4)

        self.device_combo = QComboBox()
        self.device_combo.setToolTip("Compute device used for inference.")
        device_row.addWidget(self.device_combo)

        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(28)
        self.refresh_btn.setToolTip("Re-scan for available compute devices.")
        style_icon_btn(self.refresh_btn)
        device_row.addWidget(self.refresh_btn)

        self.form.addRow("Device", device_row)

        self.device_hint = QLabel("")
        self.device_hint.setWordWrap(True)
        self.device_hint.setVisible(False)
        self.form.addRow("", self.device_hint)

        # --- Overlap % -------------------------------------------------------
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 50)
        self.overlap_spin.setValue(0)
        self.overlap_spin.setSuffix(" %")
        self.overlap_spin.setToolTip(
            "Percentage of the tile size used as overlap between adjacent tiles.\n"
            "0 % — no overlap, fastest, may show seam artefacts at tile edges.\n"
            "25–50 % — overlapping tiles are averaged, smoother result.\n"
            "Higher overlap increases tile count and processing time.")
        self.overlap_spin.valueChanged.connect(self._update_tile_estimate)
        self.form.addRow("Overlap", self.overlap_spin)

        # --- Threshold -------------------------------------------------------
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.99)
        self.threshold_spin.setValue(0.50)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setToolTip(
            "Sigmoid probability threshold for converting the model output\n"
            "to a binary mask.\n"
            "0.50 — balanced (equal weight to false positives / negatives).\n"
            "Higher → fewer detections, lower false-positive rate.\n"
            "Lower  → more detections, higher recall."
        )
        self.form.addRow("Threshold", self.threshold_spin)

        # --- Estimated tile count (read-only) --------------------------------
        self.tile_count_lbl = QLabel("—")
        self.tile_count_lbl.setToolTip(
            "Approximate number of tiles that will be processed.\n"
            "Updates automatically when the raster and overlap are set."
        )
        self.form.addRow("Est. Tiles", self.tile_count_lbl)

        # Internal state for tile count calculation
        self._img_size = None
        self._raster_w = None
        self._raster_h = None

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(self.content)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

        self.refresh_btn.clicked.connect(self._refresh_devices)
        self._refresh_devices()

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
        self.device_hint.setText(
            f"<span style='color:{color}'>{message}</span>")
        self.device_hint.setVisible(True)

    def _update_tile_estimate(self):
        if (self._img_size is None
                or self._raster_w is None
                or self._raster_h is None):
            self.tile_count_lbl.setText("—")
            return

        overlap_pct = self.overlap_spin.value() / 100.0
        stride = max(1, int(self._img_size * (1.0 - overlap_pct)))
        n_x = len(range(0, self._raster_w, stride))
        n_y = len(range(0, self._raster_h, stride))
        self.tile_count_lbl.setText(f"~{n_x * n_y:,}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def update_tile_estimate(
            self,
            img_size: int,
            raster_w: int,
            raster_h: int):
        """Called by Tab4Widget whenever model or raster selection changes."""
        self._img_size = img_size
        self._raster_w = raster_w
        self._raster_h = raster_h
        self._update_tile_estimate()

    def get_settings_config(self) -> dict:
        return {
            "device": self.device_combo.currentData(),
            "overlap_pct": self.overlap_spin.value() / 100.0,
            "threshold": self.threshold_spin.value(),
        }
