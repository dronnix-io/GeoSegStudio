"""
module: tab2_hardware.py

Hardware section for the Train tab.

Lets the user select the compute device (CPU or a detected CUDA GPU)
and the number of dataloader workers.
"""
import os

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QSpinBox, QLabel, QPushButton,
)
from qgis.PyQt.QtCore import Qt

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .styles import style_icon_btn


def _detect_devices() -> tuple:
    """Detects available compute devices.

    Returns
    -------
    devices : list of (display_label, device_key)
        CPU is always present; CUDA entries are appended when available.
    status  : (message, is_error)
        Human-readable explanation of what was found or what went wrong.
    """
    devices = [("CPU", "cpu")]

    try:
        import torch
    except ImportError:
        return devices, (
            "PyTorch is not installed in the QGIS Python environment.\n"
            "Install PyTorch to enable GPU support.",
            True,
        )

    if not torch.cuda.is_available():
        return devices, ("PyTorch found but CUDA is not available.\n"
                         "A CPU-only version of PyTorch may be installed, or GPU drivers "
                         "are missing. Re-install PyTorch with CUDA support to use your GPU.", True, )

    count = torch.cuda.device_count()
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        devices.append((f"CUDA:{i}  ({name})", f"cuda:{i}"))

    msg = (  # nosec B608
        f"{count} GPU{'s' if count > 1 else ''} detected. "
        "Select one from the dropdown above."
    )
    return devices, (msg, False)


class HardwareWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Hardware")
        self.content = SectionContentWidget()
        self.form = self.content.layout()

        # --- Device ----------------------------------------------------------
        device_row = QHBoxLayout()
        device_row.setContentsMargins(0, 0, 0, 0)
        device_row.setSpacing(4)

        self.device_combo = QComboBox()
        self.device_combo.setToolTip(
            "Compute device used for training.\n"
            "GPU training is significantly faster than CPU.\n"
            "Click ↻ to re-scan if your GPU is not listed."
        )
        device_row.addWidget(self.device_combo)

        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(28)
        self.refresh_btn.setToolTip("Re-scan for available compute devices.")
        style_icon_btn(self.refresh_btn)
        device_row.addWidget(self.refresh_btn)

        self.form.addRow("Device", device_row)

        # Detection status hint
        self.device_hint = QLabel("")
        self.device_hint.setWordWrap(True)
        self.device_hint.setVisible(False)
        self.form.addRow("", self.device_hint)

        # --- Dataloader workers ----------------------------------------------
        # Hidden from the UI for now: worker processes always crash inside
        # QGIS because sys.executable is qgis.exe, not python.exe.
        # num_workers is hardcoded to 0 in dataset.py until this is resolved.
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(0)
        self.workers_spin.setVisible(False)
        # (label row intentionally not added to form)

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(self.content)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)
        self.setLayout(layout)

        # --- Connections -----------------------------------------------------
        self.refresh_btn.clicked.connect(self._refresh_devices)

        # Populate on startup
        self._refresh_devices()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _refresh_devices(self):
        """Re-detects compute devices and repopulates the combo box."""
        devices, (message, is_error) = _detect_devices()

        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for label, key in devices:
            self.device_combo.addItem(label, key)

        # Default to the first GPU if one was found
        if len(devices) > 1:
            self.device_combo.setCurrentIndex(1)
        self.device_combo.blockSignals(False)

        # Show status hint
        color = "red" if is_error else "green"
        self.device_hint.setText(
            f"<span style='color:{color}'>{message}</span>")
        self.device_hint.setVisible(True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_hardware_config(self) -> dict:
        """Returns the hardware configuration as a dict."""
        return {
            "device": self.device_combo.currentData(),
            "num_workers": self.workers_spin.value(),
        }
