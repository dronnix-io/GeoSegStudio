"""
module: tab1_clipping.py
"""
import multiprocessing
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QComboBox, QPushButton, QProgressBar
)
from qgis.PyQt.QtCore import Qt
from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .styles import style_primary_btn, style_progress_bar
from ..DL.constants import SUPPORTED_SIZES


class ClippingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Clipping")

        self.content = SectionContentWidget()
        form = self.content.layout()

        # Tile Size — restricted to sizes supported by all model
        # architectures
        self.window_size = QComboBox()
        for size in SUPPORTED_SIZES:
            self.window_size.addItem(str(size), size)
        self.window_size.setCurrentIndex(
            SUPPORTED_SIZES.index(256))   # default 256
        self.window_size.setToolTip(
            "Tile size in pixels. Only sizes supported by the model "
            "architectures are available: " +
            ", ".join(
                str(s) for s in SUPPORTED_SIZES))
        form.addRow("Tile Size (px)", self.window_size)

        # Stride
        self.stride = QSpinBox()
        self.stride.setRange(1, 10000)
        self.stride.setValue(128)
        form.addRow("Stride (px)", self.stride)

        # Burn Value
        self.burn_value = QSpinBox()
        self.burn_value.setRange(1, 255)
        self.burn_value.setValue(1)
        form.addRow("Burn Value (1–255)", self.burn_value)

        # Output Format
        self.output_format = QComboBox()
        self.output_format.addItems(["geocoded", "array"])
        form.addRow("Output Format", self.output_format)

        # Number of CPUs
        max_cpus = max(1, multiprocessing.cpu_count() - 2)
        self.cpu_spin = QSpinBox()
        self.cpu_spin.setMinimum(1)
        self.cpu_spin.setMaximum(max_cpus)
        form.addRow("Number of CPUs", self.cpu_spin)

        # --- Apply button + progress bar -------------------------------------
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(0, 6, 0, 0)
        bottom_layout.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.setContentsMargins(0, 0, 0, 0)

        self.apply_btn = QPushButton("Apply Clipping")
        style_primary_btn(self.apply_btn)
        btn_row.addWidget(self.apply_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        from .styles import style_danger_btn
        style_danger_btn(self.stop_btn)
        btn_row.addWidget(self.stop_btn)

        bottom_layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        style_progress_bar(self.progress_bar)
        bottom_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setVisible(False)
        bottom_layout.addWidget(self.status_label)

        # Assemble section
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(self.content)
        section_layout.addLayout(bottom_layout)

        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)
        self.setLayout(layout)

    # -------------------------------------------------------------------------
    # State helpers  (called by tab1.py)
    # -------------------------------------------------------------------------

    def set_running(self, running: bool):
        self.apply_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.progress_bar.setVisible(running)
        self.status_label.setVisible(False)
        if running:
            self.progress_bar.setValue(0)

    def set_status(self, message: str, error: bool = False):
        color = "red" if error else "green"
        self.status_label.setText(
            f"<span style='color:{color}'>{message}</span>"
        )
        self.status_label.setVisible(True)
        self.progress_bar.setVisible(False)

    # -------------------------------------------------------------------------
    # Data getter
    # -------------------------------------------------------------------------

    def get_clipping_params(self) -> dict:
        return {
            "window_size": self.window_size.currentData(),   # int from SUPPORTED_SIZES
            "stride": self.stride.value(),
            "burn_value": self.burn_value.value(),
            "output_format": self.output_format.currentText(),
            "cpu_count": self.cpu_spin.value(),
        }
