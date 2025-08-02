'''
module: tab1_clipping.py
'''
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QLineEdit
)
from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget


class ClippingSection(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Clipping")

        self.content = SectionContentWidget()
        form = self.content.layout()

        # Window Size (max 10000)
        self.window_size = QSpinBox()
        self.window_size.setRange(1, 10000)
        form.addRow("Window Size (px)", self.window_size)

        # Stride
        self.stride = QSpinBox()
        self.stride.setRange(1, 10000)
        form.addRow("Stride (px)", self.stride)

        # Output Pixel Size
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.0001, 10000)
        self.pixel_size.setDecimals(4)
        form.addRow("Output Pixel Size", self.pixel_size)

        # Burn Value
        self.burn_value = QSpinBox()
        self.burn_value.setRange(1, 255)
        form.addRow("Burn Value (1–255)", self.burn_value)

        # Output Format (geocoded or array)
        self.output_format = QComboBox()
        self.output_format.addItems(["geocoded", "array"])
        form.addRow("Output Format", self.output_format)

        # Number of CPUs
        self.cpu_count = QSpinBox()
        self.cpu_count.setRange(1, 64)  # adjust based on target system
        form.addRow("Number of CPUs", self.cpu_count)

        # Name Prefix
        self.prefix = QLineEdit()
        form.addRow("Name Prefix", self.prefix)

        self.section.setContentLayout(QVBoxLayout())
        self.section.content_area.layout().addWidget(self.content)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)
        self.setLayout(layout)

    def get_clipping_params(self):
        return {
            "window_size": self.window_size.value(),
            "stride": self.stride.value(),
            "pixel_size": self.pixel_size.value(),
            "burn_value": self.burn_value.value(),
            "output_format": self.output_format.currentText(),
            "cpu_count": self.cpu_count.value(),
            "prefix": self.prefix.text()
        }