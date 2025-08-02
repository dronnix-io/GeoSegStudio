'''
module: tab1_splitting.py
'''

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QSpinBox, QLabel, QFormLayout, QMessageBox
)
from qgis.PyQt.QtCore import Qt
from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget


class SplittingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Splitting")
        self.section.setContentLayout(QVBoxLayout())

        self.content = SectionContentWidget()
        self.form = self.content.layout()

        # Spin boxes
        self.train_spin = QSpinBox()
        self.valid_spin = QSpinBox()
        self.test_spin = QSpinBox()

        for spin in (self.train_spin, self.valid_spin, self.test_spin):
            spin.setRange(0, 100)
            spin.setSuffix("%")

        # Initial value
        self.train_spin.setValue(80)
        self.valid_spin.setValue(10)
        self.test_spin.setValue(10)

        self.form.addRow("Training %", self.train_spin)
        self.form.addRow("Validation %", self.valid_spin)
        self.form.addRow("Testing %", self.test_spin)

        # Connect change signals
        self.train_spin.valueChanged.connect(self._validate_total)
        self.valid_spin.valueChanged.connect(self._validate_total)
        self.test_spin.valueChanged.connect(self._validate_total)

        self.section.content_area.layout().addWidget(self.content)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)
        self.setLayout(layout)

    def _validate_total(self):
        total = self.train_spin.value() + self.valid_spin.value() + self.test_spin.value()

        if total != 100:
            self.section.title_label.setText(
                f"<b>Splitting</b> <span style='color:red'>(Total: {total}%)</span>"
            )
        else:
            self.section.title_label.setText("<b>Splitting</b>")

    def get_split_percentages(self):
        return {
            "train": self.train_spin.value(),
            "valid": self.valid_spin.value(),
            "test": self.test_spin.value()
        }
