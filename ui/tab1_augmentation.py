'''
module: tab1_augmentation.py
'''
from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QCheckBox
from qgis.PyQt.QtCore import Qt
from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget


class AugmentationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Augmentation")
        self.section.setContentLayout(QVBoxLayout())

        self.content = SectionContentWidget()
        self.grid_layout = QGridLayout()
        self.content.layout().addRow(self.grid_layout)

        self.section.content_area.layout().addWidget(self.content)

        layout = QVBoxLayout()
        layout.addWidget(self.section)
        self.setLayout(layout)

        self.augmentations = [
            ("Original", True),
            ("Rotate 90", False),
            ("Rotate 180", False),
            ("Rotate 270", False),
            ("Mirror", False),
            ("Flip H", False),
            ("Flip V", False),
        ]

        self.checkboxes = {}
        self._init_checkboxes()

    def _init_checkboxes(self):
        self.checkbox_all = QCheckBox("All")
        self.checkbox_all.stateChanged.connect(self._handle_all_checkbox)
        self.grid_layout.addWidget(self.checkbox_all, 0, 0)

        for i, (label, always_checked) in enumerate(self.augmentations):
            cb = QCheckBox(label)
            cb.setChecked(always_checked)
            if always_checked:
                cb.setEnabled(False)
            else:
                cb.stateChanged.connect(self._handle_individual_checkbox)
            row = (i + 1) // 2
            col = (i + 1) % 2
            self.grid_layout.addWidget(cb, row, col)
            self.checkboxes[label] = cb

    def _handle_all_checkbox(self, state):
        if state == Qt.Checked:
            for label, cb in self.checkboxes.items():
                if cb.isEnabled():
                    cb.setChecked(True)
        else:
            for label, cb in self.checkboxes.items():
                if cb.isEnabled():
                    cb.setChecked(False)

    def _handle_individual_checkbox(self):
        all_checked = all(
            cb.isChecked() or not cb.isEnabled()
            for label, cb in self.checkboxes.items()
        )
        self.checkbox_all.blockSignals(True)
        self.checkbox_all.setChecked(all_checked)
        self.checkbox_all.blockSignals(False)

    def selected_methods(self):
        return [label for label, cb in self.checkboxes.items() if cb.isChecked()]
