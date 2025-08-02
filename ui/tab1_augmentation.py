'''
module: tab1_augmentation.py
'''
from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QCheckBox
from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget


class AugmentationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Augmentation")

        self.content = SectionContentWidget()
        self.checkboxes = {}

        # Augmentation options
        options = [
            "Stretched",
            "Inversed",
            "Edge map",
            "Sharpened",
            "Blurred"
        ]

        # "All" checkbox
        self.check_all = QCheckBox("All")
        self.check_all.stateChanged.connect(self._toggle_all)
        self.content.layout().addRow(self.check_all)

        # Individual checkboxes
        for opt in options:
            cb = QCheckBox(opt)
            self.checkboxes[opt] = cb
            cb.stateChanged.connect(self._check_if_all_selected)
            self.content.layout().addRow(cb)

        self.section.setContentLayout(QVBoxLayout())
        self.section.content_area.layout().addWidget(self.content)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)
        self.setLayout(layout)

    def _toggle_all(self, state):
        checked = state == 2  # Qt.Checked
        for cb in self.checkboxes.values():
            cb.setChecked(checked)

    def _check_if_all_selected(self):
        all_selected = all(cb.isChecked() for cb in self.checkboxes.values())
        self.check_all.blockSignals(True)
        self.check_all.setChecked(all_selected)
        self.check_all.blockSignals(False)

    def get_selected_augmentations(self):
        return [name for name, cb in self.checkboxes.items() if cb.isChecked()]