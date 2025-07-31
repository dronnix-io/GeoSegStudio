from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QLabel
from qgis.PyQt.QtCore import Qt  
from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget


class Tab1Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        # ----------------------------
        # Ins & Outs (non-expandable, but shows arrow)
        # ----------------------------
        ins_section = ExpandableGroupBox("Ins & Outs")
        ins_section.toggle_button.setChecked(True)           # Always expanded
        ins_section.toggle_button.setArrowType(Qt.DownArrow)
        ins_section.toggle_button.setEnabled(False)          # Disable arrow click

        ins_content = SectionContentWidget()
        ins_content.layout().addRow(QLabel("Controls for Ins & Outs go here."))

        ins_section.setContentLayout(QVBoxLayout())
        ins_section.content_area.layout().addWidget(ins_content)
        layout.addWidget(ins_section)

        # ----------------------------
        # Expandable Sections
        # ----------------------------
        layout.addWidget(self._create_expandable_section("Clipping"))
        layout.addWidget(self._create_expandable_section("Splitting"))
        layout.addWidget(self._create_expandable_section("Augmentation"))
        layout.addWidget(self._create_expandable_section("Channel Stacking"))

        layout.addStretch()
        self.setLayout(layout)

    def _create_expandable_section(self, title):
        section = ExpandableGroupBox(title)

        content = SectionContentWidget()
        content.layout().addRow(QLabel(f"Controls for {title} go here."))

        section.setContentLayout(QVBoxLayout())
        section.content_area.layout().addWidget(content)

        return section
