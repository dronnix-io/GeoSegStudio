'''
module: tab1.py
'''
from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QLabel
from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .tab1_ins_outs import InsAndOutsWidget 


class Tab1Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        # ----------------------------
        # Ins & Outs section (non-expandable, handled in its own widget)
        # ----------------------------
        layout.addWidget(self._create_ins_outs_section())

        # ----------------------------
        # Expandable Sections
        # ----------------------------
        layout.addWidget(self._create_expandable_section("Clipping"))
        layout.addWidget(self._create_expandable_section("Splitting"))
        layout.addWidget(self._create_expandable_section("Augmentation"))
        layout.addWidget(self._create_expandable_section("Channel Stacking"))

        layout.addStretch()
        self.setLayout(layout)

    def _create_ins_outs_section(self):
        return InsAndOutsWidget()

    def _create_expandable_section(self, title):
        section = ExpandableGroupBox(title)

        content = SectionContentWidget()
        content.layout().addRow(QLabel(f"Controls for {title} go here."))

        section.setContentLayout(QVBoxLayout())
        section.content_area.layout().addWidget(content)

        return section
