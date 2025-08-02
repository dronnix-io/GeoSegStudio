'''
module: tab1.py
'''
from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QLabel
from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .tab1_ins_outs import InsAndOutsWidget
from .tab1_clipping import ClippingWidget
from .tab1_splitting import SplittingWidget
from .tab1_augmentation import AugmentationWidget


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
        layout.addWidget(self._create_clipping_section())
        layout.addWidget(self._create_splitting_section())
        layout.addWidget(self._create_augmentation_section())
        layout.addWidget(self._create_expandable_section("Channel Stacking"))

        layout.addStretch()
        self.setLayout(layout)

    def _create_ins_outs_section(self):
        return InsAndOutsWidget()
    
    def _create_clipping_section(self):
        return ClippingWidget()

    def _create_splitting_section(self):
        return SplittingWidget()
    
    def _create_augmentation_section(self):
        return AugmentationWidget()

    def _create_expandable_section(self, title):
        section = ExpandableGroupBox(title)

        content = SectionContentWidget()
        content.layout().addRow(QLabel(f"Controls for {title} go here."))

        section.setContentLayout(QVBoxLayout())
        section.content_area.layout().addWidget(content)

        return section
