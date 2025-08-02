'''
module: tab1.py
'''
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QSizePolicy
)
from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .tab1_ins_outs import InsAndOutsWidget
from .tab1_clipping import ClippingWidget
from .tab1_splitting import SplittingWidget
from .tab1_augmentation import AugmentationWidget
from .tab1_channel_stacking import ChannelStackingWidget


class Tab1Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Inner content widget inside the scroll area ---
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(12)

        # Section: Ins & Outs (non-expandable)
        content_layout.addWidget(InsAndOutsWidget())

        # Section: Clipping
        content_layout.addWidget(self._wrap_in_expandable("Clipping", ClippingWidget()))

        # Section: Splitting
        content_layout.addWidget(self._wrap_in_expandable("Splitting", SplittingWidget()))

        # Section: Augmentation
        content_layout.addWidget(self._wrap_in_expandable("Augmentation", AugmentationWidget()))

        # Section: Channel Stacking
        content_layout.addWidget(self._wrap_in_expandable("Channel Stacking", ChannelStackingWidget()))

        content_layout.addStretch()

        # --- Scroll area to wrap the content ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Final layout of Tab1 ---
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def _wrap_in_expandable(self, title, inner_widget):
        """Utility to create a section with a title and inner content widget"""
        section = ExpandableGroupBox(title)
        section.setContentLayout(QVBoxLayout())
        section.content_area.layout().addWidget(inner_widget)
        return section