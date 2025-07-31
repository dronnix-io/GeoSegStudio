from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QToolButton, QFrame, QSizePolicy, QHBoxLayout, QLabel
from qgis.PyQt.QtCore import Qt
from .section_content_widget import SectionContentWidget

class ExpandableGroupBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = QToolButton()
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.clicked.connect(self.toggle_content)

        self.title_label = QLabel(f"<b>{title}</b>")

        # Header layout: arrow + label
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(10, 0, 0, 0)
        header_layout.setSpacing(6)
        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        # Content area
        self.content_area = QFrame()
        self.content_area.setFrameShape(QFrame.NoFrame)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_area.setVisible(True)

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addLayout(header_layout)
        self.main_layout.addWidget(self.content_area)

    def toggle_content(self):
        visible = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if visible else Qt.RightArrow)
        self.content_area.setVisible(visible)

    def setContentLayout(self, layout):
        self.content_area.setLayout(layout)
