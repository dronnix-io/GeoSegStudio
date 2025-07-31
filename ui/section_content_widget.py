from qgis.PyQt.QtWidgets import QWidget, QFormLayout
from qgis.PyQt.QtCore import Qt


class SectionContentWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create a consistent layout and style
        layout = QFormLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.setLayout(layout)

        # Optional styling to mimic QGroupBox content
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)

    def layout(self):
        return super().layout()  # For type hinting or direct access
