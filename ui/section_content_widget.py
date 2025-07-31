'''
module: section_content_widget.py
'''
from qgis.PyQt.QtWidgets import QWidget, QFormLayout
from qgis.PyQt.QtCore import Qt
from qgis.utils import iface
from qgis.PyQt.QtGui import QFont



class SectionContentWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(iface.mainWindow().font())  # Good here!

        # Inherit font from parent/QGIS
        # self.setFont(self.parent().font() if self.parent() else self.font())
        app_font = iface.mainWindow().font()
        self.setFont(app_font)

        # Create a consistent layout and style
        layout = QFormLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.setLayout(layout)

        # Optional styling to mimic QGroupBox content
        self.setStyleSheet("""
            SectionContentWidget  {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        
        SectionContentWidget QLabel,
        SectionContentWidget QComboBox,
        SectionContentWidget QLineEdit {
                font-size: 10pt;
            }
        """)

    def layout(self):
        return super().layout()  # For type hinting or direct access
