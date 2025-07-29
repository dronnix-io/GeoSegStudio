from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QLabel

class Tab1Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Tab 1 - Model"))
        self.setLayout(layout)
