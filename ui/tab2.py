from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QLabel

class Tab2Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Tab 2 - Inference"))
        self.setLayout(layout)
