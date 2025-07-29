from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QLabel

class Tab3Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Tab 3 - Results"))
        self.setLayout(layout)
