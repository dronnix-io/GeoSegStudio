from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QLabel

class Tab4Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Predict — coming soon"))
        self.setLayout(layout)
