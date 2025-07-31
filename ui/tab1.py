from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QLabel, QScrollArea, QFormLayout

class Tab1Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Main vertical layout
        layout = QVBoxLayout()

        # Create each section
        layout.addWidget(self._create_section("Ins & Outs"))
        layout.addWidget(self._create_section("Clipping"))
        layout.addWidget(self._create_section("Splitting"))
        layout.addWidget(self._create_section("Augmentation"))
        layout.addWidget(self._create_section("Channel Stacking"))

        layout.addStretch()
        self.setLayout(layout)

    def _create_section(self, title):
        group = QGroupBox(title)
        group.setCheckable(False)  # Set to True if you want collapse/expand via checkbox
        inner_layout = QFormLayout()

        # Placeholder label for now — replace with actual widgets later
        inner_layout.addRow(QLabel(f"Controls for {title} go here."))

        group.setLayout(inner_layout)
        return group
