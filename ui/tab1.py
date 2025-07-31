from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QFormLayout
)
from qgis.PyQt.QtCore import Qt


class Tab1Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        # Add sections: Ins & Outs is NOT expandable, others are
        layout.addWidget(self._create_section("Ins & Outs", expandable=False))
        layout.addWidget(self._create_section("Clipping", expandable=True))
        layout.addWidget(self._create_section("Splitting", expandable=True))
        layout.addWidget(self._create_section("Augmentation", expandable=True))
        layout.addWidget(self._create_section("Channel Stacking", expandable=True))

        layout.addStretch()
        self.setLayout(layout)

    def _create_section(self, title, expandable=True):
        group = QGroupBox(title)

        if expandable:
            group.setCheckable(True)
            group.setChecked(True)  # Expanded by default

        content_widget = QWidget()
        content_layout = QFormLayout()
        content_layout.addRow(QLabel(f"Controls for {title} go here."))
        content_widget.setLayout(content_layout)

        wrapper_layout = QVBoxLayout()
        wrapper_layout.addWidget(content_widget)
        group.setLayout(wrapper_layout)

        if expandable:
            # Toggle visibility based on checkbox
            group.toggled.connect(content_widget.setVisible)
            content_widget.setVisible(True)

        return group
