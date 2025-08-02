'''
module: tab1_channel_stacking.py
'''
from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel
from qgis.PyQt.QtCore import Qt
from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget


class ChannelStackingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.methods = [
            "Stretched",
            "Inversed",
            "Edge Map",
            "Sharpened",
            "Blurred"
        ]

        self.method_checkboxes = []
        self.order_labels = []

        self.section = ExpandableGroupBox("Channel Stacking")
        self.section.setContentLayout(QVBoxLayout())

        self.content = SectionContentWidget()
        self.form_layout = self.content.layout()

        for method in self.methods:
            checkbox = QCheckBox(method)
            checkbox.stateChanged.connect(self._update_orders)

            order_label = QLabel("")
            order_label.setFixedWidth(20)
            order_label.setAlignment(Qt.AlignCenter)

            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_layout.addWidget(checkbox)
            row_layout.addStretch()
            row_layout.addWidget(QLabel("Order:"))
            row_layout.addWidget(order_label)

            container = QWidget()
            container.setLayout(row_layout)

            self.method_checkboxes.append(checkbox)
            self.order_labels.append(order_label)

            self.form_layout.addRow(container)

        self.section.content_area.layout().addWidget(self.content)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)
        self.setLayout(layout)

    def _update_orders(self):
        checked_indexes = [
            i for i, checkbox in enumerate(self.method_checkboxes) if checkbox.isChecked()
        ]
        for i, label in enumerate(self.order_labels):
            if i in checked_indexes:
                label.setText(str(checked_indexes.index(i) + 1))
            else:
                label.setText("")

    def get_selected_methods_with_order(self):
        """Returns list of selected method names in their assigned order"""
        selected = []
        for i, checkbox in enumerate(self.method_checkboxes):
            if checkbox.isChecked():
                selected.append(self.methods[i])
        return selected
