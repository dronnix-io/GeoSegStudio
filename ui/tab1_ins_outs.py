'''
module: tab1_ins_outs.py
'''
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QLineEdit, QPushButton, QLabel,
    QHBoxLayout, QFileDialog
)
from qgis.PyQt.QtCore import Qt
from qgis.core import QgsMapLayer, QgsProject

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget


class InsAndOutsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Main expandable-style section with disabled toggle
        self.section = ExpandableGroupBox("Ins & Outs")
        self.section.toggle_button.setChecked(True)
        self.section.toggle_button.setArrowType(Qt.DownArrow)
        self.section.toggle_button.setEnabled(False)  # Disables user clicking the arrow

        # Section content layout
        self.content = SectionContentWidget()
        self.form = self.content.layout()

        # Raster layer dropdown
        self.raster_combo = QComboBox()
        self.form.addRow("Raster Lyr", self.raster_combo)

        # Vector layer dropdown
        self.vector_combo = QComboBox()
        self.form.addRow("Vector Lyr", self.vector_combo)

        # Output directory selector (LineEdit + Button)
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("...")
        self.output_dir_button.setFixedWidth(30)

        output_layout = QHBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(4)
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.output_dir_button)

        self.form.addRow("Output Dir", output_layout)

        # Add content to section and main layout
        self.section.setContentLayout(QVBoxLayout())
        self.section.content_area.layout().addWidget(self.content)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)
        self.setLayout(layout)

        # Populate dropdowns
        self._populate_layer_combos()

        # Connect file dialog button
        self.output_dir_button.clicked.connect(self._select_output_directory)

        # Connect to QGIS layer changes
        QgsProject.instance().layersAdded.connect(self._populate_layer_combos)
        QgsProject.instance().layersRemoved.connect(self._populate_layer_combos)

    def _populate_layer_combos(self):
        """Fills the combo boxes with available raster and vector layers."""
        self.raster_combo.clear()
        self.vector_combo.clear()

        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() == QgsMapLayer.RasterLayer:
                self.raster_combo.addItem(layer.name(), layer.id())
            elif layer.type() == QgsMapLayer.VectorLayer:
                self.vector_combo.addItem(layer.name(), layer.id())

    def _select_output_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir_edit.setText(folder)

    def get_selected_inputs(self):
        """Returns selected layer IDs and output path"""
        raster_id = self.raster_combo.currentData()
        vector_id = self.vector_combo.currentData()
        output_path = self.output_dir_edit.text()
        return raster_id, vector_id, output_path

