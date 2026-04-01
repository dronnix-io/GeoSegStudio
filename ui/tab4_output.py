"""
module: ui/tab4_output.py

Output section for the Predict tab.

Follows QGIS conventions:
  - Output Folder : directory picker
  - Output Name   : base filename, no extension (user types it)
  - Format        : Vector / Raster / Both  (determines extension(s))
  - Preview       : read-only label showing the exact file path(s) to be written
  - Load into QGIS: checkbox — adds the result layer to the map on success
"""
import os

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLineEdit,
    QLabel, QCheckBox, QFileDialog,
)

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .styles import style_icon_btn


_FORMATS = [
    ("Vector — GeoPackage (.gpkg)",  "vector"),
    ("Raster — GeoTIFF (.tif)",      "raster"),
    ("Both — GeoPackage + GeoTIFF",  "both"),
]

_EXT = {
    "vector": [".gpkg"],
    "raster": [".tif"],
    "both":   [".gpkg", ".tif"],
}


class PredictOutputWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Output")
        self.content = SectionContentWidget()
        self.form    = self.content.layout()

        # --- Output folder ---------------------------------------------------
        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.setSpacing(4)

        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Select output folder …")
        self.dir_edit.setReadOnly(True)
        dir_row.addWidget(self.dir_edit)

        self.dir_btn = QPushButton("…")
        self.dir_btn.setFixedWidth(30)
        style_icon_btn(self.dir_btn)
        self.dir_btn.setToolTip("Choose the folder where the output file(s) will be saved.")
        dir_row.addWidget(self.dir_btn)

        self.form.addRow("Output Folder", dir_row)

        # --- Output name (no extension) --------------------------------------
        self.name_edit = QLineEdit("prediction")
        self.name_edit.setToolTip(
            "Base filename without extension.\n"
            "The correct extension is added automatically based on the format.\n"
            "Example:  solar_panels  →  solar_panels.gpkg"
        )
        self.form.addRow("Output Name", self.name_edit)

        # --- Output format ---------------------------------------------------
        self.format_combo = QComboBox()
        for label, key in _FORMATS:
            self.format_combo.addItem(label, key)
        self.format_combo.setToolTip(
            "Vector (GeoPackage): polygonizes the binary prediction into polygons.\n"
            "  Recommended — matches the original vector ground-truth format.\n\n"
            "Raster (GeoTIFF): single-band uint8 mask (0 = background, 255 = positive).\n\n"
            "Both: saves a GeoPackage and a GeoTIFF with the same base name."
        )
        self.form.addRow("Format", self.format_combo)

        # --- Preview (read-only) ---------------------------------------------
        self.preview_lbl = QLabel("—")
        self.preview_lbl.setWordWrap(True)
        self.preview_lbl.setToolTip("Exact file path(s) that will be written.")
        self.form.addRow("Will save as", self.preview_lbl)

        # --- Load into QGIS --------------------------------------------------
        self.load_check = QCheckBox("Load result into QGIS after prediction")
        self.load_check.setChecked(True)
        self.load_check.setToolTip(
            "Automatically adds the output file as a new layer in the QGIS\n"
            "map canvas when prediction finishes.\n"
            "For 'Both' format, the vector layer is loaded."
        )
        self.form.addRow("", self.load_check)

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(self.content)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

        # --- Connections -----------------------------------------------------
        self.dir_btn.clicked.connect(self._browse_dir)
        self.name_edit.textChanged.connect(self._update_preview)
        self.format_combo.currentIndexChanged.connect(self._update_preview)

        self._update_preview()

    # -------------------------------------------------------------------------

    def _browse_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder",
            self.dir_edit.text() or "",
        )
        if folder:
            self.dir_edit.setText(folder)
            self._update_preview()

    def _update_preview(self):
        folder = self.dir_edit.text().strip()
        name   = self.name_edit.text().strip() or "prediction"
        fmt    = self.format_combo.currentData()
        exts   = _EXT.get(fmt, [".gpkg"])

        if not folder:
            self.preview_lbl.setText("—")
            return

        paths = [os.path.join(folder, name + ext) for ext in exts]
        self.preview_lbl.setText("\n".join(paths))

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_output_config(self) -> dict:
        """
        Returns the output configuration dict.

        output_path is the base path (folder + name, no extension).
        The worker appends the correct extension(s) based on output_format.
        """
        folder = self.dir_edit.text().strip()
        name   = self.name_edit.text().strip() or "prediction"
        return {
            "output_format":  self.format_combo.currentData(),
            "output_path":    os.path.join(folder, name) if folder else "",
            "load_into_qgis": self.load_check.isChecked(),
        }
