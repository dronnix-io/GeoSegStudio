"""
module: ui/tab4_output.py

Output section for the Predict tab.

Lets the user choose:
  - Output format : Vector — GeoPackage  (default, recommended)
                    Raster — GeoTIFF
                    Both   — GeoPackage + GeoTIFF side-by-side
  - Output path   : file path (extension is fixed / updated by format choice)
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


_FORMATS = [
    ("Vector — GeoPackage (.gpkg)",         "vector",  ".gpkg"),
    ("Raster — GeoTIFF (.tif)",             "raster",  ".tif"),
    ("Both — GeoPackage + GeoTIFF",         "both",    ""),
]


class PredictOutputWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Output")
        self.content = SectionContentWidget()
        self.form    = self.content.layout()

        # --- Output format ---------------------------------------------------
        self.format_combo = QComboBox()
        for label, key, _ in _FORMATS:
            self.format_combo.addItem(label, key)
        self.format_combo.setToolTip(
            "Vector (GeoPackage): polygonizes the binary prediction mask.\n"
            "  Recommended — matches the original vector ground-truth format.\n\n"
            "Raster (GeoTIFF): single-band uint8 mask (0 = background, 255 = positive).\n\n"
            "Both: saves a GeoPackage and a GeoTIFF with the same base name."
        )
        self.form.addRow("Format", self.format_combo)

        # --- Output path -----------------------------------------------------
        path_row = QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.setSpacing(4)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self._update_placeholder()
        path_row.addWidget(self.path_edit)

        self.browse_btn = QPushButton("…")
        self.browse_btn.setFixedWidth(30)
        self.browse_btn.setToolTip("Choose where to save the prediction output.")
        path_row.addWidget(self.browse_btn)

        self.form.addRow("Output Path", path_row)

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

        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        self.browse_btn.clicked.connect(self._browse_output)

    # -------------------------------------------------------------------------

    def _current_ext(self) -> str:
        idx = self.format_combo.currentIndex()
        return _FORMATS[idx][2]

    def _update_placeholder(self):
        ext = self._current_ext()
        if ext:
            self.path_edit.setPlaceholderText(
                f"Choose save location (*{ext}) …"
            )
        else:
            self.path_edit.setPlaceholderText(
                "Choose base save path (extensions added automatically) …"
            )

    def _on_format_changed(self):
        self._update_placeholder()
        # If a path is already set, update its extension to match new format
        current = self.path_edit.text().strip()
        if not current:
            return
        ext = self._current_ext()
        if ext:
            base = os.path.splitext(current)[0]
            self.path_edit.setText(base + ext)
        else:
            # "both" — strip any extension so the user sees the base path
            self.path_edit.setText(os.path.splitext(current)[0])

    def _browse_output(self):
        fmt_key = self.format_combo.currentData()
        ext     = self._current_ext()

        if fmt_key == "vector":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Prediction As",
                self.path_edit.text() or "prediction",
                "GeoPackage (*.gpkg)",
            )
        elif fmt_key == "raster":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Prediction As",
                self.path_edit.text() or "prediction",
                "GeoTIFF (*.tif *.tiff)",
            )
        else:
            # "both" — ask for a base name (no extension filter)
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Prediction — Choose Base Name",
                self.path_edit.text() or "prediction",
                "All files (*.*)",
            )
            if path:
                path = os.path.splitext(path)[0]   # strip any extension

        if path:
            self.path_edit.setText(path)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_output_config(self) -> dict:
        return {
            "output_format": self.format_combo.currentData(),
            "output_path":   self.path_edit.text().strip(),
            "load_into_qgis": self.load_check.isChecked(),
        }
