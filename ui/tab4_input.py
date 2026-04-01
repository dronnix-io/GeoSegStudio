"""
module: ui/tab4_input.py

Input Raster section for the Predict tab.

Supports two source modes toggled by radio buttons:

  QGIS Layer  — picks from raster layers already loaded in the current
                 QGIS project via QgsMapLayerComboBox.  Updates live as
                 layers are added or removed.

  From File   — classic file-browser for rasters not yet loaded in QGIS.

Either way the underlying path is resolved to a file on disk and the same
read-only info panel (dimensions, band count, CRS, pixel size) is shown as
a compact MetaCardGrid.
"""

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog,
    QRadioButton, QButtonGroup,
)

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .info_card import MetaCardGrid
from .styles import style_icon_btn


class PredictInputWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._band_count = None
        self._raster_w   = None
        self._raster_h   = None

        self.section = ExpandableGroupBox("Input Raster")
        self.content = SectionContentWidget()
        self.form    = self.content.layout()

        # --- Source mode toggle ----------------------------------------------
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(12)

        self.radio_layer = QRadioButton("QGIS layer")
        self.radio_file  = QRadioButton("From file")
        self.radio_layer.setChecked(True)

        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.radio_layer)
        self._mode_group.addButton(self.radio_file)

        mode_row.addWidget(self.radio_layer)
        mode_row.addWidget(self.radio_file)
        mode_row.addStretch()
        self.form.addRow("Source", mode_row)

        # --- QGIS layer combo ------------------------------------------------
        self._layer_combo_ok = False
        try:
            from qgis.gui import QgsMapLayerComboBox
            from qgis.core import QgsMapLayerProxyModel

            self.layer_combo = QgsMapLayerComboBox()
            self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
            self.layer_combo.setAllowEmptyLayer(True)
            self.layer_combo.setToolTip(
                "Select a raster layer already loaded in the QGIS project."
            )
            self._layer_combo_ok = True
        except Exception:
            self.layer_combo = QLabel("QGIS layer selector unavailable.")

        self.form.addRow("QGIS Layer", self.layer_combo)

        # --- File browser row ------------------------------------------------
        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(4)

        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Browse to a GeoTIFF raster …")
        self.file_edit.setReadOnly(True)
        file_row.addWidget(self.file_edit)

        self.browse_btn = QPushButton("…")
        self.browse_btn.setFixedWidth(30)
        self.browse_btn.setToolTip("Select a GeoTIFF raster file from disk.")
        style_icon_btn(self.browse_btn)
        file_row.addWidget(self.browse_btn)

        self.form.addRow("File", file_row)

        # --- Raster info cards -----------------------------------------------
        self.info_cards = MetaCardGrid(cols_per_row=2)
        self.form.addRow(self.info_cards)

        self.hint_lbl = QLabel("")
        self.hint_lbl.setWordWrap(True)
        self.hint_lbl.setVisible(False)
        self.form.addRow("", self.hint_lbl)

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(self.content)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

        # --- Connections -----------------------------------------------------
        self.radio_layer.toggled.connect(self._on_mode_changed)
        self.browse_btn.clicked.connect(self._browse_file)

        if self._layer_combo_ok:
            self.layer_combo.layerChanged.connect(self._on_layer_changed)

        self._on_mode_changed()

    # -------------------------------------------------------------------------

    def _on_mode_changed(self):
        use_layer = self.radio_layer.isChecked()
        self.layer_combo.setEnabled(use_layer)
        self.file_edit.setEnabled(not use_layer)
        self.browse_btn.setEnabled(not use_layer)

        if use_layer:
            self._on_layer_changed()
        else:
            path = self.file_edit.text().strip()
            if path:
                self._load_raster_info(path)
            else:
                self._clear_info()

    def _on_layer_changed(self):
        if not self._layer_combo_ok:
            return
        layer = self.layer_combo.currentLayer()
        if layer is None:
            self._clear_info()
            return
        path = layer.source().split("|")[0].strip()
        if path:
            self._load_raster_info(path)
        else:
            self._clear_info()

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Raster", "",
            "GeoTIFF (*.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        self.file_edit.setText(path)
        self._load_raster_info(path)

    def _load_raster_info(self, path: str):
        self._clear_info()
        self.hint_lbl.setVisible(False)

        try:
            from osgeo import gdal, osr
            gdal.UseExceptions()

            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds is None:
                raise IOError("GDAL could not open the file.")

            w     = ds.RasterXSize
            h     = ds.RasterYSize
            bands = ds.RasterCount
            gt    = ds.GetGeoTransform()
            wkt   = ds.GetProjection()
            ds    = None

            crs_name = "Unknown"
            if wkt:
                srs = osr.SpatialReference()
                srs.ImportFromWkt(wkt)
                name = srs.GetAttrValue("PROJCS") or srs.GetAttrValue("GEOGCS")
                if name:
                    crs_name = name

            px     = abs(gt[1]) if gt else None
            px_str = f"{px:.6f} mu" if px is not None else "?"

            self._band_count = bands
            self._raster_w   = w
            self._raster_h   = h

            self.info_cards.set_cards([
                ("Dimensions", f"{w} × {h} px"),
                ("Bands",      str(bands)),
                ("CRS",        crs_name),
                ("Pixel Size", px_str),
            ])

        except Exception as exc:
            self.hint_lbl.setText(
                f"<span style='color:red'>Could not read raster: {exc}</span>"
            )
            self.hint_lbl.setVisible(True)

    def _clear_info(self):
        self._band_count = None
        self._raster_w   = None
        self._raster_h   = None
        self.info_cards.clear_cards()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_raster_path(self) -> str:
        """Returns the resolved file path of the selected raster."""
        if self.radio_layer.isChecked() and self._layer_combo_ok:
            layer = self.layer_combo.currentLayer()
            if layer:
                return layer.source().split("|")[0].strip()
            return ""
        return self.file_edit.text().strip()

    def get_band_count(self):
        return self._band_count

    def get_raster_size(self):
        return self._raster_w, self._raster_h
