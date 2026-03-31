"""
module: ui/tab4_input.py

Input Raster section for the Predict tab.

Lets the user browse to any GeoTIFF.  Once a file is selected, reads and
displays read-only metadata: width × height, band count, CRS name, and
pixel size.  The band count is used by Tab4Widget to validate that the
raster matches the model's expected in_channels.
"""
import os

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QFrame,
)

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget


class PredictInputWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._band_count = None   # int, set after a file is loaded
        self._raster_w   = None
        self._raster_h   = None

        self.section = ExpandableGroupBox("Input Raster")
        self.content = SectionContentWidget()
        self.form    = self.content.layout()

        # --- File picker -----------------------------------------------------
        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(4)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Browse to a GeoTIFF raster …")
        self.path_edit.setReadOnly(True)
        file_row.addWidget(self.path_edit)

        self.browse_btn = QPushButton("…")
        self.browse_btn.setFixedWidth(30)
        self.browse_btn.setToolTip(
            "Select the full-resolution GeoTIFF raster to run prediction on.\n"
            "The band count must match the model's expected input bands."
        )
        file_row.addWidget(self.browse_btn)

        self.form.addRow("Raster File", file_row)

        self.hint_lbl = QLabel("")
        self.hint_lbl.setWordWrap(True)
        self.hint_lbl.setVisible(False)
        self.form.addRow("", self.hint_lbl)

        # --- Separator -------------------------------------------------------
        self.sep = QFrame()
        self.sep.setFrameShape(QFrame.HLine)
        self.sep.setFrameShadow(QFrame.Sunken)
        self.sep.setVisible(False)
        self.form.addRow(self.sep)

        # --- Read-only info --------------------------------------------------
        self.size_lbl       = QLabel("—")
        self.bands_lbl      = QLabel("—")
        self.crs_lbl        = QLabel("—")
        self.pixel_size_lbl = QLabel("—")

        self.form.addRow("Dimensions",  self.size_lbl)
        self.form.addRow("Bands",       self.bands_lbl)
        self.form.addRow("CRS",         self.crs_lbl)
        self.form.addRow("Pixel Size",  self.pixel_size_lbl)

        self._set_info_visible(False)

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(self.content)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

        self.browse_btn.clicked.connect(self._browse_raster)

    # -------------------------------------------------------------------------

    def _set_info_visible(self, visible: bool):
        self.sep.setVisible(visible)
        for lbl in (self.size_lbl, self.bands_lbl,
                    self.crs_lbl, self.pixel_size_lbl):
            lbl.setVisible(visible)
        form = self.form
        for row in range(form.rowCount()):
            lbl_item = form.itemAt(row, form.LabelRole)
            fld_item = form.itemAt(row, form.FieldRole)
            if fld_item and fld_item.widget() in (
                self.size_lbl, self.bands_lbl,
                self.crs_lbl, self.pixel_size_lbl,
            ):
                if lbl_item and lbl_item.widget():
                    lbl_item.widget().setVisible(visible)

    def _browse_raster(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Raster", "",
            "GeoTIFF (*.tif *.tiff);;All files (*.*)"
        )
        if not path:
            return
        self.path_edit.setText(path)
        self._load_raster_info(path)

    def _load_raster_info(self, path: str):
        self._band_count = None
        self._raster_w   = None
        self._raster_h   = None
        self._set_info_visible(False)
        self.hint_lbl.setVisible(False)

        try:
            from osgeo import gdal, osr
            gdal.UseExceptions()

            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds is None:
                raise IOError("GDAL could not open the file.")

            w       = ds.RasterXSize
            h       = ds.RasterYSize
            bands   = ds.RasterCount
            gt      = ds.GetGeoTransform()
            wkt     = ds.GetProjection()
            ds = None

            # CRS name
            crs_name = "Unknown"
            if wkt:
                srs = osr.SpatialReference()
                srs.ImportFromWkt(wkt)
                name = srs.GetAttrValue("PROJCS") or srs.GetAttrValue("GEOGCS")
                if name:
                    crs_name = name

            # Pixel size (absolute value of x-pixel size)
            px = abs(gt[1]) if gt else None
            px_str = f"{px:.6f} map units" if px is not None else "?"

            self._band_count = bands
            self._raster_w   = w
            self._raster_h   = h

            self.size_lbl.setText(f"{w} × {h} px")
            self.bands_lbl.setText(str(bands))
            self.crs_lbl.setText(crs_name)
            self.pixel_size_lbl.setText(px_str)

            self._set_info_visible(True)

        except Exception as exc:
            color = "red"
            self.hint_lbl.setText(
                f"<span style='color:{color}'>Could not read raster: {exc}</span>"
            )
            self.hint_lbl.setVisible(True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_raster_path(self) -> str:
        return self.path_edit.text().strip()

    def get_band_count(self):
        """Returns the band count as int, or None if no file is loaded."""
        return self._band_count

    def get_raster_size(self):
        """Returns (width, height) or (None, None)."""
        return self._raster_w, self._raster_h
