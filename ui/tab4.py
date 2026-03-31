"""
module: ui/tab4.py

Predict tab — orchestrates all prediction-related sections and wires the
background PredictionWorker to the Run widget.

Sections (top to bottom)
------------------------
PredictModelWidget    — checkpoint file picker + metadata display
PredictInputWidget    — input raster picker + raster info
PredictSettingsWidget — device, overlap %, threshold, estimated tile count
PredictOutputWidget   — output format, path, load-into-QGIS option
PredictRunWidget      — run/stop, tile progress bar, status
"""
import os

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QSizePolicy, QMessageBox,
)

from .tab4_model    import PredictModelWidget
from .tab4_input    import PredictInputWidget
from .tab4_settings import PredictSettingsWidget
from .tab4_output   import PredictOutputWidget
from .tab4_run      import PredictRunWidget


class Tab4Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.model    = PredictModelWidget()
        self.input    = PredictInputWidget()
        self.settings = PredictSettingsWidget()
        self.output   = PredictOutputWidget()
        self.run      = PredictRunWidget()

        self._worker = None   # PredictionWorker while running

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)

        content_layout.addWidget(self.model)
        content_layout.addWidget(self.input)
        content_layout.addWidget(self.settings)
        content_layout.addWidget(self.output)
        content_layout.addWidget(self.run)
        content_layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # --- Connections — keep tile estimate up to date ---------------------
        # Re-calculate whenever model or raster selection changes
        self.model.browse_btn.clicked.connect(self._refresh_tile_estimate)
        self.input.browse_btn.clicked.connect(self._refresh_tile_estimate)

        # Run / Stop
        self.run.run_btn.clicked.connect(self._start_prediction)
        self.run.stop_btn.clicked.connect(self._stop_prediction)

    # -------------------------------------------------------------------------
    # Tile estimate helper
    # -------------------------------------------------------------------------

    def _refresh_tile_estimate(self):
        meta      = self.model.get_metadata()
        img_size  = meta.get("img_size")
        rw, rh    = self.input.get_raster_size()
        if img_size and rw and rh:
            self.settings.update_tile_estimate(img_size, rw, rh)

    # -------------------------------------------------------------------------
    # Prediction control
    # -------------------------------------------------------------------------

    def _start_prediction(self):
        config, error = self._build_config()
        if error:
            QMessageBox.warning(self, "Cannot Start Prediction", error)
            return

        from ..DL.predictor import PredictionWorker

        self._worker = PredictionWorker(config, parent=self)
        self._worker.phase_update.connect(self.run.update_phase)
        self._worker.tile_done.connect(
            lambda cur, tot: self.run.update_tile_progress(cur, tot)
        )
        self._worker.prediction_finished.connect(self._on_prediction_finished)

        self.run.set_running(True, total_tiles=0)
        self._worker.start()

    def _stop_prediction(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()

    def _on_prediction_finished(
        self, success: bool, results: object, message: str
    ):
        self.run.set_running(False)

        if not success:
            self.run.set_status(message, error=True)
            self._worker = None
            return

        output_paths  = results.get("output_paths", [])
        output_format = results.get("output_format", "")
        tile_count    = results.get("tile_count", 0)

        paths_str = "  |  ".join(output_paths)
        self.run.set_status(
            f"Done — {tile_count} tile(s) processed.   {paths_str}",
            error=False,
        )

        # Load result into QGIS if requested
        out_cfg = self.output.get_output_config()
        if out_cfg["load_into_qgis"] and output_paths:
            self._load_layers(output_paths, output_format)

        self._worker = None

    # -------------------------------------------------------------------------
    # Load output layer(s) into QGIS
    # -------------------------------------------------------------------------

    def _load_layers(self, paths: list, output_format: str):
        try:
            from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer

            for path in paths:
                ext = os.path.splitext(path)[1].lower()
                name = os.path.splitext(os.path.basename(path))[0]

                if ext in (".tif", ".tiff"):
                    layer = QgsRasterLayer(path, name)
                else:
                    layer = QgsVectorLayer(path, name, "ogr")

                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
        except Exception:
            pass  # Loading is optional — never block on failure

    # -------------------------------------------------------------------------
    # Config assembly & validation
    # -------------------------------------------------------------------------

    def _build_config(self) -> tuple:
        """
        Collects settings from all section widgets into one config dict.

        Returns
        -------
        (config, None)   on success
        (None, message)  when validation fails
        """
        # Model
        checkpoint_path = self.model.get_checkpoint_path()
        meta            = self.model.get_metadata()

        if not checkpoint_path:
            return None, "Please select a checkpoint file."
        if not meta:
            return None, (
                "No checkpoint metadata found.\n"
                "Select a valid .pth checkpoint saved by this plugin."
            )

        in_channels = meta["in_channels"]
        img_size    = meta["img_size"]

        # Input raster
        raster_path = self.input.get_raster_path()
        band_count  = self.input.get_band_count()

        if not raster_path:
            return None, "Please select an input raster file."
        if band_count is None:
            return None, "Could not read the input raster. Select a valid GeoTIFF."

        if band_count != in_channels:
            return None, (
                f"Band count mismatch:\n"
                f"  Input raster has  : {band_count} band(s)\n"
                f"  Model expects     : {in_channels} band(s)\n\n"
                f"Select a raster with {in_channels} band(s), "
                f"or choose a checkpoint trained on {band_count}-band data."
            )

        # Settings
        settings_cfg = self.settings.get_settings_config()

        # Output
        out_cfg = self.output.get_output_config()

        if not out_cfg["output_path"]:
            return None, (
                "Please set an output folder and name.\n"
                "Use the Output section to select a folder and enter a base filename."
            )

        config = {
            "checkpoint_path": checkpoint_path,
            "input_raster":    raster_path,
            "img_size":        img_size,
            "overlap_pct":     settings_cfg["overlap_pct"],
            "threshold":       settings_cfg["threshold"],
            "device":          settings_cfg["device"],
            "output_format":   out_cfg["output_format"],
            "output_path":     out_cfg["output_path"],
        }

        return config, None
