"""
module: ui/tab4.py

Predict tab — orchestrates all prediction-related sections and wires the
background PredictionWorker and PostProcessWorker to their widgets.

Sections (top to bottom)
------------------------
PredictModelWidget    — checkpoint file picker + metadata display
PredictInputWidget    — input raster picker (QGIS layer or file)
PredictSettingsWidget — device, overlap %, threshold, estimated tile count
PredictOutputWidget   — output format, folder, name, load-into-QGIS option
PredictRunWidget      — run/stop, tile progress bar, status
PostProcessWidget     — post-processing operations (enabled after prediction)
"""
import os

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QSizePolicy, QMessageBox,
)

from .tab4_model import PredictModelWidget
from .tab4_input import PredictInputWidget
from .tab4_settings import PredictSettingsWidget
from .tab4_output import PredictOutputWidget
from .tab4_run import PredictRunWidget
from .tab4_postprocess import PostProcessWidget


class Tab4Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.model = PredictModelWidget()
        self.input = PredictInputWidget()
        self.settings = PredictSettingsWidget()
        self.output = PredictOutputWidget()
        self.run = PredictRunWidget()
        self.postprocess = PostProcessWidget()

        self._worker = None   # PredictionWorker while running
        self._worker_pp = None   # PostProcessWorker while running

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
        content_layout.addWidget(self.postprocess)
        content_layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # --- Connections — keep tile estimate up to date ---------------------
        self.model.browse_btn.clicked.connect(self._refresh_tile_estimate)
        self.input.browse_btn.clicked.connect(self._refresh_tile_estimate)

        # Prediction run / stop
        self.run.run_btn.clicked.connect(self._start_prediction)
        self.run.stop_btn.clicked.connect(self._stop_prediction)

        # Post-processing apply / stop
        self.postprocess.apply_btn.clicked.connect(self._start_postprocess)
        self.postprocess.stop_btn.clicked.connect(self._stop_postprocess)

    # -------------------------------------------------------------------------
    # Tile estimate helper
    # -------------------------------------------------------------------------

    def _refresh_tile_estimate(self):
        meta = self.model.get_metadata()
        img_size = meta.get("img_size")
        rw, rh = self.input.get_raster_size()
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

        output_paths = results.get("output_paths", [])
        output_format = results.get("output_format", "")
        tile_count = results.get("tile_count", 0)

        paths_str = "  |  ".join(output_paths)
        self.run.set_status(
            f"Done — {tile_count} tile(s) processed.   {paths_str}",
            error=False,
        )

        # Load result into QGIS if requested
        out_cfg = self.output.get_output_config()
        if out_cfg["load_into_qgis"] and output_paths:
            self._load_layers(output_paths, output_format)

        # Auto-fill input path if a vector output was produced
        vector_path = next(
            (p for p in output_paths if p.endswith((".gpkg", ".shp"))),
            None,
        )
        if vector_path:
            self.postprocess.set_input_path(vector_path)

        self._worker = None

    # -------------------------------------------------------------------------
    # Post-processing control
    # -------------------------------------------------------------------------

    def _start_postprocess(self):
        pp_cfg, error = self._build_postprocess_config()
        if error:
            QMessageBox.warning(self, "Cannot Start Post-Processing", error)
            return

        from ..DL.postprocessor import PostProcessWorker

        self._worker_pp = PostProcessWorker(pp_cfg, parent=self)
        self._worker_pp.phase_update.connect(self.postprocess.update_phase)
        self._worker_pp.feature_done.connect(
            lambda cur, tot: self.postprocess.update_progress(cur, tot)
        )
        self._worker_pp.postprocess_finished.connect(
            self._on_postprocess_finished
        )

        self.postprocess.set_running(True)
        self._worker_pp.start()

    def _stop_postprocess(self):
        if self._worker_pp and self._worker_pp.isRunning():
            self._worker_pp.stop()

    def _on_postprocess_finished(
        self, success: bool, results: object, message: str
    ):
        self.postprocess.set_running(False)

        if not success:
            self.postprocess.set_status(message, error=True)
            self._worker_pp = None
            return

        output_path = results.get("output_path", "")
        input_count = results.get("input_count", 0)
        output_count = results.get("output_count", 0)

        self.postprocess.set_status(
            f"Done — {input_count} → {output_count} feature(s).   {output_path}",
            error=False,
        )

        # Load into QGIS
        if output_path:
            self._load_layers([output_path], "vector")

        self._worker_pp = None

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
            pass

    # -------------------------------------------------------------------------
    # Config assembly & validation
    # -------------------------------------------------------------------------

    def _build_config(self) -> tuple:
        # Model
        checkpoint_path = self.model.get_checkpoint_path()
        meta = self.model.get_metadata()

        if not checkpoint_path:
            return None, "Please select a checkpoint file."
        if not meta:
            return None, (
                "No checkpoint metadata found.\n"
                "Select a valid .pth checkpoint saved by this plugin."
            )

        in_channels = meta["in_channels"]
        img_size = meta["img_size"]

        # Input raster
        raster_path = self.input.get_raster_path()
        band_count = self.input.get_band_count()

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
            return None, ("Please set an output folder and name.\n"
                          "Use the Output section to select a folder and enter a base filename.")

        return {
            "checkpoint_path": checkpoint_path,
            "input_raster": raster_path,
            "img_size": img_size,
            "overlap_pct": settings_cfg["overlap_pct"],
            "threshold": settings_cfg["threshold"],
            "device": settings_cfg["device"],
            "output_format": out_cfg["output_format"],
            "output_path": out_cfg["output_path"],
        }, None

    def _build_postprocess_config(self) -> tuple:
        cfg = self.postprocess.get_config()

        if not cfg["input_path"]:
            return None, "No input vector selected.\nRun a prediction first, or browse to an existing vector file."

        if not os.path.isfile(cfg["input_path"]):
            return None, (
                f"Input vector not found:\n{cfg['input_path']}\n\n"
                "Run a prediction first, or browse to an existing vector file."
            )

        if not cfg["output_path"]:
            return None, "Could not determine output path. Check the Output Name field."

        # At least one operation must be enabled
        ops = ("merge_touching", "fill_holes", "filter_min_area",
               "filter_max_area", "simplify", "smooth")
        if not any(cfg.get(op) for op in ops):
            return None, (
                "No operations are enabled.\n"
                "Tick at least one operation checkbox before applying."
            )

        return cfg, None
