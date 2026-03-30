"""
module: tab1.py
"""
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QSizePolicy,
    QPushButton, QMessageBox
)
from qgis.PyQt.QtCore import QCoreApplication

from .tab1_ins_outs import InsAndOutsWidget
from .tab1_clipping import ClippingWidget
from .tab1_splitting import SplittingWidget
from .tab1_augmentation import AugmentationWidget

from ..DL.data_preparation import (
    ValidationError,
    run_clipping,
    run_splitting,
    run_augmentation,
    run_all,
    get_dataset_dir,
    get_clipping_versions,
    get_splitting_versions,
    version_label,
)


class Tab1Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ins_outs    = InsAndOutsWidget()
        self.clipping    = ClippingWidget()
        self.splitting   = SplittingWidget()
        self.augmentation = AugmentationWidget()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)

        content_layout.addWidget(self.ins_outs)
        content_layout.addWidget(self.clipping)
        content_layout.addWidget(self.splitting)
        content_layout.addWidget(self.augmentation)

        # Run All button
        self.run_all_btn = QPushButton("Run All")
        self.run_all_btn.setToolTip(
            "Runs Clipping → Splitting → Augmentation in sequence "
            "using the current settings."
        )
        self.run_all_btn.clicked.connect(self._on_run_all)
        content_layout.addWidget(self.run_all_btn)

        content_layout.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # Wire individual Apply buttons
        self.clipping.apply_btn.clicked.connect(self._on_apply_clipping)
        self.splitting.apply_btn.clicked.connect(self._on_apply_splitting)
        self.augmentation.apply_btn.clicked.connect(self._on_apply_augmentation)

        # Wire refresh buttons on version selectors
        self.splitting.refresh_btn.clicked.connect(self._refresh_clipping_versions)
        self.augmentation.refresh_btn.clicked.connect(self._refresh_splitting_versions)

    # -------------------------------------------------------------------------
    # Config builders
    # -------------------------------------------------------------------------

    def _clipping_config(self) -> dict:
        raster_id, vector_id, output_dir = self.ins_outs.get_selected_inputs()
        return {
            "raster_id":  raster_id,
            "vector_id":  vector_id,
            "output_dir": output_dir,
            "clip_params": self.clipping.get_clipping_params(),
        }

    def _splitting_config(self) -> dict:
        _, _, output_dir = self.ins_outs.get_selected_inputs()
        return {
            "output_dir":        output_dir,
            "prefix":            self._prefix(),
            "split_percentages": self.splitting.get_split_percentages(),
            "clipping_version":  self.splitting.get_selected_clipping_version(),
            "cpu_count":         self.clipping.get_clipping_params()["cpu_count"],
        }

    def _augmentation_config(self) -> dict:
        _, _, output_dir = self.ins_outs.get_selected_inputs()
        return {
            "output_dir":        output_dir,
            "prefix":            self._prefix(),
            "augmentations":     self.augmentation.selected_methods(),
            "splitting_version": self.augmentation.get_selected_splitting_version(),
            "cpu_count":         self.clipping.get_clipping_params()["cpu_count"],
        }

    def _prefix(self) -> str:
        """Derives the dataset folder prefix from the selected raster layer name,
        using the same sanitisation logic as clipper.py."""
        from qgis.core import QgsProject
        raster_id, _, _ = self.ins_outs.get_selected_inputs()
        layer = QgsProject.instance().mapLayer(raster_id) if raster_id else None
        name  = layer.name() if layer else ""
        sanitised = "".join(
            c if c.isalnum() or c in "_-" else "_"
            for c in name
        ).strip("_")
        return sanitised or "dataset"

    # -------------------------------------------------------------------------
    # Version selector refresh helpers
    # -------------------------------------------------------------------------

    def _dataset_dir(self) -> str:
        _, _, output_dir = self.ins_outs.get_selected_inputs()
        return get_dataset_dir(output_dir, self._prefix())

    def _refresh_clipping_versions(self):
        """Re-scans disk and repopulates the Splitting version selector."""
        try:
            dataset_dir = self._dataset_dir()
            versions    = get_clipping_versions(dataset_dir)
            labelled    = [
                {"version": v["version"], "info": v["info"],
                 "label": version_label(v)}
                for v in versions
            ]
            self.splitting.populate_clipping_versions(labelled)
        except Exception as e:
            self.splitting.set_status(f"Refresh failed: {e}", error=True)

    def _refresh_splitting_versions(self):
        """Re-scans disk and repopulates the Augmentation version selector."""
        try:
            dataset_dir = self._dataset_dir()
            versions    = get_splitting_versions(dataset_dir)
            labelled    = [
                {"version": v["version"], "info": v["info"],
                 "label": version_label(v)}
                for v in versions
            ]
            self.augmentation.populate_splitting_versions(labelled)
        except Exception as e:
            self.augmentation.set_status(f"Refresh failed: {e}", error=True)

    # -------------------------------------------------------------------------
    # Apply Clipping
    # -------------------------------------------------------------------------

    def _on_apply_clipping(self):
        config = self._clipping_config()
        self.clipping.set_running(True)

        def on_progress(pct):
            self.clipping.progress_bar.setValue(pct)
            QCoreApplication.processEvents()

        try:
            result   = run_clipping(config, progress_callback=on_progress)
            warnings = result.get("warnings", [])

            if warnings:
                QMessageBox.warning(self, "Clipping Warnings",
                                    "\n\n".join(warnings))

            self.clipping.set_status(
                f"✓ v{result['version']} — "
                f"{result['tile_count']} tiles saved, "
                f"{result['skipped_count']} skipped (empty masks)"
            )
            # Auto-refresh the splitting version selector
            self._refresh_clipping_versions()

        except ValidationError as e:
            self.clipping.set_running(False)
            QMessageBox.critical(self, "Clipping — Validation Error", str(e))
            self.clipping.set_status("Validation failed.", error=True)

        except Exception as e:
            self.clipping.set_running(False)
            QMessageBox.critical(self, "Clipping Error", str(e))
            self.clipping.set_status("Clipping failed.", error=True)

        else:
            self.clipping.set_running(False)

    # -------------------------------------------------------------------------
    # Apply Splitting
    # -------------------------------------------------------------------------

    def _on_apply_splitting(self):
        config = self._splitting_config()
        self.splitting.set_running(True)

        def on_progress(pct):
            self.splitting.progress_bar.setValue(pct)
            QCoreApplication.processEvents()

        try:
            result   = run_splitting(config, progress_callback=on_progress)
            warnings = result.get("warnings", [])

            if warnings:
                QMessageBox.warning(self, "Splitting Warnings",
                                    "\n\n".join(warnings))

            c = result["counts"]
            self.splitting.set_status(
                f"✓ v{result['version']} — "
                f"train {c['train']} / valid {c['valid']} / test {c['test']}"
            )
            # Auto-refresh the augmentation version selector
            self._refresh_splitting_versions()

        except ValidationError as e:
            self.splitting.set_running(False)
            QMessageBox.critical(self, "Splitting — Validation Error", str(e))
            self.splitting.set_status("Validation failed.", error=True)

        except Exception as e:
            self.splitting.set_running(False)
            QMessageBox.critical(self, "Splitting Error", str(e))
            self.splitting.set_status("Splitting failed.", error=True)

        else:
            self.splitting.set_running(False)

    # -------------------------------------------------------------------------
    # Apply Augmentation
    # -------------------------------------------------------------------------

    def _on_apply_augmentation(self):
        config = self._augmentation_config()
        self.augmentation.set_running(True)

        def on_progress(pct):
            self.augmentation.progress_bar.setValue(pct)
            QCoreApplication.processEvents()

        try:
            result   = run_augmentation(config, progress_callback=on_progress)
            warnings = result.get("warnings", [])

            if warnings:
                QMessageBox.warning(self, "Augmentation Warnings",
                                    "\n\n".join(warnings))

            c = result["counts"]
            self.augmentation.set_status(
                f"✓ v{result['version']} — "
                f"train {c['train']} / valid {c['valid']} / test {c['test']} tiles"
            )

        except ValidationError as e:
            self.augmentation.set_running(False)
            QMessageBox.critical(self, "Augmentation — Validation Error", str(e))
            self.augmentation.set_status("Validation failed.", error=True)

        except Exception as e:
            self.augmentation.set_running(False)
            QMessageBox.critical(self, "Augmentation Error", str(e))
            self.augmentation.set_status("Augmentation failed.", error=True)

        else:
            self.augmentation.set_running(False)

    # -------------------------------------------------------------------------
    # Run All
    # -------------------------------------------------------------------------

    def _on_run_all(self):
        config = self._clipping_config()
        config["split_percentages"] = self.splitting.get_split_percentages()
        config["augmentations"]     = self.augmentation.selected_methods()

        self.clipping.set_running(True)
        self.splitting.set_running(True)
        self.augmentation.set_running(True)
        self.run_all_btn.setEnabled(False)

        def clip_progress(pct):
            self.clipping.progress_bar.setValue(pct)
            QCoreApplication.processEvents()

        def split_progress(pct):
            self.splitting.progress_bar.setValue(pct)
            QCoreApplication.processEvents()

        def aug_progress(pct):
            self.augmentation.progress_bar.setValue(pct)
            QCoreApplication.processEvents()

        try:
            results = run_all(
                config,
                clipping_progress=clip_progress,
                splitting_progress=split_progress,
                augmentation_progress=aug_progress,
            )

            cr = results["clipping"]
            sr = results["splitting"]
            ar = results["augmentation"]
            sc = sr["counts"]
            ac = ar["counts"]

            self.clipping.set_status(
                f"✓ v{cr['version']} — {cr['tile_count']} tiles"
            )
            self.splitting.set_status(
                f"✓ v{sr['version']} — "
                f"train {sc['train']} / valid {sc['valid']} / test {sc['test']}"
            )
            self.augmentation.set_status(
                f"✓ v{ar['version']} — "
                f"train {ac['train']} / valid {ac['valid']} / test {ac['test']}"
            )

            self._refresh_clipping_versions()
            self._refresh_splitting_versions()

            QMessageBox.information(
                self, "Run All Complete",
                f"Clipping  : v{cr['version']}  ({cr['tile_count']} tiles)\n"
                f"Splitting : v{sr['version']}  "
                f"(train {sc['train']} / valid {sc['valid']} / test {sc['test']})\n"
                f"Augmented : v{ar['version']}  "
                f"(train {ac['train']} / valid {ac['valid']} / test {ac['test']})"
            )

        except ValidationError as e:
            QMessageBox.critical(self, "Run All — Validation Error", str(e))

        except Exception as e:
            QMessageBox.critical(self, "Run All Error", str(e))

        finally:
            self.clipping.set_running(False)
            self.splitting.set_running(False)
            self.augmentation.set_running(False)
            self.run_all_btn.setEnabled(True)
