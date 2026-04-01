"""
module: ui/tab4_postprocess.py

Post-Processing section for the Predict tab.

Appears below the Run section and is disabled until a prediction run
produces a vector output (or the user manually browses to an existing
vector file).  Runs independently of prediction — the user inspects the
raw result in QGIS first, then tunes the parameters here and clicks Apply.

Operations (each individually enabled by a checkbox):
  1. Merge touching / overlapping polygons
  2. Fill holes
  3. Remove small polygons  (min area threshold)
  4. Remove large polygons  (max area threshold)
  5. Simplify               (Douglas-Peucker tolerance)
  6. Smooth edges           (Chaikin corner-cutting iterations)

Output is always a GeoPackage saved alongside the input file.
"""
import os

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QProgressBar, QLabel, QLineEdit, QFileDialog,
    QCheckBox, QDoubleSpinBox, QSpinBox, QSizePolicy,
    QFrame,
)
from qgis.PyQt.QtCore import Qt

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .styles import style_primary_btn, style_danger_btn, style_icon_btn, style_progress_bar


class PostProcessWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Post-Processing")
        self.section.toggle_button.setChecked(True)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(10, 10, 10, 10)
        inner_layout.setSpacing(10)

        # ── Input vector ─────────────────────────────────────────────────────
        input_content = SectionContentWidget()
        input_form    = input_content.layout()

        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(4)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText(
            "Auto-filled after prediction, or browse to an existing vector …"
        )
        self.input_edit.setReadOnly(True)
        file_row.addWidget(self.input_edit)

        self.input_browse_btn = QPushButton("…")
        self.input_browse_btn.setFixedWidth(30)
        style_icon_btn(self.input_browse_btn)
        self.input_browse_btn.setToolTip(
            "Select an existing predicted vector file (GeoPackage or Shapefile)."
        )
        file_row.addWidget(self.input_browse_btn)

        input_form.addRow("Input Vector", file_row)

        # Output name (saved in same folder as input)
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("e.g. prediction_postprocessed")
        self.output_name_edit.setToolTip(
            "Base name for the output GeoPackage.\n"
            "Saved in the same folder as the input vector.\n"
            "Example:  solar_panels_clean  →  solar_panels_clean.gpkg"
        )
        input_form.addRow("Output Name", self.output_name_edit)

        inner_layout.addWidget(input_content)

        # ── Separator ────────────────────────────────────────────────────────
        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        inner_layout.addWidget(sep1)

        # ── Operations ───────────────────────────────────────────────────────
        ops_content = SectionContentWidget()
        ops_form    = ops_content.layout()

        # 1. Merge touching
        self.merge_check = QCheckBox("Merge touching / overlapping polygons")
        self.merge_check.setToolTip(
            "Dissolves adjacent or overlapping polygons into single features.\n"
            "Run this first so area thresholds apply to the merged shapes."
        )
        ops_form.addRow(self.merge_check)

        # 2. Fill holes
        self._add_sep(ops_form)
        self.fill_check = QCheckBox("Fill holes")
        self.fill_check.setToolTip(
            "Removes interior rings (holes) from polygons.\n"
            "Set Max hole area to 0 to fill every hole regardless of size."
        )
        ops_form.addRow(self.fill_check)

        self.fill_spin = QDoubleSpinBox()
        self.fill_spin.setRange(0, 1_000_000)
        self.fill_spin.setValue(0)
        self.fill_spin.setDecimals(1)
        self.fill_spin.setSuffix(" map units²")
        self.fill_spin.setToolTip(
            "Holes smaller than this area are removed.\n"
            "0 = fill all holes regardless of size."
        )
        self.fill_spin.setEnabled(False)
        ops_form.addRow("  Max hole area", self.fill_spin)

        # 3. Remove small polygons
        self._add_sep(ops_form)
        self.min_area_check = QCheckBox("Remove small polygons")
        self.min_area_check.setToolTip(
            "Discards polygons whose area is below the threshold.\n"
            "Effective for removing noise and false positives."
        )
        ops_form.addRow(self.min_area_check)

        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(0.1, 1_000_000)
        self.min_area_spin.setValue(10.0)
        self.min_area_spin.setDecimals(1)
        self.min_area_spin.setSuffix(" map units²")
        self.min_area_spin.setToolTip(
            "Polygons with area smaller than this value are removed."
        )
        self.min_area_spin.setEnabled(False)
        ops_form.addRow("  Min area", self.min_area_spin)

        # 4. Remove large polygons
        self._add_sep(ops_form)
        self.max_area_check = QCheckBox("Remove large polygons")
        self.max_area_check.setToolTip(
            "Discards polygons whose area exceeds the threshold.\n"
            "Useful for removing large false-positive detections."
        )
        ops_form.addRow(self.max_area_check)

        self.max_area_spin = QDoubleSpinBox()
        self.max_area_spin.setRange(0.1, 1_000_000_000)
        self.max_area_spin.setValue(10_000.0)
        self.max_area_spin.setDecimals(1)
        self.max_area_spin.setSuffix(" map units²")
        self.max_area_spin.setToolTip(
            "Polygons with area larger than this value are removed."
        )
        self.max_area_spin.setEnabled(False)
        ops_form.addRow("  Max area", self.max_area_spin)

        # 5. Simplify
        self._add_sep(ops_form)
        self.simplify_check = QCheckBox("Simplify geometries")
        self.simplify_check.setToolTip(
            "Applies Douglas-Peucker simplification to reduce vertex count.\n"
            "Preserves topology — polygons will not self-intersect."
        )
        ops_form.addRow(self.simplify_check)

        self.simplify_spin = QDoubleSpinBox()
        self.simplify_spin.setRange(0.01, 1_000)
        self.simplify_spin.setValue(0.5)
        self.simplify_spin.setDecimals(2)
        self.simplify_spin.setSuffix(" map units")
        self.simplify_spin.setToolTip(
            "Maximum allowed deviation from the original geometry.\n"
            "Larger values = more simplification, fewer vertices.\n"
            "Typical starting point: ~0.5 × pixel size."
        )
        self.simplify_spin.setEnabled(False)
        ops_form.addRow("  Tolerance", self.simplify_spin)

        # 6. Smooth edges
        self._add_sep(ops_form)
        self.smooth_check = QCheckBox("Smooth edges")
        self.smooth_check.setToolTip(
            "Applies Chaikin's corner-cutting algorithm to soften jagged,\n"
            "pixelated edges. Each iteration doubles the vertex count and\n"
            "rounds corners. 2–4 iterations is usually sufficient."
        )
        ops_form.addRow(self.smooth_check)

        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(1, 8)
        self.smooth_spin.setValue(3)
        self.smooth_spin.setToolTip(
            "Number of Chaikin smoothing iterations.\n"
            "More iterations = smoother edges but more vertices.\n"
            "Recommended: 2–4."
        )
        self.smooth_spin.setEnabled(False)
        ops_form.addRow("  Iterations", self.smooth_spin)

        inner_layout.addWidget(ops_content)

        # ── Separator ────────────────────────────────────────────────────────
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        inner_layout.addWidget(sep2)

        # ── Apply / Stop buttons ──────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.apply_btn = QPushButton("Apply Post-Processing")
        self.apply_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        style_primary_btn(self.apply_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.stop_btn.setEnabled(False)
        style_danger_btn(self.stop_btn)

        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.stop_btn)
        inner_layout.addLayout(btn_row)

        # ── Phase label + progress bar + status ───────────────────────────────
        self.phase_label = QLabel("")
        self.phase_label.setAlignment(Qt.AlignCenter)
        self.phase_label.setVisible(False)
        inner_layout.addWidget(self.phase_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Feature %v / %m")
        self.progress_bar.setVisible(False)
        style_progress_bar(self.progress_bar)
        inner_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        inner_layout.addWidget(self.status_label)

        # ── Assemble section ──────────────────────────────────────────────────
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(inner)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

        # ── Connections ───────────────────────────────────────────────────────
        self.input_browse_btn.clicked.connect(self._browse_input)

        # Enable parameter spinboxes only when their checkbox is ticked
        self.fill_check.toggled.connect(self.fill_spin.setEnabled)
        self.min_area_check.toggled.connect(self.min_area_spin.setEnabled)
        self.max_area_check.toggled.connect(self.max_area_spin.setEnabled)
        self.simplify_check.toggled.connect(self.simplify_spin.setEnabled)
        self.smooth_check.toggled.connect(self.smooth_spin.setEnabled)

        # Start disabled — enabled when a vector input is available
        self.setEnabled(False)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _add_sep(form):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setStyleSheet("color: #e0e0e0;")
        form.addRow(sep)

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Predicted Vector File", "",
            "GeoPackage (*.gpkg);;Shapefile (*.shp);;All files (*.*)",
        )
        if path:
            self.set_input_path(path)

    def _default_output_name(self, input_path: str) -> str:
        base = os.path.splitext(os.path.basename(input_path))[0]
        return base + "_postprocessed"

    # -------------------------------------------------------------------------
    # Public API — called by Tab4Widget
    # -------------------------------------------------------------------------

    def set_input_path(self, path: str):
        """Auto-fills the input path and derives a default output name."""
        self.input_edit.setText(path)
        if not self.output_name_edit.text().strip():
            self.output_name_edit.setText(self._default_output_name(path))
        self.setEnabled(True)

    def get_config(self) -> dict:
        """Returns the full post-processing configuration dict."""
        input_path  = self.input_edit.text().strip()
        output_name = self.output_name_edit.text().strip() or "prediction_postprocessed"
        output_dir  = os.path.dirname(input_path) if input_path else ""
        output_path = os.path.join(output_dir, output_name + ".gpkg") if output_dir else ""

        return {
            "input_path":         input_path,
            "output_path":        output_path,
            "merge_touching":     self.merge_check.isChecked(),
            "fill_holes":         self.fill_check.isChecked(),
            "min_hole_area":      self.fill_spin.value(),
            "filter_min_area":    self.min_area_check.isChecked(),
            "min_area":           self.min_area_spin.value(),
            "filter_max_area":    self.max_area_check.isChecked(),
            "max_area":           self.max_area_spin.value(),
            "simplify":           self.simplify_check.isChecked(),
            "simplify_tolerance": self.simplify_spin.value(),
            "smooth":             self.smooth_check.isChecked(),
            "smooth_iterations":  self.smooth_spin.value(),
        }

    # -------------------------------------------------------------------------
    # Public API — UI state control (called by Tab4Widget)
    # -------------------------------------------------------------------------

    def set_running(self, running: bool, total: int = 0):
        self.apply_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(0)
            self.phase_label.setText("Starting…")
            self.phase_label.setVisible(True)
            self.status_label.setVisible(False)
        else:
            self.phase_label.setVisible(False)

    def update_phase(self, message: str):
        self.phase_label.setText(message)
        self.phase_label.setVisible(True)
        # Hide progress bar for non-per-feature phases
        if "Smoothing" not in message:
            self.progress_bar.setVisible(False)

    def update_progress(self, current: int, total: int):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.progress_bar.setVisible(True)

    def set_status(self, message: str, error: bool = False):
        color = "red" if error else "green"
        self.status_label.setText(
            f"<span style='color:{color}'>{message}</span>"
        )
        self.status_label.setVisible(True)
        self.progress_bar.setVisible(False)
        self.phase_label.setVisible(False)
