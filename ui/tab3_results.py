"""
module: ui/tab3_results.py

Results section for the Evaluate tab.

Shows after a successful evaluation run:
  - Aggregate metrics table  (IoU, F1/Dice, Precision, Recall, Pixel Accuracy)
  - Confusion matrix counts  (TP, FP, TN, FN)
  - Matplotlib confusion-matrix heatmap (optional — gracefully omitted if
    matplotlib is unavailable)
  - "Export CSV" status hint showing the saved CSV path (if output_dir was set)

The section is hidden until show_results() is called.
"""
from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QFileDialog,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor

from .expandable_groupbox import ExpandableGroupBox


# Metric rows: (display label, results key)
_METRIC_ROWS = [
    ("IoU",            "IoU"),
    ("F1 / Dice",      "F1 / Dice"),
    ("Precision",      "Precision"),
    ("Recall",         "Recall"),
    ("Pixel Accuracy", "Pixel Accuracy"),
]


class EvalResultsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._results = None   # cached for CSV export

        self.section = ExpandableGroupBox("Results")
        self.section.toggle_button.setChecked(True)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(10, 10, 10, 10)
        inner_layout.setSpacing(10)

        # --- Aggregate metrics table -----------------------------------------
        metrics_lbl = QLabel("<b>Aggregate Metrics</b>")
        inner_layout.addWidget(metrics_lbl)

        self.metrics_table = QTableWidget(len(_METRIC_ROWS), 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setSelectionMode(QTableWidget.NoSelection)
        self.metrics_table.setMaximumHeight(175)

        for row, (label, _) in enumerate(_METRIC_ROWS):
            item = QTableWidgetItem(label)
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.metrics_table.setItem(row, 0, item)
            val_item = QTableWidgetItem("—")
            val_item.setTextAlignment(Qt.AlignCenter)
            self.metrics_table.setItem(row, 1, val_item)

        inner_layout.addWidget(self.metrics_table)

        # --- Confusion matrix ------------------------------------------------
        cm_lbl = QLabel("<b>Confusion Matrix</b>")
        inner_layout.addWidget(cm_lbl)

        self._matplotlib_ok = False
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
            self._cm_figure = Figure(figsize=(3.2, 2.6))
            self._cm_canvas = FigureCanvas(self._cm_figure)
            self._cm_canvas.setMinimumHeight(200)
            self._cm_ax = self._cm_figure.add_subplot(111)
            self._draw_empty_cm()
            self._matplotlib_ok = True
            inner_layout.addWidget(self._cm_canvas)
        except Exception:
            # Fallback: plain 2×2 label grid
            self._cm_widget = self._build_cm_fallback()
            inner_layout.addWidget(self._cm_widget)

        # --- CSV path hint ---------------------------------------------------
        self.csv_hint = QLabel("")
        self.csv_hint.setWordWrap(True)
        self.csv_hint.setVisible(False)
        inner_layout.addWidget(self.csv_hint)

        # --- Total tiles label -----------------------------------------------
        self.total_lbl = QLabel("")
        self.total_lbl.setAlignment(Qt.AlignRight)
        self.total_lbl.setVisible(False)
        inner_layout.addWidget(self.total_lbl)

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(inner)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

        # Start hidden — revealed once results arrive
        self.setVisible(False)

    # -------------------------------------------------------------------------
    # Confusion matrix fallback widget (no matplotlib)
    # -------------------------------------------------------------------------

    def _build_cm_fallback(self):
        """2×2 text grid showing TP / FP / TN / FN."""
        w = QWidget()
        grid = QHBoxLayout(w)
        grid.setSpacing(6)

        for name in ("TP", "FP", "TN", "FN"):
            box = QWidget()
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(6, 6, 6, 6)
            box_layout.setSpacing(2)

            title = QLabel(f"<b>{name}</b>")
            title.setAlignment(Qt.AlignCenter)

            val = QLabel("—")
            val.setAlignment(Qt.AlignCenter)
            val.setObjectName(f"cm_{name}")

            box_layout.addWidget(title)
            box_layout.addWidget(val)
            box.setStyleSheet(
                "QWidget { border: 1px solid #bbb; border-radius: 4px; background: #f9f9f9; }"
            )
            grid.addWidget(box)

        return w

    def _update_cm_fallback(self, cm: dict):
        for name in ("TP", "FP", "TN", "FN"):
            lbl = self._cm_widget.findChild(QLabel, f"cm_{name}")
            if lbl:
                lbl.setText(f"{cm[name]:,}")

    # -------------------------------------------------------------------------
    # Matplotlib confusion matrix
    # -------------------------------------------------------------------------

    def _draw_empty_cm(self):
        import numpy as np
        ax = self._cm_ax
        ax.clear()
        data = np.zeros((2, 2))
        ax.imshow(data, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Act. 0", "Act. 1"], fontsize=8)
        ax.set_title("Confusion Matrix", fontsize=9)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, "—", ha="center", va="center", fontsize=8)
        self._cm_figure.tight_layout()
        self._cm_canvas.draw()

    def _draw_cm(self, cm: dict):
        import numpy as np
        # Layout: rows = Actual (0=neg,1=pos), cols = Predicted (0=neg,1=pos)
        data = np.array([
            [cm["TN"], cm["FP"]],
            [cm["FN"], cm["TP"]],
        ], dtype=float)

        ax = self._cm_ax
        ax.clear()

        # Normalise for colour scale but show raw counts as text
        total = data.sum() or 1
        ax.imshow(data / total, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Act. 0", "Act. 1"], fontsize=8)
        ax.set_title("Confusion Matrix", fontsize=9)

        labels = [["TN", "FP"], ["FN", "TP"]]
        for r in range(2):
            for c in range(2):
                val = int(data[r, c])
                txt = f"{labels[r][c]}\n{val:,}"
                color = "white" if data[r, c] / total > 0.5 else "black"
                ax.text(c, r, txt, ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

        self._cm_figure.tight_layout()
        self._cm_canvas.draw()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def show_results(self, results: dict):
        """Populates all result widgets from the EvaluationWorker results dict."""
        self._results = results

        metrics = results.get("metrics", {})
        for row, (_, key) in enumerate(_METRIC_ROWS):
            val = metrics.get(key)
            text = f"{val:.4f}" if val is not None else "—"
            item = self.metrics_table.item(row, 1)
            if item:
                item.setText(text)
                # Colour IoU and F1 green/amber/red by threshold
                if key in ("IoU", "F1 / Dice") and val is not None:
                    if val >= 0.7:
                        item.setBackground(QColor("#d4edda"))   # green
                    elif val >= 0.5:
                        item.setBackground(QColor("#fff3cd"))   # amber
                    else:
                        item.setBackground(QColor("#f8d7da"))   # red

        cm = results.get("confusion", {})
        if self._matplotlib_ok:
            self._draw_cm(cm)
        else:
            self._update_cm_fallback(cm)

        csv_path = results.get("csv_path")
        if csv_path:
            self.csv_hint.setText(
                f"<span style='color:green'>CSV saved: {csv_path}</span>"
            )
            self.csv_hint.setVisible(True)
        else:
            self.csv_hint.setVisible(False)

        total = results.get("total_tiles", 0)
        split = results.get("split", "")
        self.total_lbl.setText(
            f"Evaluated {total} tile{'s' if total != 1 else ''} "
            f"from the <b>{split}</b> split."
        )
        self.total_lbl.setVisible(True)

        self.setVisible(True)

    def reset(self):
        """Hides the section and clears all data."""
        self._results = None

        for row in range(self.metrics_table.rowCount()):
            item = self.metrics_table.item(row, 1)
            if item:
                item.setText("—")
                item.setBackground(QColor("#ffffff"))

        if self._matplotlib_ok:
            self._draw_empty_cm()
        else:
            self._update_cm_fallback({"TP": 0, "FP": 0, "TN": 0, "FN": 0})

        self.csv_hint.setVisible(False)
        self.total_lbl.setVisible(False)
        self.setVisible(False)
