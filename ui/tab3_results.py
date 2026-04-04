"""
module: ui/tab3_results.py

Results section for the Evaluate tab.

Shows after a successful evaluation run:
  - Aggregate metrics table  (IoU, F1/Dice, Precision, Recall, Pixel Accuracy)
  - Compact TP / FP / TN / FN counts as a single text line
  - CSV path hint (if output_dir was set)

The section is hidden until show_results() is called.
"""
from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor

from .expandable_groupbox import ExpandableGroupBox


# Metric rows: (display label, results key)
_METRIC_ROWS = [
    ("IoU", "IoU"),
    ("F1 / Dice", "F1 / Dice"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("Pixel Accuracy", "Pixel Accuracy"),
]


class EvalResultsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Results")
        self.section.toggle_button.setChecked(True)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(10, 10, 10, 10)
        inner_layout.setSpacing(10)

        # --- Aggregate metrics table -----------------------------------------
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

        # --- Compact confusion counts ----------------------------------------
        self.cm_lbl = QLabel("")
        self.cm_lbl.setAlignment(Qt.AlignCenter)
        self.cm_lbl.setVisible(False)
        inner_layout.addWidget(self.cm_lbl)

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
    # Public API
    # -------------------------------------------------------------------------

    def show_results(self, results: dict):
        """Populates all result widgets from the EvaluationWorker results dict."""
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
        if cm:
            self.cm_lbl.setText(
                f"TP: {cm.get('TP', 0):,}   |   "
                f"FP: {cm.get('FP', 0):,}   |   "
                f"TN: {cm.get('TN', 0):,}   |   "
                f"FN: {cm.get('FN', 0):,}"
            )
            self.cm_lbl.setVisible(True)

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
        for row in range(self.metrics_table.rowCount()):
            item = self.metrics_table.item(row, 1)
            if item:
                item.setText("—")
                item.setBackground(QColor("#ffffff"))

        self.cm_lbl.setVisible(False)
        self.csv_hint.setVisible(False)
        self.total_lbl.setVisible(False)
        self.setVisible(False)
