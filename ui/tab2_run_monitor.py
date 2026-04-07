"""
module: tab2_run_monitor.py

Run & Monitor section for the Train tab.

Always visible (non-collapsible). Contains Start / Stop controls,
an epoch progress bar, a live-updating metrics table, and a status label.

The training backend will call the public update methods on this widget
via Qt signals once the worker thread is wired up.
"""
import csv
import os
import time

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QProgressBar, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy,
)
from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtGui import QColor

from .expandable_groupbox import ExpandableGroupBox
from .styles import style_primary_btn, style_danger_btn, style_progress_bar

_CSV_HEADERS = ["Epoch", "Train Loss", "Val Loss", "Val IoU", "Val F1"]


# Metrics table column definitions: (header label, alignment)
_COLUMNS = [
    ("Epoch", Qt.AlignCenter),
    ("Train Loss", Qt.AlignCenter),
    ("Val Loss", Qt.AlignCenter),
    ("Val IoU", Qt.AlignCenter),
    ("Val F1", Qt.AlignCenter),
]


class RunMonitorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._csv_path = None   # set by set_output_paths() before each run

        # Non-collapsible section — toggle button disabled like InsAndOuts
        self.section = ExpandableGroupBox("Run & Monitor")
        self.section.toggle_button.setChecked(True)
        self.section.toggle_button.setEnabled(False)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(10, 10, 10, 10)
        inner_layout.setSpacing(8)

        # --- Start / Stop buttons --------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.start_btn = QPushButton("Start Training")
        style_primary_btn(self.start_btn)
        self.start_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.stop_btn.setEnabled(False)
        style_danger_btn(self.stop_btn)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        inner_layout.addLayout(btn_row)

        # --- Phase status label ----------------------------------------------
        self.phase_label = QLabel("")
        self.phase_label.setAlignment(Qt.AlignCenter)
        self.phase_label.setVisible(False)
        inner_layout.addWidget(self.phase_label)

        # --- Timer label -----------------------------------------------------
        self.timer_label = QLabel("")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setVisible(False)
        inner_layout.addWidget(self.timer_label)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._update_timer)
        self._start_time = None

        # --- Epoch progress bar ----------------------------------------------
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Epoch %v / %m")
        self.progress_bar.setVisible(False)
        style_progress_bar(self.progress_bar)
        inner_layout.addWidget(self.progress_bar)

        # --- Batch progress bar (within-epoch) -------------------------------
        self.batch_bar = QProgressBar()
        self.batch_bar.setRange(0, 100)
        self.batch_bar.setValue(0)
        self.batch_bar.setFormat("Train — Batch %v / %m")
        self.batch_bar.setVisible(False)
        style_progress_bar(self.batch_bar)
        inner_layout.addWidget(self.batch_bar)

        # --- Status label ----------------------------------------------------
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        inner_layout.addWidget(self.status_label)

        # --- Metrics table ---------------------------------------------------
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels([c[0] for c in _COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setMinimumHeight(160)
        self.table.setVisible(False)
        inner_layout.addWidget(self.table)

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(inner)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)
        self.setLayout(layout)

    # -------------------------------------------------------------------------
    # Public API — called by the training worker
    # -------------------------------------------------------------------------

    def set_running(self, running: bool, total_epochs: int = 0):
        """Switches the UI between idle and running states."""
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setRange(0, total_epochs)
            self.progress_bar.setValue(0)
            self.batch_bar.setValue(0)
            self.batch_bar.setVisible(True)
            self.phase_label.setText("Starting...")
            self.phase_label.setVisible(True)
            self.status_label.setVisible(False)
            self._start_time = time.time()
            self.timer_label.setText("Elapsed: 0:00:00")
            self.timer_label.setVisible(True)
            self._timer.start()
        else:
            self.phase_label.setVisible(False)
            self.batch_bar.setVisible(False)
            self._timer.stop()

    def add_epoch_row(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_iou: float,
        val_f1: float,
    ):
        """Appends one row to the metrics table and scrolls to it."""
        self.table.setVisible(True)
        row = self.table.rowCount()
        self.table.insertRow(row)

        values = [epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                  f"{val_iou:.4f}", f"{val_f1:.4f}"]

        for col, (value, (_, alignment)) in enumerate(zip(values, _COLUMNS)):
            item = QTableWidgetItem(str(value))
            item.setTextAlignment(alignment)
            self.table.setItem(row, col, item)

        # Highlight the best val IoU row in light green
        self._highlight_best_iou()

        self.progress_bar.setValue(epoch)
        self.batch_bar.setValue(0)   # reset batch bar at start of next epoch
        self.table.scrollToBottom()

        # Append row to CSV log
        self._write_csv_row(epoch, train_loss, val_loss, val_iou, val_f1)

    def set_status(self, message: str, error: bool = False):
        """Shows a status message below the progress bar."""
        self._timer.stop()
        elapsed = self._elapsed_str()
        color = "red" if error else "green"
        suffix = f"  ({elapsed})" if elapsed else ""
        self.status_label.setText(
            f"<span style='color:{color}'>{message}{suffix}</span>"
        )
        self.status_label.setVisible(True)
        self.progress_bar.setVisible(False)
        self.batch_bar.setVisible(False)
        self.phase_label.setVisible(False)
        self.timer_label.setVisible(False)

    def set_output_paths(self, output_dir: str, model_name: str):
        """
        Called before training starts to set up the CSV log file.
        Creates the file and writes the header row.
        """
        if not output_dir:
            self._csv_path = None
            return
        os.makedirs(output_dir, exist_ok=True)
        self._csv_path = os.path.join(
            output_dir, f"{model_name}_training_log.csv")
        try:
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(_CSV_HEADERS)
        except Exception:
            self._csv_path = None

    def update_phase(self, message: str):
        """Updates the phase status label (e.g. 'Building model...')."""
        self.phase_label.setText(message)
        self.phase_label.setVisible(True)

    def update_batch_progress(self, current: int, total: int, phase: str):
        """Updates the within-epoch batch progress bar."""
        self.batch_bar.setRange(0, total)
        self.batch_bar.setValue(current)
        self.batch_bar.setFormat(f"{phase} — Batch %v / %m")

    def reset_monitor(self):
        """Clears the table and resets all monitor widgets to idle state."""
        self._timer.stop()
        self._start_time = None
        self.table.setRowCount(0)
        self.table.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.batch_bar.setValue(0)
        self.batch_bar.setVisible(False)
        self.phase_label.setText("")
        self.phase_label.setVisible(False)
        self.timer_label.setText("")
        self.timer_label.setVisible(False)
        self.status_label.setVisible(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _write_csv_row(self, epoch, train_loss, val_loss, val_iou, val_f1):
        """Appends one epoch row to the CSV log file."""
        if not self._csv_path:
            return
        try:
            with open(self._csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_iou:.6f}",
                    f"{val_f1:.6f}",
                ])
        except Exception:
            pass

    def _update_timer(self):
        """Called every second by QTimer to update the elapsed time label."""
        self.timer_label.setText(f"Elapsed: {self._elapsed_str()}")

    def _elapsed_str(self) -> str:
        """Returns elapsed time as H:MM:SS string, or empty if not started."""
        if self._start_time is None:
            return ""
        elapsed = int(time.time() - self._start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        return f"{h}:{m:02d}:{s:02d}"

    def _highlight_best_iou(self):
        """Colours the row with the highest Val IoU light green."""
        iou_col = 3   # Val IoU column index
        best_row = -1
        best_val = -1.0

        for row in range(self.table.rowCount()):
            item = self.table.item(row, iou_col)
            if item:
                try:
                    val = float(item.text())
                    if val > best_val:
                        best_val = val
                        best_row = row
                except ValueError:
                    pass

        highlight = QColor("#d4edda")   # light green
        default = QColor("#ffffff")   # white

        for row in range(self.table.rowCount()):
            colour = highlight if row == best_row else default
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item:
                    item.setBackground(colour)
