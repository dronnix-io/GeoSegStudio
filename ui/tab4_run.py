"""
module: ui/tab4_run.py

Run & Monitor section for the Predict tab.

Always visible (non-collapsible).  Contains:
  - Run / Stop buttons
  - Tile progress bar
  - Phase status label
  - Final status label (success / error) with clickable output path(s)
"""
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QProgressBar, QLabel, QSizePolicy,
)
from qgis.PyQt.QtCore import Qt

from .expandable_groupbox import ExpandableGroupBox
from .styles import style_primary_btn, style_danger_btn, style_progress_bar


class PredictRunWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Non-collapsible section
        self.section = ExpandableGroupBox("Run")
        self.section.toggle_button.setChecked(True)
        self.section.toggle_button.setEnabled(False)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(10, 10, 10, 10)
        inner_layout.setSpacing(8)

        # --- Run / Stop buttons ----------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.run_btn = QPushButton("Run Prediction")
        self.run_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        style_primary_btn(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.stop_btn.setEnabled(False)
        style_danger_btn(self.stop_btn)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        inner_layout.addLayout(btn_row)

        # --- Phase label -----------------------------------------------------
        self.phase_label = QLabel("")
        self.phase_label.setAlignment(Qt.AlignCenter)
        self.phase_label.setVisible(False)
        inner_layout.addWidget(self.phase_label)

        # --- Tile progress bar -----------------------------------------------
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Tile %v / %m")
        self.progress_bar.setVisible(False)
        style_progress_bar(self.progress_bar)
        inner_layout.addWidget(self.progress_bar)

        # --- Status label ----------------------------------------------------
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        inner_layout.addWidget(self.status_label)

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(inner)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_running(self, running: bool, total_tiles: int = 0):
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setRange(0, total_tiles)
            self.progress_bar.setValue(0)
            self.phase_label.setText("Starting…")
            self.phase_label.setVisible(True)
            self.status_label.setVisible(False)
        else:
            self.phase_label.setVisible(False)

    def update_phase(self, message: str):
        self.phase_label.setText(message)
        self.phase_label.setVisible(True)

    def update_tile_progress(self, current: int, total: int):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)

    def set_status(self, message: str, error: bool = False):
        color = "red" if error else "green"
        self.status_label.setText(
            f"<span style='color:{color}'>{message}</span>"
        )
        self.status_label.setVisible(True)
        self.progress_bar.setVisible(False)
        self.phase_label.setVisible(False)

    def reset(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.phase_label.setText("")
        self.phase_label.setVisible(False)
        self.status_label.setVisible(False)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
