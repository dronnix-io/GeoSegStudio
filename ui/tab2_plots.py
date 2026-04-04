"""
module: ui/tab2_plots.py

Live training plot widget for the Train tab.

Shows two subplots that update live after every completed epoch:
  Left  — Loss curves:    Train Loss (blue)  +  Val Loss (orange)
  Right — Metric curves:  Val IoU   (green)  +  Val F1   (pink)

A "Save Plots" button exports the figure as PNG, PDF, or SVG.
The section is collapsed by default to save vertical space; the user
can expand it at any time during or after training.
"""
from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel,
)
from qgis.PyQt.QtCore import Qt

from .expandable_groupbox import ExpandableGroupBox


class TrainingPlotWidget(QWidget):
    """
    Collapsible section that embeds a live-updating matplotlib canvas.

    Public API
    ----------
    add_epoch(epoch, train_loss, val_loss, val_iou, val_f1)
        Append one epoch's data and redraw.  Connected directly to the
        worker's epoch_done signal from Tab2Widget.
    reset()
        Clear all stored data and wipe the canvas (called before each run).
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._epochs = []
        self._train_losses = []
        self._val_losses = []
        self._val_ious = []
        self._val_f1s = []

        self.section = ExpandableGroupBox("Training Plots")
        # Collapsed by default — user opens when they want to inspect curves
        self.section.toggle_button.setChecked(False)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(6, 6, 6, 6)
        inner_layout.setSpacing(6)

        # --- Matplotlib canvas -----------------------------------------------
        self._matplotlib_ok = False
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg as FigureCanvas,
            )

            self._figure = Figure(figsize=(6, 3))
            self._canvas = FigureCanvas(self._figure)
            self._canvas.setMinimumHeight(230)
            self._ax_loss, self._ax_metrics = self._figure.subplots(1, 2)
            self._draw_empty()
            self._matplotlib_ok = True
            inner_layout.addWidget(self._canvas)

        except Exception as exc:
            lbl = QLabel(f"Plots unavailable: {exc}")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setWordWrap(True)
            inner_layout.addWidget(lbl)

        # --- Save button -----------------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.save_btn = QPushButton("Save Plots")
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip(
            "Export the training curves as an image or document.\n"
            "Supported formats: PNG, PDF, SVG."
        )
        self.save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self.save_btn)
        inner_layout.addLayout(btn_row)

        # --- Assemble --------------------------------------------------------
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

    def add_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_iou: float,
        val_f1: float,
    ):
        """Append one epoch's metrics and refresh the canvas."""
        self._epochs.append(epoch)
        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)
        self._val_ious.append(val_iou)
        self._val_f1s.append(val_f1)
        self._redraw()

    def reset(self):
        """Clear all data and reset canvas to empty state."""
        self._epochs.clear()
        self._train_losses.clear()
        self._val_losses.clear()
        self._val_ious.clear()
        self._val_f1s.clear()
        if self._matplotlib_ok:
            self._draw_empty()
            self.save_btn.setEnabled(False)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _draw_empty(self):
        """Render placeholder axes with labels but no data."""
        for ax, title in [
            (self._ax_loss, "Loss"),
            (self._ax_metrics, "Val. Accuracy"),
        ]:
            ax.clear()
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, linewidth=0.4, alpha=0.5)

        self._figure.tight_layout()
        self._canvas.draw()

    def _redraw(self):
        if not self._matplotlib_ok:
            return

        ep = self._epochs

        # --- Loss subplot ----------------------------------------------------
        ax = self._ax_loss
        ax.clear()
        ax.plot(ep, self._train_losses,
                color="#2196F3", linewidth=1.5, label="Train Loss")
        ax.plot(ep, self._val_losses,
                color="#FF9800", linewidth=1.5, label="Val Loss")
        ax.set_title("Loss", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, linewidth=0.4, alpha=0.5)

        # --- Metrics subplot -------------------------------------------------
        ax = self._ax_metrics
        ax.clear()
        ax.plot(ep, self._val_ious,
                color="#4CAF50", linewidth=1.5, label="Val IoU")
        ax.plot(ep, self._val_f1s,
                color="#E91E63", linewidth=1.5, label="Val F1")
        ax.set_title("Val. Accuracy", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, linewidth=0.4, alpha=0.5)

        self._figure.tight_layout()
        self._canvas.draw()
        self.save_btn.setEnabled(True)

    def _on_save(self):
        if not self._matplotlib_ok:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Training Plots",
            "training_plots",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg)",
        )
        if path:
            self._figure.savefig(path, dpi=150, bbox_inches="tight")
