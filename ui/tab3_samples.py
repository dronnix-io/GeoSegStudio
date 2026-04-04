"""
module: ui/tab3_samples.py

Sample Predictions section for the Evaluate tab.

Renders a visual grid of sample tiles after evaluation completes.
For each sample tile, three panels are shown side-by-side:
  [Input image (RGB composite)]  [Ground-truth mask]  [Predicted mask]

The filename and per-tile IoU are shown above each group.

The section is hidden until show_samples() is called.
Requires matplotlib; if unavailable, shows a plain text fallback.
"""
from __future__ import annotations

import numpy as np

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QSizePolicy,
)
from qgis.PyQt.QtCore import Qt

from .expandable_groupbox import ExpandableGroupBox


class EvalSamplesWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Sample Predictions")
        self.section.toggle_button.setChecked(True)

        # --- Scroll area wrapping the grid -----------------------------------
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred)

        self._grid_widget = QWidget()
        self._grid_layout = QVBoxLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(4, 4, 4, 4)
        self._grid_layout.setSpacing(12)

        self._placeholder = QLabel("No samples yet.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._grid_layout.addWidget(self._placeholder)

        self._scroll.setWidget(self._grid_widget)

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(self._scroll)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

        # Start hidden
        self.setVisible(False)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def show_samples(self, samples: list):
        """
        Renders one row per sample.

        Parameters
        ----------
        samples : list of dict
            Each dict has keys:
              filename  str          original tile filename
              image     ndarray      (C, H, W) float32 [0-1]
              gt_mask   ndarray      (1, H, W) float32 binary
              pred_mask ndarray      (1, H, W) float32 binary
              iou       float        per-tile IoU
        """
        self._clear_grid()

        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
            matplotlib_ok = True
        except Exception:
            matplotlib_ok = False

        if not matplotlib_ok:
            lbl = QLabel(
                "matplotlib is not available — cannot render sample images.\n"
                "Install matplotlib into the plugin's Python environment."
            )
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setWordWrap(True)
            self._grid_layout.addWidget(lbl)
            self.setVisible(True)
            return

        for sample in samples:
            row_widget = self._build_sample_row(
                sample, Figure, FigureCanvas
            )
            self._grid_layout.addWidget(row_widget)

        self._grid_layout.addStretch()
        self.setVisible(True)

    def reset(self):
        """Clears all rendered samples and hides the section."""
        self._clear_grid()
        self.setVisible(False)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _clear_grid(self):
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _build_sample_row(self, sample: dict, Figure, FigureCanvas) -> QWidget:
        """Builds a single row widget (title + 3-panel figure)."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Title bar
        fname = sample.get("filename", "")
        iou = sample.get("iou", None)
        iou_str = f"  —  IoU: {iou:.4f}" if iou is not None else ""
        title = QLabel(f"<b>{fname}</b>{iou_str}")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Figure with 3 subplots
        fig = Figure(figsize=(7.5, 2.9))
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(220)
        axes = fig.subplots(1, 3)

        image = sample.get("image", None)   # (C,H,W)
        gt_mask = sample.get("gt_mask", None)   # (1,H,W)
        pred_mask = sample.get("pred_mask", None)  # (1,H,W)

        # --- Input image (show RGB if ≥3 bands, otherwise first band) --------
        ax = axes[0]
        if image is not None:
            ax.imshow(
                _to_display(image),
                cmap=None if image.shape[0] >= 3 else "gray")
        ax.set_title("Input Image", fontsize=8)
        ax.axis("off")

        # --- Ground truth mask -----------------------------------------------
        ax = axes[1]
        if gt_mask is not None:
            ax.imshow(gt_mask.squeeze(0), cmap="gray", vmin=0, vmax=1)
        ax.set_title("Ground Truth", fontsize=8)
        ax.axis("off")

        # --- Predicted mask --------------------------------------------------
        ax = axes[2]
        if pred_mask is not None:
            ax.imshow(pred_mask.squeeze(0), cmap="gray", vmin=0, vmax=1)
        ax.set_title("Prediction", fontsize=8)
        ax.axis("off")

        # rect leaves headroom at the top so subplot titles are not clipped
        fig.tight_layout(pad=0.8, rect=[0, 0, 1, 0.93])
        canvas.draw()
        layout.addWidget(canvas)

        return container


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_display(image: np.ndarray) -> np.ndarray:
    """
    Converts a (C, H, W) float32 image tensor to (H, W, 3) uint8 for display.

    If C >= 3 the first three bands are used as R, G, B.
    If C == 1 or 2, the first band is replicated to all three channels.
    Values are assumed to be in [0, 1] and clipped before conversion.
    """
    c, h, w = image.shape
    if c >= 3:
        rgb = image[:3].transpose(1, 2, 0)          # (H, W, 3)
    else:
        band = image[0]                              # (H, W)
        rgb = np.stack([band, band, band], axis=-1)  # (H, W, 3)

    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)
