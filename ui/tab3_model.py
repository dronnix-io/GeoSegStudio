"""
module: ui/tab3_model.py

Model section for the Evaluate tab.

Lets the user browse to a trained .pth checkpoint file.
Once a valid file is loaded, shows read-only metadata extracted from the
checkpoint (architecture, epoch, val IoU, input bands, tile size).
The metadata is also used by Tab3Widget for validation when the user
clicks Run.
"""
import os

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QFrame,
)

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget


class EvalModelWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Metadata cached after a successful checkpoint read
        self._meta = {}   # keys: architecture, epoch, val_iou, in_channels, img_size

        self.section = ExpandableGroupBox("Model")
        self.content = SectionContentWidget()
        self.form    = self.content.layout()

        # --- Checkpoint file picker ------------------------------------------
        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(4)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Browse to a trained .pth checkpoint …")
        self.path_edit.setReadOnly(True)
        file_row.addWidget(self.path_edit)

        self.browse_btn = QPushButton("…")
        self.browse_btn.setFixedWidth(30)
        self.browse_btn.setToolTip(
            "Select a .pth checkpoint file saved by the Train tab.\n"
            "The checkpoint must contain model weights and metadata\n"
            "(architecture, in_channels, img_size)."
        )
        file_row.addWidget(self.browse_btn)

        self.form.addRow("Checkpoint", file_row)

        # Hint shown below the path (error or metadata summary)
        self.hint_lbl = QLabel("")
        self.hint_lbl.setWordWrap(True)
        self.hint_lbl.setVisible(False)
        self.form.addRow("", self.hint_lbl)

        # --- Separator -------------------------------------------------------
        self.sep = QFrame()
        self.sep.setFrameShape(QFrame.HLine)
        self.sep.setFrameShadow(QFrame.Sunken)
        self.sep.setVisible(False)
        self.form.addRow(self.sep)

        # --- Read-only metadata ----------------------------------------------
        self.arch_lbl      = QLabel("—")
        self.epoch_lbl     = QLabel("—")
        self.val_iou_lbl   = QLabel("—")
        self.in_ch_lbl     = QLabel("—")
        self.img_size_lbl  = QLabel("—")

        self.form.addRow("Architecture",  self.arch_lbl)
        self.form.addRow("Epoch",         self.epoch_lbl)
        self.form.addRow("Val IoU",       self.val_iou_lbl)
        self.form.addRow("Input Bands",   self.in_ch_lbl)
        self.form.addRow("Tile Size",     self.img_size_lbl)

        # Hide metadata rows until a checkpoint is loaded
        self._set_metadata_visible(False)

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(self.content)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)
        self.setLayout(layout)

        # --- Connections -----------------------------------------------------
        self.browse_btn.clicked.connect(self._browse_checkpoint)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _set_metadata_visible(self, visible: bool):
        self.sep.setVisible(visible)
        for lbl in (self.arch_lbl, self.epoch_lbl, self.val_iou_lbl,
                    self.in_ch_lbl, self.img_size_lbl):
            lbl.setVisible(visible)

        # Also hide the corresponding form row labels
        form = self.form
        for row in range(form.rowCount()):
            lbl_item = form.itemAt(row, form.LabelRole)
            fld_item = form.itemAt(row, form.FieldRole)
            if fld_item and fld_item.widget() in (
                self.arch_lbl, self.epoch_lbl, self.val_iou_lbl,
                self.in_ch_lbl, self.img_size_lbl,
            ):
                if lbl_item and lbl_item.widget():
                    lbl_item.widget().setVisible(visible)

    def _browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint File", "", "PyTorch checkpoint (*.pth)"
        )
        if not path:
            return
        self.path_edit.setText(path)
        self._load_metadata(path)

    def _load_metadata(self, path: str):
        """Reads checkpoint metadata and updates the display."""
        self._meta = {}
        self._set_metadata_visible(False)

        try:
            import torch
            data = torch.load(path, map_location="cpu")
        except ImportError:
            self._show_hint(
                "PyTorch is not installed — cannot read checkpoint metadata.",
                error=True,
            )
            return
        except Exception as exc:
            self._show_hint(
                f"Could not read checkpoint: {exc}\n"
                "Make sure it is a valid .pth file saved by this plugin.",
                error=True,
            )
            return

        saved = data.get("config", {})
        arch     = data.get("architecture") or saved.get("architecture")
        epoch    = data.get("epoch")
        val_iou  = data.get("val_iou")
        in_ch    = saved.get("in_channels")
        img_size = saved.get("img_size")

        missing = []
        if not arch:     missing.append("architecture")
        if in_ch is None: missing.append("in_channels")
        if img_size is None: missing.append("img_size")

        if missing:
            self._show_hint(
                "Checkpoint is missing required metadata: "
                + ", ".join(missing)
                + ".\nIt may have been saved by an older version of this plugin.",
                error=True,
            )
            return

        # Cache for use by Tab3Widget when building the evaluation config
        self._meta = {
            "architecture": arch,
            "epoch":        epoch,
            "val_iou":      val_iou,
            "in_channels":  in_ch,
            "img_size":     img_size,
        }

        # Update display labels
        self.arch_lbl.setText(arch)
        self.epoch_lbl.setText(str(epoch) if epoch is not None else "—")
        self.val_iou_lbl.setText(
            f"{val_iou:.4f}" if val_iou is not None else "—"
        )
        self.in_ch_lbl.setText(str(in_ch))
        self.img_size_lbl.setText(f"{img_size} × {img_size} px")

        self._show_hint("Checkpoint loaded successfully.", error=False)
        self._set_metadata_visible(True)

    def _show_hint(self, message: str, error: bool):
        color = "red" if error else "green"
        self.hint_lbl.setText(f"<span style='color:{color}'>{message}</span>")
        self.hint_lbl.setVisible(True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_checkpoint_path(self) -> str:
        """Returns the currently selected checkpoint file path (may be empty)."""
        return self.path_edit.text().strip()

    def get_metadata(self) -> dict:
        """Returns the cached checkpoint metadata dict (empty if no checkpoint loaded)."""
        return dict(self._meta)
