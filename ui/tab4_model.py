"""
module: ui/tab4_model.py

Model section for the Predict tab.

Identical responsibility to tab3_model.py: browse to a trained .pth
checkpoint and display its metadata as a compact MetaCardGrid.
Kept as a separate file so Tab 4 has no import coupling to the Evaluate tab.
"""

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog,
)

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .info_card import MetaCardGrid
from .styles import style_icon_btn


class PredictModelWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._meta = {}   # architecture, in_channels, img_size, epoch, val_iou

        self.section = ExpandableGroupBox("Model")
        self.content = SectionContentWidget()
        self.form = self.content.layout()

        # --- Checkpoint file picker ------------------------------------------
        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(4)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(
            "Browse to a trained .pth checkpoint …")
        self.path_edit.setReadOnly(True)
        file_row.addWidget(self.path_edit)

        self.browse_btn = QPushButton("…")
        self.browse_btn.setFixedWidth(30)
        self.browse_btn.setToolTip(
            "Select a .pth checkpoint file saved by the Train tab."
        )
        style_icon_btn(self.browse_btn)
        file_row.addWidget(self.browse_btn)

        self.form.addRow("Checkpoint", file_row)

        self.hint_lbl = QLabel("")
        self.hint_lbl.setWordWrap(True)
        self.hint_lbl.setVisible(False)
        self.form.addRow("", self.hint_lbl)

        # --- Metadata cards --------------------------------------------------
        self.meta_cards = MetaCardGrid(cols_per_row=3)
        self.form.addRow(self.meta_cards)

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(self.content)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

        self.browse_btn.clicked.connect(self._browse_checkpoint)

    # -------------------------------------------------------------------------

    def _browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint File", "", "PyTorch checkpoint (*.pth)"
        )
        if not path:
            return
        self.path_edit.setText(path)
        self._load_metadata(path)

    def _load_metadata(self, path: str):
        self._meta = {}
        self.meta_cards.clear_cards()

        try:
            import torch
            data = torch.load(
                path,
                map_location="cpu",
                weights_only=False)  # nosec B614
        except ImportError:
            self._show_hint(
                "PyTorch is not installed — cannot read checkpoint metadata.",
                error=True,
            )
            return
        except Exception as exc:
            self._show_hint(f"Could not read checkpoint: {exc}", error=True)
            return

        saved = data.get("config", {})
        arch = data.get("architecture") or saved.get("architecture")
        epoch = data.get("epoch")
        val_iou = data.get("val_iou")
        in_ch = saved.get("in_channels")
        img_size = saved.get("img_size")

        missing = []
        if not arch:
            missing.append("architecture")
        if in_ch is None:
            missing.append("in_channels")
        if img_size is None:
            missing.append("img_size")

        if missing:
            self._show_hint(
                "Checkpoint is missing required metadata: "
                + ", ".join(missing) + ".",
                error=True,
            )
            return

        self._meta = {
            "architecture": arch,
            "epoch": epoch,
            "val_iou": val_iou,
            "in_channels": in_ch,
            "img_size": img_size,
        }

        self.meta_cards.set_cards([
            ("Architecture", arch),
            ("Epoch", str(epoch) if epoch is not None else "—"),
            ("Val IoU", f"{val_iou:.4f}" if val_iou is not None else "—"),
            ("Input Bands", str(in_ch)),
            ("Tile Size", f"{img_size} × {img_size} px"),
        ])

        self._show_hint("Checkpoint loaded successfully.", error=False)

    def _show_hint(self, message: str, error: bool):
        color = "red" if error else "green"
        self.hint_lbl.setText(f"<span style='color:{color}'>{message}</span>")
        self.hint_lbl.setVisible(True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_checkpoint_path(self) -> str:
        return self.path_edit.text().strip()

    def get_metadata(self) -> dict:
        return dict(self._meta)
