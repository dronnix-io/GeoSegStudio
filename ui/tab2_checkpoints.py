"""
module: tab2_checkpoints.py

Checkpoints section for the Train tab.

Two responsibilities:
  Save   — where to write the trained model and how often.
  Resume — optionally load an existing .pth file to continue training
           or fine-tune a previously trained model.
"""
import os

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QComboBox, QSpinBox,
    QLabel, QFileDialog, QFrame, QCheckBox,
)
from qgis.PyQt.QtCore import Qt

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .styles import style_icon_btn


_SAVE_STRATEGIES = [
    ("Best only  (highest val IoU)", "best"),
    ("Every N epochs", "every_n"),
]


class CheckpointsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Checkpoints")
        self.content = SectionContentWidget()
        self.form = self.content.layout()

        # ── Save ─────────────────────────────────────────────────────────────

        save_header = QLabel("<b>Save</b>")
        self.form.addRow(save_header)

        # Output directory
        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.setSpacing(4)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText(
            "Folder where the .pth file will be saved …")
        self.output_dir_edit.setReadOnly(True)
        dir_row.addWidget(self.output_dir_edit)

        self.output_dir_btn = QPushButton("…")
        self.output_dir_btn.setFixedWidth(30)
        style_icon_btn(self.output_dir_btn)
        self.output_dir_btn.setToolTip(
            "Select the folder where the trained model will be saved.")
        dir_row.addWidget(self.output_dir_btn)

        self.form.addRow("Output Dir", dir_row)

        # Model name
        self.model_name_edit = QLineEdit("model")
        self.model_name_edit.setToolTip(
            "Base name for the saved file.\n"
            "The final filename will be:  <name>.pth\n"
            "Example:  building_detector  →  building_detector.pth"
        )
        self.form.addRow("Model Name", self.model_name_edit)

        # Save strategy
        self.strategy_combo = QComboBox()
        for label, key in _SAVE_STRATEGIES:
            self.strategy_combo.addItem(label, key)
        self.strategy_combo.setToolTip(
            "Best only:     saves the model only when val IoU improves.\n"
            "Every N epochs: saves a checkpoint at regular intervals."
        )
        self.form.addRow("Save Strategy", self.strategy_combo)

        # Every-N spinbox (visible only when strategy = every_n)
        self.every_n_spin = QSpinBox()
        self.every_n_spin.setRange(1, 1000)
        self.every_n_spin.setValue(10)
        self.every_n_spin.setToolTip(
            "Save a checkpoint every this many epochs.")
        self.every_n_spin.setVisible(False)
        self.every_n_label = QLabel("Save Every (ep.)")
        self.every_n_label.setVisible(False)
        self.form.addRow(self.every_n_label, self.every_n_spin)

        # ── Separator ────────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.form.addRow(sep)

        # ── Resume (optional) ────────────────────────────────────────────────

        resume_header = QLabel("<b>Resume  </b><small>(optional)</small>")
        self.form.addRow(resume_header)

        # Enable checkbox
        self.resume_check = QCheckBox(
            "Load a checkpoint to resume or fine-tune")
        self.resume_check.setChecked(False)
        self.form.addRow(self.resume_check)

        # Checkpoint file picker (hidden until checkbox is ticked)
        self.resume_widget = QWidget()
        resume_layout = QFormLayout(self.resume_widget)
        resume_layout.setContentsMargins(0, 0, 0, 0)
        resume_layout.setSpacing(8)

        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(4)

        self.checkpoint_edit = QLineEdit()
        self.checkpoint_edit.setPlaceholderText("Browse to a .pth file …")
        self.checkpoint_edit.setReadOnly(True)
        file_row.addWidget(self.checkpoint_edit)

        self.checkpoint_btn = QPushButton("…")
        self.checkpoint_btn.setFixedWidth(30)
        style_icon_btn(self.checkpoint_btn)
        self.checkpoint_btn.setToolTip(
            "Select a previously saved .pth checkpoint file.")
        file_row.addWidget(self.checkpoint_btn)

        resume_layout.addRow("Checkpoint", file_row)

        # Read-only hint showing metadata from the loaded file
        self.checkpoint_hint = QLabel("")
        self.checkpoint_hint.setWordWrap(True)
        self.checkpoint_hint.setVisible(False)
        resume_layout.addRow("", self.checkpoint_hint)

        self.resume_widget.setVisible(False)
        self.form.addRow(self.resume_widget)

        # ── Assemble section ─────────────────────────────────────────────────
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(self.content)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)
        self.setLayout(layout)

        # ── Connections ──────────────────────────────────────────────────────
        self.output_dir_btn.clicked.connect(self._browse_output_dir)
        self.strategy_combo.currentIndexChanged.connect(
            self._on_strategy_changed)
        self.resume_check.toggled.connect(self._on_resume_toggled)
        self.checkpoint_btn.clicked.connect(self._browse_checkpoint)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _browse_output_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        if folder:
            self.output_dir_edit.setText(folder)

    def _on_strategy_changed(self):
        is_every_n = self.strategy_combo.currentData() == "every_n"
        self.every_n_label.setVisible(is_every_n)
        self.every_n_spin.setVisible(is_every_n)

    def _on_resume_toggled(self, checked: bool):
        self.resume_widget.setVisible(checked)
        if not checked:
            self.checkpoint_edit.clear()
            self.checkpoint_hint.setVisible(False)

    def _browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint File", "", "PyTorch checkpoint (*.pth)"
        )
        if not path:
            return

        self.checkpoint_edit.setText(path)
        self._load_checkpoint_metadata(path)

    def _load_checkpoint_metadata(self, path: str):
        """Reads the checkpoint file and shows a summary hint."""
        try:
            import torch
            data = torch.load(
                path,
                map_location="cpu",
                weights_only=False)  # nosec B614

            arch = data.get("architecture", "unknown")
            epoch = data.get("epoch", "?")
            val_iou = data.get("val_iou", None)

            iou_str = f"  |  Val IoU: {
                val_iou:.4f}" if val_iou is not None else ""
            hint = (
                f"<span style='color:green'>"
                f"Architecture: {arch}  |  Epoch: {epoch}{iou_str}"
                f"</span>"
            )

        except ImportError:
            hint = (
                "<span style='color:red'>"
                "PyTorch is not installed — cannot read checkpoint metadata."
                "</span>"
            )
        except Exception:
            hint = (
                "<span style='color:red'>"
                "Could not read the checkpoint file. "
                "Make sure it is a valid .pth file saved by this plugin."
                "</span>"
            )

        self.checkpoint_hint.setText(hint)
        self.checkpoint_hint.setVisible(True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_checkpoint_config(self) -> dict:
        """Returns the checkpoint configuration as a dict."""
        config = {
            "output_dir": self.output_dir_edit.text().strip(),
            "model_name": self.model_name_edit.text().strip() or "model",
            "save_strategy": self.strategy_combo.currentData(),
            "resume_path": None,
        }

        if self.strategy_combo.currentData() == "every_n":
            config["every_n"] = self.every_n_spin.value()

        if self.resume_check.isChecked():
            path = self.checkpoint_edit.text().strip()
            config["resume_path"] = path if path else None

        return config
