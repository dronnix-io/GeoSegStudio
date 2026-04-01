"""
module: ui/tab3_dataset.py

Dataset section for the Evaluate tab.

Lets the user select the same *_dataset folder that was used during training,
choose an augmented version, and pick which split to evaluate on
(test / valid / train).  Shows a read-only tile-count summary so the user
can confirm the right dataset is loaded.
"""
import json
import os

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QLabel, QLineEdit, QFileDialog, QFrame,
)

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .styles import style_icon_btn


_SPLITS = [
    ("Test  (recommended)",  "test"),
    ("Validation",           "valid"),
    ("Train",                "train"),
]


class EvalDatasetWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Dataset")
        self.content = SectionContentWidget()
        self.form    = self.content.layout()

        # --- Dataset directory -----------------------------------------------
        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.setSpacing(4)

        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("e.g.  …/output/BuildingName_dataset")
        self.dir_edit.setReadOnly(True)
        dir_row.addWidget(self.dir_edit)

        self.browse_btn = QPushButton("…")
        self.browse_btn.setFixedWidth(30)
        style_icon_btn(self.browse_btn)
        self.browse_btn.setToolTip(
            "Select the top-level dataset folder created by the Prepare tab.\n"
            "Use the same folder that was used when training the checkpoint."
        )
        dir_row.addWidget(self.browse_btn)

        self.form.addRow("Dataset Dir", dir_row)

        self.dir_hint_lbl = QLabel("")
        self.dir_hint_lbl.setWordWrap(True)
        self.dir_hint_lbl.setVisible(False)
        self.form.addRow("", self.dir_hint_lbl)

        # --- Augmented version -----------------------------------------------
        version_row = QHBoxLayout()
        version_row.setContentsMargins(0, 0, 0, 0)
        version_row.setSpacing(4)

        self.version_combo = QComboBox()
        self.version_combo.setToolTip(
            "Select the augmented version to evaluate.\n"
            "Should match the version used during training."
        )
        version_row.addWidget(self.version_combo)

        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(28)
        self.refresh_btn.setToolTip("Re-scan the dataset folder for augmented versions.")
        style_icon_btn(self.refresh_btn)
        version_row.addWidget(self.refresh_btn)

        self.form.addRow("Aug. Version", version_row)

        # --- Split -----------------------------------------------------------
        self.split_combo = QComboBox()
        for label, key in _SPLITS:
            self.split_combo.addItem(label, key)
        self.split_combo.setToolTip(
            "Which data split to run evaluation on.\n"
            "Test is recommended — it was not seen during training."
        )
        self.form.addRow("Split", self.split_combo)

        # --- Separator -------------------------------------------------------
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.form.addRow(sep)

        # --- Read-only summary -----------------------------------------------
        self.train_lbl = QLabel("—")
        self.valid_lbl = QLabel("—")
        self.test_lbl  = QLabel("—")

        self.form.addRow("Train tiles", self.train_lbl)
        self.form.addRow("Val tiles",   self.valid_lbl)
        self.form.addRow("Test tiles",  self.test_lbl)

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
        self.browse_btn.clicked.connect(self._browse_dataset_dir)
        self.refresh_btn.clicked.connect(self._refresh_versions)
        self.version_combo.currentIndexChanged.connect(self._on_version_changed)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _browse_dataset_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.dir_edit.setText(folder)
            self._refresh_versions()

    def _refresh_versions(self):
        from ..DL.data_preparation import get_augmented_versions, version_label

        dataset_dir = self.dir_edit.text().strip()
        self.version_combo.blockSignals(True)
        self.version_combo.clear()
        self._clear_summary()

        if not dataset_dir or not os.path.isdir(dataset_dir):
            self.version_combo.addItem("No dataset folder selected", None)
            self.dir_hint_lbl.setVisible(False)
            self.version_combo.blockSignals(False)
            return

        if not os.path.isdir(os.path.join(dataset_dir, "augmented")):
            folder_name = os.path.basename(dataset_dir)
            if folder_name in ("augmented", "clipping", "splitting"):
                hint = (
                    f"You selected the '{folder_name}/' subfolder. "
                    "Go one level up and select the '_dataset' folder instead."
                )
            else:
                hint = (
                    "This does not look like a dataset folder. "
                    "Select the folder ending with '_dataset'."
                )
            self.dir_hint_lbl.setText(f"<span style='color:red'>{hint}</span>")
            self.dir_hint_lbl.setVisible(True)
            self.version_combo.addItem("No augmented versions found", None)
            self.version_combo.blockSignals(False)
            return

        self.dir_hint_lbl.setVisible(False)

        try:
            versions = get_augmented_versions(dataset_dir)
        except Exception:
            versions = []

        if not versions:
            self.version_combo.addItem("No augmented versions found", None)
        else:
            for v in versions:
                self.version_combo.addItem(version_label(v), v["version"])
            self.version_combo.setCurrentIndex(self.version_combo.count() - 1)

        self.version_combo.blockSignals(False)
        self._on_version_changed()

    def _on_version_changed(self):
        self._clear_summary()
        dataset_dir = self.dir_edit.text().strip()
        version     = self.version_combo.currentData()

        if not dataset_dir or version is None:
            return

        try:
            aug_path = os.path.join(
                dataset_dir, "augmented", f"v{version}", "augmentation_info.json"
            )
            with open(aug_path) as f:
                aug = json.load(f)
            self.train_lbl.setText(str(aug.get("train_count", "?")))
            self.valid_lbl.setText(str(aug.get("valid_count", "?")))
            self.test_lbl.setText(str(aug.get("test_count",  "?")))
        except Exception:
            self._clear_summary()

    def _clear_summary(self):
        for lbl in (self.train_lbl, self.valid_lbl, self.test_lbl):
            lbl.setText("—")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_dataset_dir(self) -> str:
        return self.dir_edit.text().strip()

    def get_selected_version(self):
        return self.version_combo.currentData()

    def get_split(self) -> str:
        return self.split_combo.currentData()
