"""
module: tab2_dataset.py

Dataset section for the Train tab.

Lets the user point to a *_dataset folder produced by the Prepare tab,
select an augmented version, and shows a read-only summary of that version
(tile size, band count, train / val / test tile counts).
"""
import json
import os

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QLabel, QLineEdit, QFileDialog, QFrame,
)
from qgis.PyQt.QtCore import pyqtSignal

from .expandable_groupbox import ExpandableGroupBox
from .info_card import MetaCardGrid
from .section_content_widget import SectionContentWidget
from .styles import style_icon_btn


class DatasetWidget(QWidget):
    # Emitted whenever the summary is refreshed (version selected or cleared)
    summary_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Raw integer values populated when a version is selected
        self._tile_size_int = None
        self._band_count_int = None

        self.section = ExpandableGroupBox("Dataset")
        self.content = SectionContentWidget()
        self.form = self.content.layout()

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
            "Its name ends with '_dataset' and it contains the subfolders:\n"
            "  clipping/   splitting/   augmented/\n\n"
            "Example:  D:/output/Building2_dataset\n"
            "Do NOT select a subfolder such as 'augmented' or 'clipping'."
        )
        dir_row.addWidget(self.browse_btn)

        ver_sep = QFrame()
        ver_sep.setFrameShape(QFrame.VLine)
        ver_sep.setFrameShadow(QFrame.Sunken)
        dir_row.addWidget(ver_sep)

        self.version_combo = QComboBox()
        self.version_combo.setFixedWidth(110)
        self.version_combo.setToolTip(
            "Select the augmented version to train on. "
            "Run the Prepare tab first if no versions appear."
        )
        dir_row.addWidget(self.version_combo)

        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(28)
        self.refresh_btn.setToolTip(
            "Re-scan the dataset folder for augmented versions")
        style_icon_btn(self.refresh_btn)
        dir_row.addWidget(self.refresh_btn)

        self.form.addRow("Dataset Dir", dir_row)

        self.dir_hint_lbl = QLabel("")
        self.dir_hint_lbl.setVisible(False)
        self.form.addRow("", self.dir_hint_lbl)

        # --- Read-only summary -----------------------------------------------
        self.summary_cards = MetaCardGrid(cols_per_row=3)
        self.form.addRow(self.summary_cards)

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
        self.version_combo.currentIndexChanged.connect(
            self._on_version_changed)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _browse_dataset_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder")
        if folder:
            self.dir_edit.setText(folder)
            self._refresh_versions()

    def _refresh_versions(self):
        """Scans the dataset folder and repopulates the augmented-version combo."""
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

        # Validate folder structure — must contain an 'augmented/' subfolder
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
                    "Select the folder ending with '_dataset' that contains "
                    "the 'augmented/', 'clipping/', and 'splitting/' subfolders.")
            self.dir_hint_lbl.setText(f"<span style='color:red'>{hint}</span>")
            self.dir_hint_lbl.setWordWrap(True)
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
        """Reads the JSON metadata for the selected version and refreshes summary."""
        self._clear_summary()

        dataset_dir = self.dir_edit.text().strip()
        version = self.version_combo.currentData()

        if not dataset_dir or version is None:
            self.summary_updated.emit()
            return

        try:
            aug_path = os.path.join(
                dataset_dir,
                "augmented",
                f"v{version}",
                "augmentation_info.json")
            with open(aug_path) as f:
                aug = json.load(f)

            # Tile size and band count live in the linked clipping_info.json
            tile_size = "—"
            band_count = "—"
            clip_ver = aug.get("based_on_clipping_version")
            if clip_ver is not None:
                clip_path = os.path.join(
                    dataset_dir,
                    "clipping",
                    f"v{clip_ver}",
                    "clipping_info.json")
                if os.path.isfile(clip_path):
                    with open(clip_path) as f:
                        clip = json.load(f)
                    ws = clip.get("window_size")
                    bc = clip.get("band_count")
                    self._tile_size_int = int(ws) if ws is not None else None
                    self._band_count_int = int(bc) if bc is not None else None
                    tile_size = f"{ws} × {ws} px" if ws else "?"
                    band_count = str(bc) if bc else "?"

            self.summary_cards.set_cards([
                ("Tile Size", tile_size),
                ("Bands", band_count),
                ("Train tiles", str(aug.get("train_count", "?"))),
                ("Val tiles", str(aug.get("valid_count", "?"))),
                ("Test tiles", str(aug.get("test_count", "?"))),
            ])

        except Exception:
            self._clear_summary()

        self.summary_updated.emit()

    def _clear_summary(self):
        self._tile_size_int = None
        self._band_count_int = None
        self.summary_cards.clear_cards()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_dataset_dir(self) -> str:
        """Returns the currently entered dataset folder path."""
        return self.dir_edit.text().strip()

    def get_selected_version(self):
        """Returns the selected augmented version number (int) or None."""
        return self.version_combo.currentData()

    def get_tile_size_int(self):
        """Returns the tile size in pixels as an int, or None if unavailable."""
        return self._tile_size_int

    def get_band_count_int(self):
        """Returns the band count as an int, or None if unavailable."""
        return self._band_count_int

    def get_summary(self) -> dict:
        """Returns the current summary as a dict built from internal state."""
        return {
            "tile_size": f"{
                self._tile_size_int} × {
                self._tile_size_int} px" if self._tile_size_int else "—",
            "band_count": str(
                self._band_count_int) if self._band_count_int else "—",
        }
