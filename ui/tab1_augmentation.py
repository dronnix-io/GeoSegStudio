"""
module: tab1_augmentation.py
"""
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QCheckBox,
    QComboBox, QPushButton, QProgressBar, QLabel, QHBoxLayout
)
from qgis.PyQt.QtCore import Qt
from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .styles import style_primary_btn, style_icon_btn, style_progress_bar


class AugmentationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Augmentation")

        self.content = SectionContentWidget()
        form_layout = self.content.layout()

        # --- Splitting version selector -------------------------------------
        version_row = QHBoxLayout()
        version_row.setContentsMargins(0, 0, 0, 0)
        version_row.setSpacing(4)

        self.splitting_version_combo = QComboBox()
        self.splitting_version_combo.setToolTip(
            "Select the splitting version to augment. "
            "Run Apply Splitting first if no versions appear."
        )
        version_row.addWidget(self.splitting_version_combo)

        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(28)
        self.refresh_btn.setToolTip("Refresh splitting version list")
        style_icon_btn(self.refresh_btn)
        version_row.addWidget(self.refresh_btn)

        form_layout.addRow("Splitting Ver.", version_row)

        # --- Augmentation method checkboxes ---------------------------------
        self.grid_layout = QGridLayout()
        form_layout.addRow(self.grid_layout)

        self.augmentations = [
            ("Original", True),
            ("Rotate 90",  False),
            ("Rotate 180", False),
            ("Rotate 270", False),
            ("Mirror",  False),
            ("Flip H",  False),
            ("Flip V",  False),
        ]

        self.checkboxes = {}
        self._init_checkboxes()

        # --- Apply button + progress bar ------------------------------------
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(0, 6, 0, 0)
        bottom_layout.setSpacing(4)

        self.apply_btn = QPushButton("Apply Augmentation")
        style_primary_btn(self.apply_btn)
        bottom_layout.addWidget(self.apply_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        style_progress_bar(self.progress_bar)
        bottom_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setVisible(False)
        bottom_layout.addWidget(self.status_label)

        # Assemble section
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(self.content)
        section_layout.addLayout(bottom_layout)

        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)
        self.setLayout(layout)

    # -------------------------------------------------------------------------
    # Checkbox setup
    # -------------------------------------------------------------------------

    def _init_checkboxes(self):
        self.checkbox_all = QCheckBox("All")
        self.checkbox_all.stateChanged.connect(self._handle_all_checkbox)
        self.grid_layout.addWidget(self.checkbox_all, 0, 0)

        for i, (label, always_checked) in enumerate(self.augmentations):
            cb = QCheckBox(label)
            cb.setChecked(always_checked)
            if always_checked:
                cb.setEnabled(False)
            else:
                cb.stateChanged.connect(self._handle_individual_checkbox)
            row = (i + 1) // 2
            col = (i + 1) % 2
            self.grid_layout.addWidget(cb, row, col)
            self.checkboxes[label] = cb

    def _handle_all_checkbox(self, state):
        for cb in self.checkboxes.values():
            if cb.isEnabled():
                cb.setChecked(state == Qt.Checked)

    def _handle_individual_checkbox(self):
        all_checked = all(
            cb.isChecked() or not cb.isEnabled()
            for cb in self.checkboxes.values()
        )
        self.checkbox_all.blockSignals(True)
        self.checkbox_all.setChecked(all_checked)
        self.checkbox_all.blockSignals(False)

    # -------------------------------------------------------------------------
    # Version selector
    # -------------------------------------------------------------------------

    def populate_splitting_versions(self, versions: list):
        """
        Fills the splitting version combo box.

        Parameters
        ----------
        versions : list[dict]
            Each dict: {version (int), info (dict), label (str)}.
        """
        self.splitting_version_combo.clear()
        if not versions:
            self.splitting_version_combo.addItem("No splitting versions found", None)
            self.apply_btn.setEnabled(False)
        else:
            for v in versions:
                self.splitting_version_combo.addItem(v["label"], v["version"])
            self.splitting_version_combo.setCurrentIndex(
                self.splitting_version_combo.count() - 1
            )
            self.apply_btn.setEnabled(True)

    def get_selected_splitting_version(self):
        """Returns the currently selected splitting version number (int) or None."""
        return self.splitting_version_combo.currentData()

    # -------------------------------------------------------------------------
    # State helpers
    # -------------------------------------------------------------------------

    def set_running(self, running: bool):
        self.apply_btn.setEnabled(not running)
        self.refresh_btn.setEnabled(not running)
        self.progress_bar.setVisible(running)
        self.status_label.setVisible(False)
        if running:
            self.progress_bar.setValue(0)

    def set_status(self, message: str, error: bool = False):
        color = "red" if error else "green"
        self.status_label.setText(
            f"<span style='color:{color}'>{message}</span>"
        )
        self.status_label.setVisible(True)
        self.progress_bar.setVisible(False)

    # -------------------------------------------------------------------------
    # Data getter
    # -------------------------------------------------------------------------

    def selected_methods(self) -> list:
        return [label for label, cb in self.checkboxes.items() if cb.isChecked()]
