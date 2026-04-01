"""
module: tab1_splitting.py
"""
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QSpinBox, QComboBox,
    QPushButton, QProgressBar, QLabel, QHBoxLayout
)
from qgis.PyQt.QtCore import Qt
from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget
from .styles import style_primary_btn, style_icon_btn, style_progress_bar


class SplittingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Splitting")

        self.content = SectionContentWidget()
        self.form = self.content.layout()

        # --- Clipping version selector ---------------------------------------
        version_row = QHBoxLayout()
        version_row.setContentsMargins(0, 0, 0, 0)
        version_row.setSpacing(4)

        self.clipping_version_combo = QComboBox()
        self.clipping_version_combo.setToolTip(
            "Select the clipping version to split. "
            "Run Apply Clipping first if no versions appear."
        )
        version_row.addWidget(self.clipping_version_combo)

        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(28)
        self.refresh_btn.setToolTip("Refresh clipping version list")
        style_icon_btn(self.refresh_btn)
        version_row.addWidget(self.refresh_btn)

        self.form.addRow("Clipping Ver.", version_row)

        # --- Split percentage spin boxes ------------------------------------
        self.train_spin = QSpinBox()
        self.valid_spin = QSpinBox()
        self.test_spin  = QSpinBox()

        for spin in (self.train_spin, self.valid_spin, self.test_spin):
            spin.setRange(0, 100)
            spin.setSuffix("%")

        self.train_spin.setValue(80)
        self.valid_spin.setValue(10)
        self.test_spin.setValue(10)

        self.form.addRow("Training %",   self.train_spin)
        self.form.addRow("Validation %", self.valid_spin)
        self.form.addRow("Testing %",    self.test_spin)

        self.train_spin.valueChanged.connect(self._validate_total)
        self.valid_spin.valueChanged.connect(self._validate_total)
        self.test_spin.valueChanged.connect(self._validate_total)

        # --- Apply button + progress bar ------------------------------------
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(0, 6, 0, 0)
        bottom_layout.setSpacing(4)

        self.apply_btn = QPushButton("Apply Splitting")
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
    # Version selector
    # -------------------------------------------------------------------------

    def populate_clipping_versions(self, versions: list):
        """
        Fills the clipping version combo box.

        Parameters
        ----------
        versions : list[dict]
            Each dict: {version (int), info (dict), label (str)}
            As returned by pipeline.get_clipping_versions() + version_label().
        """
        self.clipping_version_combo.clear()
        if not versions:
            self.clipping_version_combo.addItem("No clipping versions found", None)
            self.apply_btn.setEnabled(False)
        else:
            for v in versions:
                self.clipping_version_combo.addItem(v["label"], v["version"])
            # Default to the latest version
            self.clipping_version_combo.setCurrentIndex(
                self.clipping_version_combo.count() - 1
            )
            self.apply_btn.setEnabled(True)

    def get_selected_clipping_version(self):
        """Returns the currently selected clipping version number (int) or None."""
        return self.clipping_version_combo.currentData()

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

    def get_split_percentages(self) -> dict:
        return {
            "train": self.train_spin.value(),
            "valid": self.valid_spin.value(),
            "test":  self.test_spin.value(),
        }

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _validate_total(self):
        total = (self.train_spin.value()
                 + self.valid_spin.value()
                 + self.test_spin.value())
        if total != 100:
            self.section.title_label.setText(
                f"<b>Splitting</b> "
                f"<span style='color:red'>(Total: {total}%)</span>"
            )
        else:
            self.section.title_label.setText("<b>Splitting</b>")
