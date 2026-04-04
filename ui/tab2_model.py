"""
module: tab2_model.py

Model section for the Train tab.

Lets the user pick a segmentation architecture.
Input channels and tile size are owned by the Dataset section and read
from there when building the training config — they are not repeated here.
An Advanced toggle reveals the base_channels parameter for CNN-based models.
"""
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QPushButton, QSpinBox,
)
from qgis.PyQt.QtCore import Qt

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget


# Display name → registry key  (order matches the architectures.py listing)
_ARCHITECTURES = [
    ("UNet", "unet"),
    ("Attention UNet", "attention_unet"),
    ("UNet++", "unet_pp"),
    ("Swin UNet", "swin_unet"),
    ("LinkNet", "linknet"),
    ("DeepLabV3+", "deeplabv3plus"),
    ("SegFormer", "segformer"),
]


class ModelWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Model")
        self.content = SectionContentWidget()
        self.form = self.content.layout()

        # --- Architecture ----------------------------------------------------
        self.arch_combo = QComboBox()
        for display_name, key in _ARCHITECTURES:
            self.arch_combo.addItem(display_name, key)
        self.arch_combo.setToolTip("Segmentation model architecture to train.")
        self.form.addRow("Architecture", self.arch_combo)

        # --- Advanced toggle -------------------------------------------------
        self.advanced_btn = QPushButton("▸  Advanced")
        self.advanced_btn.setFlat(True)
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setChecked(False)
        self.advanced_btn.setCursor(Qt.PointingHandCursor)
        self.advanced_btn.clicked.connect(self._toggle_advanced)
        self.form.addRow("", self.advanced_btn)

        # --- Advanced content (hidden by default) ----------------------------
        self.adv_content = SectionContentWidget()
        adv_form = self.adv_content.layout()

        self.base_ch_spin = QSpinBox()
        self.base_ch_spin.setRange(8, 512)
        self.base_ch_spin.setValue(64)
        self.base_ch_spin.setSingleStep(8)
        self.base_ch_spin.setToolTip(
            "Feature-map width at the first encoder stage (doubles each stage).\n"
            "Applies to: UNet, Attention UNet, UNet++, LinkNet, DeepLabV3+.\n"
            "Has no effect on Swin UNet and SegFormer (they use embed_dim).")
        adv_form.addRow("Base Channels", self.base_ch_spin)

        self.adv_content.setVisible(False)

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(0)
        section_layout.addWidget(self.content)
        section_layout.addWidget(self.adv_content)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)
        self.setLayout(layout)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _toggle_advanced(self, checked: bool):
        self.advanced_btn.setText("▾  Advanced" if checked else "▸  Advanced")
        self.adv_content.setVisible(checked)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_model_config(self) -> dict:
        """Returns the model configuration owned by this section.

        Note: in_channels and img_size are intentionally excluded — they are
        derived from the Dataset section and added by the Tab2Widget when
        assembling the full training config.
        """
        return {
            "architecture": self.arch_combo.currentData(),
            "base_channels": self.base_ch_spin.value(),
        }
