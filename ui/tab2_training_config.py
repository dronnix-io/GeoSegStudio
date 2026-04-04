"""
module: tab2_training_config.py

Training Configuration section for the Train tab.

Covers loss function, optimizer, learning rate, batch size, epochs, and
LR scheduler selection. Scheduler parameters are handled internally with
sensible defaults — the user only picks which scheduler to use.
"""
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
)

from .expandable_groupbox import ExpandableGroupBox
from .section_content_widget import SectionContentWidget


# (Display label, data key)
_LOSS_OPTIONS = [
    ("BCE", "bce"),
    ("Dice", "dice"),
    ("BCE + Dice", "bce_dice"),
]

_OPTIMIZER_OPTIONS = [
    ("Adam", "adam"),
    ("AdamW", "adamw"),
    ("SGD", "sgd"),
]

_SCHEDULER_OPTIONS = [
    ("None", None),
    ("StepLR", "StepLR"),
    ("Cosine Annealing", "CosineAnnealing"),
    ("Reduce on Plateau", "ReduceLROnPlateau"),
]


class TrainingConfigWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.section = ExpandableGroupBox("Training Configuration")
        self.content = SectionContentWidget()
        self.form = self.content.layout()

        # --- Loss function ---------------------------------------------------
        self.loss_combo = QComboBox()
        for label, key in _LOSS_OPTIONS:
            self.loss_combo.addItem(label, key)
        self.loss_combo.setCurrentIndex(2)          # default: BCE + Dice
        self.loss_combo.setToolTip(
            "BCE: Binary Cross-Entropy — penalises each pixel independently.\n"
            "Dice: Dice loss — optimises overlap between prediction and mask.\n"
            "BCE + Dice: combines both; recommended for most cases.")
        self.form.addRow("Loss Function", self.loss_combo)

        # --- Optimizer -------------------------------------------------------
        self.optimizer_combo = QComboBox()
        for label, key in _OPTIMIZER_OPTIONS:
            self.optimizer_combo.addItem(label, key)
        self.optimizer_combo.setToolTip(
            "Adam:  adaptive learning rate — good general-purpose default.\n"
            "AdamW: Adam with weight decay — often better on transformers.\n"
            "SGD:   stochastic gradient descent — robust but slower to converge.")
        self.form.addRow("Optimizer", self.optimizer_combo)

        # --- Learning rate ---------------------------------------------------
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.0001)
        self.lr_spin.setToolTip(
            "Initial learning rate.\n"
            "Controls the size of weight updates each step.\n"
            "Typical range: 0.0001 – 0.001 for Adam / AdamW."
        )
        self.form.addRow("Learning Rate", self.lr_spin)

        # --- Batch size ------------------------------------------------------
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(8)
        self.batch_spin.setToolTip(
            "Number of tiles processed together in each training step.\n"
            "Larger batches train faster but require more GPU memory.\n"
            "Reduce if you run out of memory."
        )
        self.form.addRow("Batch Size", self.batch_spin)

        # --- Epochs ----------------------------------------------------------
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setToolTip(
            "Number of full passes through the training dataset.\n"
            "More epochs can improve accuracy but may overfit if too high."
        )
        self.form.addRow("Epochs", self.epochs_spin)

        # --- LR Scheduler ----------------------------------------------------
        self.scheduler_combo = QComboBox()
        for label, key in _SCHEDULER_OPTIONS:
            self.scheduler_combo.addItem(label, key)
        self.scheduler_combo.setToolTip(
            "Automatically adjusts the learning rate during training.\n"
            "None:              learning rate stays fixed throughout.\n"
            "StepLR:            reduces LR by half every 10 epochs.\n"
            "Cosine Annealing:  smoothly reduces LR following a cosine curve.\n"
            "Reduce on Plateau: reduces LR when validation loss stops improving.")
        self.form.addRow("LR Scheduler", self.scheduler_combo)

        # --- Assemble section ------------------------------------------------
        section_layout = QVBoxLayout()
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.addWidget(self.content)
        self.section.setContentLayout(section_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)
        self.setLayout(layout)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_training_config(self) -> dict:
        """Returns the training configuration as a dict."""
        return {
            "loss": self.loss_combo.currentData(),
            "optimizer": self.optimizer_combo.currentData(),
            "lr": self.lr_spin.value(),
            "batch_size": self.batch_spin.value(),
            "epochs": self.epochs_spin.value(),
            "scheduler": self.scheduler_combo.currentData(),
        }
