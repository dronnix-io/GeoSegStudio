"""
module: tab2.py

Train tab — orchestrates all training-related sections.
"""
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QSizePolicy,
)

from .tab2_dataset         import DatasetWidget
from .tab2_model           import ModelWidget
from .tab2_training_config import TrainingConfigWidget
from .tab2_checkpoints     import CheckpointsWidget
from .tab2_hardware        import HardwareWidget
from .tab2_run_monitor     import RunMonitorWidget


class Tab2Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.dataset         = DatasetWidget()
        self.model           = ModelWidget()
        self.training_config = TrainingConfigWidget()
        self.checkpoints     = CheckpointsWidget()
        self.hardware        = HardwareWidget()
        self.run_monitor     = RunMonitorWidget()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)

        content_layout.addWidget(self.dataset)
        content_layout.addWidget(self.model)
        content_layout.addWidget(self.training_config)
        content_layout.addWidget(self.checkpoints)
        content_layout.addWidget(self.hardware)
        content_layout.addWidget(self.run_monitor)
        content_layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
