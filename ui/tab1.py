'''
module: tab1.py
'''
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QSizePolicy, QPushButton, QMessageBox
)
from .tab1_ins_outs import InsAndOutsWidget
from .tab1_clipping import ClippingWidget
from .tab1_splitting import SplittingWidget
from .tab1_augmentation import AugmentationWidget
from .tab1_channel_stacking import ChannelStackingWidget


class Tab1Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ins_outs = InsAndOutsWidget()
        self.clipping = ClippingWidget()
        self.splitting = SplittingWidget()
        self.augmentation = AugmentationWidget()
        self.channel_stacking = ChannelStackingWidget()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)

        content_layout.addWidget(self.ins_outs)
        content_layout.addWidget(self.clipping)
        content_layout.addWidget(self.splitting)
        content_layout.addWidget(self.augmentation)
        content_layout.addWidget(self.channel_stacking)

        # --- Run button ---
        run_button = QPushButton("Run")
        run_button.clicked.connect(self.on_run_clicked)
        content_layout.addWidget(run_button)

        content_layout.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    def on_run_clicked(self):
        try:
            raster_id, vector_id, output_dir = self.ins_outs.get_selected_inputs()
            clip_params = self.clipping.get_clipping_params()
            split_percentages = self.splitting.get_split_percentages()
            augmentations = self.augmentation.selected_methods()
            stacking_methods = self.channel_stacking.get_selected_methods_with_order()

            # Placeholder: you can now pass them to your backend logic
            print("=== RUN CONFIGURATION ===")
            print("Raster ID:", raster_id)
            print("Vector ID:", vector_id)
            print("Output Dir:", output_dir)
            print("Clipping Params:", clip_params)
            print("Splitting:", split_percentages)
            print("Augmentations:", augmentations)
            print("Channel Stacking:", stacking_methods)

            QMessageBox.information(self, "Run Triggered", "Run configuration collected successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Run Error", f"Error while running: {str(e)}")
