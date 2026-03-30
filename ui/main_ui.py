'''
module: main_ui.py
'''
from qgis.PyQt.QtWidgets import QDockWidget, QTabWidget, QVBoxLayout, QWidget
from .tab1 import Tab1Widget
from .tab2 import Tab2Widget
from .tab3 import Tab3Widget
from .tab4 import Tab4Widget

class DeepLearningDockWidget(QDockWidget):
    def __init__(self, iface, parent=None):
        super().__init__("Deep Learning Plugin", parent)
        self.setFont(iface.mainWindow().font())  # Good here!
        self.iface = iface
        self.setObjectName("DeepLearningDockWidget")

        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)

        self.tabs = QTabWidget()
        self.tabs.addTab(Tab1Widget(),      "Prepare")
        self.tabs.addTab(Tab2Widget(),      "Train")
        self.tabs.addTab(Tab3Widget(),      "Evaluate")
        self.tabs.addTab(Tab4Widget(),      "Predict")

        layout.addWidget(self.tabs)
        self.setWidget(main_widget)
