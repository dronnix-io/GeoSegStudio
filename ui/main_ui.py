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
        self.setMinimumWidth(500)

        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)

        self.tab2 = Tab2Widget()
        self.tab3 = Tab3Widget()
        self.tab4 = Tab4Widget()

        self.tabs = QTabWidget()
        self.tabs.addTab(Tab1Widget(),  "Prepare")
        self.tabs.addTab(self.tab2,     "Train")
        self.tabs.addTab(self.tab3,     "Evaluate")
        self.tabs.addTab(self.tab4,     "Predict")

        layout.addWidget(self.tabs)
        self.setWidget(main_widget)
