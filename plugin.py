from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon
from .ui.main_ui import DeepLearningDockWidget

class DeepLearningPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.dock_widget = None
        self.action = None

    def initGui(self):
        self.action = QAction(QIcon(":/plugins/DeepLearningPlugin/icon.png"), "Deep Learning Plugin", self.iface.mainWindow())
        self.action.triggered.connect(self.show_dock)
        self.iface.addPluginToMenu("Deep Learning Plugin", self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        self.iface.removePluginMenu("Deep Learning Plugin", self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.dock_widget:
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget = None

    def show_dock(self):
        if self.dock_widget is None:
            self.dock_widget = DeepLearningDockWidget(self.iface)
            self.iface.addDockWidget(Qt.LeftDockWidgetArea, self.dock_widget)
        self.dock_widget.show()
        self.dock_widget.raise_()
