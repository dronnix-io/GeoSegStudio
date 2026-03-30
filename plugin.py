from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon
from .ui.main_ui import DeepLearningDockWidget
from qgis.PyQt.QtCore import Qt


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
        # Stop any running training worker before destroying the dock widget.
        # Destroying Qt objects while a QThread is alive causes a crash.
        if self.dock_widget is not None:
            try:
                worker = self.dock_widget.tab2._worker
                if worker is not None and worker.isRunning():
                    worker.stop()
                    worker.wait(3000)  # give it up to 3 s to finish cleanly
            except Exception:
                pass

        self.iface.removePluginMenu("Deep Learning Plugin", self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.dock_widget:
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget = None

    def show_dock(self):
        # Check / bootstrap the PyTorch environment on first open
        self._ensure_env()

        if self.dock_widget is None:
            self.dock_widget = DeepLearningDockWidget(self.iface)
            self.iface.addDockWidget(Qt.LeftDockWidgetArea, self.dock_widget)
        self.dock_widget.show()
        self.dock_widget.raise_()

    # -------------------------------------------------------------------------
    # Environment bootstrap
    # -------------------------------------------------------------------------

    def _ensure_env(self):
        """
        Shows the install dialog if the env is not ready yet.
        Patches sys.path when the env already exists so torch is importable.
        """
        try:
            from .DL.env_manager import is_env_ready, patch_sys_path
        except Exception:
            return  # env_manager unavailable — skip silently

        if is_env_ready():
            patch_sys_path()
            return

        # Env not ready — show the setup dialog
        try:
            from .ui.install_dialog import InstallDialog
        except Exception:
            return

        dlg = InstallDialog(self.iface.mainWindow())
        dlg.exec_()

        if dlg.was_installed():
            patch_sys_path()
