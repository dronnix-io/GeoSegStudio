from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon
from .ui.main_ui import GeoSegStudioDockWidget
from qgis.PyQt.QtCore import Qt


class GeoSegStudioPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.dock_widget = None
        self.action = None

    def initGui(self):
        self.action = QAction(QIcon(":/plugins/GeoSegStudio/icon.png"), "GeoSeg Studio", self.iface.mainWindow())
        self.action.triggered.connect(self.show_dock)
        self.iface.addPluginToMenu("GeoSeg Studio", self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        # Stop any running training worker before destroying the dock widget.
        # Destroying Qt objects while a QThread is alive causes a crash.
        if self.dock_widget is not None:
            for attr in ("tab2", "tab3", "tab4"):
                tab = getattr(self.dock_widget, attr, None)
                if tab is None:
                    continue
                for worker_attr in ("_worker", "_worker_pp"):
                    try:
                        worker = getattr(tab, worker_attr, None)
                        if worker is not None and worker.isRunning():
                            worker.stop()
                            worker.wait(3000)
                    except Exception:
                        pass

        self.iface.removePluginMenu("GeoSeg Studio", self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.dock_widget:
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget = None

    def show_dock(self):
        # Check / bootstrap the PyTorch environment on first open
        self._ensure_env()

        if self.dock_widget is None:
            self.dock_widget = GeoSegStudioDockWidget(self.iface)
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)
        self.dock_widget.show()
        self.dock_widget.raise_()

    # -------------------------------------------------------------------------
    # Environment bootstrap
    # -------------------------------------------------------------------------

    def _ensure_env(self):
        """
        Patches sys.path when the env already exists so torch is importable.
        Falls back to showing the install dialog only when the env folder
        itself is missing (i.e. never been installed).
        """
        try:
            from .DL.env_manager import patch_sys_path
        except Exception:
            return  # env_manager unavailable — skip silently

        # If the env site-packages exist, patch sys.path unconditionally.
        # is_env_ready() spawns a subprocess which can fail inside QGIS even
        # when torch is perfectly importable, so we skip that check here.
        if patch_sys_path():
            return

        # Env folder is genuinely missing — show the setup dialog
        try:
            from .ui.install_dialog import InstallDialog
        except Exception:
            return

        dlg = InstallDialog(self.iface.mainWindow())
        dlg.exec_()

        if dlg.was_installed():
            patch_sys_path()
