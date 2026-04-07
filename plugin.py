from qgis.PyQt.QtWidgets import QAction, QMessageBox
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

        self._offer_env_removal()

    def _offer_env_removal(self):
        """
        Asks the user whether to delete the PyTorch virtual environment
        when the plugin is unloaded (i.e. uninstalled via Plugin Manager).
        The env folder can be several GB — skipping it silently is the
        main reason reinstalls fail or leave orphaned data.
        """
        try:
            from .DL.env_manager import ENV_DIR
        except Exception:
            return

        if not ENV_DIR.exists():
            return

        reply = QMessageBox.question(
            None,
            "GeoSeg Studio — Remove PyTorch Environment?",
            "The PyTorch environment installed by GeoSeg Studio is still present:\n\n"
            f"  {ENV_DIR}\n\n"
            "Do you want to delete it now?\n\n"
            "• Yes — removes the environment (~3–5 GB freed)\n"
            "• No  — keeps it (useful if you plan to reinstall)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            try:
                import shutil
                shutil.rmtree(ENV_DIR, ignore_errors=True)
            except Exception:
                pass

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
