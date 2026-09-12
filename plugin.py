import os

from qgis.PyQt.QtWidgets import QAction, QApplication
from qgis.PyQt.QtGui import QIcon
from .ui.main_ui import GeoSegStudioDockWidget
from qgis.PyQt.QtCore import Qt
from qgis.core import Qgis
from .log_utils import log_warning

# The GeoSeg Studio logo, loaded from disk.
#
# This used to be QIcon(":/plugins/GeoSegStudio/icon.png"). The ":/" prefix is
# a Qt *resource* path, which only resolves once a resources.py compiled from a
# .qrc file has been imported — this plugin has neither, so that QIcon was null
# and the toolbar button rendered as blank space with nothing to click on.
ICON_PATH = os.path.join(os.path.dirname(__file__), "icon.png")


class GeoSegStudioPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.dock_widget = None
        self.action = None
        self.setup_action = None

    def initGui(self):
        icon = QIcon(ICON_PATH)
        self.action = QAction(icon, "GeoSeg Studio", self.iface.mainWindow())
        self.action.setToolTip("GeoSeg Studio — deep learning segmentation")
        self.action.triggered.connect(self.show_dock)
        self.iface.addPluginToMenu("GeoSeg Studio", self.action)
        self.iface.addToolBarIcon(self.action)

        # Always-reachable way back into the PyTorch setup. Without this the
        # install dialog only ever appeared when no environment existed, so a
        # user who accepted a CPU-only build — or who later updated their
        # NVIDIA driver — had no way to switch to a GPU build.
        self.setup_action = QAction(
            icon, "GeoSeg Studio — PyTorch Setup…", self.iface.mainWindow())
        self.setup_action.triggered.connect(self.show_setup)
        self.iface.addPluginToMenu("GeoSeg Studio", self.setup_action)

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
                    except Exception as exc:
                        log_warning(
                            f"Could not stop the {attr}.{worker_attr} "
                            "worker during unload", exc)

        self.iface.removePluginMenu("GeoSeg Studio", self.action)
        if self.setup_action is not None:
            self.iface.removePluginMenu("GeoSeg Studio", self.setup_action)
        self.iface.removeToolBarIcon(self.action)
        if self.dock_widget:
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget = None

    def show_dock(self):
        # Building the panel the first time can block for a few seconds while
        # the environment is located and the tabs are constructed. That work is
        # on the GUI thread, so without this the window simply freezes with no
        # explanation. Show a wait cursor and a message bar entry, and flush the
        # event queue once so both actually paint before the blocking starts.
        first_open = self.dock_widget is None
        bar = None
        if first_open:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                bar = self.iface.messageBar().createMessage(
                    "GeoSeg Studio",
                    "Starting up — this can take a few seconds on first open.",
                )
                self.iface.messageBar().pushWidget(bar, Qgis.MessageLevel.Info)
                QApplication.processEvents()
            except Exception as exc:
                log_warning("Could not show the startup message", exc)

        try:
            # Check / bootstrap the PyTorch environment on first open
            self._ensure_env()

            if self.dock_widget is None:
                self.dock_widget = GeoSegStudioDockWidget(self.iface)
                self.iface.addDockWidget(
                    Qt.DockWidgetArea.RightDockWidgetArea, self.dock_widget)
            self.dock_widget.show()
            self.dock_widget.raise_()
        finally:
            if first_open:
                QApplication.restoreOverrideCursor()
                try:
                    if bar is not None:
                        self.iface.messageBar().popWidget(bar)
                except Exception as exc:
                    # Usually means the user already dismissed it, or it timed
                    # out. Harmless either way, but worth a line in the log
                    # rather than swallowing it silently.
                    log_warning("Could not dismiss the startup message", exc)

    def show_setup(self):
        """Opens the PyTorch setup dialog on demand, rebuilding if needed."""
        try:
            from .ui.install_dialog import InstallDialog
            from .DL.env_manager import patch_sys_path
        except Exception as exc:
            log_warning("Could not open the PyTorch setup dialog", exc)
            return

        dlg = InstallDialog(self.iface.mainWindow())
        dlg.exec()
        if dlg.was_installed():
            patch_sys_path()

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
            from .DL.env_manager import migrate_legacy_env, patch_sys_path
        except Exception:
            return  # env_manager unavailable — skip silently

        # Pre-1.0.1 installs kept the env inside the plugin folder. Move it out
        # before anything imports torch, while nothing in it is loaded.
        try:
            migrate_legacy_env()
        except Exception as exc:
            log_warning("Legacy environment migration failed", exc)

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
        dlg.exec()

        if dlg.was_installed():
            patch_sys_path()
