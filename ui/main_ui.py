'''
module: main_ui.py
'''
from qgis.PyQt.QtWidgets import QDockWidget, QTabWidget, QVBoxLayout, QWidget
from qgis.PyQt.QtCore import QTimer
from .tab1 import Tab1Widget
from .tab2 import Tab2Widget
from .tab3 import Tab3Widget
from .tab4 import Tab4Widget
from .footer_links import LinksFooter


class GeoSegStudioDockWidget(QDockWidget):
    def __init__(self, iface, parent=None):
        super().__init__("GeoSeg Studio", parent)
        self.setFont(iface.mainWindow().font())  # Good here!
        self.iface = iface
        self.setObjectName("GeoSegStudioDockWidget")
        self.setMinimumWidth(500)

        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)

        self.tab2 = Tab2Widget()
        self.tab3 = Tab3Widget()
        self.tab4 = Tab4Widget()

        self.tabs = QTabWidget()
        self.tabs.addTab(Tab1Widget(), "Prepare")
        self.tabs.addTab(self.tab2, "Train")
        self.tabs.addTab(self.tab3, "Evaluate")
        self.tabs.addTab(self.tab4, "Predict")

        layout.addWidget(self.tabs)

        # Community links pinned below the tabs. The rightmost link follows the
        # active tab, so the handbook chapter for whatever the user is doing is
        # always one click away.
        self.footer = LinksFooter()
        layout.addWidget(self.footer)

        self.setWidget(main_widget)

        # The Train/Evaluate/Predict tabs each detect compute devices, which
        # imports torch and costs several seconds. Detection is deferred until
        # its tab is first opened so this panel appears immediately; the
        # single-shot timer lets the tab paint (showing "Detecting devices…")
        # before the blocking import starts.
        self.tabs.currentChanged.connect(self._on_tab_changed)
        QTimer.singleShot(0, lambda: self._on_tab_changed(self.tabs.currentIndex()))

    def _on_tab_changed(self, index):
        """Points the footer at the right chapter and triggers device detection."""
        self.footer.set_tab(self.tabs.tabText(index))

        widget = self.tabs.widget(index)
        if widget is None:
            return
        for attr in ("hardware", "run", "settings"):
            target = getattr(widget, attr, None)
            loader = getattr(target, "ensure_devices_loaded", None)
            if loader is not None:
                QTimer.singleShot(0, loader)
