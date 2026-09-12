'''
module: main_ui.py
'''
from qgis.PyQt.QtWidgets import (
    QDockWidget, QTabWidget, QHBoxLayout, QWidget,
)
from qgis.PyQt.QtCore import QTimer
from .tab1 import Tab1Widget
from .tab2 import Tab2Widget
from .tab3 import Tab3Widget
from .tab4 import Tab4Widget
from .links_rail import LinksRail


class GeoSegStudioDockWidget(QDockWidget):
    def __init__(self, iface, parent=None):
        super().__init__("GeoSeg Studio", parent)
        self.setFont(iface.mainWindow().font())  # Good here!
        self.iface = iface
        self.setObjectName("GeoSegStudioDockWidget")
        self.setMinimumWidth(500)

        main_widget = QWidget()
        # Horizontal: a narrow links rail pinned down the left edge, then the
        # tabs filling the rest. No margins or spacing, so the rail is flush
        # against the panel edge and reads as part of the panel rather than
        # something floating next to it.
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.rail = LinksRail()
        layout.addWidget(self.rail)

        self.tab2 = Tab2Widget()
        self.tab3 = Tab3Widget()
        self.tab4 = Tab4Widget()

        self.tabs = QTabWidget()
        self.tabs.addTab(Tab1Widget(), "Prepare")
        self.tabs.addTab(self.tab2, "Train")
        self.tabs.addTab(self.tab3, "Evaluate")
        self.tabs.addTab(self.tab4, "Predict")

        layout.addWidget(self.tabs, 1)

        self.setWidget(main_widget)

        # The Train/Evaluate/Predict tabs each detect compute devices, which
        # imports torch and costs several seconds. Detection is deferred until
        # its tab is first opened so this panel appears immediately; the
        # single-shot timer lets the tab paint (showing "Detecting devices…")
        # before the blocking import starts.
        self.tabs.currentChanged.connect(self._on_tab_changed)
        QTimer.singleShot(0, lambda: self._on_tab_changed(self.tabs.currentIndex()))

    def _on_tab_changed(self, index):
        """Points the rail at the right chapter and triggers device detection."""
        self.rail.set_tab(self.tabs.tabText(index))

        widget = self.tabs.widget(index)
        if widget is None:
            return
        for attr in ("hardware", "run", "settings"):
            target = getattr(widget, attr, None)
            loader = getattr(target, "ensure_devices_loaded", None)
            if loader is not None:
                QTimer.singleShot(0, loader)
