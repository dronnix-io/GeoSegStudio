"""
module: ui/links_rail.py

The community links rail down the left edge of the GeoSeg Studio panel.

GeoSeg Studio is free and open source, so this gives users a one-click path
back to the project: the repository, report a bug, request a feature, or open
the handbook. The bottom icon is context-aware — it points at the handbook
chapter for whichever tab is currently open.

The rail is icon-only by design. Rotated text is hard to read at this width
and vertical labels would force the rail wide enough to eat into the panel, so
every action is an icon with a tooltip instead.

The icons are bundled SVGs rather than emoji. In an icon-only rail there is no
label to fall back on, so on a system without an emoji font the whole strip
would degrade to a column of empty squares explaining nothing.
"""

from __future__ import annotations

import os

from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QToolButton, QFrame
from qgis.PyQt.QtGui import QIcon, QDesktopServices
from qgis.PyQt.QtCore import QUrl, QSize, Qt

from .styles import Palette

_REPO_URL = "https://github.com/dronnix-io/GeoSegStudio"
_BUG_URL = _REPO_URL + "/issues/new?labels=bug"
_FEATURE_URL = _REPO_URL + "/issues/new?labels=enhancement"
_HANDBOOK_URL = _REPO_URL + "/tree/main/docs/handbook"

# Tab name -> (tooltip subject, handbook chapter). The chapters live in the
# repository, so these resolve on GitHub with no site or login required.
_CHAPTER = _REPO_URL + "/blob/main/docs/handbook/"
_TAB_GUIDES = {
    "Prepare":  ("Preparing a training dataset", _CHAPTER + "03_data_preparation.md"),
    "Train":    ("Training that converges", _CHAPTER + "05_training.md"),
    "Evaluate": ("Evaluating your model honestly", _CHAPTER + "06_evaluation.md"),
    "Predict":  ("From prediction to GIS product", _CHAPTER + "07_prediction.md"),
}
_DEFAULT_GUIDE = ("Read the handbook", _HANDBOOK_URL)

RAIL_WIDTH = 34


class LinksRail(QWidget):
    """A narrow vertical strip of project links for the left edge of the panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("linksRail")
        self.setFixedWidth(RAIL_WIDTH)

        col = QVBoxLayout(self)
        col.setContentsMargins(3, 6, 3, 6)
        col.setSpacing(2)

        col.addWidget(self._button(
            "github.svg", "Open the GeoSeg Studio repository on GitHub",
            _REPO_URL))
        col.addWidget(self._rule())
        col.addWidget(self._button(
            "icon_bug.svg", "Report a bug — opens a new GitHub issue", _BUG_URL))
        col.addWidget(self._button(
            "icon_feature.svg", "Request a feature — opens a new GitHub issue",
            _FEATURE_URL))
        col.addWidget(self._button(
            "icon_handbook.svg", "Read the GeoSeg Studio handbook", _HANDBOOK_URL))

        col.addStretch()

        # Context link, updated by set_tab() as the user moves between tabs.
        col.addWidget(self._rule())
        self._guide_btn = self._button(
            "icon_guide.svg", "", _HANDBOOK_URL, name="railGuide")
        col.addWidget(self._guide_btn)
        self.set_tab(None)

        self.setStyleSheet(f"""
            QWidget#linksRail {{
                background-color: {Palette.CARD_BG};
                border-right: 1px solid {Palette.CONTENT_BORDER};
            }}
            QToolButton {{
                color: {Palette.CARD_LABEL}; background: transparent;
                border: none; border-radius: 4px;
                font-size: 13px; padding: 0px;
            }}
            QToolButton:hover {{ background-color: {Palette.PRIMARY_LIGHT}; }}
            QToolButton:pressed {{ background-color: {Palette.HEADER_BG}; }}
            QToolButton#railGuide {{ font-size: 13px; }}
        """)

    # -------------------------------------------------------------------------

    def _button(self, icon_file, tip, url, name=None):
        """One flat square button that opens `url` in the system browser."""
        btn = QToolButton()
        path = os.path.join(os.path.dirname(__file__), icon_file)
        if os.path.isfile(path):
            btn.setIcon(QIcon(path))
            btn.setIconSize(QSize(16, 16))
        if name:
            btn.setObjectName(name)
        btn.setFixedSize(28, 26)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tip)
        btn.setAutoRaise(True)
        btn.clicked.connect(
            lambda _=False, u=url: QDesktopServices.openUrl(QUrl(u)))
        return btn

    def _rule(self):
        """A hairline separator between icon groups."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setFixedHeight(1)
        line.setStyleSheet(
            f"background-color: {Palette.CONTENT_BORDER}; border: none;")
        return line

    def set_tab(self, tab_name):
        """Points the bottom icon at the handbook chapter for `tab_name`."""
        subject, url = _TAB_GUIDES.get(tab_name, _DEFAULT_GUIDE)
        self._guide_btn.setToolTip(f"Handbook: {subject}")
        try:
            self._guide_btn.clicked.disconnect()
        except TypeError:
            pass  # nothing connected yet
        self._guide_btn.clicked.connect(
            lambda _=False, u=url: QDesktopServices.openUrl(QUrl(u)))
