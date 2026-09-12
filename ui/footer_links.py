"""
module: ui/footer_links.py

The community links strip pinned at the bottom of the GeoSeg Studio panel.

GeoSeg Studio is free and open source, so this gives users a one-click path
back to the project: report a bug, request a feature, or open the handbook.
The rightmost link is context-aware — it points at the handbook chapter for
whichever tab is currently open, so "where do I read about this?" is always
one click away rather than a search.

Everything opens in the system browser. Styled from the shared Palette as flat
borderless links so the strip reads as a footer and does not compete with the
controls above it.
"""

from __future__ import annotations

import os

from qgis.PyQt.QtWidgets import QWidget, QHBoxLayout, QPushButton, QFrame
from qgis.PyQt.QtGui import QIcon, QDesktopServices
from qgis.PyQt.QtCore import QUrl, QSize, Qt

from .styles import Palette

_REPO_URL = "https://github.com/dronnix-io/GeoSegStudio"
_BUG_URL = _REPO_URL + "/issues/new?labels=bug"
_FEATURE_URL = _REPO_URL + "/issues/new?labels=enhancement"
_HANDBOOK_URL = _REPO_URL + "/tree/main/docs/handbook"

# Tab name -> (link label, handbook chapter). The chapters live in the repo,
# so these resolve on GitHub for anyone, with no site or login required.
_CHAPTER = _REPO_URL + "/blob/main/docs/handbook/"
_TAB_GUIDES = {
    "Prepare":  ("Preparing data", _CHAPTER + "03_data_preparation.md"),
    "Train":    ("Training guide", _CHAPTER + "05_training.md"),
    "Evaluate": ("Evaluation guide", _CHAPTER + "06_evaluation.md"),
    "Predict":  ("Prediction guide", _CHAPTER + "07_prediction.md"),
}
_DEFAULT_GUIDE = ("Handbook", _HANDBOOK_URL)


class LinksFooter(QWidget):
    """A slim strip of project links, with a tab-aware guide link on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("linksFooter")

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 3, 8, 4)
        row.setSpacing(2)

        icon_path = os.path.join(os.path.dirname(__file__), "github.svg")
        gh_icon = QIcon(icon_path) if os.path.isfile(icon_path) else None

        # The GitHub mark plus label is itself a link to the repository, and
        # names the group of links that follow it.
        row.addWidget(self._link(
            " GitHub", "Open the GeoSeg Studio repository on GitHub",
            _REPO_URL, object_name="footerRepo", icon=gh_icon))

        for label, tip, url in (
            ("🐛 Report Bug", "Report a bug — opens a new GitHub issue", _BUG_URL),
            ("💡 Request Feature", "Suggest a feature — opens a new GitHub issue", _FEATURE_URL),
            ("📖 Handbook", "Read the GeoSeg Studio handbook", _HANDBOOK_URL),
        ):
            row.addWidget(self._link(label, tip, url))

        row.addStretch()

        # Context link, updated by set_tab() as the user moves between tabs.
        self._guide_btn = self._link(
            "", "", _HANDBOOK_URL, object_name="footerGuide")
        row.addWidget(self._guide_btn)
        self.set_tab(None)

        self.setStyleSheet(f"""
            QWidget#linksFooter {{
                background-color: {Palette.CARD_BG};
                border-top: 1px solid {Palette.SEPARATOR};
            }}
            QPushButton#footerRepo {{
                color: {Palette.CARD_VALUE}; background: transparent;
                border: none; font-size: 11px; font-weight: bold;
                padding: 1px 2px; text-align: left;
            }}
            QPushButton#footerRepo:hover {{ color: {Palette.PRIMARY}; }}
            QPushButton#footerLink {{
                color: {Palette.CARD_LABEL}; background: transparent;
                border: none; font-size: 10px; padding: 1px 6px;
                text-align: left;
            }}
            QPushButton#footerLink:hover {{ color: {Palette.PRIMARY}; }}
            QPushButton#footerGuide {{
                color: {Palette.PRIMARY}; background: transparent;
                border: none; font-size: 10px; padding: 1px 2px;
                text-align: right;
            }}
            QPushButton#footerGuide:hover {{ color: {Palette.PRIMARY_DARK}; }}
        """)

    # -------------------------------------------------------------------------

    def _link(self, text, tip, url, object_name="footerLink", icon=None):
        """One flat, borderless link that opens `url` in the system browser."""
        btn = QPushButton(text)
        btn.setObjectName(object_name)
        btn.setFlat(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tip)
        if icon is not None:
            btn.setIcon(icon)
            btn.setIconSize(QSize(15, 15))
        btn.clicked.connect(
            lambda _=False, u=url: QDesktopServices.openUrl(QUrl(u)))
        return btn

    def set_tab(self, tab_name):
        """Points the right-hand link at the handbook chapter for `tab_name`."""
        label, url = _TAB_GUIDES.get(tab_name, _DEFAULT_GUIDE)
        self._guide_btn.setText(f"📘 {label}  ↗")
        self._guide_btn.setToolTip(f"Open the handbook: {label}")
        try:
            self._guide_btn.clicked.disconnect()
        except TypeError:
            pass  # nothing connected yet
        self._guide_btn.clicked.connect(
            lambda _=False, u=url: QDesktopServices.openUrl(QUrl(u)))


def build_separator() -> QFrame:
    """A hairline rule to sit above the footer."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    line.setStyleSheet(f"color: {Palette.SEPARATOR};")
    return line
