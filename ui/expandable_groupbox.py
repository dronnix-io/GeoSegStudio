"""
module: ui/expandable_groupbox.py

Collapsible section widget with a styled header bar.

The header has:
  - A 4 px sky-blue left-border accent (visible on dark background)
  - Dark slate background (Palette.HEADER_BG = #1E293B)
  - Bold title in near-white (Palette.HEADER_TEXT = #F1F5F9)
  - Collapse arrow on the right (▾ / ▸)
  - Clicking anywhere on the header toggles the section

Usage
-----
    box = ExpandableGroupBox("Model Settings")
    box.setContentLayout(my_layout)

    # Start collapsed:
    box.toggle_button.setChecked(False)
    box.toggle_content()
"""

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QFrame, QSizePolicy, QHBoxLayout, QLabel,
    QToolButton,
)
from qgis.PyQt.QtCore import Qt

from .styles import Palette


class ExpandableGroupBox(QWidget):
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)

        # ── Header bar ────────────────────────────────────────────────────────
        self._header = QFrame()
        self._header.setObjectName("EGB_header")
        self._header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._header.setStyleSheet(f"""
            QFrame#EGB_header {{
                background-color: {Palette.HEADER_BG};
                border-left: 4px solid {Palette.HEADER_BORDER};
                border-top: 1px solid {Palette.HEADER_SEP};
                border-bottom: 1px solid {Palette.HEADER_SEP};
                border-right: none;
            }}
        """)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"""
            QLabel {{
                color: {Palette.HEADER_TEXT};
                font-weight: bold;
                font-size: 12px;
                background: transparent;
                border: none;
                padding: 6px 4px;
            }}
        """)

        # Arrow toggle — sits on the right
        self.toggle_button = QToolButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setText("▾")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.toggle_button.setStyleSheet(f"""
            QToolButton {{
                border: none;
                background: transparent;
                color: {Palette.HEADER_TEXT};
                font-size: 14px;
                padding: 4px 8px;
            }}
            QToolButton:hover {{
                color: white;
            }}
        """)
        self.toggle_button.clicked.connect(self.toggle_content)

        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(8, 0, 4, 0)
        header_layout.setSpacing(0)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.toggle_button)

        # Make the whole header clickable
        self._header.mousePressEvent = self._header_clicked

        # ── Content area ──────────────────────────────────────────────────────
        self.content_area = QFrame()
        self.content_area.setObjectName("EGB_content")
        self.content_area.setFrameShape(QFrame.NoFrame)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_area.setStyleSheet(f"""
            QFrame#EGB_content {{
                background-color: {Palette.CONTENT_BG};
                border-left: 4px solid {Palette.HEADER_BORDER};
                border-bottom: 1px solid {Palette.CONTENT_BORDER};
                border-right: 1px solid {Palette.CONTENT_BORDER};
            }}
        """)
        self.content_area.setVisible(True)

        # ── Outer layout ──────────────────────────────────────────────────────
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.main_layout.addWidget(self._header)
        self.main_layout.addWidget(self.content_area)

    # -------------------------------------------------------------------------

    def _header_clicked(self, event):
        """Toggle when the user clicks anywhere on the header bar."""
        self.toggle_button.setChecked(not self.toggle_button.isChecked())
        self.toggle_content()

    def toggle_content(self):
        visible = self.toggle_button.isChecked()
        self.toggle_button.setText("▾" if visible else "▸")
        self.content_area.setVisible(visible)

    def setContentLayout(self, layout):
        self.content_area.setLayout(layout)
