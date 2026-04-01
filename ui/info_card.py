"""
module: ui/info_card.py

MetaCardGrid — a horizontal flow of labelled key-value badge cards.

Replaces sparse QFormLayout metadata rows with compact visual cards that
make good use of horizontal space and are easier to scan at a glance.

Usage
-----
    grid = MetaCardGrid()
    grid.set_cards([
        ("Architecture", "UNet"),
        ("Channels", "4"),
        ("Image Size", "512 px"),
        ("Epoch", "50"),
        ("Val IoU", "0.847"),
    ])

    # Update a single card value (label must already exist):
    grid.update_card("Val IoU", "0.912")

    # Clear all cards (e.g. when a new file is selected):
    grid.clear_cards()
"""

from qgis.PyQt.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy, QFrame,
)
from qgis.PyQt.QtCore import Qt

from .styles import Palette


class _Card(QFrame):
    """Single key-value badge card."""

    def __init__(self, label: str, value: str, parent=None):
        super().__init__(parent)
        self.setObjectName("MetaCard")
        self.setStyleSheet(f"""
            QFrame#MetaCard {{
                background-color: {Palette.CARD_BG};
                border: 1px solid {Palette.CARD_BORDER};
                border-radius: 6px;
                padding: 0px;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(1)

        self._label_widget = QLabel(label)
        self._label_widget.setStyleSheet(f"""
            QLabel {{
                color: {Palette.CARD_LABEL};
                font-size: 10px;
                background: transparent;
                border: none;
            }}
        """)

        self._value_widget = QLabel(value)
        self._value_widget.setStyleSheet(f"""
            QLabel {{
                color: {Palette.CARD_VALUE};
                font-weight: bold;
                font-size: 11px;
                background: transparent;
                border: none;
            }}
        """)

        layout.addWidget(self._label_widget)
        layout.addWidget(self._value_widget)

    def set_value(self, value: str):
        self._value_widget.setText(value)

    def key(self) -> str:
        return self._label_widget.text()


class MetaCardGrid(QWidget):
    """
    Wrapping flow of MetaCards.  Cards are laid out left-to-right in rows
    of up to `cols_per_row` items (default 3).  The widget is hidden when
    empty and shown automatically once cards are added.
    """

    def __init__(self, cols_per_row: int = 3, parent=None):
        super().__init__(parent)
        self._cols = cols_per_row
        self._cards: list[_Card] = []

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(6)

        self.setVisible(False)

    # -------------------------------------------------------------------------

    def set_cards(self, items: list[tuple[str, str]]):
        """Replace all cards with a new set of (label, value) pairs."""
        self.clear_cards()
        for label, value in items:
            card = _Card(label, value)
            self._cards.append(card)

        self._rebuild_layout()
        self.setVisible(bool(self._cards))

    def update_card(self, label: str, value: str):
        """Update the value of an existing card (no-op if label not found)."""
        for card in self._cards:
            if card.key() == label:
                card.set_value(value)
                return

    def clear_cards(self):
        """Remove all cards and hide the widget."""
        for card in self._cards:
            card.deleteLater()
        self._cards.clear()

        # Remove all rows from outer layout
        while self._outer.count():
            item = self._outer.takeAt(0)
            if item.layout():
                # clear the row layout
                while item.layout().count():
                    item.layout().takeAt(0)

        self.setVisible(False)

    # -------------------------------------------------------------------------

    def _rebuild_layout(self):
        """Arrange cards into rows of `_cols` per row."""
        row_layout = None
        for i, card in enumerate(self._cards):
            if i % self._cols == 0:
                row_layout = QHBoxLayout()
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                self._outer.addLayout(row_layout)
            row_layout.addWidget(card)

        # Pad the last row so cards don't stretch full width
        if row_layout is not None:
            remainder = len(self._cards) % self._cols
            if remainder:
                for _ in range(self._cols - remainder):
                    row_layout.addStretch(1)
