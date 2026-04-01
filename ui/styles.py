"""
module: ui/styles.py

Central design system for the Deep Learning Plugin UI.

Defines the color palette and provides helper functions that apply
consistent stylesheet rules to specific widget types.  Import and
call these helpers in widget __init__ methods rather than writing
inline stylesheets in every file.

Usage
-----
    from .styles import Palette, style_primary_btn, style_run_progress

    style_primary_btn(self.run_btn)
    style_danger_btn(self.stop_btn)
    style_secondary_btn(self.browse_btn)
    style_run_progress(self.progress_bar)
"""


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

class Palette:
    # Brand / accent
    PRIMARY        = "#2563EB"   # blue — buttons, left-border accents
    PRIMARY_DARK   = "#1D4ED8"   # hover / pressed state
    PRIMARY_LIGHT  = "#EFF6FF"   # tinted backgrounds

    # Section header
    HEADER_BG      = "#DBEAFE"   # blue-100 — very light blue tint, subtle/transparent feel
    HEADER_BORDER  = "#93C5FD"   # blue-300 left-border accent — matches pale header tone
    HEADER_TEXT    = "#1E3A5F"   # dark navy text — readable on light blue bg
    HEADER_SEP     = "#BFDBFE"   # blue-200 separator

    # Content area
    CONTENT_BG     = "#FFFFFF"
    CONTENT_BORDER = "#E2E8F0"   # subtle border around content

    # Metadata cards
    CARD_BG        = "#F8FAFC"
    CARD_BORDER    = "#E2E8F0"
    CARD_LABEL     = "#64748B"   # secondary text (slate-500)
    CARD_VALUE     = "#1E293B"   # primary text

    # Status colours
    SUCCESS        = "#16A34A"
    SUCCESS_BG     = "#F0FDF4"
    SUCCESS_BORDER = "#86EFAC"

    WARNING        = "#D97706"
    WARNING_BG     = "#FFFBEB"
    WARNING_BORDER = "#FCD34D"

    ERROR          = "#DC2626"
    ERROR_BG       = "#FEF2F2"
    ERROR_BORDER   = "#FCA5A5"

    # Neutral
    SEPARATOR      = "#E2E8F0"
    DISABLED_TEXT  = "#94A3B8"
    DISABLED_BG    = "#F1F5F9"


# ---------------------------------------------------------------------------
# Button helpers
# ---------------------------------------------------------------------------

def style_primary_btn(btn):
    """Filled slate button — primary action (Run, Apply, Start Training …).
    Light default, darkens on hover."""
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: #6366F1;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0px 10px;
            min-height: 32px;
            max-height: 32px;
            font-weight: bold;
            font-size: 11px;
        }}
        QPushButton:hover {{
            background-color: #4F46E5;
        }}
        QPushButton:pressed {{
            background-color: #4338CA;
        }}
        QPushButton:disabled {{
            background-color: {Palette.DISABLED_BG};
            color: {Palette.DISABLED_TEXT};
        }}
    """)


def style_danger_btn(btn):
    """Outlined red button — destructive / stop action."""
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: white;
            color: {Palette.ERROR};
            border: 1px solid #FCA5A5;
            border-radius: 4px;
            padding: 0px 14px;
            min-height: 32px;
            max-height: 32px;
            font-size: 11px;
        }}
        QPushButton:hover {{
            background-color: {Palette.ERROR_BG};
            border-color: {Palette.ERROR};
        }}
        QPushButton:pressed {{
            background-color: #FEE2E2;
        }}
        QPushButton:disabled {{
            color: {Palette.DISABLED_TEXT};
            border-color: {Palette.SEPARATOR};
            background-color: white;
        }}
    """)


def style_secondary_btn(btn):
    """Outlined neutral button — secondary action (Browse, Refresh …)."""
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: white;
            color: #374151;
            border: 1px solid #D1D5DB;
            border-radius: 4px;
            padding: 5px 12px;
            font-size: 11px;
        }}
        QPushButton:hover {{
            background-color: #F9FAFB;
            border-color: #9CA3AF;
        }}
        QPushButton:pressed {{
            background-color: #F3F4F6;
        }}
        QPushButton:disabled {{
            color: {Palette.DISABLED_TEXT};
            border-color: {Palette.SEPARATOR};
        }}
    """)


def style_icon_btn(btn):
    """Small square icon-only button (…, ↻) — no visible border at rest."""
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: transparent;
            color: {Palette.PRIMARY};
            border: 1px solid transparent;
            border-radius: 4px;
            padding: 4px 6px;
            font-size: 12px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {Palette.PRIMARY_LIGHT};
            border-color: {Palette.CARD_BORDER};
        }}
        QPushButton:pressed {{
            background-color: #DBEAFE;
        }}
        QPushButton:disabled {{
            color: {Palette.DISABLED_TEXT};
        }}
    """)


# ---------------------------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------------------------

def style_progress_bar(bar, color=None):
    """Styled progress bar — accent coloured fill, rounded, slim."""
    fill = color or Palette.PRIMARY
    bar.setFixedHeight(10)
    bar.setTextVisible(False)
    bar.setStyleSheet(f"""
        QProgressBar {{
            background-color: {Palette.CARD_BG};
            border: 1px solid {Palette.CARD_BORDER};
            border-radius: 5px;
        }}
        QProgressBar::chunk {{
            background-color: {fill};
            border-radius: 5px;
        }}
    """)


def style_success_progress_bar(bar):
    style_progress_bar(bar, color=Palette.SUCCESS)
