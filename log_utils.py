"""
module: log_utils.py

Thin wrapper over the QGIS message log.

Used by the best-effort ``except`` blocks throughout the plugin. Those blocks
must not abort the surrounding operation — a checkpoint whose optimizer state
no longer matches, an optional preview mask that could not be written — but
swallowing the exception silently leaves users and bug reports with nothing to
go on. Logging keeps the failure recoverable *and* visible in
View ▸ Panels ▸ Log Messages.
"""

from __future__ import annotations

LOG_TAG = "GeoSeg Studio"


def log_warning(message: str, exc: BaseException | None = None) -> None:
    """
    Records a non-fatal problem in the QGIS log panel.

    Never raises: logging failures must not mask the original error.
    """
    text = f"{message}: {exc}" if exc is not None else message
    try:
        from qgis.core import QgsMessageLog, Qgis
        QgsMessageLog.logMessage(text, LOG_TAG, level=Qgis.MessageLevel.Warning)
    except Exception:
        # Outside QGIS (tests, scripts) fall back to stderr.
        import sys
        print(f"[{LOG_TAG}] {text}", file=sys.stderr)
