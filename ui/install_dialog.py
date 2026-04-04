"""
module: ui/install_dialog.py

First-run setup dialog — creates the plugin virtualenv and installs PyTorch.

Flow
----
1. User selects their GPU / CUDA option from a combo box.
2. Clicks "Install".
3. A QThread worker:
     a. calls env_manager.create_env()
     b. streams pip output to a QTextEdit log
4. On success the dialog closes and the caller patches sys.path.
5. "Skip" closes without installing (plugin runs without GPU support).
"""

from __future__ import annotations

import subprocess

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QTextEdit, QProgressBar,
    QSizePolicy,
)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import QFont

from ..DL.env_manager import CUDA_OPTIONS, create_env, get_pip_cmd


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _InstallWorker(QThread):
    """Creates the env, then runs pip, streaming output line-by-line."""

    log_line = pyqtSignal(str)   # one line of pip output
    finished = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, cuda_key: str, parent=None):
        super().__init__(parent)
        self._cuda_key = cuda_key

    def run(self):
        # Step 1 — create the venv
        self.log_line.emit("Creating virtual environment …")
        ok, msg = create_env()
        if not ok:
            self.finished.emit(False, f"Failed to create environment:\n{msg}")
            return
        self.log_line.emit("Virtual environment created.\n")

        # Step 2 — install packages
        cmd = get_pip_cmd(self._cuda_key)
        self.log_line.emit("Running: " + " ".join(cmd) + "\n")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                self.log_line.emit(line.rstrip())
            proc.wait()

            if proc.returncode == 0:
                self.finished.emit(True, "Installation complete.")
            else:
                self.finished.emit(
                    False, f"pip exited with code {
                        proc.returncode}.")
        except Exception as exc:
            self.finished.emit(False, str(exc))


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class InstallDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GeoSeg Studio — Setup")
        self.setMinimumWidth(560)
        self.setMinimumHeight(420)
        self._worker: _InstallWorker | None = None
        self._installed = False

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # --- Intro -----------------------------------------------------------
        intro = QLabel(
            "<b>Welcome to GeoSeg Studio.</b><br><br>"
            "PyTorch needs to be installed in an isolated environment "
            "before you can train or run models.<br>"
            "Select the option that matches your hardware, then click "
            "<b>Install</b>."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # --- CUDA option picker ----------------------------------------------
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Hardware / CUDA:"))

        self.cuda_combo = QComboBox()
        for key, (label, _) in CUDA_OPTIONS.items():
            self.cuda_combo.addItem(label, key)
        # Default to CPU (last item) so users don't accidentally pick wrong
        # CUDA
        self.cuda_combo.setCurrentIndex(len(CUDA_OPTIONS) - 1)
        opt_row.addWidget(self.cuda_combo, 1)
        layout.addLayout(opt_row)

        # --- Log window ------------------------------------------------------
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Courier New", 9))
        self.log_edit.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding)
        self.log_edit.setPlaceholderText(
            "Installation output will appear here …")
        layout.addWidget(self.log_edit)

        # --- Progress bar ----------------------------------------------------
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)   # indeterminate
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # --- Buttons ---------------------------------------------------------
        btn_row = QHBoxLayout()

        self.install_btn = QPushButton("Install")
        self.install_btn.setDefault(True)
        self.install_btn.clicked.connect(self._start_install)

        self.skip_btn = QPushButton("Skip  (install later)")
        self.skip_btn.clicked.connect(self.reject)

        self.close_btn = QPushButton("Close")
        self.close_btn.setVisible(False)
        self.close_btn.clicked.connect(self.accept)

        btn_row.addStretch()
        btn_row.addWidget(self.skip_btn)
        btn_row.addWidget(self.install_btn)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

    # -------------------------------------------------------------------------

    def _start_install(self):
        cuda_key = self.cuda_combo.currentData()

        self.install_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.cuda_combo.setEnabled(False)
        self.progress.setVisible(True)
        self.log_edit.clear()

        self._worker = _InstallWorker(cuda_key, parent=self)
        self._worker.log_line.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, line: str):
        self.log_edit.append(line)
        # Auto-scroll to bottom
        sb = self.log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, success: bool, message: str):
        self.progress.setVisible(False)
        self._installed = success

        if success:
            self._append_log(f"\n✓ {message}")
            self.close_btn.setVisible(True)
            self.skip_btn.setVisible(False)
        else:
            self._append_log(f"\n✗ {message}")
            self.install_btn.setEnabled(True)
            self.skip_btn.setEnabled(True)
            self.cuda_combo.setEnabled(True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def was_installed(self) -> bool:
        """Returns True if the installation completed successfully."""
        return self._installed
