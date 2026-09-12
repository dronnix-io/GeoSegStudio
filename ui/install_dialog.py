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

import subprocess  # nosec B404 — runs the env's pip via a fixed argv list.

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QTextEdit, QProgressBar,
    QSizePolicy,
)
from qgis.PyQt.QtCore import QThread, pyqtSignal
from qgis.PyQt.QtGui import QFont

from ..DL.env_manager import (
    CUDA_OPTIONS, ENV_DIR, create_env, get_pip_cmd, no_window_kwargs,
    recommend_cuda_key,
)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _InstallWorker(QThread):
    """Creates the env, then runs pip, streaming output line-by-line."""

    log_line = pyqtSignal(str)   # one line of pip output
    # Deliberately NOT named `finished`: QThread already defines a finished()
    # signal, and shadowing it breaks Qt's own thread teardown notifications.
    install_done = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, cuda_key: str, parent=None):
        super().__init__(parent)
        self._cuda_key = cuda_key

    def run(self):
        # Step 1 — create the venv
        self.log_line.emit("Creating virtual environment …")
        ok, msg = create_env()
        if not ok:
            self.install_done.emit(False, f"Failed to create environment:\n{msg}")
            return
        self.log_line.emit("Virtual environment created.\n")

        # Step 2 — install packages
        cmd = get_pip_cmd(self._cuda_key)
        self.log_line.emit("Running: " + " ".join(cmd) + "\n")

        try:
            # cmd is built by get_pip_cmd() from a validated key — a fixed
            # argv list, never a shell string, so nothing here is injectable.
            proc = subprocess.Popen(  # nosec B603
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                **no_window_kwargs(),
            )
            for line in proc.stdout:
                self.log_line.emit(line.rstrip())
            proc.wait()

            if proc.returncode == 0:
                self.install_done.emit(True, "Installation complete.")
            else:
                self.install_done.emit(
                    False, f"pip exited with code {proc.returncode}."
                )
        except Exception as exc:
            self.install_done.emit(False, str(exc))


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

        # Reinstalling deletes and recreates the environment. On Windows that
        # fails if torch is already loaded into this QGIS process, so say so
        # up front rather than after a 5 GB download.
        if ENV_DIR.exists():
            warn = QLabel(
                "<b>An environment already exists.</b> Installing again "
                "replaces it — use this to switch between CPU and CUDA "
                "builds, or after updating your NVIDIA driver.<br>"
                "If PyTorch has already been used in this QGIS session, "
                "restart QGIS first, then reinstall without opening the "
                "GeoSeg Studio panel; otherwise Windows keeps the files "
                "locked and the rebuild fails."
            )
            warn.setWordWrap(True)
            layout.addWidget(warn)

        # --- CUDA option picker ----------------------------------------------
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Hardware / CUDA:"))

        self.cuda_combo = QComboBox()
        for key, (label, _url, _drv) in CUDA_OPTIONS.items():
            self.cuda_combo.addItem(label, key)

        # Preselect what this machine can actually run, rather than defaulting
        # to CPU. Defaulting to CPU meant anyone who clicked straight through
        # got a CPU-only build, and since this dialog only opens when no env
        # exists, there was no way back to a GPU build afterwards.
        recommended, reason = recommend_cuda_key()
        idx = self.cuda_combo.findData(recommended)
        if idx >= 0:
            self.cuda_combo.setCurrentIndex(idx)
        opt_row.addWidget(self.cuda_combo, 1)
        layout.addLayout(opt_row)

        self.detect_label = QLabel(reason)
        self.detect_label.setWordWrap(True)
        layout.addWidget(self.detect_label)

        # --- Log window ------------------------------------------------------
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Courier New", 9))
        self.log_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding)
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
        self._worker.install_done.connect(self._on_finished)
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

    def closeEvent(self, event):
        """
        Blocks closing while the install thread is alive.

        Destroying the dialog would destroy its child QThread mid-run, which
        crashes QGIS rather than just cancelling the install.
        """
        worker = self._worker
        if worker is not None and worker.isRunning():
            self._append_log(
                "\nInstallation is still running — please wait for it to finish."
            )
            event.ignore()
            return
        super().closeEvent(event)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def was_installed(self) -> bool:
        """Returns True if the installation completed successfully."""
        return self._installed
