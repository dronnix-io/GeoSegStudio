"""
module: DL/env_manager.py

Manages the isolated Python virtual environment used by the plugin.

The env is created at  <plugin_dir>/env/  with --system-site-packages so
that QGIS built-ins (numpy, gdal, …) remain accessible.  PyTorch and
torchvision are installed on first run via the InstallDialog.

Public surface
--------------
CUDA_OPTIONS   dict  — {key: (label, whl_url)} for the install dialog
is_env_ready() bool  — True when torch is importable from the env
get_env_python()     — path to env Python executable
get_pip_cmd(cuda_key) → list[str]  — full pip install argv
patch_sys_path()     — prepend env site-packages to sys.path (called
                        once at plugin startup when env already exists)
"""

from __future__ import annotations

import os
import sys
import subprocess
import importlib.util
from pathlib import Path


# ---------------------------------------------------------------------------
# Plugin root  (this file lives at  <root>/DL/env_manager.py)
# ---------------------------------------------------------------------------
_PLUGIN_DIR = Path(__file__).resolve().parent.parent

# The isolated virtualenv lives here
ENV_DIR = _PLUGIN_DIR / "env"

# ---------------------------------------------------------------------------
# CUDA installation options shown in the install dialog
# ---------------------------------------------------------------------------
CUDA_OPTIONS: dict[str, tuple[str, str]] = {
    "cuda118": (
        "NVIDIA GPU — CUDA 11.8  (GTX/RTX 10xx, 20xx, 30xx, older drivers)",
        "https://download.pytorch.org/whl/cu118",
    ),
    "cuda121": (
        "NVIDIA GPU — CUDA 12.1  (RTX 30xx, 40xx)",
        "https://download.pytorch.org/whl/cu121",
    ),
    "cuda124": (
        "NVIDIA GPU — CUDA 12.4  (RTX 40xx, latest drivers)",
        "https://download.pytorch.org/whl/cu124",
    ),
    "cpu": (
        "CPU only  (no NVIDIA GPU / AMD / Intel / unsure)",
        "https://download.pytorch.org/whl/cpu",
    ),
}

# Packages installed regardless of CUDA choice
_BASE_PACKAGES = ["torch", "torchvision"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _site_packages() -> Path | None:
    """Returns the site-packages directory inside the env, or None."""
    if sys.platform == "win32":
        candidates = [
            ENV_DIR / "Lib" / "site-packages",
        ]
    else:
        # Linux / macOS: Lib/pythonX.Y/site-packages
        for entry in ENV_DIR.glob("lib/python*/site-packages"):
            return entry
        candidates = []

    for c in candidates:
        if c.is_dir():
            return c
    return None


def get_env_python() -> str:
    """Returns the path to the env Python executable."""
    if sys.platform == "win32":
        return str(ENV_DIR / "Scripts" / "python.exe")
    return str(ENV_DIR / "bin" / "python")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_env_ready() -> bool:
    """Returns True when the env exists and torch is importable from it."""
    python = get_env_python()
    if not Path(python).is_file():
        return False

    try:
        result = subprocess.run(
            [python, "-c", "import torch"],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def patch_sys_path() -> bool:
    """
    Prepends the env site-packages to sys.path so subsequent
    ``import torch`` calls resolve to the env installation.

    Also redirects multiprocessing.spawn to use the env Python executable
    instead of sys.executable (which inside QGIS is qgis.exe).  Without
    this, any torch/CUDA operation that spawns a worker process opens a
    new QGIS window instead of a Python process.

    Returns True if the path was added, False if not found.
    """
    sp = _site_packages()
    if sp is None:
        return False
    sp_str = str(sp)
    if sp_str not in sys.path:
        sys.path.insert(0, sp_str)

    # Redirect multiprocessing spawn to the env Python, not qgis.exe.
    try:
        import multiprocessing
        multiprocessing.set_executable(get_env_python())
    except Exception:
        pass

    return True


def get_pip_cmd(cuda_key: str) -> list[str]:
    """
    Builds the pip install command for the given CUDA option key.

    Example output (win32, cuda121):
        ['<env>/Scripts/python.exe', '-m', 'pip', 'install',
         'torch', 'torchvision',
         '--index-url', 'https://download.pytorch.org/whl/cu121']
    """
    if cuda_key not in CUDA_OPTIONS:
        raise ValueError(f"Unknown CUDA option: {cuda_key!r}")

    _, whl_url = CUDA_OPTIONS[cuda_key]
    return [
        get_env_python(),
        "-m", "pip", "install",
        *_BASE_PACKAGES,
        "--index-url", whl_url,
    ]


def create_env() -> tuple[bool, str]:
    """
    Creates the virtualenv at ENV_DIR with --system-site-packages.

    Returns (success, message).
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(ENV_DIR)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return False, result.stderr.strip() or "venv creation failed."
        # Upgrade pip silently
        subprocess.run(
            [get_env_python(), "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True,
            timeout=60,
        )
        return True, "Environment created."
    except Exception as exc:
        return False, str(exc)
