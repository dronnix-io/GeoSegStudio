"""
module: DL/env_manager.py

Manages the isolated Python virtual environment used by the plugin.

The env is created under the active QGIS profile (see _env_root) with
--system-site-packages so that QGIS built-ins (numpy, gdal, …) remain
accessible.  PyTorch and torchvision are installed on first run via the
InstallDialog.

Public surface
--------------
CUDA_OPTIONS   dict  — {key: (label, whl_url, min_driver)} for the dialog
detect_gpu()         — (name, driver) from nvidia-smi, or None
recommend_cuda_key() — (key, explanation) best option for this machine
migrate_legacy_env() — move a pre-1.0.1 env out of the plugin folder
is_env_ready() bool  — True when torch is importable from the env
get_env_python()     — path to env Python executable
get_pip_cmd(cuda_key) → list[str]  — full pip install argv
patch_sys_path()     — prepend env site-packages to sys.path (called
                        once at plugin startup when env already exists)
"""

from __future__ import annotations

import sys
# subprocess is used only to run the env's own Python interpreter and
# nvidia-smi. Every call below passes a fixed argv list, never a shell string,
# and never shell=True; the executables come from get_env_python(),
# _find_system_python() or shutil.which(), not from user input. The B603
# suppressions on those calls mark that scanner false positive - unlike a real
# finding, which belongs fixed rather than silenced.
import subprocess  # nosec B404
from pathlib import Path

from ..log_utils import log_warning


# ---------------------------------------------------------------------------
# Plugin root  (this file lives at  <root>/DL/env_manager.py)
# ---------------------------------------------------------------------------
_PLUGIN_DIR = Path(__file__).resolve().parent.parent

# Where the env used to live, up to and including v1.0.0.
_LEGACY_ENV_DIR = _PLUGIN_DIR / "env"


def _env_root() -> Path:
    """
    Returns the directory that holds the managed virtualenv.

    The env deliberately lives *outside* the plugin folder, under the active
    QGIS profile. Keeping several GB of PyTorch inside the plugin directory
    breaks plugin upgrades: QGIS' Plugin Manager deletes and re-extracts that
    folder without unloading the plugin first (QGIS issue #54968), so on
    Windows the upgrade fails outright once torch's DLLs are loaded into the
    QGIS process, and when it does succeed it throws the env away and forces a
    multi-gigabyte re-download on every release.

    Falls back to the legacy in-plugin location only when QGIS is not
    importable (tests, scripts), so the module stays usable standalone.
    """
    try:
        from qgis.core import QgsApplication
    except ImportError:
        # Not running inside QGIS (tests, scripts). Expected, not a problem.
        return _PLUGIN_DIR

    try:
        profile = Path(QgsApplication.qgisSettingsDirPath())
    except Exception as exc:
        log_warning(
            "Could not resolve the QGIS profile directory; falling back to "
            "the plugin folder for the PyTorch environment", exc)
        return _PLUGIN_DIR

    if str(profile) and profile != Path("."):
        return profile / "geoseg_studio"
    return _PLUGIN_DIR


ENV_DIR = _env_root() / "env"

# ---------------------------------------------------------------------------
# CUDA installation options shown in the install dialog
# ---------------------------------------------------------------------------
# Each entry is (label, wheel index URL, minimum NVIDIA driver version).
#
# The driver minimum is what decides whether a build can run at all: a CUDA
# wheel installs happily on an old driver and then reports no GPU, which is
# the single most confusing failure mode here. CUDA minor-version
# compatibility means a 12.6 build runs on any driver that supports 12.0, so
# separate 12.1 and 12.4 entries bought nothing and only froze users on old
# torch releases (the cu121 index stops at torch 2.5.1 and has no Python 3.13
# wheels at all).
CUDA_OPTIONS: dict[str, tuple[str, str, float]] = {
    "cuda128": (
        "NVIDIA GPU — CUDA 12.8  (RTX 50xx / Blackwell, newest drivers)",
        "https://download.pytorch.org/whl/cu128",
        570.0,
    ),
    "cuda126": (
        "NVIDIA GPU — CUDA 12.6  (RTX 20xx / 30xx / 40xx, driver 527+)",
        "https://download.pytorch.org/whl/cu126",
        527.41,
    ),
    "cuda118": (
        "NVIDIA GPU — CUDA 11.8  (GTX 10xx and older drivers, 522+)",
        "https://download.pytorch.org/whl/cu118",
        522.06,
    ),
    "cpu": (
        "CPU only  (no NVIDIA GPU / AMD / Intel / unsure)",
        "https://download.pytorch.org/whl/cpu",
        0.0,
    ),
}

# GPUs that need sm_120 kernels, which only exist in CUDA 12.8+ builds.
# A 12.6 build on one of these reports a GPU and then fails at kernel launch.
_BLACKWELL_MARKERS = ("RTX 50", "RTX PRO 6000", "B200", "GB200")


def detect_gpu() -> tuple[str, float] | None:
    """
    Returns (gpu_name, driver_version) from nvidia-smi, or None.

    nvidia-smi ships with the NVIDIA driver, so its absence is itself the
    answer: no usable NVIDIA GPU on this machine.
    """
    import shutil
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        result = subprocess.run(  # nosec B603
            [exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            **no_window_kwargs(),
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        name, driver = result.stdout.strip().splitlines()[0].split(",")
        # Driver strings look like "522.06" or occasionally "560.94.01".
        parts = driver.strip().split(".")
        return name.strip(), float(f"{parts[0]}.{parts[1]}" if len(parts) > 1 else parts[0])
    except Exception:
        return None


def recommend_cuda_key() -> tuple[str, str]:
    """
    Picks the best CUDA option for this machine.

    Returns (key, explanation). Falls back to "cpu" whenever a GPU build
    would not actually work, so the default never silently wastes a GPU or
    installs a build the driver cannot run.
    """
    gpu = detect_gpu()
    if gpu is None:
        return "cpu", (
            "No NVIDIA GPU detected (nvidia-smi is not present). "
            "Training will run on the CPU."
        )

    name, driver = gpu
    is_blackwell = any(m.lower() in name.lower() for m in _BLACKWELL_MARKERS)

    if is_blackwell:
        if driver >= CUDA_OPTIONS["cuda128"][2]:
            return "cuda128", (
                f"Detected {name} (driver {driver}). This GPU needs CUDA 12.8 "
                "builds; CUDA 12.6 would install but fail at run time."
            )
        return "cpu", (
            f"Detected {name} (driver {driver}), but this GPU requires "
            f"driver {CUDA_OPTIONS['cuda128'][2]:g} or newer for CUDA 12.8. "
            "Update your NVIDIA driver, then reinstall PyTorch to use the GPU."
        )

    for key in ("cuda128", "cuda126", "cuda118"):
        if driver >= CUDA_OPTIONS[key][2]:
            label = CUDA_OPTIONS[key][0].split("(")[0].strip()
            return key, f"Detected {name} (driver {driver}). Recommended: {label}."

    return "cpu", (
        f"Detected {name}, but driver {driver} is older than "
        f"{CUDA_OPTIONS['cuda118'][2]:g} and cannot run any current PyTorch "
        "CUDA build. Update your NVIDIA driver, then reinstall PyTorch."
    )


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


def _find_system_python() -> str:
    """
    Returns the real Python executable that runs QGIS.

    Inside QGIS, ``sys.executable`` is qgis.exe, so the interpreter has to be
    located another way.  Resolution order is deliberately most-trusted first:

    1. ``sys.executable``            — already a Python interpreter.
    2. ``sys._base_executable``      — set by CPython even when embedded.
    3. Interpreter next to ``sys.base_prefix`` — the interpreter that owns the
       running standard library, i.e. QGIS' own Python.

    PATH is intentionally *not* searched.  ``shutil.which("python")`` resolves
    against the user's PATH, so a ``python.exe`` planted earlier in PATH would
    be executed with the user's privileges, and it would in any case likely be
    an unrelated interpreter with a different ABI than QGIS.
    """
    exe = sys.executable
    if exe and "python" in Path(exe).stem.lower():
        return exe

    base_exe = getattr(sys, "_base_executable", None)
    if base_exe and "python" in Path(base_exe).stem.lower() and Path(base_exe).is_file():
        return base_exe

    # Derive the interpreter from the stdlib prefix that QGIS is running on.
    base = Path(sys.base_prefix)
    if sys.platform == "win32":
        candidates = [base / "python.exe", base / "bin" / "python.exe"]
    else:
        candidates = [
            base / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}",
            base / "bin" / f"python{sys.version_info.major}",
            base / "bin" / "python",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise RuntimeError(
        "Could not locate the Python interpreter running QGIS, so the "
        "PyTorch environment cannot be created automatically. Install "
        "torch and torchvision manually — see requirements.txt."
    )


def no_window_kwargs() -> dict:
    """
    Keeps Windows from flashing a console window for each child process.

    Returns kwargs to splat into subprocess calls; empty on non-Windows.
    """
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


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
        result = subprocess.run(  # nosec B603
            [python, "-c", "import torch"],
            capture_output=True,
            timeout=15,
            **no_window_kwargs(),
        )
        return result.returncode == 0
    except Exception:
        return False


def migrate_legacy_env() -> bool:
    """
    Relocates a pre-1.0.1 env from inside the plugin folder, if one survived.

    Only helps where the old folder is still present — a manual "Install from
    ZIP" over an existing install, or an upgrade that failed on a locked file
    and was retried. A clean Plugin Manager upgrade deletes the plugin
    directory (env included) before this code ever runs, so those users
    reinstall PyTorch once; from then on the env sits outside the plugin
    folder and survives every future upgrade.

    Runs before any torch import, so nothing in the env is loaded or locked
    yet. Returns True when an env was moved.
    """
    if ENV_DIR == _LEGACY_ENV_DIR:          # standalone fallback; nothing to do
        return False
    if not _LEGACY_ENV_DIR.is_dir() or ENV_DIR.exists():
        return False

    import shutil
    try:
        ENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(_LEGACY_ENV_DIR), str(ENV_DIR))
    except Exception as exc:
        # A partial move would leave a broken env; clear it so the install
        # dialog offers a clean rebuild rather than half a virtualenv.
        log_warning(
            f"Could not move the existing PyTorch environment out of the "
            f"plugin folder ({_LEGACY_ENV_DIR}). It will be rebuilt", exc)
        if ENV_DIR.exists():
            shutil.rmtree(ENV_DIR, ignore_errors=True)
        return False

    # A venv records absolute paths, so it must be repointed after moving.
    cfg = ENV_DIR / "pyvenv.cfg"
    if cfg.is_file():
        try:
            cfg.write_text(
                cfg.read_text(encoding="utf-8").replace(
                    str(_LEGACY_ENV_DIR), str(ENV_DIR)),
                encoding="utf-8",
            )
        except Exception as exc:
            log_warning("Could not rewrite pyvenv.cfg after moving the env", exc)
    return True


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
    except Exception as exc:
        # Non-fatal: only affects torch workers that spawn subprocesses.
        log_warning("Could not redirect multiprocessing to the env Python", exc)

    return True


def get_pip_cmd(cuda_key: str) -> list[str]:
    """
    Builds the pip install command for the given CUDA option key.

    Example output (win32, cuda121):
        ['<env>/Scripts/python.exe', '-m', 'pip', 'install',
         'torch', 'torchvision',
         '--index-url', 'https://download.pytorch.org/whl/cu121']
    """
    # cuda_key comes from the install dialog's combo box, but validate anyway:
    # only keys in CUDA_OPTIONS map to an index URL, so no caller-supplied
    # string can ever reach the pip command line.
    if cuda_key not in CUDA_OPTIONS:
        raise ValueError(f"Unknown CUDA option: {cuda_key!r}")

    _, whl_url, _ = CUDA_OPTIONS[cuda_key]
    return [
        get_env_python(),
        "-m", "pip", "install",
        *_BASE_PACKAGES,
        "--index-url", whl_url,
    ]


def _has_pip() -> bool:
    """Returns True when pip is importable from the freshly created env."""
    try:
        result = subprocess.run(  # nosec B603
            [get_env_python(), "-m", "pip", "--version"],
            capture_output=True,
            timeout=60,
            **no_window_kwargs(),
        )
        return result.returncode == 0
    except Exception:
        return False


def create_env() -> tuple[bool, str]:
    """
    Creates the virtualenv at ENV_DIR with --system-site-packages.

    pip is provisioned from the local standard library only — first by letting
    ``venv`` run ensurepip itself, and if that fails by invoking ensurepip
    inside the new env directly.  Nothing is downloaded and executed to
    bootstrap the environment: an earlier version fetched get-pip.py from the
    network and ran it, which is remote code execution with no integrity
    check.  Package downloads happen later, through pip, which verifies
    hashes and TLS.

    Returns (success, message).
    """
    import shutil

    try:
        # Remove any partial env from a previous failed attempt.
        if ENV_DIR.exists():
            shutil.rmtree(ENV_DIR)

        # Preferred path: venv provisions pip via ensurepip on its own.
        result = subprocess.run(  # nosec B603
            [
                _find_system_python(), "-m", "venv",
                "--system-site-packages",
                str(ENV_DIR),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            **no_window_kwargs(),
        )

        if result.returncode != 0 or not _has_pip():
            # Some QGIS builds ship a Python whose ensurepip is not wired into
            # venv. Recreate without pip, then run ensurepip in the env.
            if ENV_DIR.exists():
                shutil.rmtree(ENV_DIR)

            result = subprocess.run(  # nosec B603
                [
                    _find_system_python(), "-m", "venv",
                    "--system-site-packages",
                    "--without-pip",
                    str(ENV_DIR),
                ],
                capture_output=True,
                text=True,
                timeout=300,
                **no_window_kwargs(),
            )
            if result.returncode != 0:
                return False, result.stderr.strip() or "venv creation failed."

            result = subprocess.run(  # nosec B603
                [get_env_python(), "-m", "ensurepip", "--upgrade", "--default-pip"],
                capture_output=True,
                text=True,
                timeout=300,
                **no_window_kwargs(),
            )
            if result.returncode != 0 or not _has_pip():
                return False, (
                    (result.stderr.strip() or "pip bootstrap failed.")
                    + "\n\nThis QGIS installation ships a Python without a "
                    "working ensurepip. Install torch and torchvision "
                    "manually — see requirements.txt for the commands."
                )

        return True, "Environment created."
    except Exception as exc:
        return False, str(exc)
