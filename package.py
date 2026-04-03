"""
package.py
----------
Builds a distributable GeoSegStudio.zip for manual QGIS installation.

Usage:
    python package.py

Output:
    dist/GeoSegStudio.zip

The ZIP contains a single top-level folder  GeoSegStudio/  with all plugin
files inside — exactly what QGIS expects when installing from ZIP.

Excluded from the ZIP:
    - env/               (PyTorch virtual environment — installed at runtime)
    - __pycache__/       (compiled bytecode)
    - .git/              (version control internals)
    - .gitignore
    - notes/             (internal development notes)
    - docs/              (repository documentation, not needed inside QGIS)
    - package.py         (this script)
    - dist/              (output folder)
"""

import os
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLUGIN_NAME = "GeoSegStudio"
REPO_ROOT   = Path(__file__).resolve().parent
OUTPUT_DIR  = REPO_ROOT / "dist"
OUTPUT_ZIP  = OUTPUT_DIR / f"{PLUGIN_NAME}.zip"

EXCLUDE_DIRS = {
    "env",
    "__pycache__",
    ".git",
    "notes",
    "docs",
    "dist",
}

EXCLUDE_FILES = {
    ".gitignore",
    "package.py",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def should_exclude(path: Path) -> bool:
    parts = path.relative_to(REPO_ROOT).parts
    if any(part in EXCLUDE_DIRS for part in parts):
        return True
    if path.name in EXCLUDE_FILES:
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def build_zip():
    OUTPUT_DIR.mkdir(exist_ok=True)

    file_count = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(REPO_ROOT.rglob("*")):
            if not file_path.is_file():
                continue
            if should_exclude(file_path):
                continue

            # Archive path: GeoSegStudio/<relative path>
            relative = file_path.relative_to(REPO_ROOT)
            archive_path = Path(PLUGIN_NAME) / relative
            zf.write(file_path, archive_path)
            file_count += 1

    size_kb = OUTPUT_ZIP.stat().st_size // 1024
    print(f"Built {OUTPUT_ZIP.relative_to(REPO_ROOT)}")
    print(f"  {file_count} files  |  {size_kb} KB")


if __name__ == "__main__":
    build_zip()
