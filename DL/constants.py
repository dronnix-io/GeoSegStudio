"""
module: constants.py

Shared constants that can be imported anywhere in the plugin without
triggering heavy dependencies (torch, GDAL, etc.).
"""

# Square patch sizes (px) supported by all model architectures.
# Must stay in sync with SUPPORTED_SIZES in DL/architectures.py.
SUPPORTED_SIZES = [64, 128, 256, 512, 1024]
