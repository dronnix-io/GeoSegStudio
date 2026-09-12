"""
module: DL/checkpoint_io.py

Single entry point for reading .pth checkpoints.

Why this module exists
----------------------
``torch.load`` defaults to unpickling, and unpickling executes arbitrary code
embedded in the file.  A .pth checkpoint is exactly the sort of artefact users
download from model zoos, forums and colleagues, so treating it as trusted
input is not acceptable: opening a hostile checkpoint would run code with the
user's privileges inside QGIS.

Everything GeoSeg Studio writes (see ``Trainer._maybe_save``) is tensors plus
plain str/int/float, which is exactly what ``weights_only=True`` permits, so
the safe loader is used unconditionally and there is no opt-out.

Public surface
--------------
load_checkpoint(path, map_location="cpu") → dict
CheckpointError — raised with a user-facing message on any failure
"""

from __future__ import annotations

from pathlib import Path


class CheckpointError(Exception):
    """Raised when a checkpoint cannot be read safely."""


def load_checkpoint(path, map_location="cpu") -> dict:
    """
    Loads a checkpoint with pickle execution disabled.

    Parameters
    ----------
    path : str | Path
        Path to a .pth file.
    map_location : str | torch.device
        Passed through to ``torch.load``.

    Returns
    -------
    dict — the checkpoint payload.

    Raises
    ------
    CheckpointError
        If the file is missing, is not a GeoSeg Studio checkpoint, or holds
        pickled objects that the safe loader refuses to execute.
    """
    import torch

    p = Path(path)
    if not p.is_file():
        raise CheckpointError(f"Checkpoint file not found:\n  {p}")

    try:
        # weights_only=True restricts unpickling to tensors and primitives,
        # so a malicious checkpoint cannot execute code. Never set this to
        # False — see the module docstring.
        data = torch.load(str(p), map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointError(
            f"Could not read the checkpoint file:\n  {p}\n\n"
            f"{exc}\n\n"
            "GeoSeg Studio loads checkpoints in a restricted mode that only "
            "accepts model weights and plain configuration values. A "
            "checkpoint that stores custom Python objects is rejected, "
            "because loading one would run code from the file. Re-export the "
            "model as a plain state_dict, or train it in GeoSeg Studio."
        ) from exc

    if not isinstance(data, dict):
        raise CheckpointError(
            f"Unexpected checkpoint format in:\n  {p}\n\n"
            "Expected a dictionary containing 'model_state_dict'."
        )

    return data
