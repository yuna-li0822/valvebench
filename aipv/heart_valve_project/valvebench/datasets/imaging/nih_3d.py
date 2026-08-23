"""NIH 3D Print Exchange heart-valve model local loader.

Source: NIH 3D Print Exchange (https://3d.nih.gov), which hosts anatomical
STL models including pediatric heart valve models; this environment
couldn't verify a stable direct-download URL for a specific model file, so
this loader (like the others in valvebench.datasets.imaging) expects a
locally-downloaded STL rather than fetching one itself.

Uses trimesh for mesh loading and a bounding-box-based diameter estimate:
the mean of the two largest extents of the mesh's axis-aligned bounding
box. This is a coarse proxy for annulus diameter -- a real measurement
would fit a circle/ellipse to the actual annular rim geometry, which
requires identifying that specific feature on the mesh (out of scope for a
generic loader with no knowledge of a specific model's topology).
"""
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import trimesh
    _TRIMESH_AVAILABLE = True
except Exception:
    trimesh = None
    _TRIMESH_AVAILABLE = False


def _require_trimesh():
    if not _TRIMESH_AVAILABLE:
        raise RuntimeError(
            'This loader requires trimesh. Install with `pip install -e ".[imaging]"` '
            "from aipv/heart_valve_project."
        )


def bounding_box_diameter_mm(mesh) -> float:
    """Mean of the two largest axis-aligned bounding-box extents. Assumes
    the STL's units are millimeters (the common convention for anatomical
    3D-print models); verify against your specific file if unsure.
    """
    extents = np.sort(mesh.extents)[::-1]  # largest first
    return float(np.mean(extents[:2]))


def load_local(path) -> dict:
    """Loads a single local STL/mesh file and returns a dict with
    file_path, diameter_estimate_mm, is_watertight, volume_mm3 (volume is
    only meaningful if is_watertight).
    """
    _require_trimesh()
    mesh = trimesh.load(str(path), force="mesh")
    return {
        "file_path": str(path),
        "diameter_estimate_mm": bounding_box_diameter_mm(mesh),
        "is_watertight": bool(mesh.is_watertight),
        "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
    }


def load_local_directory(root, pattern: str = "*.stl") -> pd.DataFrame:
    """Loads every mesh file matching `pattern` under `root` into a single
    DataFrame (one row per file), for a directory of several downloaded
    NIH 3D models.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"'{root}' is not a directory")

    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' found under {root}.")

    rows = [load_local(p) for p in paths]
    return pd.DataFrame(rows)
