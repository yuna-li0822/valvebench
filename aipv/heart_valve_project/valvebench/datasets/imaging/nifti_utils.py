"""Generic NIfTI (.nii/.nii.gz) utilities shared by the CT-segmentation
dataset loaders (mmwhs, imagechd, segthor). Lazy-imports nibabel.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import nibabel as nib
    _NIBABEL_AVAILABLE = True
except Exception:
    nib = None
    _NIBABEL_AVAILABLE = False


def _require_nibabel():
    if not _NIBABEL_AVAILABLE:
        raise RuntimeError(
            'This loader requires nibabel. Install with `pip install -e ".[imaging]"` '
            "from aipv/heart_valve_project."
        )


def load_nifti(path):
    _require_nibabel()
    return nib.load(str(path))


def voxel_area_mm2(img) -> float:
    """Cross-sectional voxel area (mm^2) in the first two spatial axes, from
    the NIfTI header's voxel size (zooms). Assumes those two axes are the
    ones sliced in `label_diameter_at_slice` -- true for the default
    slice_axis=2 on a typical RAS-oriented volume, but NIfTI orientation
    conventions vary by acquisition; verify against your actual files.
    """
    zooms = img.header.get_zooms()[:2]
    return float(zooms[0] * zooms[1])


def equivalent_diameter_mm(mask_slice: np.ndarray, voxel_area: float) -> float:
    """Diameter of a circle with the same area as a 2D boolean mask slice
    (area = pi*r^2 => d = 2*sqrt(area/pi)). Standard way to reduce an
    irregular cross-sectional segmentation to a single representative
    diameter.
    """
    area_mm2 = float(np.count_nonzero(mask_slice)) * voxel_area
    if area_mm2 <= 0:
        return 0.0
    return float(2.0 * np.sqrt(area_mm2 / np.pi))


def label_diameter_at_slice(
    label_path,
    label_id: int,
    slice_axis: int = 2,
    slice_index: Optional[int] = None,
) -> float:
    """Loads a NIfTI segmentation, extracts the boolean mask for `label_id`,
    picks a slice along `slice_axis` (defaults to the slice with the
    largest masked area if `slice_index` isn't given), and returns the
    equivalent circular diameter in mm.

    `label_id` has no default and must be supplied explicitly: which
    integer label corresponds to the structure you care about (e.g. the
    pulmonary/aortic annulus) is specific to each dataset's annotation
    protocol and this environment has no verified copy of any of these
    datasets to confirm label conventions against. Check the dataset's
    documentation before choosing a value.
    """
    _require_nibabel()
    img = load_nifti(label_path)
    data = np.asarray(img.dataobj)
    mask = data == label_id
    if not mask.any():
        raise ValueError(f"label_id={label_id} not present in {label_path}")

    if slice_index is None:
        other_axes = tuple(a for a in range(mask.ndim) if a != slice_axis)
        areas = mask.sum(axis=other_axes)
        slice_index = int(np.argmax(areas))

    mask_slice = np.take(mask, slice_index, axis=slice_axis)
    return equivalent_diameter_mm(mask_slice, voxel_area_mm2(img))


_NIFTI_EXTENSIONS = (".nii.gz", ".nii")


def _strip_nifti_marker(filename: str, marker: str) -> Optional[str]:
    """Returns filename with `<marker><.nii or .nii.gz>` removed from the
    end, or None if it doesn't end that way. E.g.
    _strip_nifti_marker("ct_train_2001_label.nii.gz", "_label") ==
    "ct_train_2001".
    """
    for ext in _NIFTI_EXTENSIONS:
        suffix = f"{marker}{ext}"
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return None


def discover_nifti_pairs(root, image_marker: str = "_image", label_marker: str = "_label", prefix: str = ""):
    """Pairs up image/label NIfTI files under `root`: any file named
    `<case_id><label_marker>.nii` or `.nii.gz` is a label file, matched
    against a same-`case_id` `<case_id><image_marker>.nii(.gz)` if one
    exists. `case_id` is everything before `label_marker`, so it naturally
    includes any dataset-specific prefix (e.g. "ct_train_2001"). Optional
    `prefix` restricts which files are considered at all (useful if a
    directory mixes files from multiple naming conventions). Returns a list
    of {case_id, image_path, label_path} dicts (image_path is None if no
    matching image file was found).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"'{root}' is not a directory")

    cases = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if prefix and not path.name.startswith(prefix):
            continue
        case_id = _strip_nifti_marker(path.name, label_marker)
        if case_id is None:
            continue
        image_path = None
        for ext in _NIFTI_EXTENSIONS:
            candidate = root / f"{case_id}{image_marker}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        cases.append({
            "case_id": case_id,
            "image_path": str(image_path) if image_path else None,
            "label_path": str(path),
        })
    return cases


def load_local_nifti_pairs(
    root,
    label_id: int,
    image_marker: str = "_image",
    label_marker: str = "_label",
    prefix: str = "",
    slice_axis: int = 2,
    slice_index: Optional[int] = None,
) -> pd.DataFrame:
    """Discovers image/label NIfTI pairs under `root` (see
    `discover_nifti_pairs`) and computes an equivalent-diameter estimate for
    `label_id` in each case. Returns a DataFrame with columns case_id,
    annulus_estimate_mm, label_path, image_path. Cases where `label_id`
    isn't present in that case's label volume are skipped (not every case
    necessarily has every structure labeled).
    """
    cases = discover_nifti_pairs(root, image_marker, label_marker, prefix)
    if not cases:
        raise FileNotFoundError(
            f"No files matching '*{label_marker}.nii(.gz)' found under {root}. Confirm this "
            "directory actually contains the dataset and that the naming convention matches "
            "(pass custom image_marker/label_marker if your download uses different names)."
        )

    rows = []
    for case in cases:
        try:
            diameter = label_diameter_at_slice(case["label_path"], label_id, slice_axis, slice_index)
        except ValueError:
            continue
        rows.append({
            "case_id": case["case_id"],
            "annulus_estimate_mm": diameter,
            "label_path": case["label_path"],
            "image_path": case["image_path"],
        })
    return pd.DataFrame(rows)
