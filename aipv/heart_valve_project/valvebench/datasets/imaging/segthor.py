"""SegTHOR (thoracic organs at risk segmentation) local loader.

Citation: Lambert, Z. et al. (2020). "SegTHOR: Segmentation of Thoracic
Organs at Risk in CT images." IPTA 2020. Challenge page:
https://competitions.codalab.org/competitions/21145 (registration required
for the official data release; this loader does not download anything).

SegTHOR's actual distribution layout is one directory per patient
(`Patient_01/Patient_01.nii.gz` for the CT, `Patient_01/GT.nii.gz` for the
ground-truth segmentation), which doesn't fit the flat same-directory
image/label glob pattern used by the MMWHS/ImageCHD loaders -- this module
implements its own directory walk instead of reusing
nifti_utils.load_local_nifti_pairs. Labels 1-4 in SegTHOR's GT volumes are
esophagus, heart, trachea, and aorta respectively (per the challenge's
documentation) -- there is no pulmonary/aortic *annulus* label in this
dataset; the aorta label (commonly 4) is the closest available structure,
and any annulus-diameter estimate derived from it is a coarse proxy, not a
direct annulus measurement.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from . import nifti_utils as niu


def discover_cases(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"'{root}' is not a directory")

    cases = []
    for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        gt_path = patient_dir / "GT.nii.gz"
        if not gt_path.exists():
            gt_path = patient_dir / "GT.nii"
        ct_candidates = list(patient_dir.glob(f"{patient_dir.name}.nii*"))
        image_path = ct_candidates[0] if ct_candidates else None
        if gt_path.exists():
            cases.append({
                "case_id": patient_dir.name,
                "image_path": str(image_path) if image_path else None,
                "label_path": str(gt_path),
            })
    return cases


def load_local(root, label_id: int, slice_axis: int = 2, slice_index: Optional[int] = None) -> pd.DataFrame:
    """See valvebench.datasets.imaging.nifti_utils.label_diameter_at_slice
    for the label_id caveat. Returns a DataFrame with columns case_id,
    annulus_estimate_mm, label_path, image_path.
    """
    cases = discover_cases(root)
    if not cases:
        raise FileNotFoundError(
            f"No 'Patient_*/GT.nii*' structure found under {root}. Confirm this directory "
            "actually contains a SegTHOR download."
        )

    rows = []
    for case in cases:
        try:
            diameter = niu.label_diameter_at_slice(case["label_path"], label_id, slice_axis, slice_index)
        except ValueError:
            continue
        rows.append({
            "case_id": case["case_id"],
            "annulus_estimate_mm": diameter,
            "label_path": case["label_path"],
            "image_path": case["image_path"],
        })
    return pd.DataFrame(rows)
