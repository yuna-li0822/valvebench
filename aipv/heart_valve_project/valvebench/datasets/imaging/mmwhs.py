"""MMWHS (Multi-Modality Whole Heart Segmentation challenge) local loader.

Citation: Zhuang, X. & Shen, J. (2016). "Multi-scale patch and multi-modality
atlases for whole heart segmentation of MRI." Medical Image Analysis, 31,
77-87. Challenge page: https://zmiclab.github.io/zxh/0/mmwhs/ (registration
required for the official data release; this loader does not download
anything).

Default file naming (`ct_train_1001_image.nii.gz` / `ct_train_1001_label.nii.gz`)
matches how the challenge's CT training set has commonly been distributed,
but release packaging has varied -- verify against your actual download and
pass a custom `image_marker`/`label_marker` if it doesn't match.
"""
import pandas as pd

from . import nifti_utils as niu

DEFAULT_IMAGE_MARKER = "_image"
DEFAULT_LABEL_MARKER = "_label"
DEFAULT_PREFIX = "ct_train_"


def load_local(
    root,
    label_id: int,
    image_marker: str = DEFAULT_IMAGE_MARKER,
    label_marker: str = DEFAULT_LABEL_MARKER,
    prefix: str = DEFAULT_PREFIX,
    slice_axis: int = 2,
    slice_index=None,
) -> pd.DataFrame:
    """See valvebench.datasets.imaging.nifti_utils.load_local_nifti_pairs
    for the label_id caveat and return schema.
    """
    return niu.load_local_nifti_pairs(root, label_id, image_marker, label_marker, prefix, slice_axis, slice_index)
