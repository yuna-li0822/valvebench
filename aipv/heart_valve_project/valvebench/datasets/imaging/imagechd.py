"""ImageCHD (congenital heart disease CT segmentation) local loader.

Citation: Xu, X. et al. (2020). "ImageCHD: A 3D Computed Tomography Image
Dataset for Classification of Congenital Heart Disease." MICCAI 2020.

Default file naming (`ct_1001_image.nii.gz` / `ct_1001_label.nii.gz`) is a
best-effort guess based on the whole-heart-CT-segmentation convention
shared with MMWHS -- this environment had no verified copy of ImageCHD to
confirm the exact release naming against, so this is the piece most likely
to need a custom `image_marker`/`label_marker`/`prefix` for your actual
download.
"""
import pandas as pd

from . import nifti_utils as niu

DEFAULT_IMAGE_MARKER = "_image"
DEFAULT_LABEL_MARKER = "_label"
DEFAULT_PREFIX = "ct_"


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
