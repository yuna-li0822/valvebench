"""Bring-your-own-local-path imaging dataset loaders for VALVEBENCH.

None of these download anything: MMWHS and SegTHOR require challenge
registration, and this environment couldn't verify a stable, correct
download URL for any of them (ImageCHD, NIH 3D Print Exchange included) --
guessing a dataset URL and shipping it as if verified is exactly the
failure mode this package avoids. Point each loader at a directory you've
already populated via the dataset's official channel; they parse the local
files into VALVEBENCH's schema.

- nifti_utils: generic NIfTI loading + equivalent-diameter-from-mask,
  shared by the three CT-segmentation loaders below.
- mmwhs, imagechd, segthor: thin wrappers over nifti_utils.load_local_nifti_pairs
  with each dataset's citation and (best-effort, verify-before-trusting)
  default file-naming convention.
- nih_3d: STL mesh loader (NIH 3D Print Exchange heart-valve models) using
  trimesh, with a bounding-box-based diameter estimate.

Requires the `imaging` extra: `pip install -e ".[imaging]"` (nibabel, trimesh).
"""
