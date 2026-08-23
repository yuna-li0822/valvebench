import math

import numpy as np
import pytest


def nibabel_available():
    try:
        import nibabel  # noqa: F401
        return True
    except Exception:
        return False


def trimesh_available():
    try:
        import trimesh  # noqa: F401
        return True
    except Exception:
        return False


def _make_synthetic_volume(shape=(40, 40, 10), circle_radius_vox=8, label_id=5, slice_index=4, voxel_mm=1.0):
    """A volume that's all zeros except a filled circle of the given
    label_id on one axial slice -- lets us check equivalent_diameter_mm
    against a circle whose true area we know analytically.
    """
    import nibabel as nib

    data = np.zeros(shape, dtype=np.int16)
    cy, cx = shape[0] // 2, shape[1] // 2
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    circle_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= circle_radius_vox ** 2
    data[:, :, slice_index][circle_mask] = label_id

    affine = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0])
    img = nib.Nifti1Image(data, affine)
    return img, circle_mask.sum()


@pytest.mark.skipif(not nibabel_available(), reason="nibabel not installed")
def test_equivalent_diameter_matches_known_circle_area():
    from valvebench.datasets.imaging import nifti_utils as niu

    radius_vox = 8
    img, n_voxels = _make_synthetic_volume(circle_radius_vox=radius_vox)
    mask_slice = np.asarray(img.dataobj)[:, :, 4] == 5

    diameter = niu.equivalent_diameter_mm(mask_slice, niu.voxel_area_mm2(img))
    expected_diameter = 2 * radius_vox  # true circle diameter in voxels == mm here (voxel_mm=1.0)
    assert diameter == pytest.approx(expected_diameter, rel=0.05)


@pytest.mark.skipif(not nibabel_available(), reason="nibabel not installed")
def test_label_diameter_at_slice_finds_largest_area_slice(tmp_path):
    from valvebench.datasets.imaging import nifti_utils as niu
    import nibabel as nib

    img, _ = _make_synthetic_volume(circle_radius_vox=8, slice_index=4)
    path = tmp_path / "label.nii.gz"
    nib.save(img, str(path))

    diameter = niu.label_diameter_at_slice(str(path), label_id=5, slice_axis=2)
    assert diameter == pytest.approx(16.0, rel=0.05)


@pytest.mark.skipif(not nibabel_available(), reason="nibabel not installed")
def test_label_diameter_at_slice_raises_for_absent_label(tmp_path):
    from valvebench.datasets.imaging import nifti_utils as niu
    import nibabel as nib

    img, _ = _make_synthetic_volume(circle_radius_vox=8, label_id=5)
    path = tmp_path / "label.nii.gz"
    nib.save(img, str(path))

    with pytest.raises(ValueError):
        niu.label_diameter_at_slice(str(path), label_id=99)


@pytest.mark.skipif(not nibabel_available(), reason="nibabel not installed")
def test_discover_and_load_nifti_pairs(tmp_path):
    from valvebench.datasets.imaging import nifti_utils as niu
    import nibabel as nib

    for case_id in ("ct_train_1001", "ct_train_1002"):
        img, _ = _make_synthetic_volume(circle_radius_vox=6, label_id=7)
        nib.save(img, str(tmp_path / f"{case_id}_label.nii.gz"))
        nib.save(img, str(tmp_path / f"{case_id}_image.nii.gz"))

    cases = niu.discover_nifti_pairs(tmp_path, image_marker="_image", label_marker="_label")
    assert len(cases) == 2
    assert all(c["image_path"] is not None for c in cases)

    df = niu.load_local_nifti_pairs(tmp_path, label_id=7)
    assert len(df) == 2
    assert (df["annulus_estimate_mm"] > 0).all()


@pytest.mark.skipif(not nibabel_available(), reason="nibabel not installed")
def test_mmwhs_load_local_uses_its_default_naming(tmp_path):
    from valvebench.datasets.imaging import mmwhs
    import nibabel as nib

    img, _ = _make_synthetic_volume(circle_radius_vox=6, label_id=3)
    nib.save(img, str(tmp_path / "ct_train_2001_label.nii.gz"))
    nib.save(img, str(tmp_path / "ct_train_2001_image.nii.gz"))

    df = mmwhs.load_local(tmp_path, label_id=3)
    assert len(df) == 1
    assert df.iloc[0]["case_id"] == "ct_train_2001"


@pytest.mark.skipif(not nibabel_available(), reason="nibabel not installed")
def test_imagechd_load_local_uses_its_default_naming(tmp_path):
    from valvebench.datasets.imaging import imagechd
    import nibabel as nib

    img, _ = _make_synthetic_volume(circle_radius_vox=6, label_id=3)
    nib.save(img, str(tmp_path / "ct_3001_label.nii.gz"))
    nib.save(img, str(tmp_path / "ct_3001_image.nii.gz"))

    df = imagechd.load_local(tmp_path, label_id=3)
    assert len(df) == 1
    assert df.iloc[0]["case_id"] == "ct_3001"


@pytest.mark.skipif(not nibabel_available(), reason="nibabel not installed")
def test_segthor_load_local_walks_patient_directories(tmp_path):
    from valvebench.datasets.imaging import segthor
    import nibabel as nib

    patient_dir = tmp_path / "Patient_01"
    patient_dir.mkdir()
    img, _ = _make_synthetic_volume(circle_radius_vox=6, label_id=4)
    nib.save(img, str(patient_dir / "GT.nii.gz"))
    nib.save(img, str(patient_dir / "Patient_01.nii.gz"))

    df = segthor.load_local(tmp_path, label_id=4)
    assert len(df) == 1
    assert df.iloc[0]["case_id"] == "Patient_01"
    assert df.iloc[0]["image_path"] is not None


@pytest.mark.skipif(not nibabel_available(), reason="nibabel not installed")
def test_segthor_load_local_raises_on_empty_directory(tmp_path):
    from valvebench.datasets.imaging import segthor

    with pytest.raises(FileNotFoundError):
        segthor.load_local(tmp_path, label_id=4)


@pytest.mark.skipif(not trimesh_available(), reason="trimesh not installed")
def test_nih_3d_load_local_reports_bounding_box_diameter(tmp_path):
    import trimesh
    from valvebench.datasets.imaging import nih_3d

    mesh = trimesh.creation.box(extents=[10.0, 20.0, 30.0])
    path = tmp_path / "valve_model.stl"
    mesh.export(str(path))

    result = nih_3d.load_local(str(path))
    assert result["is_watertight"] is True
    # mean of the two largest extents: (30 + 20) / 2 = 25
    assert result["diameter_estimate_mm"] == pytest.approx(25.0, rel=0.01)


@pytest.mark.skipif(not trimesh_available(), reason="trimesh not installed")
def test_nih_3d_load_local_directory(tmp_path):
    import trimesh
    from valvebench.datasets.imaging import nih_3d

    for i, extents in enumerate([[10, 20, 30], [5, 8, 12]]):
        mesh = trimesh.creation.box(extents=extents)
        mesh.export(str(tmp_path / f"model_{i}.stl"))

    df = nih_3d.load_local_directory(tmp_path)
    assert len(df) == 2
    assert (df["diameter_estimate_mm"] > 0).all()


@pytest.mark.skipif(not trimesh_available(), reason="trimesh not installed")
def test_nih_3d_load_local_directory_raises_when_empty(tmp_path):
    from valvebench.datasets.imaging import nih_3d

    with pytest.raises(FileNotFoundError):
        nih_3d.load_local_directory(tmp_path)
