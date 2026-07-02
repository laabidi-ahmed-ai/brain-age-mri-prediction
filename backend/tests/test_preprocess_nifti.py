"""Tests for load_nifti_slice: format support + Windows temp-dir cleanup safety.

Adapted from the previous Django app's test suite (same 5 cases) to the new
``app.ml.preprocess`` module. Covers valid .nii, valid .nii.gz, malformed
input, invalid volume dimensions, and that the internal TemporaryDirectory is
fully removed after the call.
"""
from __future__ import annotations

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest
from nibabel.filebasedimages import ImageFileError

from app.ml import preprocess
from .conftest import make_nii_bytes


def test_load_nifti_slice_valid_nii():
    tensor, vis = preprocess.load_nifti_slice(make_nii_bytes())
    assert tensor.shape == (1, 3, 224, 224)
    assert vis.ndim == 2
    assert 0.0 <= float(vis.min()) and float(vis.max()) <= 1.0


def test_load_nifti_slice_valid_nii_gz():
    tensor, vis = preprocess.load_nifti_slice(make_nii_bytes(gz=True))
    assert tensor.shape == (1, 3, 224, 224)
    assert vis.ndim == 2


def test_load_nifti_slice_malformed_input_raises():
    with pytest.raises(ImageFileError):
        preprocess.load_nifti_slice(b"not a real nifti file" * 20)


def test_load_nifti_slice_invalid_dimensions_raises():
    vol2d = (np.random.rand(32, 32) * 10).astype(np.float32)
    bad_bytes = nib.Nifti1Image(vol2d, affine=np.eye(4)).to_bytes()
    with pytest.raises(ValueError, match="Volume must be 3D or 4D"):
        preprocess.load_nifti_slice(bad_bytes)


def test_load_nifti_slice_temp_dir_cleanup_on_windows(monkeypatch):
    """Guards the Windows PermissionError this module was patched for.

    nibabel used to keep the uploaded file mmapped/open past `get_fdata()`,
    which made `TemporaryDirectory.__exit__` raise PermissionError on Windows
    when it tried to delete the still-locked file. We assert the directory
    nibabel actually loaded from is fully gone afterwards, which can only be
    true if cleanup ran to completion without that error.
    """
    created_dirs: list[str] = []
    real_temporary_directory = tempfile.TemporaryDirectory

    class RecordingTemporaryDirectory(real_temporary_directory):
        def __enter__(self):
            path = super().__enter__()
            created_dirs.append(path)
            return path

    monkeypatch.setattr(preprocess.tempfile, "TemporaryDirectory", RecordingTemporaryDirectory)

    preprocess.load_nifti_slice(make_nii_bytes())

    assert len(created_dirs) == 1
    assert not os.path.exists(created_dirs[0]), (
        "temp directory survived the `with` block — cleanup failed "
        "(this is exactly the Windows PermissionError this fix prevents)"
    )
