"""Reading microscope files: shapes, channels, z-stacks and calibration."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import tifffile

from behavetrack.sources import StackSource, VideoSource, open_source


def _stack(t=4, h=32, w=40):
    rng = np.random.default_rng(0)
    return rng.integers(100, 4000, (t, h, w), dtype=np.uint16)


def test_imagej_tiff_reads_calibration(tmp_path):
    p = tmp_path / "movie.tif"
    tifffile.imwrite(p, _stack(), imagej=True, resolution=(1 / 0.5, 1 / 0.5),
                     metadata={"axes": "TYX", "unit": "um", "finterval": 120})
    src = open_source(p)
    assert isinstance(src, StackSource)
    assert (src.n_frames, src.height, src.width) == (4, 32, 40)
    assert src.pixel_size_um == pytest.approx(0.5)
    assert src.frame_interval_s == pytest.approx(120)


def test_channels_and_z_are_reduced(tmp_path):
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 1000, (3, 5, 2, 16, 20), dtype=np.uint16)  # T Z C Y X
    p = tmp_path / "tzc.tif"
    tifffile.imwrite(p, arr, imagej=True, metadata={"axes": "TZCYX"})
    src = open_source(p, channel=1)
    assert (src.n_frames, src.height, src.width) == (3, 16, 20)
    raw = list(src.raw_frames())
    np.testing.assert_array_equal(raw[0], arr[0, :, 1].max(axis=0).astype(np.float32))


def test_uncalibrated_tiff(tmp_path):
    p = tmp_path / "plain.tif"
    tifffile.imwrite(p, _stack(), photometric="minisblack")
    src = open_source(p)
    assert src.pixel_size_um is None and src.frame_interval_s is None
    disp, gray = next(src.frames())
    assert disp.shape == (32, 40, 3) and disp.dtype == np.uint8
    assert 0.0 <= gray.min() and gray.max() <= 1.0


def test_folder_of_images(tmp_path):
    for i, frame in enumerate(_stack(t=3)):
        cv2.imwrite(str(tmp_path / f"img_{i:03d}.png"), frame)
    src = open_source(tmp_path)
    assert src.n_frames == 3


def test_video(tmp_path):
    p = tmp_path / "v.mp4"
    wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25, (48, 32))
    for i in range(10):
        wr.write(np.full((32, 48, 3), i * 20, np.uint8))
    wr.release()
    src = open_source(p)
    assert isinstance(src, VideoSource)
    assert src.fps == pytest.approx(25) and src.n_frames == 10
    assert len(list(src.frames(2, 6))) == 4
