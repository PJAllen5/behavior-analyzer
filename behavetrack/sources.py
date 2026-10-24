"""Read frames from videos, microscope files, or folders of images.

Every source gives, for each frame:
  display  8-bit BGR image for drawing and for the motion / AI detectors
  gray     float32 image scaled 0..1, used by the cell detector

Microscope files may also carry calibration (pixel size in microns, time between frames).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
STACK_EXT = {".tif", ".tiff", ".nd2", ".czi"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ALL_EXT = VIDEO_EXT | STACK_EXT


class Source:
    fps: float = 30.0                  # frames per second of real time
    n_frames: int = 0
    width: int = 0
    height: int = 0
    pixel_size_um: float | None = None
    frame_interval_s: float | None = None   # from file metadata, if known
    is_video: bool = False

    def frames(self, start: int = 0, stop: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError


class VideoSource(Source):
    is_video = True

    def __init__(self, path: Path):
        self.path = path
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path.name}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def frames(self, start=0, stop=None):
        cap = cv2.VideoCapture(str(self.path))
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        stop = self.n_frames if stop is None else stop
        for _ in range(start, stop):
            ok, img = cap.read()
            if not ok:
                break
            yield img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        cap.release()


class StackSource(Source):
    """A time-lapse held in memory as an array shaped (T, Y, X)."""

    def __init__(self, stack: np.ndarray, pixel_size_um=None, frame_interval_s=None):
        if stack.ndim == 2:
            stack = stack[None]
        self.stack = stack
        self.n_frames, self.height, self.width = stack.shape
        self.pixel_size_um = pixel_size_um or None
        self.frame_interval_s = frame_interval_s or None
        self.fps = 1.0 / frame_interval_s if frame_interval_s else 1.0
        # Contrast range from a sample of frames, so brightness is steady across the movie.
        sample = stack[:: max(1, self.n_frames // 20)].astype(np.float32)
        self.lo, self.hi = np.percentile(sample, [0.5, 99.8])
        if self.hi <= self.lo:
            self.hi = self.lo + 1

    def frames(self, start=0, stop=None):
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        for i in range(start, stop):
            g = np.clip((self.stack[i].astype(np.float32) - self.lo) / (self.hi - self.lo), 0, 1)
            disp = cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            yield disp, g


# ------------------------------------------------------------------ readers
def _reduce_axes(arr: np.ndarray, axes: str, channel: int) -> np.ndarray:
    """Turn any microscope array into (T, Y, X): pick one channel, max-project Z."""
    axes = axes.upper()
    if "S" in axes and arr.shape[axes.index("S")] in (3, 4):  # RGB samples -> gray
        i = axes.index("S")
        arr = np.take(arr, [0, 1, 2], axis=i).mean(axis=i)
        axes = axes.replace("S", "")
    for ax in axes:
        if ax in "YXT":
            continue
        i = axes.index(ax)
        if ax == "C":
            arr = np.take(arr, min(channel, arr.shape[i] - 1), axis=i)
        elif ax == "Z":
            arr = arr.max(axis=i)
        elif ax in "IQ" and "T" not in axes:
            axes = axes.replace(ax, "T")
            continue
        else:  # positions, rotations, etc.: take the first
            arr = np.take(arr, 0, axis=i)
        axes = axes.replace(ax, "", 1)
    if "T" not in axes:
        arr, axes = arr[None], "T" + axes
    order = [axes.index(a) for a in "TYX"]
    return np.transpose(arr, order)


def _read_tiff(path: Path, channel: int) -> StackSource:
    import tifffile

    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        arr = series.asarray()
        axes = series.axes
        px, dt = None, None
        ij = tf.imagej_metadata or {}
        if ij:
            dt = ij.get("finterval")
            unit = str(ij.get("unit", "")).lower()
            page = tf.pages[0]
            xres = page.tags.get("XResolution")
            if xres and unit in {"micron", "um", "µm", "\\u00b5m", "microns"}:
                num, den = xres.value
                if num:
                    px = den / num
        if tf.ome_metadata:
            m = re.search(r'PhysicalSizeX="([\d.eE+-]+)"', tf.ome_metadata)
            if m:
                px = float(m.group(1))
                u = re.search(r'PhysicalSizeXUnit="([^"]+)"', tf.ome_metadata)
                if u and u.group(1) == "nm":
                    px /= 1000
            m = re.search(r'TimeIncrement="([\d.eE+-]+)"', tf.ome_metadata)
            if m:
                dt = float(m.group(1))
                u = re.search(r'TimeIncrementUnit="([^"]+)"', tf.ome_metadata)
                if u and u.group(1) in ("min",):
                    dt *= 60
                elif u and u.group(1) in ("ms",):
                    dt /= 1000
    return StackSource(_reduce_axes(arr, axes, channel), px, dt)


def _read_nd2(path: Path, channel: int) -> StackSource:
    import nd2

    with nd2.ND2File(path) as f:
        arr = f.asarray()
        axes = "".join(k for k in f.sizes)
        px = f.voxel_size().x if f.voxel_size() else None
        dt = None
        try:
            for loop in f.experiment:
                if loop.type == "TimeLoop":
                    dt = loop.parameters.periodMs / 1000
        except Exception:  # noqa: BLE001 - metadata is optional
            pass
    return StackSource(_reduce_axes(arr, axes, channel), px, dt)


def _read_czi(path: Path, channel: int) -> StackSource:
    import czifile

    with czifile.CziFile(path) as f:
        arr = f.asarray()
        axes = f.axes
        meta = f.metadata() or ""
    axes = axes.replace("0", "S")  # czifile names RGB samples '0'
    keep = [i for i, a in enumerate(axes) if a in "TCZYXS" or arr.shape[i] > 1]
    arr = arr.squeeze(axis=tuple(i for i in range(arr.ndim) if i not in keep))
    axes = "".join(axes[i] for i in keep)
    px = None
    m = re.search(r'<Distance Id="X">\s*<Value>([\d.eE+-]+)</Value>', meta)
    if m:
        px = float(m.group(1)) * 1e6  # metres -> microns
    return StackSource(_reduce_axes(arr, axes, channel), px, None)


def _read_folder(path: Path) -> StackSource:
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not files:
        raise ValueError(f"No images found in {path}")
    imgs = []
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        if im.ndim == 3:
            im = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
        imgs.append(im)
    return StackSource(np.stack(imgs))


def open_source(path: str | Path, channel: int = 0) -> Source:
    path = Path(path)
    if path.is_dir():
        return _read_folder(path)
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        return _read_tiff(path, channel)
    if ext == ".nd2":
        return _read_nd2(path, channel)
    if ext == ".czi":
        return _read_czi(path, channel)
    return VideoSource(path)
