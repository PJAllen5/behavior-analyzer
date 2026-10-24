"""Detectors: find objects in a single frame.

Each detector returns a list of detections: (cx, cy, w, h, class_name, confidence).

- MotionDetector: background subtraction. Works for anything that moves against a
  still camera (ants, birds in the sky, fish, people). No AI model needed.
- YoloDetector: pretrained AI model (ultralytics). Recognises people, birds,
  dogs, horses, cows, sheep, etc. Works with moving cameras and still objects.
"""
from __future__ import annotations

import cv2
import numpy as np

Detection = tuple[float, float, float, float, str, float]


class MotionDetector:
    def __init__(self, min_area: int = 15, max_area: int = 50_000,
                 sensitivity: float = 25.0, history: int = 300, label: str = "object",
                 warmup_frames: int = 15):
        self.min_area = min_area
        self.max_area = max_area
        self.label = label
        self.warmup = warmup_frames  # background model is unreliable for the first frames
        self.seen = 0
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=sensitivity, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        mask = self.bg.apply(blur)
        self.seen += 1
        if self.seen <= self.warmup:
            return []
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        dets: list[Detection] = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_area <= area <= self.max_area:
                w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                dets.append((float(cents[i][0]), float(cents[i][1]), float(w), float(h),
                             self.label, 1.0))
        return dets


class YoloDetector:
    def __init__(self, model: str = "yolo11n.pt", classes: list[str] | None = None,
                 conf: float = 0.3):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise SystemExit("The 'yolo' detector needs ultralytics: pip install ultralytics") from e
        self.model = YOLO(model)
        self.conf = conf
        names = self.model.names
        self.class_ids = None
        if classes:
            wanted = {c.strip().lower() for c in classes}
            self.class_ids = [i for i, nm in names.items() if nm.lower() in wanted]
            missing = wanted - {names[i].lower() for i in self.class_ids}
            if missing:
                raise SystemExit(f"Unknown classes {sorted(missing)}. "
                                 f"Available: {', '.join(names.values())}")

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(frame, conf=self.conf, classes=self.class_ids, verbose=False)[0]
        dets: list[Detection] = []
        for box, c, p in zip(res.boxes.xywh.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                             res.boxes.conf.cpu().numpy()):
            dets.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                         self.model.names[int(c)], float(p)))
        return dets


class CellDetector:
    """Finds cells by appearance (not motion), so still cells are found too.

    appearance: "bright" (fluorescence, nuclei), "dark" (cells darker than background),
                or "auto" (anything that differs from the background, e.g. phase contrast).
    diameter_px: typical cell diameter. 0 = estimate from the first frame.
    threshold_adjust: -0.5..+0.5. Negative finds fainter cells, positive is stricter.
    split_touching: separate cells that touch.
    """

    def __init__(self, appearance: str = "auto", diameter_px: float = 0,
                 threshold_adjust: float = 0.0, split_touching: bool = True):
        self.appearance = appearance
        self.diameter = float(diameter_px) or None
        self.adjust = threshold_adjust
        self.split = split_touching
        self.threshold = None

    def _signal(self, g: np.ndarray, diam: float) -> np.ndarray:
        from scipy import ndimage as ndi

        bg = ndi.gaussian_filter(g, sigma=max(diam * 2.0, 10))  # uneven illumination
        d = g - bg
        if self.appearance == "bright":
            s = d
        elif self.appearance == "dark":
            s = -d
        else:
            s = ndi.gaussian_filter(np.abs(d), sigma=max(diam / 8, 1))
        return ndi.gaussian_filter(s, sigma=max(diam / 12, 0.7))

    def _mask(self, s: np.ndarray) -> np.ndarray:
        from scipy import ndimage as ndi
        from skimage.filters import threshold_otsu

        t = threshold_otsu(s)
        # keep threshold steady over time: blend with the first frame's value
        self.threshold = t if self.threshold is None else 0.8 * self.threshold + 0.2 * t
        m = s > self.threshold * (1 + self.adjust)
        m = ndi.binary_opening(m, iterations=1)
        return ndi.binary_fill_holes(m)

    def _estimate_diameter(self, g: np.ndarray) -> float:
        from skimage.measure import label, regionprops

        d = 15.0
        for _ in range(2):  # refine once with the first estimate
            m = self._mask(self._signal(g, d))
            self.threshold = None
            sizes = [r.equivalent_diameter_area for r in regionprops(label(m)) if r.area >= 9]
            if sizes:
                d = float(np.median(sizes))
        return max(d, 4.0)

    def __call__(self, gray: np.ndarray) -> list:
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.measure import label, regionprops
        from skimage.segmentation import watershed

        if self.diameter is None:
            self.diameter = self._estimate_diameter(gray)
        diam = self.diameter
        s = self._signal(gray, diam)
        m = self._mask(s)
        min_area = max(4, int(np.pi * (diam * 0.25) ** 2))
        lab = label(m)
        if self.split:
            dist = ndi.distance_transform_edt(m)
            dist = ndi.gaussian_filter(dist, 1)
            pk = peak_local_max(dist, min_distance=max(2, int(diam * 0.4)), labels=lab,
                                threshold_abs=diam * 0.15, exclude_border=False)
            markers = np.zeros_like(lab)
            markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
            lab = watershed(-dist, markers, mask=m)
        dets = []
        for r in regionprops(lab, intensity_image=gray):
            if r.area < min_area:
                continue
            y0, x0, y1, x1 = r.bbox
            cy, cx = r.centroid
            dets.append((float(cx), float(cy), float(x1 - x0), float(y1 - y0), "cell", 1.0,
                         {"area_px": float(r.area), "intensity": float(r.intensity_mean),
                          "eccentricity": float(r.eccentricity),
                          "diameter_px": float(r.equivalent_diameter_area)}))
        return dets
