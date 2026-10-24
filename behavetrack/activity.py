"""Neural firing activity from calcium (or voltage) imaging movies.

Steps:
  1. drift correction: every frame is lined up with the start of the movie
  2. find neurons: groups of pixels that flicker together (local correlation), or bright
     cell bodies (which also finds neurons that never fire)
  3. fluorescence of each neuron and of the haze around it (neuropil), frame by frame
  4. ΔF/F: change in brightness relative to the neuron's resting level
  5. firing events: rises well above the neuron's own noise level
  6. network bursts: many neurons firing within a short window

The movie is read a few times rather than held in memory, so long recordings fit.

Files written:
  neurons.csv       one row per neuron: position, size, firing rate, amplitude, burst participation
  events.csv        one row per firing event: when, how big, how long
  bursts.csv        one row per network burst
  traces_dff.csv    ΔF/F of every neuron (one column per neuron)
  traces_raw.csv    raw fluorescence and neuropil of every neuron
  correlations.csv  one row per pair of neurons: distance and how alike their activity is
  frames.csv        one row per frame: how many neurons are firing
  rois.png          map of where each neuron is
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import sparse
from scipy.signal import find_peaks

QUIET, FIRING = (200, 140, 60), (40, 210, 255)  # outline colours (BGR) in the video


@dataclass
class Movie:
    src: object
    first: int
    last: int
    scale: float
    fps: float

    def raw(self):
        for r in self.src.raw_frames(self.first, self.last):
            if self.scale != 1.0:
                r = cv2.resize(r, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            yield r


def _move(img: np.ndarray, d) -> np.ndarray:
    """Shift a frame back by its drift d = (dx, dy)."""
    if d[0] == 0 and d[1] == 0:
        return img
    M = np.float32([[1, 0, -d[0]], [0, 1, -d[1]]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _structure(img: np.ndarray) -> np.ndarray:
    """Fine detail only (for lining frames up): blur out noise, remove uneven lighting."""
    a = img.astype(np.float64)
    return cv2.GaussianBlur(a, (0, 0), 1.5) - cv2.GaussianBlur(a, (0, 0), 20)


def _diameter_guess(ref: np.ndarray) -> float:
    from .detect import CellDetector

    lo, hi = np.percentile(ref, [1, 99.5])
    g = np.clip((ref - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)
    try:
        d = CellDetector("bright")._estimate_diameter(g)
    except Exception:  # noqa: BLE001 - a guess is fine
        d = 12.0
    return float(np.clip(d, 5.0, min(ref.shape) / 6))


def summary_images(mv: Movie, diam: float | None, correct_drift: bool, progress=None):
    """Pass 1: drift of every frame, plus the mean image and the local-correlation image
    (how much each pixel flickers in step with its neighbours)."""
    n_ref = min(mv.last - mv.first, max(20, int(2 * mv.fps)))
    ref_frames = []
    for i, r in enumerate(mv.raw()):
        ref_frames.append(r)
        if i + 1 >= n_ref:
            break
    ref = np.mean(ref_frames, axis=0)
    del ref_frames
    if not diam:
        diam = _diameter_guess(ref)
    H, W = ref.shape
    ref_s = _structure(ref)
    win = cv2.createHanningWindow((W, H), cv2.CV_64F)
    max_shift = max(W, H) / 8

    total = mv.last - mv.first
    shifts = np.zeros((total, 2))
    S_raw = np.zeros((H, W))
    S1, S2, S3 = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
    P = {k: np.zeros(sh) for k, sh in (("r", (H, W - 1)), ("d", (H - 1, W)),
                                        ("dr", (H - 1, W - 1)), ("dl", (H - 1, W - 1)))}
    ema = None
    alpha = 1.0 / max(5.0, 1.0 * mv.fps)   # remove slow changes (bleaching) over about 1 s
    sig_space = max(2.0 * diam, 8.0)       # remove broad haze (neuropil) from the activity map
    prev = (0.0, 0.0)
    n = 0
    for i, r in enumerate(mv.raw()):
        if correct_drift:
            # copies: OpenCV changes its inputs in place
            (dx, dy), resp = cv2.phaseCorrelate(ref_s.copy(), _structure(r), win.copy())
            if resp < 0.03 or abs(dx) > max_shift or abs(dy) > max_shift:
                dx, dy = prev
            prev = (dx, dy)
            shifts[i] = dx, dy
            r = _move(r, (dx, dy))
        r = r.astype(np.float64)
        S_raw += r
        if ema is None:
            ema = r.copy()
        hp = r - ema
        ema += alpha * (r - ema)
        v = hp - cv2.GaussianBlur(hp, (0, 0), sig_space)
        S1 += v
        v2 = v * v
        S2 += v2
        S3 += v2 * v
        P["r"] += v[:, :-1] * v[:, 1:]
        P["d"] += v[:-1] * v[1:]
        P["dr"] += v[:-1, :-1] * v[1:, 1:]
        P["dl"] += v[:-1, 1:] * v[1:, :-1]
        n += 1
        _report(n, total, "Lining up frames", progress)
    shifts = shifts[:n]
    m = S1 / n
    sd = np.sqrt(np.maximum(S2 / n - m * m, 1e-12))
    # calcium signals jump up from rest (positive skew); the rings that haze removal
    # leaves around bright neurons dip down instead (negative skew)
    skew = (S3 / n - 3 * m * S2 / n + 2 * m ** 3) / sd ** 3
    acc, cnt = np.zeros((H, W)), np.zeros((H, W))

    def add(c, a_sl, b_sl):
        acc[a_sl] += c
        acc[b_sl] += c
        cnt[a_sl] += 1
        cnt[b_sl] += 1

    def corr(Pk, a_sl, b_sl):
        return (Pk / n - m[a_sl] * m[b_sl]) / (sd[a_sl] * sd[b_sl])

    pairs = {"r": (np.s_[:, :-1], np.s_[:, 1:]), "d": (np.s_[:-1, :], np.s_[1:, :]),
             "dr": (np.s_[:-1, :-1], np.s_[1:, 1:]), "dl": (np.s_[:-1, 1:], np.s_[1:, :-1])}
    for k, (a_sl, b_sl) in pairs.items():
        add(corr(P[k], a_sl, b_sl), a_sl, b_sl)
    corr_img = acc / np.maximum(cnt, 1)
    # pixels near the edge flicker when frames are shifted back into line: ignore them
    edge = int(np.ceil(np.abs(shifts).max())) + 2 if correct_drift and n else 1
    valid = np.zeros((H, W), bool)
    valid[edge:H - edge, edge:W - edge] = True
    return {"mean": S_raw / n, "corr": corr_img, "skew": skew, "valid": valid,
            "shifts": shifts, "diameter": diam, "frames": n}


def find_neurons(img: dict, diam: float, by: str = "activity", adjust: float = 0.0) -> np.ndarray:
    """Label image of neurons (0 = none), numbered 1..N from top-left."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    if by == "brightness":
        bg = ndi.gaussian_filter(img["mean"], max(2 * diam, 10))
        score = img["mean"] - bg
    else:
        skew = ndi.gaussian_filter(img["skew"], max(diam / 6, 0.7))
        score = img["corr"] * (skew > 0)
    s = ndi.gaussian_filter(score, max(diam / 6, 0.7))
    s[~img["valid"]] = np.median(s)
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med)) + 1e-12
    k = 4.0 + 6.0 * adjust                       # slider -0.5..0.5 -> 1..7 noise levels
    thr = med + k * mad
    if by == "brightness":
        # a cell body must be at least 2% brighter than its surroundings, not just a noise bump
        thr = max(thr, med + (1 + 2 * adjust) * 0.02 * abs(float(np.median(img["mean"]))))
    else:
        # pure noise gives correlations of about 1/sqrt(frames); a neuron must stand well clear
        thr = max(thr, med + (1 + 2 * adjust) * 4.0 / np.sqrt(max(img["frames"], 1)))
    pk = peak_local_max(s, min_distance=max(2, int(diam * 0.6)), threshold_abs=thr,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros(s.shape, np.int32)
    markers = np.zeros(s.shape, np.int32)
    markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = watershed(-s, markers, mask=s > med + 0.5 * k * mad)
    yy, xx = np.indices(s.shape)
    keep = np.zeros(s.shape, np.int32)
    min_area = 0.25 * np.pi * (diam / 2) ** 2
    out_id = 0
    order = sorted(range(len(pk)), key=lambda i: (round(pk[i][0] / max(diam, 1)), pk[i][1]))
    for i in order:
        py, px = pk[i]
        reg = (lab == i + 1) & (s >= med + 0.4 * (s[py, px] - med))
        reg &= np.hypot(yy - py, xx - px) <= 0.8 * diam
        reg = ndi.binary_fill_holes(reg)
        lab_r, nr = ndi.label(reg)
        if nr > 1:  # keep the piece holding the peak
            reg = lab_r == lab_r[py, px]
        if reg.sum() < min_area:
            continue
        out_id += 1
        keep[reg & (keep == 0)] = out_id
    return keep


def roi_weights(lab: np.ndarray, diam: float):
    """Sparse matrices that average each neuron's pixels, and its neuropil ring."""
    H, W = lab.shape
    n = int(lab.max())
    others = cv2.dilate((lab > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    gap, width = 2, int(max(diam, 6))
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (gap + width) + 1,) * 2)
    rows_a, cols_a, vals_a, rows_b, cols_b, vals_b = [], [], [], [], [], []
    objs = ndi.find_objects(lab)
    pad = gap + width + 1
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - pad), min(H, sl[0].stop + pad)
        x0, x1 = max(0, sl[1].start - pad), min(W, sl[1].stop + pad)
        m = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        yy, xx = np.nonzero(m)
        idx = (yy + y0) * W + (xx + x0)
        rows_a += [i - 1] * len(idx)
        cols_a += list(idx)
        vals_a += [1.0 / len(idx)] * len(idx)
        ring = (cv2.dilate(m, k_out) > 0) & ~(cv2.dilate(m, k_in) > 0) & ~others[y0:y1, x0:x1]
        yy, xx = np.nonzero(ring)
        if len(yy) < 10:  # crowded: fall back to everything nearby that isn't a neuron
            ring = ~others[y0:y1, x0:x1]
            yy, xx = np.nonzero(ring)
        idx = (yy + y0) * W + (xx + x0)
        if len(idx):
            rows_b += [i - 1] * len(idx)
            cols_b += list(idx)
            vals_b += [1.0 / len(idx)] * len(idx)
    A = sparse.csr_matrix((vals_a, (rows_a, cols_a)), shape=(n, H * W))
    B = sparse.csr_matrix((vals_b, (rows_b, cols_b)), shape=(n, H * W))
    return A, B


def extract_traces(mv: Movie, shifts: np.ndarray, A, B, progress=None):
    """Pass 2: brightness of every neuron and its neuropil in every frame."""
    T = len(shifts)
    F = np.zeros((T, A.shape[0]), np.float32)
    Fn = np.zeros_like(F)
    for i, r in enumerate(mv.raw()):
        if i >= T:
            break
        v = _move(r, shifts[i]).ravel()
        F[i] = A @ v
        Fn[i] = B @ v
        _report(i + 1, T, "Measuring neurons", progress)
    return F, Fn


def baseline(x: np.ndarray, fps: float, window_s: float) -> np.ndarray:
    """Resting level that follows slow changes such as bleaching: smooth, then a running
    minimum followed by a running maximum over the window."""
    T = len(x)
    sig = max(1.0, 0.3 * fps)
    win = int(np.clip(window_s * fps, 3, max(3, T // 3)))
    xs = ndi.gaussian_filter1d(x, sig, axis=0, mode="nearest")
    b = ndi.minimum_filter1d(xs, win, axis=0, mode="nearest")
    b = ndi.maximum_filter1d(b, win, axis=0, mode="nearest")
    # Within half a window of either end the window runs off the recording and lags slow
    # trends such as bleaching; continue the trend from just inside instead.
    h = win // 2
    if h >= 2 and T >= 4 * h:
        s0 = (b[2 * h] - b[h]) / h
        b[:h] = b[h] + s0 * (np.arange(h) - h)[:, None]
        e = T - 1 - h
        s1 = (b[e] - b[e - h]) / h
        b[e + 1:] = b[e] + s1 * (np.arange(e + 1, T) - e)[:, None]
    return b


def delta_f(F: np.ndarray, Fn: np.ndarray, fps: float, neuropil_factor: float,
            window_s: float) -> np.ndarray:
    """ΔF/F after removing neuropil: (Fc - resting Fc) / resting F, where Fc = F - r·neuropil."""
    Fc = F - neuropil_factor * Fn
    b_raw = np.maximum(baseline(F, fps, window_s), 1e-6)
    return (Fc - baseline(Fc, fps, window_s)) / b_raw


def detect_events(dff: np.ndarray, fps: float, threshold_sd: float = 3.5,
                  min_dff: float = 0.05, smooth_s: float = 0.05):
    """Firing events in every neuron: peaks at least threshold_sd noise levels and min_dff
    above rest. An event starts and ends where the trace crosses 20% of its height.
    Returns (events table, smoothed ΔF/F, noise per neuron, frames-in-event mask)."""
    sig = smooth_s * fps
    xs = ndi.gaussian_filter1d(dff, sig, axis=0, mode="nearest") if sig >= 0.5 else dff.copy()
    T, N = xs.shape
    dist = max(1, int(round(0.2 * fps)))      # two events need at least 0.2 s between peaks
    min_w = max(1.0, 0.1 * fps)               # events last at least 0.1 s at half height
    rows, noise = [], np.zeros(N)
    in_event = np.zeros((T, N), bool)
    for j in range(N):
        x = xs[:, j]
        m = np.median(x)
        low = x[x <= m] - m
        sd = float(np.sqrt(np.mean(low * low))) if len(low) else 0.0
        sd = sd or 1e-9
        noise[j] = sd
        z = (x - m) / sd
        pk, pr = find_peaks(z, height=threshold_sd, prominence=0.8 * threshold_sd,
                            distance=dist, width=min_w, rel_height=0.5)
        prev_end = 0
        for n_, p in enumerate(pk):
            amp = float(x[p] - m)
            if amp < min_dff:
                continue
            level = m + max(0.2 * amp, sd)
            o = p
            while o > prev_end and x[o - 1] > level:
                o -= 1
            nxt = pk[n_ + 1] if n_ + 1 < len(pk) else T
            e = p
            while e + 1 < nxt and x[e + 1] > level:
                e += 1
            prev_end = e
            in_event[o:e + 1, j] = True
            rows.append({
                "neuron_id": j + 1, "onset_frame": int(o), "peak_frame": int(p), "end_frame": int(e),
                "onset_s": round(o / fps, 3), "peak_s": round(p / fps, 3),
                "rise_time_s": round((p - o) / fps, 3),
                "duration_s": round((e - o + 1) / fps, 3),
                "half_width_s": round(float(pr["widths"][n_]) / fps, 3),
                "amplitude_dff": round(amp, 4),
                "snr": round(float(z[p]), 2),
            })
    cols = ["neuron_id", "onset_frame", "peak_frame", "end_frame", "onset_s", "peak_s",
            "rise_time_s", "duration_s", "half_width_s", "amplitude_dff", "snr"]
    return pd.DataFrame(rows, columns=cols), xs, noise, in_event


def network_bursts(events: pd.DataFrame, n_neurons: int, n_active: int, T: int, fps: float,
                   window_s: float = 0.5, min_fraction: float = 0.3):
    """Moments when many neurons start firing within window_s of each other."""
    cols = ["burst", "start_frame", "peak_frame", "end_frame", "start_s", "peak_s", "duration_s",
            "neurons", "fraction_of_active", "mean_amplitude_dff"]
    w = max(1, int(round(window_s * fps)))
    need = max(3, int(np.ceil(min_fraction * max(n_active, 1))))
    if events.empty or n_active < 3:
        return pd.DataFrame(columns=cols), np.zeros(T), need, []
    on = np.zeros((T, n_neurons), np.float32)
    on[events["onset_frame"].to_numpy(), events["neuron_id"].to_numpy() - 1] = 1
    near = ndi.uniform_filter1d(on, w, axis=0, mode="constant") > 1e-6  # onset within the window
    count = near.sum(axis=1).astype(float)
    lab, n = ndi.label(count >= need)
    spans = [(sl[0].start, sl[0].stop - 1) for sl in ndi.find_objects(lab)] if n else []
    merged = []
    for a, b in spans:  # join bursts that are closer than one window
        if merged and a - merged[-1][1] <= w:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    rows, members = [], []
    for k, (a, b) in enumerate(merged, start=1):
        lo, hi = a - w // 2, b + w // 2
        ev = events[(events["onset_frame"] >= lo) & (events["onset_frame"] <= hi)]
        who = ev["neuron_id"].nunique()
        p = a + int(np.argmax(count[a:b + 1]))
        s0 = int(ev["onset_frame"].min()) if len(ev) else a
        s1 = int(ev["end_frame"].max()) if len(ev) else b
        members.append(set(ev["neuron_id"].tolist()))
        rows.append({"burst": k, "start_frame": s0, "peak_frame": p, "end_frame": s1,
                     "start_s": round(s0 / fps, 3), "peak_s": round(p / fps, 3),
                     "duration_s": round((s1 - s0 + 1) / fps, 3), "neurons": int(who),
                     "fraction_of_active": round(who / max(n_active, 1), 3),
                     "mean_amplitude_dff": round(float(ev["amplitude_dff"].mean()), 4) if len(ev) else np.nan})
    return pd.DataFrame(rows, columns=cols), count, need, members


def _intervals(times: np.ndarray) -> tuple[float, float]:
    if len(times) < 2:
        return np.nan, np.nan
    d = np.diff(np.sort(times))
    return float(d.mean()), float(d.std() / d.mean()) if d.mean() > 0 else np.nan


def run(src, out_dir: Path, s, fps: float, first: int, last: int, scale: float,
        pixel_size: float | None, progress=None) -> tuple[dict, dict]:
    """Analyze one movie. Returns (summary of results, details used for drawing the video)."""
    from .cells import units

    mv = Movie(src, first, last, scale, fps)
    diam_set = s.cell_diameter_px * scale if s.cell_diameter_px else None
    img = summary_images(mv, diam_set, s.correct_drift, progress)
    diam, T = img["diameter"], img["frames"]
    lab = find_neurons(img, diam, s.find_neurons_by, s.threshold_adjust)
    N = int(lab.max())
    k, L = units(pixel_size / scale if pixel_size else None)
    t = (np.arange(T) + first) / fps
    _write_roi_map(out_dir / "rois.png", img["mean"], img["corr"], lab)
    info = {"neurons_found": N, "diameter_px": round(diam / scale, 2), "frames": T,
            "max_drift_px": round(float(np.abs(img["shifts"]).max() / scale), 2) if T else 0.0,
            "length_unit": "µm" if L == "um" else "pixels", "calibrated": bool(pixel_size)}
    if N == 0:
        for f in ("neurons", "events", "bursts", "traces_dff", "traces_raw", "correlations", "frames"):
            pd.DataFrame().to_csv(out_dir / f"{f}.csv", index=False)
        return info, {}

    A, B = roi_weights(lab, diam)
    F, Fn = extract_traces(mv, img["shifts"], A, B, progress)
    dff = delta_f(F, Fn, fps, s.neuropil_factor, s.baseline_window_s)
    events, xs, noise, in_event = detect_events(dff, fps, s.event_threshold_sd, s.min_event_dff)
    if s.find_neurons_by == "activity" and not events.empty:
        # regions found by flicker that never fire are motion or haze, not neurons
        keep = np.sort(events["neuron_id"].unique()) - 1
        if len(keep) < N:
            remap = np.zeros(N + 1, np.int32)
            remap[keep + 1] = np.arange(1, len(keep) + 1)
            lab = remap[lab]
            F, Fn, dff, xs, noise = F[:, keep], Fn[:, keep], dff[:, keep], xs[:, keep], noise[keep]
            in_event = in_event[:, keep]
            events["neuron_id"] = remap[events["neuron_id"].to_numpy()]
            info["regions_without_firing_dropped"] = int(N - len(keep))
            N = len(keep)
            info["neurons_found"] = N
            _write_roi_map(out_dir / "rois.png", img["mean"], img["corr"], lab)
    if not events.empty:  # times relative to the start of the file
        for c in ("onset_frame", "peak_frame", "end_frame"):
            events[c] += first
        events["onset_s"] = (events["onset_frame"] / fps).round(3)
        events["peak_s"] = (events["peak_frame"] / fps).round(3)
    counts = events.groupby("neuron_id").size() if not events.empty else pd.Series(dtype=int)
    n_active = int((counts > 0).sum())
    ev_local = events.assign(onset_frame=events["onset_frame"] - first,
                             end_frame=events["end_frame"] - first) if not events.empty else events
    bursts, near_count, need, members = network_bursts(ev_local, N, n_active, T, fps, s.burst_window_s,
                                              s.burst_min_fraction)
    if not bursts.empty:
        for c in ("start_frame", "peak_frame", "end_frame"):
            bursts[c] += first
        bursts["start_s"] = (bursts["start_frame"] / fps).round(3)
        bursts["peak_s"] = (bursts["peak_frame"] / fps).round(3)
    in_burst = np.zeros(T, bool)
    for b in bursts.itertuples():
        in_burst[b.start_frame - first:b.end_frame - first + 1] = True
    if not events.empty:
        events["in_burst"] = in_burst[events["onset_frame"].to_numpy() - first]

    # --- per-neuron table
    minutes = T / fps / 60
    props = ndi.center_of_mass(np.ones_like(lab), lab, range(1, N + 1))
    areas = ndi.sum(np.ones_like(lab), lab, range(1, N + 1))
    rest = baseline(F, fps, s.baseline_window_s).mean(axis=0)
    joined = [sum((j + 1) in m for m in members) for j in range(N)]
    rows = []
    for j in range(N):
        e = events[events["neuron_id"] == j + 1] if not events.empty else events
        cy, cx = props[j]
        iei, cv = _intervals(e["onset_s"].to_numpy()) if len(e) else (np.nan, np.nan)
        rows.append({
            "neuron_id": j + 1,
            f"x_{L}": round(cx / scale * k, 2), f"y_{L}": round(cy / scale * k, 2),
            f"area_{L}2": round(float(areas[j]) / scale ** 2 * k * k, 2),
            "resting_brightness": round(float(rest[j]), 2),
            "noise_dff": round(float(noise[j]), 4),
            "active": bool(len(e) > 0),
            "events": int(len(e)),
            "events_per_min": round(len(e) / minutes, 3) if minutes else np.nan,
            "mean_amplitude_dff": round(float(e["amplitude_dff"].mean()), 4) if len(e) else np.nan,
            "max_dff": round(float(xs[:, j].max()), 4),
            "mean_half_width_s": round(float(e["half_width_s"].mean()), 3) if len(e) else np.nan,
            "mean_interval_s": round(iei, 3) if iei == iei else np.nan,
            "interval_cv": round(cv, 3) if cv == cv else np.nan,
            "first_event_s": float(e["onset_s"].min()) if len(e) else np.nan,
            "bursts_joined": joined[j],
            "burst_participation": round(joined[j] / len(bursts), 3) if len(bursts) else 0.0,
        })
    neurons = pd.DataFrame(rows)

    # --- similarity between neurons
    C = np.corrcoef(xs.T) if N > 1 else np.ones((1, 1))
    C = np.nan_to_num(C)
    iu = np.triu_indices(N, 1)
    cx_ = neurons[f"x_{L}"].to_numpy()
    cy_ = neurons[f"y_{L}"].to_numpy()
    corr = pd.DataFrame({
        "neuron_a": iu[0] + 1, "neuron_b": iu[1] + 1,
        f"distance_{L}": np.round(np.hypot(cx_[iu[0]] - cx_[iu[1]], cy_[iu[0]] - cy_[iu[1]]), 2),
        "correlation": np.round(C[iu], 4),
    }) if N > 1 else pd.DataFrame(columns=["neuron_a", "neuron_b", f"distance_{L}", "correlation"])
    act = neurons["active"].to_numpy()
    act_pairs = act[iu[0]] & act[iu[1]] if N > 1 else np.array([], bool)
    mean_corr = float(C[iu][act_pairs].mean()) if act_pairs.sum() else np.nan

    # --- per-frame table
    n_firing = in_event.sum(axis=1)
    frames = pd.DataFrame({"frame": np.arange(T) + first, "time_s": np.round(t, 4),
                           "n_firing": n_firing, "fraction_firing": np.round(n_firing / N, 4),
                           "mean_dff": np.round(xs.mean(axis=1), 5),
                           "neurons_starting_nearby": near_count.astype(int),
                           "in_burst": in_burst})

    names = [f"neuron_{j + 1}" for j in range(N)]
    tr = pd.DataFrame(np.round(dff, 5), columns=names)
    tr.insert(0, "time_s", np.round(t, 4))
    tr.insert(0, "frame", np.arange(T) + first)
    raw = pd.DataFrame(np.round(np.hstack([F, Fn]), 3),
                       columns=[f"F_{j + 1}" for j in range(N)] + [f"neuropil_{j + 1}" for j in range(N)])
    raw.insert(0, "time_s", np.round(t, 4))
    raw.insert(0, "frame", np.arange(T) + first)

    neurons.to_csv(out_dir / "neurons.csv", index=False)
    events.to_csv(out_dir / "events.csv", index=False)
    bursts.to_csv(out_dir / "bursts.csv", index=False)
    tr.to_csv(out_dir / "traces_dff.csv", index=False)
    raw.to_csv(out_dir / "traces_raw.csv", index=False)
    corr.to_csv(out_dir / "correlations.csv", index=False)
    frames.to_csv(out_dir / "frames.csv", index=False)

    a = neurons[neurons["active"]]
    secs = T / fps
    ibi, ibi_cv = _intervals(bursts["peak_s"].to_numpy()) if len(bursts) else (np.nan, np.nan)
    info.update({
        "active_neurons": n_active,
        "fraction_active": round(n_active / N, 3),
        "events": int(len(events)),
        "mean_events_per_min": round(float(a["events_per_min"].mean()), 3) if len(a) else 0.0,
        "median_amplitude_dff": round(float(events["amplitude_dff"].median()), 4) if len(events) else None,
        "median_half_width_s": round(float(events["half_width_s"].median()), 3) if len(events) else None,
        "network_bursts": int(len(bursts)),
        "bursts_per_min": round(len(bursts) / (secs / 60), 3) if secs else 0.0,
        "mean_interburst_interval_s": round(ibi, 2) if ibi == ibi else None,
        "interburst_interval_cv": round(ibi_cv, 3) if ibi_cv == ibi_cv else None,
        "mean_neurons_per_burst": round(float(bursts["neurons"].mean()), 1) if len(bursts) else None,
        "fraction_events_in_bursts": round(float(events["in_burst"].mean()), 3) if len(events) else None,
        "burst_needs_neurons": int(need),
        "mean_pairwise_correlation": round(mean_corr, 3) if mean_corr == mean_corr else None,
    })
    draw = {"labels": lab, "scale": scale, "shifts": img["shifts"], "in_event": in_event,
            "in_burst": in_burst, "n": N}
    return info, draw


# ---------------------------------------------------------------- pictures
def _stretch(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [1, 99.7])
    return (np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)


def _contours(lab: np.ndarray) -> dict[int, list]:
    out = {}
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = np.zeros(lab.shape, np.uint8)
        m[sl] = lab[sl] == i
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out[i] = cs
    return out


def _write_roi_map(path: Path, mean_img: np.ndarray, corr_img: np.ndarray, lab: np.ndarray):
    up = max(1, int(np.ceil(500 / max(lab.shape))))
    panels = []
    for a in (mean_img, corr_img):
        g = cv2.cvtColor(_stretch(a), cv2.COLOR_GRAY2BGR)
        panels.append(cv2.resize(g, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST))
    big = cv2.resize(lab.astype(np.int32).astype(np.float32), None, fx=up, fy=up,
                     interpolation=cv2.INTER_NEAREST).astype(np.int32)
    cont = _contours(big)
    fs = 0.4 * max(1, up * 0.6)
    for p in panels:
        for i, cs in cont.items():
            cv2.drawContours(p, cs, -1, FIRING, 1)
            x, y, w, h = cv2.boundingRect(cs[0])
            cv2.putText(p, str(i), (x + w, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1,
                        cv2.LINE_AA)
    gap = np.full((panels[0].shape[0], 8, 3), 255, np.uint8)
    cv2.imwrite(str(path), np.hstack([panels[0], gap, panels[1]]))


def render(src, out: Path, draw: dict, fps: float, first: int, last: int, writer, progress=None):
    """Video with each neuron outlined; outlines light up while the neuron fires."""
    lab, scale, shifts = draw["labels"], draw["scale"], draw["shifts"]
    in_event, in_burst, N = draw["in_event"], draw["in_burst"], draw["n"]
    W, H = src.width, src.height
    full = cv2.resize(lab.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    cont = _contours(full)
    thick = max(1, W // 500)
    fs = max(0.4, W / 1100)
    wr = writer(out, fps, (W, H))
    print(f"Writing {out.name}", file=sys.stderr)
    T = len(shifts)
    for k, (img, _) in enumerate(src.frames(first, last)):
        if k >= T:
            break
        img = img.copy()
        off = np.round(shifts[k] / scale).astype(int)
        for i, cs in cont.items():
            fire = in_event[k, i - 1]
            moved = [c + off for c in cs]
            cv2.drawContours(img, moved, -1, FIRING if fire else QUIET, thick * (2 if fire else 1))
        n_f = int(in_event[k].sum())
        lines = [f"t = {(k + first) / fps:7.2f} s   firing now: {n_f} / {N}"]
        if in_burst[k]:
            lines.append("NETWORK BURST")
        for j, txt in enumerate(lines):
            y = int(24 * fs / 0.55) * (j + 1)
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3 * thick)
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        FIRING if j else (255, 255, 255), thick)
        wr.write(img)
        _report(k + 1, T, "Drawing video", progress)
    print(file=sys.stderr)
    wr.release()


def _report(i: int, total: int, stage: str, cb):
    from .pipeline import _progress

    _progress(i, total, stage, cb)
