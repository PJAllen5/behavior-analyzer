"""Turn raw tracks into behavior data.

Distances are measured in "body lengths" (BL), the median object size, so the same
thresholds work for ants, birds and people regardless of zoom.

Outputs (all pandas DataFrames):
  tracks        one row per object per frame: position, velocity, speed, heading, turning
  frames        one row per frame: group metrics + a behavior label
  interactions  one row per close encounter between two objects
  behaviors     contiguous stretches of time with the same group behavior
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Group behavior labels
RESTING = "resting"                # little movement
COLLECTIVE = "collective_motion"   # moving together in one direction (flocking, schooling, marching)
MILLING = "milling"                # circling around a shared centre (vortex / torus)
DISORDERED = "disordered"          # moving, but in many directions (swarming, foraging)
TOO_FEW = "too_few"                # fewer than 3 moving objects visible


def _smooth(a: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(a) < 3:
        return a
    win = min(win, len(a) if len(a) % 2 else len(a) - 1)
    k = np.ones(win) / win
    pad = win // 2
    return np.convolve(np.pad(a, pad, mode="edge"), k, mode="valid")


def build_tracks(rows: list[tuple], fps: float, min_track_frames: int = 5,
                 smooth_seconds: float = 0.2, keep_ids: set | None = None) -> tuple[pd.DataFrame, float]:
    from .track import EXTRA_COLS
    df = pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h", "class", "conf", *EXTRA_COLS])
    df = df.drop(columns=[c for c in EXTRA_COLS if df[c].isna().all()])
    if df.empty:
        return df, 1.0
    df = df.sort_values(["id", "frame"])
    long_enough = df.groupby("id")["frame"].transform("size") >= min_track_frames
    df = df[long_enough | df["id"].isin(keep_ids or set())].copy()
    body = float(np.median(np.maximum(df["w"], df["h"]))) if len(df) else 1.0
    win = max(1, int(round(smooth_seconds * fps)) | 1)

    parts = []
    for tid, g in df.groupby("id", sort=False):
        g = g.copy()
        t = g["frame"].to_numpy() / fps
        x = _smooth(g["x"].to_numpy(float), win)
        y = _smooth(g["y"].to_numpy(float), win)
        if len(g) >= 2:
            vx, vy = np.gradient(x, t), np.gradient(y, t)
        else:
            vx = vy = np.zeros(len(g))
        speed = np.hypot(vx, vy)
        heading = np.degrees(np.arctan2(vy, vx))
        dh = np.gradient(np.unwrap(np.radians(heading)), t) if len(g) >= 2 else np.zeros(len(g))
        acc = np.gradient(speed, t) if len(g) >= 2 else np.zeros(len(g))
        g["time_s"] = t
        g["x_smooth"], g["y_smooth"] = x, y
        g["vx"], g["vy"], g["speed_px_s"] = vx, vy, speed
        g["speed_bl_s"] = speed / body
        g["heading_deg"] = heading
        g["turn_rate_deg_s"] = np.degrees(dh)
        g["accel_px_s2"] = acc
        parts.append(g)
    out = pd.concat(parts).sort_values(["frame", "id"]).reset_index(drop=True)
    return out, body


def frame_metrics(tracks: pd.DataFrame, fps: float, body: float,
                  moving_bl_s: float = 0.5, label_seconds: float = 0.5,
                  polar_thresh: float = 0.65, mill_thresh: float = 0.45) -> pd.DataFrame:
    """Per-frame group statistics.

    polarization 0..1: 1 = everyone heading the same way.
    rotation     0..1: 1 = everyone circling the group centre the same way.
    nn_dist_bl:  average distance to nearest neighbour, in body lengths.
    spread_bl:   radius of the group (radius of gyration), in body lengths.
    """
    recs = []
    for f, g in tracks.groupby("frame"):
        n = len(g)
        rec = {"frame": f, "time_s": f / fps, "n_visible": n}
        pos = g[["x_smooth", "y_smooth"]].to_numpy()
        vel = g[["vx", "vy"]].to_numpy()
        spd = g["speed_bl_s"].to_numpy()
        moving = spd >= moving_bl_s
        rec["n_moving"] = int(moving.sum())
        rec["mean_speed_bl_s"] = float(spd.mean())
        c = pos.mean(axis=0)
        rec["centroid_x"], rec["centroid_y"] = c
        rec["spread_bl"] = float(np.sqrt(((pos - c) ** 2).sum(1).mean()) / body)
        if n >= 2:
            d, _ = cKDTree(pos).query(pos, k=2)
            rec["nn_dist_bl"] = float(d[:, 1].mean() / body)
        else:
            rec["nn_dist_bl"] = np.nan
        if moving.sum() >= 2:
            u = vel[moving] / np.linalg.norm(vel[moving], axis=1, keepdims=True)
            rec["polarization"] = float(np.linalg.norm(u.mean(0)))
            r = pos[moving] - c
            rn = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
            rec["rotation"] = float(abs((rn[:, 0] * u[:, 1] - rn[:, 1] * u[:, 0]).mean()))
        else:
            rec["polarization"] = rec["rotation"] = np.nan
        recs.append(rec)
    fr = pd.DataFrame(recs)
    if fr.empty:
        return fr

    # Label each frame, using a short rolling average so labels don't flicker.
    win = max(1, int(round(label_seconds * fps)))
    roll = lambda s: s.rolling(win, center=True, min_periods=1).mean()  # noqa: E731
    P, M = roll(fr["polarization"]), roll(fr["rotation"])
    nm, frac = roll(fr["n_moving"]), roll(fr["n_moving"] / fr["n_visible"].clip(lower=1))
    labels = np.where(
        fr["n_visible"] < 3, TOO_FEW,
        np.where(frac < 0.3, RESTING,
                 np.where(nm < 3, TOO_FEW,
                          np.where(P >= polar_thresh, COLLECTIVE,
                                   np.where(M >= mill_thresh, MILLING, DISORDERED)))))
    fr["behavior"] = labels

    # Is the group bunching up or spreading out? (change in spread, BL per second)
    ds = np.gradient(roll(fr["spread_bl"]).to_numpy(), fr["time_s"].to_numpy()) if len(fr) > 1 else [0]
    fr["spread_change_bl_s"] = ds
    fr["cohesion_trend"] = np.where(ds > moving_bl_s, "dispersing", np.where(ds < -moving_bl_s, "aggregating", "steady"))
    return fr


def behavior_segments(frames: pd.DataFrame, fps: float, min_seconds: float = 1.0) -> pd.DataFrame:
    if frames.empty:
        return pd.DataFrame(columns=["behavior", "start_s", "end_s", "duration_s"])
    lab = frames["behavior"].to_numpy()
    starts = np.r_[0, np.flatnonzero(lab[1:] != lab[:-1]) + 1]
    ends = np.r_[starts[1:], len(lab)]
    recs = []
    for s, e in zip(starts, ends):
        seg = frames.iloc[s:e]
        dur = (seg["frame"].iloc[-1] - seg["frame"].iloc[0] + 1) / fps
        if dur < min_seconds:
            continue
        recs.append({
            "behavior": lab[s],
            "start_s": round(seg["time_s"].iloc[0], 3),
            "end_s": round(seg["time_s"].iloc[-1] + 1 / fps, 3),
            "duration_s": round(dur, 3),
            "start_frame": int(seg["frame"].iloc[0]),
            "end_frame": int(seg["frame"].iloc[-1]),
            "mean_individuals": round(seg["n_visible"].mean(), 1),
            "mean_polarization": round(seg["polarization"].mean(), 3),
            "mean_rotation": round(seg["rotation"].mean(), 3),
            "mean_speed_bl_s": round(seg["mean_speed_bl_s"].mean(), 3),
            "mean_nn_dist_bl": round(seg["nn_dist_bl"].mean(), 3),
        })
    return pd.DataFrame(recs)


def interactions(tracks: pd.DataFrame, fps: float, body: float,
                 contact_bl: float = 2.0, gap_seconds: float = 0.3,
                 min_seconds: float = 0.2, moving_bl_s: float = 0.5) -> pd.DataFrame:
    """Find close encounters: two objects within `contact_bl` body lengths.

    For each encounter reports duration, closeness, whether they moved in the same
    direction (alignment), who approached whom, and who was in front (leader).
    """
    radius = contact_bl * body
    close = []
    for f, g in tracks.groupby("frame"):
        if len(g) < 2:
            continue
        pos = g[["x_smooth", "y_smooth"]].to_numpy()
        ids = g["id"].to_numpy()
        for i, j in cKDTree(pos).query_pairs(radius):
            a, b = sorted((ids[i], ids[j]))
            close.append((a, b, f))
    if not close:
        return pd.DataFrame(columns=["id_a", "id_b", "start_s", "end_s", "duration_s"])

    idx = tracks.set_index(["id", "frame"])
    gap = max(1, int(gap_seconds * fps))
    cdf = pd.DataFrame(close, columns=["a", "b", "frame"]).sort_values(["a", "b", "frame"])
    recs = []
    for (a, b), g in cdf.groupby(["a", "b"]):
        fr = g["frame"].to_numpy()
        breaks = np.flatnonzero(np.diff(fr) > gap) + 1
        for seg in np.split(fr, breaks):
            f0, f1 = int(seg[0]), int(seg[-1])
            dur = (f1 - f0 + 1) / fps
            if dur < min_seconds:
                continue
            A = idx.loc[a].loc[f0:f1]
            B = idx.loc[b].loc[f0:f1]
            common = A.index.intersection(B.index)
            A, B = A.loc[common], B.loc[common]
            rel = B[["x_smooth", "y_smooth"]].to_numpy() - A[["x_smooth", "y_smooth"]].to_numpy()
            dist = np.linalg.norm(rel, axis=1)
            va, vb = A[["vx", "vy"]].to_numpy(), B[["vx", "vy"]].to_numpy()
            na, nb = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
            ok = (na > 1e-6) & (nb > 1e-6)
            align = float(((va * vb).sum(1)[ok] / (na[ok] * nb[ok])).mean()) if ok.any() else np.nan

            # Who approached? Velocity toward the other in the second before contact.
            approacher = "none"
            pre0 = max(0, f0 - max(2, int(fps)))
            try:
                Ap, Bp = idx.loc[a].loc[pre0:f0], idx.loc[b].loc[pre0:f0]
                cm = Ap.index.intersection(Bp.index)
                if len(cm) >= 2:
                    r = Bp.loc[cm, ["x_smooth", "y_smooth"]].to_numpy() - Ap.loc[cm, ["x_smooth", "y_smooth"]].to_numpy()
                    rn = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
                    toward_a = (Ap.loc[cm, ["vx", "vy"]].to_numpy() * rn).sum(1).mean() / body
                    toward_b = (-Bp.loc[cm, ["vx", "vy"]].to_numpy() * rn).sum(1).mean() / body
                    if max(toward_a, toward_b) > 0.6 * moving_bl_s:
                        if toward_a > 2 * max(toward_b, 0):
                            approacher = str(a)
                        elif toward_b > 2 * max(toward_a, 0):
                            approacher = str(b)
                        else:
                            approacher = "both"
            except KeyError:
                pass

            # Leader: who is in front along the pair's shared direction of travel.
            leader = "none"
            mv = (va + vb).mean(0)
            if np.linalg.norm(mv) / body > moving_bl_s and align > 0.5:
                front = (rel @ (mv / np.linalg.norm(mv))).mean()  # >0 means B ahead of A
                if abs(front) > 0.3 * body:
                    leader = str(b if front > 0 else a)

            recs.append({
                "id_a": a, "id_b": b,
                "class_a": A["class"].iloc[0] if len(A) else "", "class_b": B["class"].iloc[0] if len(B) else "",
                "start_s": round(f0 / fps, 3), "end_s": round((f1 + 1) / fps, 3),
                "duration_s": round(dur, 3), "start_frame": f0, "end_frame": f1,
                "min_dist_bl": round(float(dist.min() / body), 3) if len(dist) else np.nan,
                "mean_dist_bl": round(float(dist.mean() / body), 3) if len(dist) else np.nan,
                "alignment": round(align, 3),
                "relative_speed_bl_s": round(float(np.linalg.norm(va - vb, axis=1).mean() / body), 3) if len(va) else np.nan,
                "approached_by": approacher,
                "leader": leader,
            })
    return pd.DataFrame(recs).sort_values("start_s").reset_index(drop=True)
