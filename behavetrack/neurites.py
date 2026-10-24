"""Neurite outgrowth: trace axons and dendrites, link them to their cell bodies, and measure
growth over time.

For each analyzed frame:
  1. even out the lighting, then find thin lines with a ridge filter
  2. find cell bodies: blobs much thicker than a neurite
  3. thin the neurites to 1-pixel centre lines (skeleton) and trim tiny side spurs
  4. give each neurite to the cell body it connects to, by distance along the neurite
  5. measure total length, the longest path from the cell body, primary neurites,
     branch points and tips (growth cones)
Across frames, each cell body keeps its ID and growth cones are followed, to measure how
fast neurites extend or pull back.

Files written:
  neurons.csv         one row per neuron: final length, longest neurite, branching, growth rate
  neurite_frames.csv  one row per neuron per frame: length, longest, primary neurites, branches, tips
  growth_cones.csv    one row per growth cone followed over time: extension, speed, pull-backs
  growth_cone_tracks.csv  every growth cone in every frame: position, distance from the cell body
  sholl.csv           one row per neuron per ring (last frame): neurite crossings at each distance
  frames.csv          one row per frame: totals for the whole image
  neurites.png        the last frame with the traced neurites drawn on
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import dijkstra

GRAY = (150, 150, 150)


def _color(i: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(i * 7919)
    return tuple(int(c) for c in rng.integers(80, 255, 3))


# ---------------------------------------------------------------- one frame
def _signal(g: np.ndarray, look: str, soma_d: float) -> np.ndarray:
    bg = cv2.GaussianBlur(g, (0, 0), max(1.5 * soma_d, 15))
    d = g - bg
    if look == "dark":
        return -d
    if look == "auto":
        return np.abs(d)
    return d


def _noise(a: np.ndarray) -> float:
    med = np.median(a)
    return float(1.4826 * np.median(np.abs(a - med))) or 1e-9


def find_somas(d: np.ndarray, soma_d: float, width: float, adjust: float = 0.0) -> np.ndarray:
    """Cell bodies: bright regions that survive an opening wider than any neurite."""
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    ds = cv2.GaussianBlur(d, (0, 0), max(soma_d / 8, 0.8))
    nz = _noise(d)
    thr = max(threshold_otsu(ds), np.median(ds) + 6 * nz) * (1 + adjust)
    m = (ds > thr).astype(np.uint8)
    r = int(max(2, 0.3 * soma_d, 1.2 * width))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = ndi.binary_fill_holes(m)
    lab = label(m)
    areas = np.bincount(lab.ravel())
    small = areas < 0.35 * np.pi * (soma_d / 2) ** 2
    small[0] = False
    lab[small[lab]] = 0
    return label(lab > 0)


def estimate_soma_diameter(g: np.ndarray, look: str, width: float) -> float:
    d = _signal(g, look, 15.0)
    ds = cv2.GaussianBlur(d, (0, 0), 1.0)
    m = ds > np.median(ds) + 6 * _noise(d)
    dt = ndi.distance_transform_edt(m)
    lab, n = ndi.label(m)
    if not n:
        return 15.0
    peaks = ndi.maximum(dt, lab, range(1, n + 1))
    thick = [p for p in peaks if p >= max(2.0 * width, 3.0)]
    return float(np.clip(2 * np.median(thick) + 1, 6, min(g.shape) / 4)) if thick else 15.0


def neurite_mask(d: np.ndarray, width: float, adjust: float = 0.0) -> np.ndarray:
    from skimage.filters import apply_hysteresis_threshold, sato

    s0 = max(0.7, 0.29 * width)
    R = sato(d, sigmas=[s0 * 0.75, s0, s0 * 1.5], black_ridges=False)
    med, mad = np.median(R), _noise(R)
    lo = med + 3.0 * (1 + 1.5 * adjust) * mad
    hi = med + 2.5 * (lo - med)
    m = apply_hysteresis_threshold(R, lo, hi)
    return m & (d > 2.0 * (1 + adjust) * _noise(d))


def _prune(skel: np.ndarray, n: int) -> np.ndarray:
    """Remove side spurs shorter than n pixels, then let real tips grow back."""
    k = np.ones((3, 3))
    s = skel.copy()
    removed = []
    for _ in range(n):
        nb = ndi.convolve(s.astype(np.uint8), k, mode="constant") - 1
        ends = s & (nb <= 1)
        if not ends.any():
            break
        removed.append(ends)
        s &= ~ends
    for ends in reversed(removed):
        nb = ndi.convolve(s.astype(np.uint8), k, mode="constant") - 1
        tips = s & (nb <= 1)
        grow = ends & (ndi.binary_dilation(tips, np.ones((3, 3), bool)))
        s |= grow
    return s


def _graph(skel: np.ndarray):
    """Pixels of the skeleton as a graph: orthogonal steps weigh 1, diagonal steps √2
    (diagonals are skipped where an orthogonal path exists, so length isn't double counted)."""
    H, W = skel.shape
    ys, xs = np.nonzero(skel)
    idx = -np.ones((H, W), np.int64)
    idx[ys, xs] = np.arange(len(ys))
    us, vs, ws = [], [], []
    pad = np.pad(skel, 1)
    for dy, dx, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
        ok = pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] & skel
        if dy and dx:  # diagonal: skip if an orthogonal neighbour already joins them
            bridge = pad[1:1 + H, 1 + dx:1 + dx + W] | pad[1 + dy:1 + dy + H, 1:1 + W]
            ok &= ~bridge
        y, x = np.nonzero(ok)
        us.append(idx[y, x])
        vs.append(idx[y + dy, x + dx])
        ws.append(np.full(len(y), w))
    u, v, w = np.concatenate(us), np.concatenate(vs), np.concatenate(ws)
    n = len(ys)
    G = sparse.coo_matrix((w, (u, v)), shape=(n, n)).tocsr()
    return ys, xs, G, u, v, w


def analyze_frame(g: np.ndarray, look: str, soma_d: float, width: float, adjust: float = 0.0) -> dict:
    from skimage.morphology import skeletonize

    d = _signal(g, look, soma_d)
    somas = find_somas(d, soma_d, width, adjust)
    m = neurite_mask(d, width, adjust)
    m = m | (somas > 0)
    lab, n = ndi.label(m, np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    min_px = max(15, int(2 * soma_d))
    has_soma = np.zeros(n + 1, bool)
    has_soma[np.unique(lab[somas > 0])] = True
    drop = (sizes < min_px) & ~has_soma
    drop[0] = False
    m[drop[lab]] = False

    skel = skeletonize(m)
    # trim spurs while neurites are still joined to the cell body's skeleton, so a short
    # neurite's base isn't mistaken for a loose end
    skel = _prune(skel, max(3, int(round(1.5 * width + 2))))
    inside = cv2.dilate((somas > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    skel &= ~inside
    pieces, npieces = ndi.label(skel, np.ones((3, 3)))
    psize = np.bincount(pieces.ravel())
    near = cv2.dilate((somas > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    touches = np.zeros(npieces + 1, bool)
    touches[np.unique(pieces[near & skel])] = True
    tiny = (psize < max(4, width * 2)) & ~touches
    tiny[0] = False
    skel[tiny[pieces]] = False

    ys, xs, G, u, v, w = _graph(skel)
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3)), mode="constant") - 1
    owner_img = ndi.grey_dilation(somas, size=(7, 7))
    root = np.flatnonzero(near[ys, xs] & (owner_img[ys, xs] > 0)) if len(ys) else np.array([], int)
    owner = np.zeros(len(ys), np.int32)
    dist = np.full(len(ys), np.inf)
    if len(root):
        dist, _, src = dijkstra(G, directed=False, indices=root, min_only=True,
                                return_predecessors=True)
        ok = np.isfinite(dist)
        owner[ok] = owner_img[ys[src[ok]], xs[src[ok]]]
    return {"somas": somas, "ys": ys, "xs": xs, "owner": owner, "dist": dist, "nb": nb[ys, xs],
            "root": root, "edges": (u, v, w), "skel": skel}


def measure(fr: dict) -> tuple[pd.DataFrame, dict]:
    """Per-cell-body numbers for one analyzed frame (in pixels)."""
    somas, ys, xs, owner, dist, nb = (fr[k] for k in ("somas", "ys", "xs", "owner", "dist", "nb"))
    u, v, w = fr["edges"]
    n = int(somas.max())
    same = owner[u] == owner[v] if len(u) else np.array([], bool)
    length = np.bincount(owner[u][same], weights=w[same], minlength=n + 1) if len(u) else np.zeros(n + 1)
    unattached = float(length[0])
    rows = []
    cms = ndi.center_of_mass(np.ones_like(somas), somas, range(1, n + 1)) if n else []
    areas = np.bincount(somas.ravel(), minlength=n + 1)
    is_root = np.zeros(len(ys), bool)
    is_root[fr["root"]] = True
    ends = (nb == 1) & ~is_root & (dist > 3)
    branch_px = nb >= 3
    # branch points: groups of touching branch pixels count once
    bp_img = np.zeros(fr["skel"].shape, bool)
    bp_img[ys[branch_px], xs[branch_px]] = True
    bp_lab, _ = ndi.label(bp_img, np.ones((3, 3)))
    root_img = np.zeros(fr["skel"].shape, bool)
    root_img[ys[is_root], xs[is_root]] = True
    root_lab, _ = ndi.label(root_img, np.ones((3, 3)))
    tips = []
    for s in range(1, n + 1):
        mine = owner == s
        own_d = dist[mine]
        cy, cx = cms[s - 1]
        prim = len(np.unique(root_lab[ys[mine & is_root], xs[mine & is_root]]))
        bps = len(np.unique(bp_lab[ys[mine & branch_px], xs[mine & branch_px]]))
        t_idx = np.flatnonzero(mine & ends)
        for t in t_idx:
            tips.append((s, float(xs[t]), float(ys[t]), float(dist[t])))
        rows.append({"soma": s, "x": cx, "y": cy, "soma_area": float(areas[s]),
                     "total_length": float(length[s]),
                     "longest": float(own_d.max()) if len(own_d) else 0.0,
                     "primary_neurites": prim, "branch_points": bps, "tips": len(t_idx)})
    df = pd.DataFrame(rows, columns=["soma", "x", "y", "soma_area", "total_length", "longest",
                                     "primary_neurites", "branch_points", "tips"])
    return df, {"tips": tips, "unattached": unattached}


def sholl(fr: dict, soma: int, center: tuple[float, float], start: float, step: float) -> list:
    """Neurite crossings of rings around the cell body (from the skeleton's edges)."""
    ys, xs, owner = fr["ys"], fr["xs"], fr["owner"]
    u, v, _ = fr["edges"]
    keep = (owner[u] == soma) & (owner[v] == soma)
    if not keep.any():
        return []
    cx, cy = center
    r = np.hypot(xs - cx, ys - cy)
    ru, rv = r[u[keep]], r[v[keep]]
    lo, hi = np.minimum(ru, rv), np.maximum(ru, rv)
    out = []
    rad = start + step
    while rad <= hi.max():
        out.append((rad, int(((lo < rad) & (hi >= rad)).sum())))
        rad += step
    return out


# ---------------------------------------------------------------- over time
def _link(prev: dict[int, np.ndarray], cur: np.ndarray, gate: float, groups_prev=None,
          groups_cur=None) -> list[int | None]:
    """Match current points to previous ones (Hungarian, within gate). Returns the matched
    previous id for each current point, or None. Optional groups must agree (same neuron)."""
    ids = list(prev)
    out: list[int | None] = [None] * len(cur)
    if not ids or not len(cur):
        return out
    P = np.array([prev[i] for i in ids])
    cost = np.hypot(P[:, None, 0] - cur[None, :, 0], P[:, None, 1] - cur[None, :, 1])
    if groups_prev is not None:
        gp = np.array([groups_prev[i] for i in ids])
        cost[gp[:, None] != np.asarray(groups_cur)[None, :]] = 1e9
    cost[cost > gate] = 1e9
    r, c = linear_sum_assignment(cost)
    for a, b in zip(r, c):
        if cost[a, b] < 1e9:
            out[b] = ids[a]
    return out


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2 or np.ptp(t) == 0:
        return np.nan
    return float(np.polyfit(t, y, 1)[0])


def run(src, out_dir: Path, s, fps: float, first: int, last: int, scale: float,
        pixel_size: float | None, progress=None) -> tuple[dict, dict]:
    from .cells import units
    from .pipeline import _progress

    k, L = units(pixel_size / scale if pixel_size else None)
    step = max(1, int(s.frame_step))
    width = max(1.0, s.neurite_width_px * scale)
    soma_d = s.cell_diameter_px * scale if s.cell_diameter_px else None
    look = s.cell_appearance
    to_h = 1 / fps / 3600

    soma_pos: dict[int, np.ndarray] = {}
    soma_seen: dict[int, int] = {}
    next_soma = 1
    tip_pos: dict[int, np.ndarray] = {}
    tip_soma: dict[int, int] = {}
    tip_seen: dict[int, int] = {}
    next_tip = 1
    per_frame, cones, frame_rows, keep_draw = [], [], [], []
    last_fr = last_ids = None
    total = len(range(first, last, step))
    done = 0
    for f, (_, gray) in enumerate(src.frames(first, last), start=first):
        if (f - first) % step:
            continue
        g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else gray
        if soma_d is None:
            soma_d = estimate_soma_diameter(g, look, width)
        fr = analyze_frame(g, look, soma_d, width, s.threshold_adjust)
        df, extra = measure(fr)

        # cell bodies keep their IDs between frames
        pts = df[["x", "y"]].to_numpy()
        live = {i: p for i, p in soma_pos.items() if f - soma_seen[i] <= 3 * step}
        match = _link(live, pts, max(soma_d, 10))
        ids = []
        for m_ in match:
            if m_ is None:
                m_ = next_soma
                next_soma += 1
            ids.append(m_)
        for i, p in zip(ids, pts):
            soma_pos[i], soma_seen[i] = p, f
        idmap = dict(zip(df["soma"], ids))
        df["neuron_id"] = ids
        df["frame"] = f
        per_frame.append(df)

        # growth cones
        tips = extra["tips"]
        tp = np.array([[x, y] for _, x, y, _ in tips]) if tips else np.zeros((0, 2))
        tg = [idmap[s_] for s_, *_ in tips]
        live_t = {i: p for i, p in tip_pos.items() if f - tip_seen[i] <= 2 * step}
        gate = max(4 * width, soma_d)
        tmatch = _link(live_t, tp, gate, tip_soma, tg)
        for (s_, x, y, dd), m_, nid in zip(tips, tmatch, tg):
            if m_ is None:
                m_ = next_tip
                next_tip += 1
            tip_pos[m_], tip_soma[m_], tip_seen[m_] = np.array([x, y]), nid, f
            cones.append({"cone_id": m_, "neuron_id": nid, "frame": f, "x": x, "y": y, "dist": dd})

        frame_rows.append({"frame": f, "neurons": len(df), "total_length": df["total_length"].sum(),
                           "tips": int(df["tips"].sum()), "branch_points": int(df["branch_points"].sum()),
                           "unattached": extra["unattached"]})
        keep_draw.append({"frame": f, "ys": fr["ys"], "xs": fr["xs"],
                          "owner": np.array([idmap.get(o, 0) for o in fr["owner"]]) if len(fr["owner"]) else fr["owner"],
                          "somas": [(idmap[i], c) for i, c in _soma_contours(fr["somas"]).items()],
                          "tips": [(x, y, idmap[s_]) for s_, x, y, _ in tips]})
        last_fr, last_ids = fr, idmap
        done += 1
        _progress(done, total, "Tracing neurites", progress)
    print(file=sys.stderr)

    info = {"length_unit": "µm" if L == "um" else "pixels", "calibrated": bool(pixel_size),
            "frames": done, "soma_diameter_px": round(soma_d / scale, 2) if soma_d else None,
            "neurite_width_px": s.neurite_width_px}
    nf = pd.concat(per_frame, ignore_index=True) if per_frame else pd.DataFrame()
    if nf.empty:
        info["neurons_found"] = 0
        for name in ("neurons", "neurite_frames", "growth_cones", "sholl", "frames"):
            pd.DataFrame().to_csv(out_dir / f"{name}.csv", index=False)
        return info, {}

    # --- per neuron per frame, in real units
    nf["time_h"] = (nf["frame"] * to_h).round(4)
    out_nf = pd.DataFrame({
        "neuron_id": nf["neuron_id"], "frame": nf["frame"], "time_h": nf["time_h"],
        f"x_{L}": (nf["x"] / scale * k).round(2), f"y_{L}": (nf["y"] / scale * k).round(2),
        f"soma_area_{L}2": (nf["soma_area"] / scale ** 2 * k * k).round(2),
        f"total_length_{L}": (nf["total_length"] * k).round(2),
        f"longest_neurite_{L}": (nf["longest"] * k).round(2),
        "primary_neurites": nf["primary_neurites"], "branch_points": nf["branch_points"],
        "tips": nf["tips"],
    }).sort_values(["neuron_id", "frame"])

    # --- growth cones
    cn = pd.DataFrame(cones)
    cone_rows = []
    last_frame = int(nf["frame"].max())
    for cid, g in (cn.groupby("cone_id") if not cn.empty else []):
        if len(g) < 3:
            continue
        g = g.sort_values("frame")
        t = g["frame"].to_numpy() * to_h
        dd = g["dist"].to_numpy() * k
        steps = np.diff(dd)
        rate = _slope(t, dd)
        cone_rows.append({
            "cone_id": int(cid), "neuron_id": int(g["neuron_id"].iloc[0]),
            "first_frame": int(g["frame"].iloc[0]), "last_frame": int(g["frame"].iloc[-1]),
            "frames": len(g), "duration_h": round(float(t[-1] - t[0]), 3),
            f"start_distance_{L}": round(float(dd[0]), 2), f"end_distance_{L}": round(float(dd[-1]), 2),
            f"net_extension_{L}": round(float(dd[-1] - dd[0]), 2),
            f"extension_rate_{L}_per_h": round(rate, 3) if rate == rate else np.nan,
            "fraction_extending": round(float((steps > 0.5 * k).mean()), 3),
            "fraction_retracting": round(float((steps < -0.5 * k).mean()), 3),
            "behavior": ("extending" if rate > 0 and abs(rate) * (t[-1] - t[0]) > 2 * width * k else
                         "retracting" if rate < 0 and abs(rate) * (t[-1] - t[0]) > 2 * width * k else
                         "paused"),
            "fate": "present at end" if g["frame"].iloc[-1] == last_frame else "lost",
        })
    cone_cols = ["cone_id", "neuron_id", "first_frame", "last_frame", "frames", "duration_h",
                 f"start_distance_{L}", f"end_distance_{L}", f"net_extension_{L}",
                 f"extension_rate_{L}_per_h", "fraction_extending", "fraction_retracting",
                 "behavior", "fate"]
    cone_df = pd.DataFrame(cone_rows, columns=cone_cols)
    cone_track = cn.assign(time_h=(cn["frame"] * to_h).round(4), **{
        f"x_{L}": (cn["x"] / scale * k).round(2), f"y_{L}": (cn["y"] / scale * k).round(2),
        f"distance_from_soma_{L}": (cn["dist"] * k).round(2)}).drop(columns=["x", "y", "dist"]) \
        if not cn.empty else pd.DataFrame()

    # --- Sholl analysis on the last analyzed frame
    sh_rows = []
    sh_step = (s.sholl_step_um / k) if L == "um" else s.sholl_step_um
    lf = nf[nf["frame"] == last_frame]
    for r in lf.itertuples():
        rs = np.sqrt(r.soma_area / np.pi)
        for rad, c in sholl(last_fr, int(r.soma), (r.x, r.y), rs, sh_step):
            sh_rows.append({"neuron_id": int(r.neuron_id), f"radius_{L}": round(rad * k, 2),
                            f"distance_from_soma_edge_{L}": round((rad - rs) * k, 2), "crossings": c})
    sh_df = pd.DataFrame(sh_rows, columns=["neuron_id", f"radius_{L}", f"distance_from_soma_edge_{L}",
                                           "crossings"])

    # --- per neuron summary
    rows = []
    for nid, g in out_nf.groupby("neuron_id"):
        g = g.sort_values("frame")
        end = g.iloc[-1]
        th = g["time_h"].to_numpy()
        sh = sh_df[sh_df["neuron_id"] == nid]
        cones_n = cone_df[cone_df["neuron_id"] == nid]
        rows.append({
            "neuron_id": int(nid), f"x_{L}": end[f"x_{L}"], f"y_{L}": end[f"y_{L}"],
            "first_seen_h": float(th[0]), "last_seen_h": float(th[-1]), "frames_seen": len(g),
            f"soma_area_{L}2": round(float(g[f"soma_area_{L}2"].mean()), 2),
            f"total_length_{L}": end[f"total_length_{L}"],
            f"longest_neurite_{L}": end[f"longest_neurite_{L}"],
            "primary_neurites": int(end["primary_neurites"]),
            "branch_points": int(end["branch_points"]), "tips": int(end["tips"]),
            f"growth_rate_{L}_per_h": round(_slope(th, g[f"total_length_{L}"].to_numpy()), 3),
            f"longest_growth_rate_{L}_per_h": round(_slope(th, g[f"longest_neurite_{L}"].to_numpy()), 3),
            f"mean_cone_extension_{L}_per_h": (round(float(cones_n[f"extension_rate_{L}_per_h"].mean()), 3)
                                              if len(cones_n) else np.nan),
            "sholl_max_crossings": int(sh["crossings"].max()) if len(sh) else 0,
            f"sholl_peak_radius_{L}": (float(sh.loc[sh["crossings"].idxmax(), f"radius_{L}"])
                                      if len(sh) else np.nan),
        })
    neurons = pd.DataFrame(rows)

    fr_df = pd.DataFrame(frame_rows)
    fr_df.insert(1, "time_h", (fr_df["frame"] * to_h).round(4))
    fr_df[f"total_length_{L}"] = (fr_df.pop("total_length") * k).round(2)
    fr_df[f"mean_length_per_neuron_{L}"] = (fr_df[f"total_length_{L}"] / fr_df["neurons"].clip(lower=1)).round(2)
    fr_df[f"unattached_length_{L}"] = (fr_df.pop("unattached") * k).round(2)
    fr_df["time_s"] = fr_df["frame"] / fps

    neurons.to_csv(out_dir / "neurons.csv", index=False)
    out_nf.to_csv(out_dir / "neurite_frames.csv", index=False)
    cone_df.to_csv(out_dir / "growth_cones.csv", index=False)
    cone_track.to_csv(out_dir / "growth_cone_tracks.csv", index=False)
    sh_df.to_csv(out_dir / "sholl.csv", index=False)
    fr_df.to_csv(out_dir / "frames.csv", index=False)

    keep = neurons[neurons["frames_seen"] >= max(1, min(3, done // 2))]
    final = out_nf[out_nf["frame"] == last_frame]
    first_f = out_nf[out_nf["frame"] == out_nf["frame"].min()]
    rate_col = f"growth_rate_{L}_per_h"
    info.update({
        "neurons_found": int(len(final)),
        "neurons_tracked": int(len(neurons)),
        f"mean_length_per_neuron_{L}_start": round(float(first_f[f"total_length_{L}"].mean()), 2),
        f"mean_length_per_neuron_{L}_end": round(float(final[f"total_length_{L}"].mean()), 2),
        f"mean_longest_neurite_{L}": round(float(final[f"longest_neurite_{L}"].mean()), 2),
        "mean_primary_neurites": round(float(final["primary_neurites"].mean()), 2),
        "mean_branch_points": round(float(final["branch_points"].mean()), 2),
        "mean_tips": round(float(final["tips"].mean()), 2),
        f"mean_growth_rate_{L}_per_h": (round(float(keep[rate_col].mean()), 3)
                                        if len(keep) and keep[rate_col].notna().any() else None),
        "growth_cones_followed": int(len(cone_df)),
        "growth_cones_extending": int((cone_df["behavior"] == "extending").sum()),
        "growth_cones_retracting": int((cone_df["behavior"] == "retracting").sum()),
        f"mean_cone_extension_{L}_per_h": (round(float(cone_df[f"extension_rate_{L}_per_h"].mean()), 3)
                                           if len(cone_df) else None),
        f"unattached_length_{L}_end": float(fr_df[f"unattached_length_{L}"].iloc[-1]),
    })
    _write_overlay(out_dir / "neurites.png", src, last_frame, keep_draw[-1], scale)
    return info, {"frames": keep_draw, "scale": scale, "step": step}


# ---------------------------------------------------------------- pictures
def _soma_contours(lab: np.ndarray) -> dict[int, list]:
    out = {}
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = np.zeros(lab.shape, np.uint8)
        m[sl] = lab[sl] == i
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out[i] = cs
    return out


def _draw(img: np.ndarray, d: dict, scale: float):
    H, W = img.shape[:2]
    thick = max(1, W // 600)
    fs = max(0.4, W / 1500)
    inv = 1.0 / scale
    if len(d["ys"]):
        ys = np.clip((d["ys"] * inv).astype(int), 0, H - 1)
        xs = np.clip((d["xs"] * inv).astype(int), 0, W - 1)
        layer = np.zeros_like(img)
        for o in np.unique(d["owner"]):
            sel = d["owner"] == o
            layer[ys[sel], xs[sel]] = GRAY if o == 0 else _color(int(o))
        grown = cv2.dilate(layer, np.ones((2 * thick + 1,) * 2, np.uint8))
        mask = grown.any(axis=2)
        img[mask] = grown[mask]
    for nid, cs in d["somas"]:
        cs = [(c * inv).astype(np.int32) for c in cs]
        cv2.drawContours(img, cs, -1, _color(nid), thick + 1)
        x, y, w, h = cv2.boundingRect(cs[0])
        cv2.putText(img, str(nid), (x + w + 2, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3 * thick)
        cv2.putText(img, str(nid), (x + w + 2, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick)
    for x, y, _ in d["tips"]:
        cv2.circle(img, (int(x * inv), int(y * inv)), 3 * thick + 1, (255, 255, 255), thick)


def _write_overlay(path: Path, src, frame: int, d: dict, scale: float):
    for img, _ in src.frames(frame, frame + 1):
        img = img.copy()
        _draw(img, d, scale)
        cv2.imwrite(str(path), img)


def render(src, out: Path, draw: dict, fps: float, first: int, last: int, writer, progress=None):
    from .pipeline import _progress

    frames = {d["frame"]: d for d in draw["frames"]}
    W, H = src.width, src.height
    wr = writer(out, fps, (W, H))
    fs = max(0.45, W / 1300)
    thick = max(1, W // 600)
    print(f"Writing {out.name}", file=sys.stderr)
    n = 0
    for f, (img, _) in enumerate(src.frames(first, last), start=first):
        d = frames.get(f)
        if d is None:
            continue
        img = img.copy()
        _draw(img, d, draw["scale"])
        txt = f"frame {f}   neurons {len(d['somas'])}   growth cones {len(d['tips'])}"
        cv2.putText(img, txt, (10, int(26 * fs / 0.55)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3 * thick)
        cv2.putText(img, txt, (10, int(26 * fs / 0.55)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick)
        wr.write(img)
        n += 1
        _progress(n, len(frames), "Drawing video", progress)
    print(file=sys.stderr)
    wr.release()
