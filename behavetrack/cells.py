"""Cell-migration measurements, in real units (microns, minutes) when calibration is known.

cells.csv      one row per cell track: speed, path, straightness, direction, MSD exponent, fate
msd.csv        mean squared displacement vs time lag (how fast cells spread out)
lineage.csv    parent -> daughter links from cell divisions
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def units(pixel_size_um: float | None) -> tuple[float, str]:
    return (pixel_size_um, "um") if pixel_size_um else (1.0, "px")


def calibrate(tracks: pd.DataFrame, pixel_size_um: float | None) -> pd.DataFrame:
    k, L = units(pixel_size_um)
    t = tracks.copy()
    t["time_min"] = t["time_s"] / 60
    t[f"x_{L}"] = t["x_smooth"] * k
    t[f"y_{L}"] = t["y_smooth"] * k
    t[f"speed_{L}_per_min"] = t["speed_px_s"] * k * 60
    if "area_px" in t:
        t[f"area_{L}2"] = t["area_px"] * k * k
    if "diameter_px" in t:
        t[f"diameter_{L}"] = t["diameter_px"] * k
    return t


def _msd(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    return np.array([np.mean((x[l:] - x[:-l]) ** 2 + (y[l:] - y[:-l]) ** 2) for l in range(1, max_lag + 1)])


def _alpha(msd: np.ndarray, lags: np.ndarray) -> float:
    ok = msd > 0
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(lags[ok]), np.log(msd[ok]), 1)[0])


def cell_table(tracks: pd.DataFrame, lineage: dict, fps: float, pixel_size_um: float | None,
               width: int, height: int, last_frame: int) -> pd.DataFrame:
    k, L = units(pixel_size_um)
    dt_min = 1 / fps / 60
    ids = set(tracks["id"].unique())

    def generation(i):
        g, seen = 0, set()
        while i in lineage and lineage[i].get("parent") is not None and i not in seen:
            seen.add(i)
            i = lineage[i]["parent"]
            g += 1
        return g

    rows = []
    for tid, g in tracks.groupby("id"):
        g = g.sort_values("frame")
        x, y = g["x_smooth"].to_numpy() * k, g["y_smooth"].to_numpy() * k
        steps = np.hypot(np.diff(x), np.diff(y))
        path = float(steps.sum())
        dx, dy = x[-1] - x[0], y[-1] - y[0]
        net = float(np.hypot(dx, dy))
        dur = (g["frame"].iloc[-1] - g["frame"].iloc[0]) * dt_min
        n = len(g)
        alpha = np.nan
        if n >= 10:
            lags = np.arange(1, n // 3 + 1)
            alpha = _alpha(_msd(x, y, len(lags)), lags.astype(float))
        info = lineage.get(tid, {})
        parent = info.get("parent")
        divided = "divided_at" in info and any(lineage.get(c, {}).get("parent") == tid for c in ids)
        last = g.iloc[-1]
        margin = 1.0 * max(last["w"], last["h"])
        if divided:
            fate = "divided"
        elif g["frame"].iloc[-1] >= last_frame - 1:
            fate = "present at end"
        elif (last["x"] < margin or last["y"] < margin or last["x"] > width - margin
              or last["y"] > height - margin):
            fate = "left view"
        else:
            fate = "lost"
        rec = {
            "id": tid,
            "parent_id": parent if parent in ids else None,
            "generation": generation(tid),
            "start_min": round(g["time_s"].iloc[0] / 60, 2),
            "end_min": round(g["time_s"].iloc[-1] / 60, 2),
            "duration_min": round(dur, 2),
            "frames": n,
            f"path_length_{L}": round(path, 2),
            f"net_displacement_{L}": round(net, 2),
            "straightness": round(net / path, 3) if path > 0 else np.nan,
            f"mean_speed_{L}_per_min": round(path / dur, 4) if dur > 0 else np.nan,
            f"max_speed_{L}_per_min": round(float(steps.max() / dt_min), 4) if len(steps) else np.nan,
            "direction_deg": round(float(np.degrees(np.arctan2(dy, dx))), 1),
            "fmi_x": round(dx / path, 3) if path > 0 else np.nan,
            "fmi_y": round(dy / path, 3) if path > 0 else np.nan,
            "msd_alpha": round(alpha, 3) if alpha == alpha else np.nan,
            "fate": fate,
        }
        if "area_px" in g:
            rec[f"mean_area_{L}2"] = round(float(g["area_px"].mean() * k * k), 2)
        if "intensity" in g:
            rec["mean_intensity"] = round(float(g["intensity"].mean()), 4)
        rows.append(rec)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["parent_id"] = df["parent_id"].astype("Int64")
    return df


def msd_table(tracks: pd.DataFrame, fps: float, pixel_size_um: float | None,
              max_lag_frac: float = 0.25) -> pd.DataFrame:
    k, L = units(pixel_size_um)
    span = int(tracks["frame"].max() - tracks["frame"].min()) if len(tracks) else 0
    max_lag = max(1, int(span * max_lag_frac))
    sums, counts = np.zeros(max_lag), np.zeros(max_lag)
    for _, g in tracks.groupby("id"):
        g = g.set_index("frame").sort_index()
        full = g.reindex(range(g.index.min(), g.index.max() + 1))  # keep gaps as NaN
        x, y = full["x_smooth"].to_numpy() * k, full["y_smooth"].to_numpy() * k
        for l in range(1, min(max_lag, len(x) - 1) + 1):
            d = (x[l:] - x[:-l]) ** 2 + (y[l:] - y[:-l]) ** 2
            d = d[~np.isnan(d)]
            sums[l - 1] += d.sum()
            counts[l - 1] += len(d)
    ok = counts > 0
    lag = np.arange(1, max_lag + 1)
    return pd.DataFrame({
        "lag_frames": lag[ok],
        "lag_min": np.round(lag[ok] / fps / 60, 3),
        f"msd_{L}2": np.round(sums[ok] / counts[ok], 4),
        "n_measurements": counts[ok].astype(int),
    })


def lineage_table(lineage: dict, tracks: pd.DataFrame, fps: float) -> pd.DataFrame:
    ids = set(tracks["id"].unique())
    rows = []
    for child, info in lineage.items():
        p = info.get("parent")
        if child in ids and p in ids and "born_at" in info:
            rows.append({"parent_id": p, "daughter_id": child,
                         "division_frame": info["born_at"],
                         "division_min": round(info["born_at"] / fps / 60, 2)})
    return pd.DataFrame(rows, columns=["parent_id", "daughter_id", "division_frame", "division_min"])


def population_summary(cells: pd.DataFrame, msd: pd.DataFrame, lineage: pd.DataFrame,
                       frames: pd.DataFrame, pixel_size_um: float | None) -> dict:
    _, L = units(pixel_size_um)
    out: dict = {"length_unit": "µm" if L == "um" else "pixels", "calibrated": bool(pixel_size_um)}
    long = cells[cells["frames"] >= 5] if not cells.empty else cells
    if not long.empty:
        out["cells_tracked"] = int(len(cells))
        out[f"median_speed_{L}_per_min"] = round(float(long[f"mean_speed_{L}_per_min"].median()), 4)
        out["median_straightness"] = round(float(long["straightness"].median()), 3)
        out["mean_fmi_x"] = round(float(long["fmi_x"].mean()), 3)
        out["mean_fmi_y"] = round(float(long["fmi_y"].mean()), 3)
    out["divisions"] = int(lineage["parent_id"].nunique()) if not lineage.empty else 0
    if len(msd) >= 3:
        col = f"msd_{L}2"
        out["msd_alpha"] = round(_alpha(msd[col].to_numpy(), msd["lag_min"].to_numpy()), 3)
        out["diffusion_coefficient"] = round(float(msd[col].iloc[0] / (4 * msd["lag_min"].iloc[0])), 4)
        out["diffusion_coefficient_unit"] = f"{out['length_unit']}²/min"
    if not frames.empty and len(frames) >= 5:
        n = frames["n_visible"].to_numpy(float)
        t = frames["time_s"].to_numpy() / 3600
        out["cells_at_start"], out["cells_at_end"] = int(n[0]), int(n[-1])
        ok = n > 0
        if ok.sum() >= 5 and np.ptp(t[ok]) > 0:
            slope = np.polyfit(t[ok], np.log(n[ok]), 1)[0]
            if slope > 1e-3:
                out["population_doubling_time_h"] = round(float(np.log(2) / slope), 2)
    return out
