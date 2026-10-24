"""Run detection -> tracking -> analysis on one video or microscope file and write the data files."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from . import analyze as A
from . import cells as C
from .detect import CellDetector, MotionDetector, YoloDetector
from .sources import open_source
from .track import Tracker


@dataclass
class Settings:
    mode: str = "animals"               # "animals" (animals / people), "cells" or "neurons" (microscope)
    detector: str = "motion"            # animals mode: "motion" or "yolo"
    classes: list[str] = field(default_factory=list)  # yolo only, e.g. ["person", "bird"]
    yolo_model: str = "yolo11n.pt"
    conf: float = 0.3
    min_area: int = 15                  # motion only: smallest blob (pixels, after resize)
    max_area: int = 50_000
    sensitivity: float = 25.0           # motion only: lower = picks up fainter motion
    resize_width: int = 960             # process at this width (0 = original size)
    start_s: float = 0.0
    max_seconds: float = 0.0            # 0 = whole video
    max_missed: int = 10
    min_track_frames: int = 5
    contact_bl: float = 2.0             # "interaction" distance, in body lengths
    moving_bl_s: float = 0.5            # slower than this counts as not moving
    min_behavior_seconds: float = 1.0
    annotate: bool = True
    # --- cells mode
    pixel_size_um: float = 0.0          # microns per pixel (0 = read from file, else pixels)
    frame_interval_s: float = 0.0       # seconds between frames (0 = read from file / video)
    channel: int = 0                    # which colour channel of a multi-channel file
    cell_appearance: str = "auto"       # "auto", "bright" (fluorescent) or "dark"
    cell_diameter_px: float = 0.0       # 0 = estimate automatically
    threshold_adjust: float = 0.0       # -0.5..0.5, negative finds fainter cells
    split_touching: bool = True
    detect_divisions: bool = True
    moving_bl_per_frame: float = 0.05   # cells: slower than this per frame counts as still
    # --- neurons mode (also uses the calibration, channel, cell_appearance,
    #     cell_diameter_px and threshold_adjust settings above)
    neural_analysis: str = "activity"   # "activity" (calcium imaging) or "neurites" (growth)
    find_neurons_by: str = "activity"   # activity: "activity" (flicker) or "brightness" (also silent ones)
    correct_drift: bool = True          # activity: line every frame up with the first
    neuropil_factor: float = 0.7        # activity: share of the surrounding haze to subtract
    baseline_window_s: float = 60.0     # activity: resting level follows changes slower than this
    event_threshold_sd: float = 3.5     # activity: firing = rise this many noise levels above rest
    min_event_dff: float = 0.05         # activity: smallest event (ΔF/F); 0 for small voltage signals
    burst_window_s: float = 0.5         # activity: neurons starting this close together are in step
    burst_min_fraction: float = 0.3     # activity: share of active neurons needed for a burst
    neurite_width_px: float = 3.0       # neurites: typical neurite thickness
    frame_step: int = 1                 # neurites: analyze every Nth frame
    sholl_step_um: float = 10.0         # neurites: ring spacing for Sholl analysis (pixels if uncalibrated)


def cell_settings(**kw) -> Settings:
    """Defaults that suit microscope time-lapses."""
    base = dict(mode="cells", resize_width=0, max_missed=2, min_track_frames=3, contact_bl=1.5)
    base.update(kw)
    return Settings(**base)


def neuron_settings(**kw) -> Settings:
    """Defaults that suit neuron imaging (calcium activity or neurite growth)."""
    base = dict(mode="neurons", resize_width=0, cell_appearance="bright")
    base.update(kw)
    return Settings(**base)


ProgressFn = Callable[[str, float], None]  # (stage name, fraction done 0..1)


def _progress(i: int, total: int, stage: str = "", cb: ProgressFn | None = None):
    if total > 0 and (i % 10 == 0 or i == total):
        if cb:
            cb(stage, i / total)
        else:
            sys.stderr.write(f"\r  {stage} {i}/{total} ({100 * i / total:.0f}%)")
            sys.stderr.flush()


def _video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """H.264 so the video plays in web browsers; falls back to mp4v."""
    for code in ("avc1", "mp4v"):
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
        if wr.isOpened():
            return wr
    raise RuntimeError(f"Could not write video {path}")


def playback_fps(fps: float) -> float:
    """Time-lapses (few frames per second of real time) play back at 8 frames/s."""
    return fps if fps >= 5 else 8.0


def run(video: str | Path, out_dir: str | Path, s: Settings,
        progress: ProgressFn | None = None) -> dict:
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        src = open_source(video, s.channel)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    cells = s.mode == "cells"

    if s.frame_interval_s:
        fps = 1.0 / s.frame_interval_s
    elif src.frame_interval_s:
        fps = 1.0 / src.frame_interval_s
    else:
        fps = src.fps
    time_known = bool(s.frame_interval_s or src.frame_interval_s or src.is_video)
    pixel_size = s.pixel_size_um or src.pixel_size_um or None

    W, H, total = src.width, src.height, src.n_frames
    first = int(s.start_s * fps)
    last = total if not s.max_seconds else min(total, first + int(s.max_seconds * fps))
    scale = s.resize_width / W if s.resize_width and W > s.resize_width else 1.0
    if s.mode == "neurons":
        return _run_neurons(video, out_dir, s, src, fps, first, last, scale, pixel_size,
                            time_known, progress)

    if cells:
        det = CellDetector(s.cell_appearance, s.cell_diameter_px * scale, s.threshold_adjust,
                           s.split_touching)
        trk = Tracker(max_missed=s.max_missed, min_hits=2, gate_sizes=1.5,
                      divisions=s.detect_divisions)
        moving = s.moving_bl_per_frame * fps
        label_s, min_beh_s, gap_s, min_int_s = 3 / fps, 3 / fps, 1 / fps, 2 / fps
    else:
        if s.detector == "yolo":
            det = YoloDetector(s.yolo_model, s.classes or None, s.conf)
        else:
            det = MotionDetector(s.min_area, s.max_area, s.sensitivity)
        trk = Tracker(max_missed=s.max_missed)
        moving = s.moving_bl_s
        label_s, min_beh_s, gap_s, min_int_s = 0.5, s.min_behavior_seconds, 0.3, 0.2

    print(f"Analyzing {video.name}: {W}x{H}, {total} frames, mode={s.mode}", file=sys.stderr)
    f = first
    for display, gray in src.frames(first, last):
        if cells:
            g = cv2.resize(gray, None, fx=scale, fy=scale) if scale != 1.0 else gray
            raw = det(g)
        else:
            small = cv2.resize(display, None, fx=scale, fy=scale) if scale != 1.0 else display
            raw = det(small)
        dets = []
        for d in raw:
            d = list(d)
            d[0:4] = [v / scale for v in d[0:4]]
            if len(d) > 6:
                d[6] = {**d[6], "area_px": d[6]["area_px"] / scale ** 2,
                        "diameter_px": d[6]["diameter_px"] / scale}
            dets.append(tuple(d))
        trk.update(f, dets)
        f += 1
        _progress(f - first, last - first, "Tracking", progress)
    print(file=sys.stderr)

    # cells that divided are kept even if seen only briefly, so family links survive
    parents = {v["parent"] for v in trk.lineage.values() if v.get("parent") is not None}
    tracks, body = A.build_tracks(trk.rows, fps, s.min_track_frames, keep_ids=parents)
    if tracks.empty:
        hint = ("Try a lower detection threshold, a different 'cells look' setting, or set the "
                "cell diameter." if cells else
                "Try --sensitivity 15 or --min-area 5, or --detector yolo.")
        print("Nothing was found to track. " + hint, file=sys.stderr)
    frames = (A.frame_metrics(tracks, fps, body, moving, label_seconds=label_s)
              if not tracks.empty else pd.DataFrame())
    segs = A.behavior_segments(frames, fps, min_beh_s)
    inter = (A.interactions(tracks, fps, body, s.contact_bl, gap_seconds=gap_s,
                            min_seconds=min_int_s, moving_bl_s=moving)
             if not tracks.empty else pd.DataFrame())

    summary_cells = None
    if cells and not tracks.empty:
        tracks = C.calibrate(tracks, pixel_size)
        cell_df = C.cell_table(tracks, trk.lineage, fps, pixel_size, W, H, f - 1)
        msd = C.msd_table(tracks, fps, pixel_size)
        lin = C.lineage_table(trk.lineage, tracks, fps)
        cell_df.to_csv(out_dir / "cells.csv", index=False)
        msd.to_csv(out_dir / "msd.csv", index=False)
        lin.to_csv(out_dir / "lineage.csv", index=False)
        summary_cells = C.population_summary(cell_df, msd, lin, frames, pixel_size)
        if not frames.empty:
            frames.insert(2, "time_min", frames["time_s"] / 60)

    tracks.round(4).to_csv(out_dir / "tracks.csv", index=False)
    frames.round(4).to_csv(out_dir / "frames.csv", index=False)
    segs.to_csv(out_dir / "behaviors.csv", index=False)
    inter.to_csv(out_dir / "interactions.csv", index=False)

    dur = (f - first) / fps
    time_in = {}
    if not frames.empty:
        time_in = (frames["behavior"].value_counts() / fps).round(2).to_dict()
    summary = {
        "video": str(video), "mode": s.mode, "fps": fps, "width": W, "height": H,
        "analyzed_seconds": round(dur, 2), "start_s": s.start_s,
        "frames_analyzed": f - first,
        "pixel_size_um": pixel_size, "time_known": time_known,
        "body_length_px": round(body, 2),
        "individuals_tracked": int(tracks["id"].nunique()) if not tracks.empty else 0,
        "max_visible_at_once": int(frames["n_visible"].max()) if not frames.empty else 0,
        "typical_visible": int(frames["n_visible"].median()) if not frames.empty else 0,
        "seconds_in_each_behavior": time_in,
        "behavior_segments": len(segs),
        "interactions": len(inter),
    }
    if summary_cells is not None:
        summary["cells"] = summary_cells
    summary["settings"] = asdict(s)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if s.annotate and not tracks.empty:
        render(src, out_dir / "annotated.mp4", tracks, frames, fps, first, f,
               lineage=trk.lineage if cells else None, progress=progress)
    return summary


def _run_neurons(video: Path, out_dir: Path, s: Settings, src, fps: float, first: int, last: int,
                 scale: float, pixel_size: float | None, time_known: bool,
                 progress: ProgressFn | None) -> dict:
    from . import activity, neurites

    kind = s.neural_analysis
    print(f"Analyzing {video.name}: {src.width}x{src.height}, {src.n_frames} frames, "
          f"mode=neurons ({kind})", file=sys.stderr)
    mod = activity if kind == "activity" else neurites
    info, draw = mod.run(src, out_dir, s, fps, first, last, scale, pixel_size, progress)
    found = info.get("neurons_found", 0)
    if not found:
        print("No neurons were found. Try moving the detection setting toward 'fainter', "
              "or set the cell diameter.", file=sys.stderr)
    n_frames = info.get("frames", last - first)
    summary = {
        "video": str(video), "mode": "neurons", "neural_analysis": kind, "fps": fps,
        "width": src.width, "height": src.height,
        "analyzed_seconds": round((last - first) / fps, 2), "start_s": s.start_s,
        "frames_analyzed": n_frames, "pixel_size_um": pixel_size, "time_known": time_known,
        "individuals_tracked": found, "typical_visible": found,
        "neurons": info, "settings": asdict(s),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if s.annotate and draw:
        mod.render(src, out_dir / "annotated.mp4", draw, playback_fps(fps), first, last,
                   _video_writer, progress)
    return summary


def _color(i: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(i * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def _root(i: int, lineage: dict) -> int:
    seen = set()
    while lineage.get(i, {}).get("parent") is not None and i not in seen:
        seen.add(i)
        i = lineage[i]["parent"]
    return i


def render(src, out: Path, tracks: pd.DataFrame, frames: pd.DataFrame,
           fps: float, first: int, last: int, trail: int = 20,
           lineage: dict | None = None, progress: ProgressFn | None = None):
    cells = lineage is not None
    W, H = src.width, src.height
    wr = _video_writer(out, playback_fps(fps), (W, H))
    by_frame = {k: g for k, g in tracks.groupby("frame")}
    fr_idx = frames.set_index("frame") if not frames.empty else None
    history: dict[int, list] = {}
    thick = max(1, W // 640)
    fs = max(0.35, W / 1400)
    births = {i: v["born_at"] for i, v in (lineage or {}).items() if "born_at" in v}
    n_div = 0
    print(f"Writing {out.name}", file=sys.stderr)
    for f, (img, _) in enumerate(src.frames(first, last), start=first):
        img = img.copy()
        g = by_frame.get(f)
        if g is not None:
            for r in g.itertuples():
                c = _color(_root(r.id, lineage) if cells else r.id)
                h = history.setdefault(r.id, [])
                h.append((int(r.x_smooth), int(r.y_smooth)))
                del h[:-(30 if cells else trail)]
                if len(h) > 1:
                    cv2.polylines(img, [np.array(h)], False, c, thick)
                if cells:
                    rad = int(max(r.w, r.h) / 2) + 1
                    cv2.circle(img, (int(r.x), int(r.y)), rad, c, thick)
                    if r.id in births and f - births[r.id] < 4:  # just divided
                        cv2.circle(img, (int(r.x), int(r.y)), rad + 4, (255, 255, 255), thick)
                    cv2.putText(img, str(r.id), (int(r.x) + rad, int(r.y) - rad),
                                cv2.FONT_HERSHEY_SIMPLEX, fs, c, thick)
                else:
                    x0, y0 = int(r.x - r.w / 2), int(r.y - r.h / 2)
                    cv2.rectangle(img, (x0, y0), (int(x0 + r.w), int(y0 + r.h)), c, thick)
                    cv2.putText(img, str(r.id), (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4 * thick, c, thick)
        n_div += sum(1 for b in births.values() if b == f) // 2 if cells else 0
        if fr_idx is not None and f in fr_idx.index:
            m = fr_idx.loc[f]
            if cells:
                txt = [f"t = {m.time_s / 60:7.1f} min   cells = {int(m.n_visible)}   divisions = {n_div}",
                       f"{m.behavior}   alignment={m.polarization:.2f}"]
            else:
                txt = [f"t={m.time_s:6.2f}s  n={int(m.n_visible)}  behavior: {m.behavior}",
                       f"polarization={m.polarization:.2f}  rotation={m.rotation:.2f}  "
                       f"speed={m.mean_speed_bl_s:.1f} BL/s  {m.cohesion_trend}"]
            for k, t in enumerate(txt):
                y = int(22 * fs / 0.55) + k * int(22 * fs / 0.55)
                cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3 * thick)
                cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick)
        wr.write(img)
        _progress(f - first + 1, last - first, "Drawing video", progress)
    print(file=sys.stderr)
    wr.release()


def extract_clip(video: Path, out: Path, start_frame: int, end_frame: int, pad_s: float = 0.0,
                 channel: int = 0):
    src = open_source(video, channel)
    fps = playback_fps(src.fps)
    pad = int(pad_s * fps)
    start_frame = max(0, start_frame - pad)
    end_frame = min(src.n_frames - 1, end_frame + pad)
    wr = _video_writer(out, fps, (src.width, src.height))
    for img, _ in src.frames(start_frame, end_frame + 1):
        wr.write(img)
    wr.release()
