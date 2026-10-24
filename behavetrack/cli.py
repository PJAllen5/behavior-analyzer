"""Command line:

  python -m behavetrack analyze VIDEO [options]
  python -m behavetrack filter VIDEO_OR_FOLDER... --behavior collective_motion [--clips]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analyze import COLLECTIVE, DISORDERED, MILLING, RESTING
from .pipeline import Settings, cell_settings, extract_clip, run
from .sources import ALL_EXT as VIDEO_EXT
BEHAVIORS = [COLLECTIVE, MILLING, DISORDERED, RESTING, "interaction"]


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--mode", choices=["animals", "cells"], default="animals",
                   help="animals: animals/people in video (default). cells: microscope time-lapse")
    g = p.add_argument_group("cells mode")
    g.add_argument("--pixel-size", type=float, default=0.0,
                   help="microns per pixel (default: read from the file, else results in pixels)")
    g.add_argument("--frame-interval", type=float, default=0.0,
                   help="seconds between frames (default: read from the file)")
    g.add_argument("--channel", type=int, default=0, help="channel to analyze (0 = first)")
    g.add_argument("--cells-look", choices=["auto", "bright", "dark"], default="auto",
                   help="bright: fluorescent cells/nuclei; dark: darker than background; "
                        "auto: anything different from the background (e.g. phase contrast)")
    g.add_argument("--cell-diameter", type=float, default=0.0,
                   help="typical cell diameter in pixels (default: automatic)")
    g.add_argument("--threshold-adjust", type=float, default=0.0,
                   help="-0.5..0.5: negative finds fainter cells, positive is stricter")
    g.add_argument("--no-split", action="store_true", help="don't separate touching cells")
    g.add_argument("--no-divisions", action="store_true", help="don't look for cell division")
    g = p.add_argument_group("detection (animals mode)")
    g.add_argument("--detector", choices=["motion", "yolo"], default="motion",
                   help="motion: anything moving on a still camera (default). "
                        "yolo: AI model for people/birds/animals")
    g.add_argument("--classes", default="",
                   help="yolo only, comma list, e.g. 'person' or 'bird,dog'")
    g.add_argument("--yolo-model", default="yolo11n.pt")
    g.add_argument("--conf", type=float, default=0.3, help="yolo confidence cut-off")
    g.add_argument("--min-area", type=int, default=15, help="motion: smallest blob in pixels")
    g.add_argument("--max-area", type=int, default=50_000, help="motion: largest blob in pixels")
    g.add_argument("--sensitivity", type=float, default=25.0,
                   help="motion: lower picks up fainter movement (default 25)")
    g.add_argument("--resize-width", type=int, default=960,
                   help="process at this width for speed (0 = full size)")
    g = p.add_argument_group("time range")
    g.add_argument("--start", type=float, default=0.0, help="start at this second")
    g.add_argument("--max-seconds", type=float, default=0.0, help="only analyze this long")
    g = p.add_argument_group("behavior")
    g.add_argument("--contact-bl", type=float, default=2.0,
                   help="interaction distance in body lengths (default 2)")
    g.add_argument("--moving-bl-s", type=float, default=0.5,
                   help="speed (body lengths/s) below which an object counts as still")
    g.add_argument("--min-behavior-seconds", type=float, default=1.0)
    p.add_argument("--no-video", action="store_true", help="skip the annotated video")


def _settings(a) -> Settings:
    if a.mode == "cells":
        return cell_settings(
            pixel_size_um=a.pixel_size, frame_interval_s=a.frame_interval, channel=a.channel,
            cell_appearance=a.cells_look, cell_diameter_px=a.cell_diameter,
            threshold_adjust=a.threshold_adjust, split_touching=not a.no_split,
            detect_divisions=not a.no_divisions, start_s=a.start, max_seconds=a.max_seconds,
            annotate=not a.no_video)
    return Settings(
        detector=a.detector, classes=[c for c in a.classes.split(",") if c],
        yolo_model=a.yolo_model, conf=a.conf, min_area=a.min_area, max_area=a.max_area,
        sensitivity=a.sensitivity, resize_width=a.resize_width, start_s=a.start,
        max_seconds=a.max_seconds, contact_bl=a.contact_bl, moving_bl_s=a.moving_bl_s,
        min_behavior_seconds=a.min_behavior_seconds, annotate=not a.no_video)


def _videos(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            vids = sorted(q for q in p.rglob("*") if q.suffix.lower() in VIDEO_EXT)
            out += vids if vids else [p]  # a folder of images is one time-lapse
        else:
            out.append(p)
    return out


def cmd_analyze(a):
    out = Path(a.out) if a.out else Path(a.video).with_name(Path(a.video).stem + "_behavior")
    summary = run(a.video, out, _settings(a))
    summary.pop("seconds_in_each_behavior", None) if a.mode == "cells" else None
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=2))
    print(f"\nData files written to: {out}/")


def cmd_filter(a):
    s = _settings(a)
    s.annotate = a.annotated
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    matches = []
    for v in _videos(a.inputs):
        res = root / v.stem
        if (res / "summary.json").exists() and not a.rerun:
            print(f"Using earlier results for {v.name}")
        else:
            run(v, res, s)
        if a.behavior == "interaction":
            df = pd.read_csv(res / "interactions.csv") if (res / "interactions.csv").stat().st_size > 1 else pd.DataFrame()
            df = df[df["duration_s"] >= a.min_seconds] if not df.empty else df
        else:
            df = pd.read_csv(res / "behaviors.csv") if (res / "behaviors.csv").stat().st_size > 1 else pd.DataFrame()
            df = df[(df["behavior"] == a.behavior) & (df["duration_s"] >= a.min_seconds)] if not df.empty else df
        for i, r in enumerate(df.to_dict("records")):
            r = {"video": str(v), **r}
            if a.clips:
                clip = root / "clips" / f"{v.stem}_{a.behavior}_{i + 1:03d}.mp4"
                clip.parent.mkdir(exist_ok=True)
                extract_clip(v, clip, int(r["start_frame"]), int(r["end_frame"]), a.pad)
                r["clip"] = str(clip)
            matches.append(r)
    m = pd.DataFrame(matches)
    m.to_csv(root / f"matches_{a.behavior}.csv", index=False)
    vids = m["video"].nunique() if len(m) else 0
    print(f"\nFound {len(m)} '{a.behavior}' segments in {vids} video(s).")
    print(f"List: {root / f'matches_{a.behavior}.csv'}")
    if a.clips and len(m):
        print(f"Clips: {root / 'clips'}/")


def main(argv=None):
    p = argparse.ArgumentParser(prog="behavetrack",
                                description="Track animals/people in video and export behavior data.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze", help="analyze one video and write data files")
    pa.add_argument("video")
    pa.add_argument("-o", "--out", help="output folder (default: <video>_behavior)")
    _add_common(pa)
    pa.set_defaults(func=cmd_analyze)

    pf = sub.add_parser("filter", help="search videos for a behavior and list/cut matching clips")
    pf.add_argument("inputs", nargs="+", help="video files or folders")
    pf.add_argument("--behavior", required=True, choices=BEHAVIORS)
    pf.add_argument("--min-seconds", type=float, default=1.0, help="shortest segment to keep")
    pf.add_argument("--clips", action="store_true", help="cut each match into its own video")
    pf.add_argument("--pad", type=float, default=0.5, help="seconds of padding around clips")
    pf.add_argument("--annotated", action="store_true", help="also write annotated videos")
    pf.add_argument("--rerun", action="store_true", help="ignore earlier results")
    pf.add_argument("-o", "--out", default="behavior_results")
    _add_common(pf)
    pf.set_defaults(func=cmd_filter)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
