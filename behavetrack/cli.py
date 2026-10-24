"""Command line:

  behavetrack app                                   open the browser app
  behavetrack analyze VIDEO [options]
  behavetrack filter VIDEO_OR_FOLDER... --behavior collective_motion [--clips]
  behavetrack analyze MOVIE.tif --mode neurons --analysis activity|neurites

(`python -m behavetrack ...` works the same if the command isn't installed.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .analyze import COLLECTIVE, DISORDERED, MILLING, RESTING
from .pipeline import extract_clip, run
from .settings import MODES, NEURAL_ANALYSES, Settings, defaults, with_changes
from .sources import ALL_EXT as VIDEO_EXT
BEHAVIORS = [COLLECTIVE, MILLING, DISORDERED, RESTING, "interaction", "network_burst"]


# Defaults are never written here: they come from settings.py, per mode, so the command
# line and the app always agree. Options left out stay None and keep the mode's default.
_A, _C = defaults("animals"), defaults("cells")
_NA, _NN = defaults("neurons", "activity"), defaults("neurons", "neurites")

# command-line option -> Settings field
_OPTION_FIELDS = {
    "pixel_size": "pixel_size_um", "frame_interval": "frame_interval_s", "channel": "channel",
    "cells_look": "cell_appearance", "cell_diameter": "cell_diameter_px",
    "threshold_adjust": "threshold_adjust", "analysis": "neural_analysis",
    "find_by": "find_neurons_by", "neuropil": "neuropil_factor", "event_sd": "event_threshold_sd",
    "min_event": "min_event_dff", "baseline_window": "baseline_window_s",
    "burst_window": "burst_window_s", "burst_fraction": "burst_min_fraction",
    "neurite_width": "neurite_width_px", "frame_step": "frame_step", "sholl_step": "sholl_step_um",
    "detector": "detector", "yolo_model": "yolo_model", "conf": "conf", "min_area": "min_area",
    "max_area": "max_area", "sensitivity": "sensitivity", "resize_width": "resize_width",
    "start": "start_s", "max_seconds": "max_seconds", "contact_bl": "contact_bl",
    "moving_bl_s": "moving_bl_s", "min_behavior_seconds": "min_behavior_seconds",
}
# on/off switches: option -> (Settings field, value when the switch is given)
_SWITCHES = {"no_split": ("split_touching", False), "no_divisions": ("detect_divisions", False),
             "no_drift": ("correct_drift", False), "no_video": ("annotate", False)}


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--mode", choices=list(MODES), default="animals",
                   help="animals: animals/people in video (default). cells: microscope time-lapse. "
                        "neurons: calcium imaging or neurite growth")
    g = p.add_argument_group("cells and neurons modes")
    g.add_argument("--pixel-size", type=float,
                   help="microns per pixel (default: read from the file, else results in pixels)")
    g.add_argument("--frame-interval", type=float,
                   help="seconds between frames (default: read from the file)")
    g.add_argument("--channel", type=int, help=f"channel to analyze (default {_C.channel} = first)")
    g.add_argument("--cells-look", choices=["auto", "bright", "dark"],
                   help="bright: fluorescent cells/nuclei; dark: darker than background; "
                        "auto: anything different from the background (e.g. phase contrast). "
                        f"Default: {_C.cell_appearance} in cells mode, {_NA.cell_appearance} in neurons mode")
    g.add_argument("--cell-diameter", type=float,
                   help="typical cell diameter in pixels (default: automatic)")
    g.add_argument("--threshold-adjust", type=float,
                   help=f"-0.5..0.5: negative finds fainter cells, positive is stricter (default {_C.threshold_adjust})")
    g.add_argument("--no-split", action="store_true", help="don't separate touching cells")
    g.add_argument("--no-divisions", action="store_true", help="don't look for cell division")
    g = p.add_argument_group("neurons mode (--cells-look, --cell-diameter and --threshold-adjust "
                             "also apply; cell diameter means the cell body)")
    g.add_argument("--analysis", choices=list(NEURAL_ANALYSES),
                   help=f"activity: firing in calcium/voltage imaging (default {_NA.neural_analysis}). "
                        "neurites: axon and dendrite growth")
    g.add_argument("--find-by", choices=["activity", "brightness"],
                   help=f"activity: find neurons that flicker. brightness: find bright cell bodies, "
                        f"which also counts neurons that never fire (default {_NA.find_neurons_by})")
    g.add_argument("--no-drift", action="store_true", help="activity: don't correct stage drift")
    g.add_argument("--neuropil", type=float,
                   help=f"activity: share of the surrounding haze to subtract (0 = none, default {_NA.neuropil_factor})")
    g.add_argument("--event-sd", type=float,
                   help=f"activity: an event must rise this many noise levels above rest (default {_NA.event_threshold_sd})")
    g.add_argument("--min-event", type=float,
                   help=f"activity: smallest event in ΔF/F (default {_NA.min_event_dff}; use 0 for voltage dyes)")
    g.add_argument("--baseline-window", type=float,
                   help=f"activity: seconds over which the resting level is estimated (default {_NA.baseline_window_s:g})")
    g.add_argument("--burst-window", type=float,
                   help=f"activity: neurons starting within this many seconds fire together (default {_NA.burst_window_s})")
    g.add_argument("--burst-fraction", type=float,
                   help=f"activity: share of active neurons needed for a network burst (default {_NA.burst_min_fraction})")
    g.add_argument("--neurite-width", type=float,
                   help=f"neurites: typical neurite thickness in pixels (default {_NN.neurite_width_px:g})")
    g.add_argument("--frame-step", type=int, help=f"neurites: analyze every Nth frame (default {_NN.frame_step})")
    g.add_argument("--sholl-step", type=float,
                   help=f"neurites: spacing of Sholl rings in µm (pixels if uncalibrated; default {_NN.sholl_step_um:g})")
    g = p.add_argument_group("detection (animals mode)")
    g.add_argument("--detector", choices=["motion", "yolo"],
                   help=f"motion: anything moving on a still camera. yolo: AI model for "
                        f"people/birds/animals (default {_A.detector})")
    g.add_argument("--classes", default="",
                   help="yolo only, comma list, e.g. 'person' or 'bird,dog'")
    g.add_argument("--yolo-model", help=f"default {_A.yolo_model}")
    g.add_argument("--conf", type=float, help=f"yolo confidence cut-off (default {_A.conf})")
    g.add_argument("--min-area", type=int, help=f"motion: smallest blob in pixels (default {_A.min_area})")
    g.add_argument("--max-area", type=int, help=f"motion: largest blob in pixels (default {_A.max_area})")
    g.add_argument("--sensitivity", type=float,
                   help=f"motion: lower picks up fainter movement (default {_A.sensitivity:g})")
    g.add_argument("--resize-width", type=int,
                   help=f"process at this width for speed (0 = full size; default {_A.resize_width}, "
                        f"{_C.resize_width} for microscope modes)")
    g = p.add_argument_group("time range")
    g.add_argument("--start", type=float, help="start at this second")
    g.add_argument("--max-seconds", type=float, help="only analyze this long (default: all)")
    g = p.add_argument_group("behavior")
    g.add_argument("--contact-bl", type=float,
                   help=f"interaction distance in body lengths (default {_A.contact_bl:g}; "
                        f"{_C.contact_bl:g} cell diameters in cells mode)")
    g.add_argument("--moving-bl-s", type=float,
                   help=f"speed (body lengths/s) below which an object counts as still (default {_A.moving_bl_s})")
    g.add_argument("--min-behavior-seconds", type=float, help=f"default {_A.min_behavior_seconds:g}")
    p.add_argument("--no-video", action="store_true", help="skip the annotated video")


def _settings(a) -> Settings:
    """Settings for the chosen mode, with only the options the user actually typed changed."""
    changes = {field: getattr(a, opt) for opt, field in _OPTION_FIELDS.items()}
    for opt, (field, value) in _SWITCHES.items():
        if getattr(a, opt):
            changes[field] = value
    if a.classes:
        changes["classes"] = [c.strip() for c in a.classes.split(",") if c.strip()]
    base = defaults(a.mode, a.analysis if a.mode == "neurons" else None)
    return with_changes(base, **changes)


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
    summary.pop("seconds_in_each_behavior", None) if a.mode != "animals" else None
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
        if a.behavior in ("interaction", "network_burst"):
            f = res / ("interactions.csv" if a.behavior == "interaction" else "bursts.csv")
            df = pd.read_csv(f) if f.exists() and f.stat().st_size > 1 else pd.DataFrame()
            df = df[df["duration_s"] >= a.min_seconds] if not df.empty else df
        else:
            f = res / "behaviors.csv"
            df = pd.read_csv(f) if f.exists() and f.stat().st_size > 1 else pd.DataFrame()
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


def cmd_app(a):
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")),
                "--server.maxUploadSize", "4096", "--browser.gatherUsageStats", "false",
                # only this computer can open the app (not other devices on the network)
                "--server.address", "localhost",
                "--server.headless", "true" if a.no_browser else "false"]
    if a.port:
        sys.argv += ["--server.port", str(a.port)]
    sys.exit(stcli.main())


def app_main():
    """Entry point for the `behavetrack-app` command."""
    main(["app", *sys.argv[1:]])


def main(argv=None):
    p = argparse.ArgumentParser(prog="behavetrack",
                                description="Track animals, people, cells and neurons in video and export behavior data.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("app", help="open the browser app")
    pp.add_argument("--port", type=int, help="port to serve on (default 8501)")
    pp.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    pp.set_defaults(func=cmd_app)

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
