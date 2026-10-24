"""All analysis settings, and the default values for each mode, in one place.

The command line (cli.py) and the browser app (app.py) both take their defaults from
here, so changing a default in this file changes it everywhere.

    defaults("cells")                   -> Settings with the cell-mode defaults
    defaults("neurons", "neurites")     -> neurite-growth defaults
    with_changes(base, threshold_adjust=-0.1, pixel_size_um=None)
        -> a copy of base with the given values; None means "keep the default"
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

MODES = ("animals", "cells", "neurons")
NEURAL_ANALYSES = ("activity", "neurites")


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


def defaults(mode: str = "animals", analysis: str | None = None) -> Settings:
    """The default settings for a mode (and, in neurons mode, an analysis)."""
    if mode == "cells":
        return cell_settings()
    if mode == "neurons":
        return neuron_settings(**({"neural_analysis": analysis} if analysis else {}))
    if mode != "animals":
        raise ValueError(f"Unknown mode {mode!r}; choose from {', '.join(MODES)}")
    return Settings()


def with_changes(base: Settings, **changes) -> Settings:
    """Copy of `base` with `changes` applied. Values of None are ignored (keep the default)."""
    known = {f.name for f in fields(Settings)}
    unknown = set(changes) - known
    if unknown:
        raise TypeError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
    return replace(base, **{k: v for k, v in changes.items() if v is not None})
