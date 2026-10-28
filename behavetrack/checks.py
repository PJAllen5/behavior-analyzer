"""Data checks: flag results that are implausible for their field or recorded in a way that
can't support them.

Thresholds, explanations and sources live in reference/checks.yaml; this file computes the
value each check looks at. Results go to checks.json in the results folder, with a count in
summary.json. Checks only advise: nothing is blocked or changed.

Each check returns a status:
  ok    the value is within the expected range
  warn  worth a second look before reporting (the message says why and what to do)
  info  couldn't be checked, for example because a detail such as the indicator is missing
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from . import metadata as md

REFERENCE = Path(__file__).with_name("reference")

# Experiment details each mode should have; listed by the details_recorded check.
KEY_DETAILS = {
    "animals": ["organism"],
    "cells": ["cell_kind", "cell_line"],
    "neurons": ["organism", "preparation", "indicator"],
}


@dataclass
class Check:
    id: str
    title: str
    kind: str      # setup or biology
    status: str    # ok, warn or info
    message: str
    why: str
    value: float | str | None = None


def load_rules() -> list[dict]:
    import yaml

    return yaml.safe_load((REFERENCE / "checks.yaml").read_text(encoding="utf-8"))["checks"]


class _Result:
    """Lazy access to one results folder."""

    def __init__(self, folder: Path, summary: dict | None = None, meta: dict | None = None):
        self.dir = Path(folder)
        self.summary = summary if summary is not None else json.loads((self.dir / "summary.json").read_text())
        self.meta = meta if meta is not None else md.load(self.dir)

    def csv(self, name: str) -> pd.DataFrame:
        p = self.dir / name
        return pd.read_csv(p) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()

    @property
    def mode(self) -> str:
        return self.summary.get("mode", "animals")

    @property
    def analysis(self) -> str | None:
        return self.summary.get("neural_analysis")

    @property
    def px(self) -> float | None:
        return self.summary.get("pixel_size_um")

    @property
    def fps(self) -> float:
        return float(self.summary.get("fps") or 0)

    @cached_property
    def cell_diameter_um(self) -> float | None:
        cells = self.csv("cells.csv")
        if "mean_area_um2" not in cells or cells.empty:
            return None
        return float(np.median(2 * np.sqrt(cells["mean_area_um2"].dropna() / math.pi)))


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    digits = 2 if abs(v) < 10 else 1 if abs(v) < 100 else 0
    return f"{v:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{v:.0f}"


def _range_text(rule: dict) -> str:
    return f"{_fmt(rule['low'])}–{_fmt(rule['high'])} µm"


# ---------------------------------------------------------------- one function per check id
# Each returns (status, value, extra text) and may return None to leave the check out entirely.

def _details_recorded(r: _Result, rule):
    labels = [md.FIELD_BY_KEY[k].label.lower() for k in KEY_DETAILS.get(r.mode, [])
              if not r.meta.get(k) and r.mode in md.FIELD_BY_KEY[k].modes]
    if r.mode == "neurons" and r.analysis == "neurites":
        labels = [l for l in labels if l != md.FIELD_BY_KEY["indicator"].label.lower()]
    return ("warn", ", ".join(labels), "") if labels else ("ok", None, "")


def _animals_id_breaks(r: _Result, rule):
    seen = r.summary.get("typical_visible") or 0
    if not seen:
        return None
    ratio = r.summary.get("individuals_tracked", 0) / seen
    return ("warn" if ratio > rule["limit"] else "ok"), ratio, ""


def _animals_expected_count(r: _Result, rule):
    expected = r.meta.get("expected_individuals")
    if not expected:
        return "info", None, "Enter the number of individuals under Experiment details to check this."
    seen = r.summary.get("typical_visible", 0)
    off = abs(seen - expected) / expected
    return ("warn" if off > rule["limit"] else "ok"), seen, f" Group size entered: {_fmt(expected)}."


def _animals_size(r: _Result, rule):
    body = r.summary.get("body_length_px")
    if not body:
        return None
    return ("warn" if body < rule["limit"] else "ok"), body, ""


def _animals_step(r: _Result, rule):
    tr = r.csv("tracks.csv")
    if tr.empty or "speed_bl_s" not in tr or not r.fps:
        return None
    step = float(tr["speed_bl_s"].median() / r.fps)
    return ("warn" if step > rule["limit"] else "ok"), step, ""


def _calibrated(r: _Result, rule):
    return ("ok", r.px, "") if r.px else ("warn", None, "")


def _time_known(r: _Result, rule):
    if not r.summary.get("time_known"):
        return "warn", None, ""
    if r.mode == "neurons" and r.analysis == "activity":
        return "ok", f"{_fmt(r.fps)} frames per second", ""
    dt = 1 / r.fps if r.fps else 0
    text = f"{_fmt(dt / 60)} min" if dt >= 60 else f"{_fmt(dt)} s"
    return "ok", f"{text} between frames", ""


def _kind_gate(r: _Result, rule):
    """Skip biology checks that don't apply to the kind of cells; note when it's unknown."""
    kind = r.meta.get("cell_kind")
    if "only_kinds" in rule:
        if not kind:
            return "info", None, "Set the kind of cells under Experiment details to check this."
        if kind not in rule["only_kinds"]:
            return "info", None, f"Not checked for {kind}: the published values used here are for " \
                                 f"{' and '.join(rule['only_kinds'])}."
    if kind and kind in rule.get("skip_kinds", []):
        return "info", None, f"Not checked for {kind}: the published values used here don't cover them."
    return None


def _needs_calibration(r: _Result, rule):
    if rule.get("needs_calibration") and not r.px:
        return "info", None, "Needs the pixel size."
    return None


def _cells_size(r: _Result, rule):
    gate = _kind_gate(r, rule) or _needs_calibration(r, rule)
    if gate:
        return gate
    d = r.cell_diameter_um
    if d is None:
        return None
    return ("warn" if not rule["low"] <= d <= rule["high"] else "ok"), d, ""


def _cells_speed(r: _Result, rule):
    gate = _kind_gate(r, rule) or _needs_calibration(r, rule)
    if gate:
        return gate
    speed = r.summary.get("cells", {}).get("median_speed_um_per_min")
    if speed is None:
        return None
    note = "" if r.meta.get("cell_kind") else " Compared with adherent cells, since the kind of cells isn't set."
    return ("warn" if speed > rule["limit"] else "ok"), speed, note


def _cells_step(r: _Result, rule):
    gate = _needs_calibration(r, rule)
    if gate:
        return gate
    tr, d = r.csv("tracks.csv"), r.cell_diameter_um
    if tr.empty or "speed_um_per_min" not in tr or not d or not r.fps:
        return None
    step_um = float(tr["speed_um_per_min"].median() * (1 / r.fps) / 60)
    ratio = step_um / d
    return ("warn" if ratio > rule["limit"] else "ok"), ratio, ""


def _cells_track_length(r: _Result, rule):
    cells = r.csv("cells.csv")
    if cells.empty or "frames" not in cells:
        return None
    n = float(cells["frames"].median())
    return ("warn" if n < rule["limit"] else "ok"), n, ""


def _cells_doubling(r: _Result, rule):
    gate = _kind_gate(r, rule)
    if gate:
        return gate
    dbl = r.summary.get("cells", {}).get("population_doubling_time_h")
    if not dbl:
        return "info", None, "The cell count didn't rise during the recording, so there's no doubling time to check."
    return ("warn" if dbl < rule["limit"] else "ok"), dbl, ""


def _neurons_frame_rate(r: _Result, rule):
    if not r.summary.get("time_known"):
        return "info", None, "Needs the frame rate."
    indicator = r.meta.get("indicator")
    half = rule["half_decay_s"].get(indicator) if indicator else None
    need = 2 / half if half else 2.0
    status = "warn" if r.fps < need else "ok"
    if not indicator:
        note = " Set the indicator under Experiment details for an exact check."
        return status, r.fps, note
    return status, r.fps, ""


def _neurons_size(r: _Result, rule):
    gate = _needs_calibration(r, rule)
    if gate:
        return gate
    nn = r.csv("neurons.csv")
    if nn.empty or "area_um2" not in nn:
        return None
    d = float(np.median(2 * np.sqrt(nn["area_um2"].dropna() / math.pi)))
    return ("warn" if not rule["low"] <= d <= rule["high"] else "ok"), d, ""


def _neurons_duration(r: _Result, rule):
    window = r.summary.get("settings", {}).get("baseline_window_s")
    if not window or not r.summary.get("time_known"):
        return None
    ratio = r.summary.get("analyzed_seconds", 0) / window
    return ("warn" if ratio < rule["limit"] else "ok"), ratio, ""


def _drift_reasons(r: _Result, rule) -> list[str]:
    """Why a drift correction looks suspicious (empty if it doesn't)."""
    n = r.summary.get("neurons", {})
    biggest, step = n.get("max_drift_px") or 0, n.get("max_drift_step_px")
    diam, skipped = n.get("diameter_px"), n.get("drift_frames_skipped")
    side = min(r.summary.get("width") or 0, r.summary.get("height") or 0)
    frames = r.summary.get("frames_analyzed") or 0
    out = []
    if side and biggest > rule["image_share"] * side:
        out.append(f"that is {_fmt(100 * biggest / side)}% of the image (above {_fmt(100 * rule['image_share'])}%)")
    if diam and biggest > rule["diameters"] * diam:
        out.append(f"{_fmt(biggest / diam)} neuron diameters (above {_fmt(rule['diameters'])})")
    if diam and step is not None and step > rule["step_diameters"] * diam:
        out.append(f"it jumped by up to {_fmt(step / diam)} neuron diameters from one frame to the next "
                   f"(above {_fmt(rule['step_diameters'])})")
    if frames and skipped and skipped > rule["skipped_share"] * frames:
        out.append(f"{_fmt(100 * skipped / frames)}% of frames couldn't be lined up "
                   f"(above {_fmt(100 * rule['skipped_share'])}%)")
    return out


def _neurons_drift(r: _Result, rule):
    if not r.summary.get("settings", {}).get("correct_drift", True):
        return None
    n = r.summary.get("neurons", {})
    if "max_drift_px" not in n:
        return None
    skipped = n.get("drift_frames_skipped") or 0
    note = (f" {skipped} frame{'s' if skipped != 1 else ''} couldn't be lined up (blank, or too different "
            "from the rest) and took the drift of their neighbours.") if skipped else ""
    return ("warn" if _drift_reasons(r, rule) else "ok"), float(n["max_drift_px"]), note


def _neurites_frames(r: _Result, rule):
    n = r.summary.get("frames_analyzed", 0)
    return ("warn" if n < rule["limit"] else "ok"), n, ""


COMPUTE = {
    "details_recorded": _details_recorded,
    "animals_id_breaks": _animals_id_breaks,
    "animals_expected_count": _animals_expected_count,
    "animals_size": _animals_size,
    "animals_step": _animals_step,
    "cells_calibrated": _calibrated,
    "cells_time_known": _time_known,
    "cells_size": _cells_size,
    "cells_speed": _cells_speed,
    "cells_step": _cells_step,
    "cells_track_length": _cells_track_length,
    "cells_doubling": _cells_doubling,
    "neurons_calibrated": _calibrated,
    "neurons_time_known": _time_known,
    "neurons_frame_rate": _neurons_frame_rate,
    "neurons_size": _neurons_size,
    "neurons_duration": _neurons_duration,
    "neurons_drift": _neurons_drift,
    "neurites_frames": _neurites_frames,
}


def _limit_text(rule: dict, r: _Result) -> str:
    if "low" in rule:
        return _range_text(rule)
    if rule["id"] == "neurons_drift":
        return "; ".join(_drift_reasons(r, rule))
    if rule["id"] == "neurons_frame_rate":
        half = rule["half_decay_s"].get(r.meta.get("indicator") or "")
        return f"{_fmt(2 / half if half else 2.0)} per second"
    return _fmt(rule["limit"]) if "limit" in rule else ""


def run_checks(results_dir: str | Path, summary: dict | None = None, metadata: dict | None = None) -> list[Check]:
    r = _Result(Path(results_dir), summary, metadata)
    out = []
    for rule in load_rules():
        if r.mode not in rule["modes"]:
            continue
        if "analysis" in rule and r.analysis != rule["analysis"]:
            continue
        got = COMPUTE[rule["id"]](r, rule)
        if got is None:
            continue
        status, value, extra = got
        shown = _fmt(value) if isinstance(value, (int, float)) and value is not None else (value or "")
        if status == "info":
            message = extra.strip()
        else:
            template = rule["warn"] if status == "warn" else rule["ok"]
            message = template.format(value=shown, limit=_limit_text(rule, r)) + extra
        out.append(Check(rule["id"], rule["title"], rule["kind"], status, message, rule["why"],
                         round(value, 4) if isinstance(value, float) else value))
    return out


def counts(checks: list[Check]) -> dict:
    return {s: sum(c.status == s for c in checks) for s in ("warn", "ok", "info")}


def save(results_dir: str | Path, checks: list[Check]) -> dict:
    """Write checks.json and record the counts in summary.json."""
    results_dir = Path(results_dir)
    (results_dir / "checks.json").write_text(json.dumps([asdict(c) for c in checks], indent=2))
    path = results_dir / "summary.json"
    summary = json.loads(path.read_text())
    summary["checks"] = counts(checks)
    path.write_text(json.dumps(summary, indent=2, default=str))
    return summary["checks"]


def recheck(results_dir: str | Path, metadata: dict | None = None) -> list[Check]:
    """Save new experiment details (if given) and run the checks again. No re-analysis needed."""
    results_dir = Path(results_dir)
    if metadata is not None:
        meta = md.save(results_dir, metadata)
        path = results_dir / "summary.json"
        summary = json.loads(path.read_text())
        summary["metadata"] = meta
        path.write_text(json.dumps(summary, indent=2, default=str))
    checks = run_checks(results_dir)
    save(results_dir, checks)
    return checks


def load(results_dir: str | Path) -> list[Check]:
    p = Path(results_dir) / "checks.json"
    return [Check(**c) for c in json.loads(p.read_text())] if p.exists() else []
