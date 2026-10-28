"""Data checks and experiment details."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import pytest
import yaml

from behavetrack import checks, metadata

REF = Path(checks.__file__).with_name("reference")


# ------------------------------------------------------------------ the checks file itself

def _bib_entries() -> list[dict]:
    text = (REF / "references.bib").read_text(encoding="utf-8")
    out = []
    for block in re.split(r"\n(?=@\w+\{)", text):
        key = re.search(r"@\w+\{([^,\s]+),", block)
        author = re.search(r"^\s*author\s*=\s*\{(.*)\},\s*$", block, re.MULTILINE)
        year = re.search(r"year\s*=\s*\{(\d{4})\}", block)
        if key and author and year:
            names = [_norm(a.split(",")[0]) for a in author.group(1).split(" and ")]
            out.append({"key": key.group(1), "surnames": names, "year": int(year.group(1))})
    return out


def _norm(name: str) -> str:
    name = re.sub(r"\{\\['\"^~`]?([a-zA-Z])\}", r"\1", name).replace("{", "").replace("}", "")
    name = unicodedata.normalize("NFKD", name)
    return "".join(c for c in name if not unicodedata.combining(c)).strip().lower()


def _citations(text: str) -> list[tuple[list[str], int, bool, str]]:
    found = []
    for group in re.findall(r"\(([^()]*\b\d{4}\b[^()]*)\)", text):
        for item in group.split(";"):
            m = re.match(r"^(.*?),\s*(\d{4})$", item.strip())
            if not m:
                continue
            authors, year = m.group(1), int(m.group(2))
            et_al = "et al." in authors
            authors = authors.replace("et al.", "").strip()
            names = [_norm(a) for a in authors.split("&")]
            found.append((names, year, et_al, item.strip()))
    return found


def test_every_citation_matches_the_bibliography():
    entries = _bib_entries()
    text = (REF / "checks.yaml").read_text(encoding="utf-8") + (Path(metadata.__file__).read_text(encoding="utf-8"))
    cites = _citations(text)
    assert len(cites) >= 9
    missing = []
    for names, year, et_al, raw in cites:
        ok = any(e["year"] == year and (names[0] in e["surnames"] if et_al else all(n in e["surnames"] for n in names))
                 for e in entries)
        if not ok:
            missing.append(raw)
    assert not missing, f"No bibliography entry for: {missing}"


def test_rules_and_code_agree():
    rules = checks.load_rules()
    ids = [r["id"] for r in rules]
    assert len(ids) == len(set(ids)), "duplicate check ids"
    assert set(ids) == set(checks.COMPUTE), "every rule needs code, and every piece of code a rule"
    for r in rules:
        for field in ("title", "kind", "warn", "ok", "why", "modes"):
            assert field in r, f"{r['id']} is missing '{field}'"
        assert r["kind"] in ("setup", "biology")
        for key in r.get("needs", []):
            assert key in metadata.FIELD_BY_KEY, f"{r['id']} needs unknown detail {key}"
        for kind in r.get("skip_kinds", []) + r.get("only_kinds", []):
            assert kind in metadata.CELL_KINDS, f"{r['id']}: unknown cell kind {kind!r}"
        for ind in r.get("half_decay_s", {}):
            assert ind in metadata.INDICATORS, f"{r['id']}: unknown indicator {ind!r}"


# ------------------------------------------------------------------ experiment details

def test_clean_converts_and_rejects():
    got = metadata.clean({"expected_individuals": "40", "organism": " zebrafish ", "notes": ""})
    assert got == {"expected_individuals": 40, "organism": "zebrafish"}
    with pytest.raises(ValueError, match="Unknown"):
        metadata.clean({"colour": "blue"})
    with pytest.raises(ValueError, match="must be one of"):
        metadata.clean({"indicator": "GFP"})
    with pytest.raises(ValueError, match="number"):
        metadata.clean({"temperature_c": "warm"})
    assert metadata.parse_pairs(["a=b", "c=d=e"]) == {"a": "b", "c": "d=e"}


# ------------------------------------------------------------------ checks on hand-made results

def _results(tmp_path: Path, mode: str, **summary) -> Path:
    d = tmp_path / mode
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"mode": mode, "fps": 20.0, "time_known": True, **summary}))
    return d


def _by_id(found) -> dict:
    return {c.id: c for c in found}


def test_cells_uncalibrated(tmp_path):
    d = _results(tmp_path, "cells", fps=1 / 300, pixel_size_um=None, cells={})
    got = _by_id(checks.run_checks(d))
    assert got["cells_calibrated"].status == "warn"
    assert got["cells_size"].status == "info" and "pixel size" in got["cells_size"].message


@pytest.mark.parametrize("kind,speed,expect", [
    ("adherent mammalian cells", 0.5, "ok"),
    ("adherent mammalian cells", 25.0, "warn"),
    (None, 25.0, "warn"),                       # unknown kind: compared with adherent cells
    ("immune cells (leukocytes)", 25.0, "info"),  # not covered by the published comparison
])
def test_cell_speed(tmp_path, kind, speed, expect):
    d = _results(tmp_path, "cells", fps=1 / 300, pixel_size_um=0.65, cells={"median_speed_um_per_min": speed})
    meta = {"cell_kind": kind} if kind else {}
    assert _by_id(checks.run_checks(d, metadata=meta))["cells_speed"].status == expect


@pytest.mark.parametrize("indicator,fps,expect", [
    ("GCaMP6s", 10, "ok"), ("GCaMP6f", 10, "warn"), ("GCaMP6f", 30, "ok"),
    (None, 1, "warn"), (None, 10, "ok"),
])
def test_frame_rate_for_indicator(tmp_path, indicator, fps, expect):
    d = _results(tmp_path, "neurons", neural_analysis="activity", fps=fps, pixel_size_um=0.8,
                 analyzed_seconds=600, settings={"baseline_window_s": 60})
    meta = {"indicator": indicator} if indicator else {}
    c = _by_id(checks.run_checks(d, metadata=meta))["neurons_frame_rate"]
    assert c.status == expect
    if not indicator:
        assert "Set the indicator" in c.message


def test_animals_expected_count_and_identity(tmp_path):
    d = _results(tmp_path, "animals", typical_visible=40, individuals_tracked=90, body_length_px=17)
    got = _by_id(checks.run_checks(d, metadata={"organism": "fish", "expected_individuals": 60}))
    assert got["animals_expected_count"].status == "warn"
    assert got["animals_id_breaks"].status == "warn"
    assert got["details_recorded"].status == "ok"
    got = _by_id(checks.run_checks(d, metadata={}))
    assert got["animals_expected_count"].status == "info"
    assert got["details_recorded"].status == "warn" and "species" in got["details_recorded"].message


def test_recheck_saves_details_and_counts(tmp_path):
    d = _results(tmp_path, "animals", typical_visible=40, individuals_tracked=42, body_length_px=17)
    found = checks.recheck(d, {"organism": "starling", "expected_individuals": 40})
    assert json.loads((d / "metadata.json").read_text())["organism"] == "starling"
    summary = json.loads((d / "summary.json").read_text())
    assert summary["metadata"]["expected_individuals"] == 40
    assert summary["checks"] == checks.counts(found)
    assert {c.id for c in checks.load(d)} == {c.id for c in found}


def test_check_command(tmp_path):
    d = _results(tmp_path, "animals", typical_visible=40, individuals_tracked=42, body_length_px=17)
    run = subprocess.run([sys.executable, "-m", "behavetrack", "check", str(d), "--meta", "organism=starling",
                          "--meta", "expected_individuals=60"], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "1 to look at" in run.stdout
    bad = subprocess.run([sys.executable, "-m", "behavetrack", "check", str(d), "--meta", "colour=blue"],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and "Unknown experiment detail" in bad.stderr


# ------------------------------------------------------------------ on the real test movies

@pytest.mark.slow
@pytest.mark.parametrize("name", ["flock", "cells", "neurons", "neurites"])
def test_every_analysis_writes_checks(analyzed, name):
    out = analyzed(name)["out"]
    found = checks.load(out)
    assert found, "checks.json missing"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["checks"] == checks.counts(found)
    assert (out / "metadata.json").exists()
    by_id = _by_id(found)
    # the generated movies are recorded and calibrated well, so the setup checks pass
    for cid in ("cells_calibrated", "cells_time_known", "cells_step", "neurons_calibrated",
                "neurons_time_known", "neurons_drift", "animals_size", "animals_step", "animals_id_breaks"):
        if cid in by_id:
            assert by_id[cid].status == "ok", (cid, by_id[cid].message)


def _drift_results(tmp_path, **neurons):
    base = {"max_drift_px": 1.5, "max_drift_step_px": 1.2, "drift_frames_skipped": 87, "diameter_px": 12.0}
    return _results(tmp_path, "neurons", neural_analysis="activity", width=512, height=512,
                    frames_analyzed=3024, settings={"correct_drift": True}, neurons={**base, **neurons})


@pytest.mark.parametrize("neurons,expect,says", [
    ({}, "ok", "87 frames couldn't be lined up"),             # Neurofinder 00.00 after the fix
    ({"max_drift_px": 52.96}, "warn", "% of the image"),      # ... and before it
    ({"max_drift_px": 52.96, "max_drift_step_px": None}, "warn", "neuron diameters"),  # older results
    ({"max_drift_step_px": 9.0}, "warn", "from one frame to the next"),
    ({"drift_frames_skipped": 400}, "warn", "couldn't be lined up (above 5%)"),
])
def test_drift_check(tmp_path, neurons, expect, says):
    c = _by_id(checks.run_checks(_drift_results(tmp_path, **neurons)))["neurons_drift"]
    assert c.status == expect and says in c.message, c.message
    if expect == "warn":
        assert "--no-drift" in c.message


def test_drift_check_skipped_without_drift_correction(tmp_path):
    d = _drift_results(tmp_path)
    s = json.loads((d / "summary.json").read_text())
    s["settings"]["correct_drift"] = False
    (d / "summary.json").write_text(json.dumps(s))
    assert "neurons_drift" not in _by_id(checks.run_checks(d))
