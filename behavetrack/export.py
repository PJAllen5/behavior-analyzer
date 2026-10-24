"""Export results for complexity-engine (https://github.com/PJAllen5/complexity-engine).

Turns one or more results folders into the files complexity-engine's behavior profiles
expect, and can run the engine on them:

  collective_timeseries.csv  animals and cells: group alignment, circling, speed and spacing
                             per frame                         -> profile collective_behavior
  neural_timeseries.csv      neuron activity: Delta F/F per neuron per frame, plus the
                             fraction of neurons firing        -> profile neural_activity
  contact_network.csv        animals and cells: who came near whom, weighted by seconds
                             in contact                        -> profile animal_social_network
  functional_network.csv     neuron activity: neuron pairs correlated above a threshold
                             -> profile neuronal_network

Neurite-growth results are skipped: a growth time-lapse has too few frames for entropy or
DFA, and neurites are not a network of interactions.

Each results folder is one recording (`session`). Its `condition` is the folder name unless
given as FOLDER:CONDITION. Time series keep conditions as given, so several recordings can
share one and be compared as a group. In network files every recording is its own network
(condition "CONDITION / session" when the names differ), since track IDs from different
videos are different individuals.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

COLLECTIVE_SIGNALS = ["polarization", "rotation", "mean_speed_bl_s", "nn_dist_bl", "spread_bl", "n_visible"]
NEURAL_SIGNALS = ["dff", "fraction_firing"]

# file -> (engine pipeline, domain profile, signal columns, unit columns)
ENGINE_RUNS = {
    "collective_timeseries.csv": ("timeseries", "collective_behavior",
                                  ["polarization", "rotation", "mean_speed_bl_s", "nn_dist_bl"], ["session"]),
    "neural_timeseries.csv": ("timeseries", "neural_activity", NEURAL_SIGNALS, ["session", "neuron_id"]),
    "contact_network.csv": ("network", "animal_social_network", None, None),
    "functional_network.csv": ("network", "neuronal_network", None, None),
}


@dataclass
class ExportResult:
    files: dict[str, Path] = field(default_factory=dict)      # file name -> path written
    skipped: list[str] = field(default_factory=list)           # why a results folder was left out
    reports: dict[str, Path] = field(default_factory=dict)     # file name -> engine report folder


def parse_input(spec: str) -> tuple[Path, str]:
    """'results/video1' or 'results/video1:control' -> (folder, condition)."""
    path = Path(spec)
    if not path.exists() and ":" in spec:
        folder, condition = spec.rsplit(":", 1)
        if Path(folder).exists() and condition.strip():
            return Path(folder), condition.strip()
    return path, path.name


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 1 else pd.DataFrame()


def _network_condition(condition: str, session: str) -> str:
    return session if condition == session else f"{condition} / {session}"


def _collective(res: Path, session: str, condition: str) -> pd.DataFrame:
    fr = _read(res / "frames.csv")
    if fr.empty:
        return fr
    cols = [c for c in COLLECTIVE_SIGNALS if c in fr]
    out = fr[["time_s", *cols]].rename(columns={"time_s": "time"})
    out.insert(1, "session", session)
    out.insert(2, "condition", condition)
    return out


def _contacts(res: Path, session: str, condition: str) -> pd.DataFrame:
    it = _read(res / "interactions.csv")
    if it.empty:
        return it
    a, b = it[["id_a", "id_b"]].min(axis=1), it[["id_a", "id_b"]].max(axis=1)
    g = it.assign(source=a, target=b).groupby(["source", "target"])["duration_s"]
    out = pd.DataFrame({"weight": g.sum().round(3), "contacts": g.size()}).reset_index()
    out["condition"] = _network_condition(condition, session)
    return out


def _neural(res: Path, session: str, condition: str) -> pd.DataFrame:
    tr = _read(res / "traces_dff.csv")
    if tr.empty:
        return tr
    neuron_cols = [c for c in tr if c.startswith("neuron_")]
    long = tr.melt(id_vars=["frame", "time_s"], value_vars=neuron_cols, var_name="neuron_id", value_name="dff")
    long["neuron_id"] = long["neuron_id"].str.removeprefix("neuron_").astype(int)
    fr = _read(res / "frames.csv")
    if "fraction_firing" in fr:
        long = long.merge(fr[["frame", "fraction_firing"]], on="frame", how="left")
    long = long.rename(columns={"time_s": "time"}).drop(columns="frame")
    long.insert(1, "session", session)
    long.insert(3, "condition", condition)
    return long.sort_values(["neuron_id", "time"]).reset_index(drop=True)


def _functional(res: Path, session: str, condition: str, min_correlation: float) -> pd.DataFrame:
    co = _read(res / "correlations.csv")
    if co.empty:
        return co
    co = co[co["correlation"] >= min_correlation]
    out = co.rename(columns={"neuron_a": "source", "neuron_b": "target", "correlation": "weight"})
    cols = ["source", "target", "weight"] + (["distance_um"] if "distance_um" in out else [])
    out = out[cols].copy()
    out["condition"] = _network_condition(condition, session)
    return out


def export(inputs: list[str | tuple[Path, str]], out_dir: str | Path, min_correlation: float = 0.3) -> ExportResult:
    """Write the complexity-engine files for these results folders. Returns what was written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in ENGINE_RUNS}
    result = ExportResult()
    for item in inputs:
        res, condition = parse_input(item) if isinstance(item, str) else item
        summary_path = res / "summary.json"
        if not summary_path.exists():
            result.skipped.append(f"{res}: not a results folder (no summary.json)")
            continue
        summary = json.loads(summary_path.read_text())
        session, mode = res.name, summary.get("mode", "animals")
        if mode == "neurons" and summary.get("neural_analysis") == "neurites":
            result.skipped.append(f"{session}: neurite growth has too few frames for complexity measures")
            continue
        if mode == "neurons":
            pieces = {"neural_timeseries.csv": _neural(res, session, condition),
                      "functional_network.csv": _functional(res, session, condition, min_correlation)}
        else:
            pieces = {"collective_timeseries.csv": _collective(res, session, condition),
                      "contact_network.csv": _contacts(res, session, condition)}
        for name, df in pieces.items():
            if df.empty:
                result.skipped.append(f"{session}: nothing for {name}")
            else:
                parts[name].append(df)
    for name, dfs in parts.items():
        if dfs:
            path = out_dir / name
            pd.concat(dfs, ignore_index=True).to_csv(path, index=False)
            result.files[name] = path
    return result


def engine_available() -> bool:
    try:
        import complexity_engine  # noqa: F401
    except ImportError:
        return False
    return True


ENGINE_INSTALL_HINT = ('complexity-engine is not installed. Install it with:\n'
                       '  pip install -e ".[complexity]"\n'
                       'or from a local copy:  pip install -e /path/to/complexity-engine')


def run_engine(result: ExportResult, out_dir: str | Path) -> ExportResult:
    """Run complexity-engine with the matching behavior profile on each exported file."""
    if not engine_available():
        raise RuntimeError(ENGINE_INSTALL_HINT)
    from complexity_engine.analysis import analyze_dataset
    from complexity_engine.network_analysis import analyze_network

    reports = Path(out_dir) / "reports"
    for name, path in result.files.items():
        pipeline, domain, signals, units = ENGINE_RUNS[name]
        target = reports / Path(name).stem
        if pipeline == "timeseries":
            present = pd.read_csv(path, nrows=1).columns
            analyze_dataset(path, signal_cols=[s for s in signals if s in present], condition_col="condition",
                            domain=domain, unit_cols=units, output_dir=target)
        else:
            analyze_network(path, domain=domain, condition_col="condition", weight_col="weight",
                            output_dir=target)
        result.reports[name] = target
    return result
