"""Experiment details (metadata) saved with each result.

The fields follow two community reporting standards: REMBI, recommended metadata for
biological images (Sarkans et al., 2021), and MIACME, the minimum information about a cell
migration experiment (Gonzalez-Beltran et al., 2020). They are kept short on purpose: the
details that other researchers need to interpret the numbers, and that the data checks use
(for example the calcium indicator sets how fast a movie must be recorded).

Saved as metadata.json in the results folder, and copied into summary.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ALL_MODES = ("animals", "cells", "neurons")


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    help: str
    modes: tuple[str, ...] = ALL_MODES
    choices: tuple[str, ...] = ()     # empty = free text
    number: bool = False
    standard: str = ""                # which reporting standard asks for it


# Choices that the data checks read. Keep these labels stable: checks.yaml refers to them.
CELL_KINDS = ("adherent mammalian cells", "immune cells (leukocytes)", "amoeba (e.g. Dictyostelium)",
              "bacteria", "other")
INDICATORS = ("GCaMP6s", "GCaMP6m", "GCaMP6f", "other genetically encoded indicator",
              "synthetic dye (e.g. Fluo-4, Cal-520)", "voltage indicator")

FIELDS: tuple[Field, ...] = (
    # --- study (all modes)
    Field("study", "Study or experiment name", "A short name that identifies the experiment.",
          standard="REMBI: study"),
    Field("condition", "Condition or treatment", "What makes this recording different from the others, "
          "such as control, drug and dose, or temperature.", standard="REMBI: study component; MIACME"),
    Field("date", "Date recorded", "When the recording was made (YYYY-MM-DD).", standard="REMBI: image acquisition"),
    # --- sample
    Field("organism", "Species", "For example zebrafish, starling, Homo sapiens, Mus musculus.",
          standard="REMBI: biosample; MIACME"),
    Field("expected_individuals", "Number of individuals in the group", "How many animals were actually "
          "there, if known. Compared with how many the tracker followed.", modes=("animals",), number=True),
    Field("setting", "Setting", "Where it was filmed, such as a lab arena, aquarium, or field site, "
          "and its size.", modes=("animals",)),
    Field("cell_kind", "Kind of cells", "Used to judge whether speeds and division times are typical.",
          modes=("cells",), choices=CELL_KINDS, standard="MIACME"),
    Field("cell_line", "Cell line or cell type", "For example HeLa, MDA-MB-231, primary fibroblasts.",
          modes=("cells",), standard="REMBI: biosample; MIACME"),
    Field("substrate", "Substrate", "What the cells move on or in, such as glass, a fibronectin coating, "
          "or a 3D collagen gel.", modes=("cells",), standard="MIACME"),
    Field("preparation", "Preparation", "Culture, brain slice, or in vivo.", modes=("neurons",),
          choices=("dissociated culture", "organotypic or acute slice", "in vivo", "organoid", "other"),
          standard="REMBI: specimen"),
    Field("brain_region", "Brain region or neuron type", "For example hippocampus, cortex layer 2/3, "
          "motor neurons.", modes=("neurons",), standard="REMBI: biosample"),
    Field("days_in_vitro", "Days in culture", "Age of the culture in days (DIV), if a culture.",
          modes=("neurons",), number=True),
    Field("indicator", "Indicator or label", "The calcium or voltage indicator, or the label used to see "
          "neurites. The data checks use it to judge whether the frame rate is fast enough.",
          modes=("neurons",), choices=INDICATORS, standard="REMBI: specimen"),
    Field("label", "Label or stain", "For example H2B-GFP, phase contrast (no label), DAPI.",
          modes=("cells",), standard="REMBI: specimen"),
    # --- imaging (all modes)
    Field("temperature_c", "Temperature (°C)", "Temperature during recording.", number=True,
          standard="MIACME"),
    Field("imaging", "Imaging method", "Camera or microscope type.",
          choices=("video camera", "widefield microscope", "confocal microscope", "two-photon microscope",
                   "light-sheet microscope", "phase contrast", "brightfield / DIC", "other"),
          standard="REMBI: image acquisition"),
    Field("objective", "Objective (magnification / NA)", "For example 20x / 0.75.",
          modes=("cells", "neurons"), standard="REMBI: image acquisition"),
    Field("notes", "Notes", "Anything else someone would need to know to interpret the results."),
)

FIELD_BY_KEY = {f.key: f for f in FIELDS}


def fields_for(mode: str) -> list[Field]:
    return [f for f in FIELDS if mode in f.modes]


def clean(metadata: dict | None, mode: str | None = None) -> dict:
    """Drop empty values, convert numbers, and reject unknown keys and choices."""
    out = {}
    for key, value in (metadata or {}).items():
        if key not in FIELD_BY_KEY:
            raise ValueError(f"Unknown experiment detail '{key}'. Known: {', '.join(FIELD_BY_KEY)}")
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        f = FIELD_BY_KEY[key]
        if f.number:
            try:
                value = float(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"'{key}' must be a number, got {value!r}") from e
            value = int(value) if value.is_integer() else value
        elif isinstance(value, str):
            value = value.strip()
        if f.choices and value not in f.choices:
            raise ValueError(f"'{key}' must be one of: {', '.join(f.choices)} (got {value!r})")
        out[key] = value
    return out


def parse_pairs(pairs: list[str]) -> dict:
    """['organism=zebrafish', 'expected_individuals=40'] -> dict."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"Experiment details look like key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load(results_dir: Path) -> dict:
    p = Path(results_dir) / "metadata.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save(results_dir: Path, metadata: dict) -> dict:
    results_dir = Path(results_dir)
    metadata = clean(metadata)
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
