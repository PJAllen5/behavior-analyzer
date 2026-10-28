"""Download real benchmark data, with expert answer keys, to test behavetrack on.

  python tools/fetch_benchmarks.py list                 what's available, with sizes
  python tools/fetch_benchmarks.py info DIC-C2DH-HeLa   details, citation, how to analyze it
  python tools/fetch_benchmarks.py get DIC-C2DH-HeLa    download (asks first, shows the size)

Sources:
  Cell Tracking Challenge  real time-lapses of cells, with reference cell positions and
                           lineages (celltrackingchallenge.net)
  Neurofinder              calcium-imaging movies with hand-labelled neurons
                           (github.com/codeneuro/neurofinder)
  idtracker.ai             videos of fish, flies and ants (Google Drive only, so these are
                           downloaded by hand; `info` says where)

Files go to benchmarks/ (ignored by git). Each dataset gets an ABOUT.txt with its source,
citation, answer-key location and the exact `behavetrack analyze` command to run on it.
Nothing is downloaded without your confirmation, unless you pass --yes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "benchmarks"

CTC_URL = "https://data.celltrackingchallenge.net/training-datasets/{}.zip"
CTC_CITE = ("Maška, M., Ulman, V., Delgado-Rodriguez, P. et al. The Cell Tracking Challenge: 10 years of "
            "objective benchmarking. Nature Methods 20, 1010-1020 (2023).")
NF_URL = "https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder.{}.zip"
NF_CITE = "Neurofinder benchmark (CodeNeuro): https://github.com/codeneuro/neurofinder"
NF_LABS = {"00": "Janelia (Svoboda lab)", "01": "UCL (Häusser lab)", "02": "Janelia (Svoboda lab)",
           "03": "Columbia (Losonczy lab)", "04": "Harvard (Harvey lab)"}
IDT_FOLDER = "https://drive.google.com/drive/folders/1kAB2CDMmgoMtgFQ_q1e8Y4jhIdbxKhUv"
IDT_CITE = ("Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G. idtracker.ai: "
            "tracking all individuals in small or large collectives of unmarked animals. "
            "Nature Methods 16, 179-182 (2019).")


@dataclass
class Dataset:
    name: str
    source: str
    mode: str                         # behavetrack mode: animals, cells or neurons
    about: str
    url: str | None = None            # None = download by hand (see `manual`)
    size_mb: float | None = None      # from the source's page; checked live before downloading
    citation: str = ""
    answer_key: str = ""
    manual: str = ""
    options: list[str] = field(default_factory=list)   # extra `behavetrack analyze` options
    details: dict = field(default_factory=dict)          # experiment details (--meta)


def _ctc(name, size_mb, about, pixel_um, minutes, look, details):
    return Dataset(
        name=name, source="Cell Tracking Challenge", mode="cells", url=CTC_URL.format(name), size_mb=size_mb,
        about=about, citation=CTC_CITE,
        answer_key="Sequences 01 and 02, with reference tracks in 01_GT/TRA and 02_GT/TRA (man_track.txt "
                   "lists each track's first and last frame and its parent, so divisions are included).",
        options=["--mode", "cells", "--pixel-size", str(pixel_um), "--frame-interval", str(minutes * 60),
                 "--cells-look", look],
        details=details)


_ADHERENT = "adherent mammalian cells"
DATASETS: dict[str, Dataset] = {d.name: d for d in [
    _ctc("DIC-C2DH-HeLa", 37, "HeLa cells on glass, DIC; 0.19 µm pixels, 10 min between frames.",
         0.19, 10, "auto", {"cell_kind": _ADHERENT, "cell_line": "HeLa", "imaging": "brightfield / DIC",
                            "organism": "Homo sapiens"}),
    _ctc("PhC-C2DH-U373", 40, "U373 glioblastoma-astrocytoma cells on polyacrylamide, phase contrast; "
         "0.65 µm, 15 min.", 0.65, 15, "auto",
         {"cell_kind": _ADHERENT, "cell_line": "U373", "imaging": "phase contrast", "organism": "Homo sapiens",
          "substrate": "polyacrylamide"}),
    _ctc("Fluo-N2DH-GOWT1", 53, "GFP-GOWT1 mouse stem cell nuclei, fluorescence; 0.24 µm, 5 min.",
         0.24, 5, "bright", {"cell_kind": _ADHERENT, "cell_line": "GOWT1 mouse stem cells",
                             "imaging": "widefield microscope", "organism": "Mus musculus", "label": "GFP"}),
    _ctc("Fluo-N2DL-HeLa", 182, "HeLa nuclei (H2B-GFP), fluorescence, many cells and divisions; "
         "0.645 µm, 30 min.", 0.645, 30, "bright",
         {"cell_kind": _ADHERENT, "cell_line": "HeLa", "label": "H2B-GFP", "organism": "Homo sapiens",
          "imaging": "widefield microscope"}),
    _ctc("Fluo-C2DL-Huh7", 36, "Huh7 hepatocarcinoma cells (YFP-TIA-1), fluorescence; 0.65 µm, 15 min.",
         0.65, 15, "bright", {"cell_kind": _ADHERENT, "cell_line": "Huh7", "label": "YFP-TIA-1",
                              "organism": "Homo sapiens"}),
    _ctc("Fluo-C2DL-MSC", 72, "Rat mesenchymal stem cells on polyacrylamide, fluorescence; 0.3 µm, "
         "20 min (one sequence uses 0.3977 µm and 30 min; see the source page).", 0.3, 20, "bright",
         {"cell_kind": _ADHERENT, "cell_line": "rat mesenchymal stem cells", "organism": "Rattus norvegicus",
          "substrate": "polyacrylamide"}),
    _ctc("PhC-C2DL-PSC", 124, "Pancreatic stem cells on polystyrene, phase contrast, crowded; "
         "1.6 µm, 10 min.", 1.6, 10, "auto",
         {"cell_kind": _ADHERENT, "cell_line": "pancreatic stem cells", "imaging": "phase contrast",
          "substrate": "polystyrene"}),
    _ctc("Fluo-N2DH-SIM+", 91, "Simulated HL60 nuclei (computer-generated, with perfect answer keys); "
         "0.125 µm, 29 min.", 0.125, 29, "bright",
         {"cell_kind": "other", "cell_line": "simulated HL60 nuclei"}),
    _ctc("BF-C2DL-HSC", 1600, "Mouse hematopoietic stem cells in hydrogel microwells, brightfield, long; "
         "0.645 µm, 5 min.", 0.645, 5, "auto",
         {"cell_kind": "other", "cell_line": "mouse hematopoietic stem cells", "imaging": "brightfield / DIC",
          "organism": "Mus musculus"}),
    _ctc("BF-C2DL-MuSC", 1200, "Mouse muscle stem cells in hydrogel microwells, brightfield, long; "
         "0.645 µm, 5 min.", 0.645, 5, "auto",
         {"cell_kind": "other", "cell_line": "mouse muscle stem cells", "imaging": "brightfield / DIC",
          "organism": "Mus musculus"}),
] + [
    Dataset(name=f"neurofinder.{n}", source="Neurofinder", mode="neurons", url=NF_URL.format(n),
            size_mb={"00.00": 994, "01.00": 944, "02.00": 1730, "03.00": 929, "04.00": 934}.get(n),
            about=f"Two-photon calcium imaging, {NF_LABS[n[:2]]}. Motion-corrected 16-bit TIFF frames.",
            citation=NF_CITE,
            answer_key="regions/regions.json: hand-labelled neurons (pixel coordinates of each).",
            options=["--mode", "neurons", "--analysis", "activity", "--find-by", "brightness"],
            details={"preparation": "in vivo", "imaging": "two-photon microscope", "organism": "Mus musculus"})
    for n in ["00.00", "00.01", "00.02", "00.03", "00.04", "00.05", "00.06", "00.07", "00.08", "00.09",
              "00.10", "00.11", "01.00", "01.01", "02.00", "02.01", "03.00", "04.00", "04.01"]
] + [
    Dataset(name=f"idtracker-{n}", source="idtracker.ai", mode="animals", about=about, citation=IDT_CITE,
            manual=f"Download it by hand from the idtracker.ai data folder ({IDT_FOLDER}), "
                   f"save it in benchmarks/idtracker-{n}/, then run the command below.",
            answer_key="No position-by-position answer key; the number of animals is known, so compare "
                       "the tracker's count and identity breaks with it.",
            options=["--mode", "animals"], details=details)
    for n, about, details in [
        ("zebrafish-8", "8 adult zebrafish with complex behaviour.",
         {"organism": "Danio rerio", "expected_individuals": 8}),
        ("zebrafish-60", "60 zebrafish (30 days old) in a 70 cm circular arena.",
         {"organism": "Danio rerio", "expected_individuals": 60}),
        ("zebrafish-100", "100 zebrafish (30 days old) in a 70 cm circular arena.",
         {"organism": "Danio rerio", "expected_individuals": 100}),
        ("flies-72", "72 fruit flies.", {"organism": "Drosophila melanogaster", "expected_individuals": 72}),
        ("flies-100", "100 fruit flies.", {"organism": "Drosophila melanogaster", "expected_individuals": 100}),
        ("ants-14", "14 ants (compressed colour video).", {"organism": "ants", "expected_individuals": 14}),
    ]
]}


# ------------------------------------------------------------------ helpers

def _mb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.1f} GB" if n_bytes >= 1e9 else f"{n_bytes / 1e6:.0f} MB"


def remote_size(url: str, timeout: float = 20) -> int | None:
    """Size in bytes, from the server's headers (nothing is downloaded). None if unknown."""
    for method, headers in (("HEAD", {}), ("GET", {"Range": "bytes=0-0"})):
        try:
            req = urllib.request.Request(url, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                rng = r.headers.get("Content-Range")
                if rng and "/" in rng:
                    return int(rng.rsplit("/", 1)[1])
                if r.headers.get("Content-Length") and method == "HEAD":
                    return int(r.headers["Content-Length"])
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            continue
    return None


def download(url: str, target: Path, expected: int | None = None, timeout: float = 30,
             progress: bool = True) -> Path:
    """Download to target, resuming a partial .part file. Raises RuntimeError with advice on failure."""
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            if have and r.status != 206:  # server ignored the resume request: start again
                have = 0
            mode = "ab" if have else "wb"
            total = expected or (have + int(r.headers.get("Content-Length", 0))) or None
            done, last = have, 0.0
            with open(part, mode) as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if progress and time.time() - last > 0.5:
                        pct = f" ({100 * done / total:.0f}%)" if total else ""
                        sys.stderr.write(f"\r  {_mb(done)}{pct}   ")
                        last = time.time()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(
            f"The server didn't respond ({getattr(e, 'reason', e)}). It may be busy or down: try again "
            f"later (the download resumes where it stopped), or download it in a browser from {url} "
            f"and put the file at {target}.") from e
    if progress:
        sys.stderr.write("\n")
    if expected and part.stat().st_size != expected:
        raise RuntimeError(f"Download incomplete ({_mb(part.stat().st_size)} of {_mb(expected)}); "
                           "run the same command again to resume.")
    part.rename(target)
    return target


def safe_extract(zip_path: Path, dest: Path) -> list[Path]:
    """Unzip into dest, refusing any entry that would land outside it."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            out = (dest / info.filename).resolve()
            if dest != out and dest not in out.parents:
                raise RuntimeError(f"Refusing to unpack {info.filename!r}: it would land outside {dest}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError(f"Refusing to unpack {info.filename!r}: it is a symbolic link")
        z.extractall(dest)
    return [p for p in dest.iterdir()]


def sequences(folder: Path, d: Dataset) -> list[Path]:
    """The image folders inside an unpacked dataset that behavetrack can analyze."""
    if d.source == "Cell Tracking Challenge":
        return sorted(p for p in folder.rglob("*") if p.is_dir() and p.name.isdigit())
    if d.source == "Neurofinder":
        return sorted(p for p in folder.rglob("images") if p.is_dir())
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})


def neurofinder_info(folder: Path) -> dict:
    """Frame rate, pixel size, indicator and brain region from a Neurofinder info.json, when present."""
    for info in folder.rglob("info.json"):
        try:
            meta = json.loads(info.read_text())
        except (OSError, ValueError):
            continue
        rate = meta.get("rate-hz") or meta.get("rate_hz")
        ppm = meta.get("pixels-per-micron") or meta.get("pixels_per_micron")
        return {"rate_hz": float(rate) if rate else None,
                "pixel_size_um": 1 / float(ppm) if ppm else None,
                "indicator": meta.get("indicator"), "region": meta.get("region")}
    return {}


def analyze_command(d: Dataset, target: str, folder: Path | None = None) -> str:
    from shlex import quote

    opts, details = list(d.options), dict(d.details)
    if d.source == "Neurofinder" and folder is not None:
        info = neurofinder_info(folder)
        if info.get("rate_hz"):
            opts += ["--frame-interval", f"{1 / info['rate_hz']:.5f}"]
        if info.get("pixel_size_um"):
            opts += ["--pixel-size", f"{info['pixel_size_um']:.4f}"]
        if info.get("indicator") in ("GCaMP6s", "GCaMP6m", "GCaMP6f"):
            details["indicator"] = info["indicator"]
        if info.get("region"):
            details["brain_region"] = info["region"]
    meta = [x for k, v in details.items() for x in ("--meta", f"{k}={v}")]
    out = f"results/{d.name}" + (f"-{Path(target).parent.name}-{Path(target).name}" if folder else "")
    return " ".join(quote(x) for x in ["behavetrack", "analyze", target, *opts, *meta, "-o", out])


def write_about(d: Dataset, folder: Path) -> Path:
    seqs = sequences(folder, d)
    lines = [d.name, "=" * len(d.name), "", f"Source: {d.source}", d.about, "", f"Cite: {d.citation}", "",
             f"Answer key: {d.answer_key}", "", "To analyze (from the project folder):", ""]
    lines += [f"  {analyze_command(d, str(s.relative_to(ROOT) if ROOT in s.parents else s), folder)}"
              for s in seqs] or [f"  {analyze_command(d, str(folder))}"]
    path = folder / "ABOUT.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


# ------------------------------------------------------------------ commands

def cmd_list(a):
    rows = [d for d in DATASETS.values() if not a.source or a.source.lower() in d.source.lower()]
    print(f"{'name':24s} {'mode':8s} {'size':>8s}  about")
    for d in rows:
        size = _mb(d.size_mb * 1e6) if d.size_mb else ("by hand" if d.manual else "~1 GB")
        print(f"{d.name:24s} {d.mode:8s} {size:>8s}  {d.about}")
    print("\nDetails: python tools/fetch_benchmarks.py info NAME    Download: ... get NAME")


def cmd_info(a):
    for name in a.names:
        d = _get(name)
        print(f"{d.name} ({d.source}, {d.mode} mode)\n  {d.about}\n  Answer key: {d.answer_key}\n"
              f"  Cite: {d.citation}")
        print(f"  {'Download by hand: ' + d.manual if d.manual else 'Download: ' + d.url}")
        print(f"  Analyze: {analyze_command(d, f'benchmarks/{d.name}/...')}\n")


def cmd_get(a):
    chosen = [_get(n) for n in a.names]
    manual = [d for d in chosen if d.manual]
    for d in manual:
        print(f"{d.name}: {d.manual}\n  {analyze_command(d, f'benchmarks/{d.name}/VIDEO_FILE')}")
    chosen = [d for d in chosen if not d.manual]
    if not chosen:
        return
    dest = Path(a.dest)
    print("Checking sizes (nothing is downloaded yet)…")
    sizes = {}
    for d in chosen:
        live = remote_size(d.url)
        sizes[d.name] = live
        shown = _mb(live) if live else (f"about {_mb(d.size_mb * 1e6)} (server didn't say)" if d.size_mb
                                        else "unknown (server didn't say)")
        print(f"  {d.name:24s} {shown:>12s}  from {d.url}")
    known = sum(s or (d.size_mb or 0) * 1e6 for d, s in zip(chosen, sizes.values()))
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    print(f"Total about {_mb(known)} (plus about the same again once unpacked). Free space: {_mb(free)}.")
    if known * 2.2 > free:
        sys.exit("Not enough free disk space for this download.")
    if not a.yes:
        if input(f"Download to {dest}/? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing downloaded.")
            return
    failed = 0
    for d in chosen:
        folder = dest / d.name
        folder.mkdir(parents=True, exist_ok=True)
        zip_path = folder / Path(d.url).name
        print(f"{d.name}:")
        try:
            if not zip_path.exists():
                download(d.url, zip_path, sizes[d.name])
            safe_extract(zip_path, folder)
        except (RuntimeError, zipfile.BadZipFile) as e:
            failed += 1
            print(f"  {e}", file=sys.stderr)
            continue
        if not a.keep_zip:
            zip_path.unlink()
        about = write_about(d, folder)
        print(f"  Ready. How to analyze it: {about}")
    if failed:
        sys.exit(f"{failed} download(s) failed; see the messages above.")


def _get(name: str) -> Dataset:
    if name not in DATASETS:
        sys.exit(f"Unknown dataset {name!r}. Run `python tools/fetch_benchmarks.py list`.")
    return DATASETS[name]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list", help="what's available, with sizes")
    pl.add_argument("--source", help="only one source, e.g. neurofinder")
    pl.set_defaults(func=cmd_list)
    pi = sub.add_parser("info", help="details, citation and how to analyze a dataset")
    pi.add_argument("names", nargs="+")
    pi.set_defaults(func=cmd_info)
    pg = sub.add_parser("get", help="download and unpack datasets (asks first)")
    pg.add_argument("names", nargs="+")
    pg.add_argument("--dest", default=str(DEFAULT_DEST), help="where to put them (default benchmarks/)")
    pg.add_argument("--yes", action="store_true", help="don't ask before downloading")
    pg.add_argument("--keep-zip", action="store_true", help="keep the downloaded .zip after unpacking")
    pg.set_defaults(func=cmd_get)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
