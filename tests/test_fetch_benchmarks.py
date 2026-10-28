"""tools/fetch_benchmarks.py, tested against a small local web server (no internet needed)."""
from __future__ import annotations

import argparse
import shlex
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import fetch_benchmarks as fb
from behavetrack import cli, metadata


# ------------------------------------------------------------------ a local server with resume support

class _Handler(BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, body: bool):
        data = self.files.get(self.path)
        if data is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        if rng:
            start, _, end = rng.removeprefix("bytes=").partition("-")
            start, end = int(start), int(end) if end else len(data) - 1
            chunk = data[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            chunk = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        if body:
            self.wfile.write(chunk)

    def do_GET(self):
        self._send(True)

    def do_HEAD(self):
        self._send(False)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", _Handler.files
    srv.shutdown()
    _Handler.files = {}


def _ctc_zip(path: Path) -> bytes:
    """A miniature Cell Tracking Challenge zip: two image sequences plus answer keys."""
    with zipfile.ZipFile(path, "w") as z:
        for seq in ("01", "02"):
            for t in range(3):
                z.writestr(f"Mini/{seq}/t{t:03d}.tif", b"fake")
            z.writestr(f"Mini/{seq}_GT/TRA/man_track.txt", "1 0 2 0\n")
    return path.read_bytes()


# ------------------------------------------------------------------ the catalog

def test_catalog_is_valid():
    assert len(fb.DATASETS) >= 30
    for d in fb.DATASETS.values():
        assert (d.url is None) == bool(d.manual), d.name
        if d.url:
            assert d.url.startswith("https://"), d.name
        metadata.clean(d.details)  # every experiment detail is a real field with a valid value
        argv = shlex.split(fb.analyze_command(d, "x"))[2:]
        p = argparse.ArgumentParser()
        p.add_argument("video")
        p.add_argument("-o")
        cli._add_common(p)
        assert cli._settings(p.parse_args(argv)).mode == d.mode, d.name


def test_list_and_info_run(capsys):
    fb.main(["list"])
    fb.main(["info", "DIC-C2DH-HeLa", "idtracker-ants-14"])
    out = capsys.readouterr().out
    assert "DIC-C2DH-HeLa" in out and "--pixel-size 0.19" in out and "by hand" in out.lower()


# ------------------------------------------------------------------ downloading and unpacking

def test_remote_size_and_resumed_download(server, tmp_path):
    base, files = server
    files["/big.bin"] = bytes(range(256)) * 4000
    assert fb.remote_size(f"{base}/big.bin") == len(files["/big.bin"])
    target = tmp_path / "big.bin"
    part = tmp_path / "big.bin.part"
    part.write_bytes(files["/big.bin"][:1000])  # an interrupted earlier download
    fb.download(f"{base}/big.bin", target, len(files["/big.bin"]), progress=False)
    assert target.read_bytes() == files["/big.bin"] and not part.exists()


def test_unreachable_server_gives_advice(tmp_path):
    with pytest.raises(RuntimeError, match="try again later"):
        fb.download("http://127.0.0.1:9/nothing.zip", tmp_path / "x.zip", timeout=2, progress=False)


def test_safe_extract_refuses_escaping_paths(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../outside.txt", "nope")
    with pytest.raises(RuntimeError, match="outside"):
        fb.safe_extract(bad, tmp_path / "dest")
    assert not (tmp_path / "outside.txt").exists()


def test_get_asks_first_and_downloads_nothing_on_no(server, tmp_path, monkeypatch, capsys):
    base, files = server
    files["/Mini.zip"] = _ctc_zip(tmp_path / "src.zip")
    monkeypatch.setitem(fb.DATASETS, "DIC-C2DH-HeLa",
                        fb.DATASETS["DIC-C2DH-HeLa"].__class__(**{**fb.DATASETS["DIC-C2DH-HeLa"].__dict__,
                                                                  "url": f"{base}/Mini.zip"}))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    fb.main(["get", "DIC-C2DH-HeLa", "--dest", str(tmp_path / "bench")])
    assert "Nothing downloaded" in capsys.readouterr().out
    assert not list((tmp_path / "bench").rglob("*.zip"))


def test_get_downloads_unpacks_and_explains(server, tmp_path, monkeypatch):
    base, files = server
    files["/Mini.zip"] = _ctc_zip(tmp_path / "src.zip")
    d = fb.DATASETS["Fluo-N2DL-HeLa"]
    monkeypatch.setitem(fb.DATASETS, d.name, d.__class__(**{**d.__dict__, "url": f"{base}/Mini.zip"}))
    dest = tmp_path / "bench"
    fb.main(["get", d.name, "--dest", str(dest), "--yes"])
    folder = dest / d.name
    assert (folder / "Mini" / "01" / "t000.tif").exists()
    assert not list(folder.glob("*.zip"))  # removed after unpacking
    about = (folder / "ABOUT.txt").read_text()
    assert about.count("behavetrack analyze") == 2  # one per image sequence (01 and 02)
    assert "--pixel-size 0.645" in about and "--frame-interval 1800" in about
    assert "cell_line=HeLa" in about and "man_track.txt" in about


def test_neurofinder_command_uses_dataset_info(tmp_path):
    folder = tmp_path / "neurofinder.00.00"
    (folder / "neurofinder.00.00" / "images").mkdir(parents=True)
    (folder / "neurofinder.00.00" / "info.json").write_text(
        '{"rate-hz": 8, "indicator": "GCaMP6s", "pixels-per-micron": 1.15, "region": "vS1"}')
    d = fb.DATASETS["neurofinder.00.00"]
    cmd = fb.analyze_command(d, "x", folder)
    assert "--frame-interval 0.12500" in cmd and "indicator=GCaMP6s" in cmd
    assert "--pixel-size 0.8696" in cmd and "brain_region=vS1" in cmd
