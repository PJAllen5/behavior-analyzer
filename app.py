"""Browser interface for behavetrack.

Start it with:   .venv/bin/python -m streamlit run app.py   (or double-click "Start Behavior Tracker.command")
"""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from behavetrack.pipeline import Settings, cell_settings, extract_clip, playback_fps, run

DATA = Path(__file__).parent / "app_data"
UPLOADS, RESULTS = DATA / "uploads", DATA / "results"
for d in (UPLOADS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

BEHAVIOR_INFO = {
    "collective_motion": ("Moving together", "Most individuals head the same way: flocking, schooling, marching."),
    "milling": ("Circling", "Individuals circle a shared centre: fish tornadoes, ant mills."),
    "disordered": ("Scattered movement", "Moving, but in many directions: swarming, foraging, a busy crowd."),
    "resting": ("Resting", "Fewer than 30% of individuals are moving."),
    "too_few": ("Too few visible", "Fewer than 3 individuals in view."),
}
# Reference categorical palette, slots 1-4 in fixed order; gray for "too few".
COLORS = {"collective_motion": "#2a78d6", "milling": "#eb6834", "disordered": "#1baf7a",
          "resting": "#eda100", "too_few": "#a3a29c"}
SERIES = "#2a78d6"
YOLO_CLASSES = ["person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                "bear", "zebra", "giraffe"]
FILE_TYPES = ["mp4", "mov", "avi", "mkv", "m4v", "webm", "tif", "tiff", "nd2", "czi"]

st.set_page_config(page_title="Behavior Tracker", page_icon=":material/timeline:", layout="wide")


# ---------------------------------------------------------------- helpers
def analyses() -> list[str]:
    runs = [p for p in RESULTS.iterdir() if (p / "summary.json").exists()]
    return [p.name for p in sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)]


def _csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()


@st.cache_data(max_entries=20)
def load(name: str, stamp: float) -> dict:  # stamp: re-load when results change
    d = RESULTS / name
    out = {"summary": json.loads((d / "summary.json").read_text())}
    for f in ("frames", "behaviors", "interactions", "tracks", "cells", "msd", "lineage"):
        out[f] = _csv(d / f"{f}.csv")
    return out


def get(name: str) -> dict:
    d = RESULTS / name
    r = load(name, (d / "summary.json").stat().st_mtime)
    return {**r, "dir": d}


def zip_folder(d: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in d.iterdir():
            if f.is_file() and f.suffix in {".csv", ".json"}:
                z.write(f, f.name)
    return buf.getvalue()


def label(b: str) -> str:
    return BEHAVIOR_INFO.get(b, (b, ""))[0]


def source_path(summary: dict) -> Path:
    p = Path(summary["video"])
    return p if p.exists() else UPLOADS / p.name  # folder may have been moved


def timeline_chart(frames: pd.DataFrame, fps: float, minutes: bool) -> alt.Chart:
    k = 60 if minutes else 1
    unit = "minutes" if minutes else "seconds"
    lab = frames["behavior"].to_numpy()
    t = frames["time_s"].to_numpy() / k
    step = 1 / fps / k
    starts = [0] + [i for i in range(1, len(lab)) if lab[i] != lab[i - 1]]
    ends = starts[1:] + [len(lab)]
    segs = pd.DataFrame([{
        "Behavior": label(lab[s]), "start": t[s], "end": t[e - 1] + step,
        "Duration": round(t[e - 1] + step - t[s], 2),
    } for s, e in zip(starts, ends)])
    order = [b for b in COLORS if b in set(lab)]
    return alt.Chart(segs).mark_bar(height=34, cornerRadius=4, stroke="white", strokeWidth=2).encode(
        x=alt.X("start:Q", title=f"Time ({unit})", scale=alt.Scale(nice=False)),
        x2="end:Q",
        color=alt.Color("Behavior:N", title=None,
                        scale=alt.Scale(domain=[label(b) for b in order], range=[COLORS[b] for b in order]),
                        legend=alt.Legend(orient="top", direction="horizontal")),
        tooltip=["Behavior", alt.Tooltip("start:Q", title="Starts at", format=".1f"),
                 alt.Tooltip("end:Q", title="Ends at", format=".1f"),
                 alt.Tooltip("Duration:Q", title=f"Duration ({unit})")],
    ).properties(height=90)


def line_chart(df: pd.DataFrame, x: str, cols: dict[str, str], x_title: str, y_title: str,
               colors: list[str], domain: tuple | None = None, log: bool = False) -> alt.Chart:
    """Lines with a hover crosshair and tooltip. One shared y-axis."""
    wide = df[[x, *cols]].rename(columns=cols)
    long = wide.melt(x, var_name="Measure", value_name="v")
    scale = alt.Scale(type="log") if log else (alt.Scale(domain=domain) if domain else alt.Scale(zero=False))
    xs = alt.Scale(type="log") if log else alt.Scale(nice=False)
    base = alt.Chart(long).encode(
        x=alt.X(f"{x}:Q", title=x_title, scale=xs),
        y=alt.Y("v:Q", title=y_title, scale=scale),
        color=alt.Color("Measure:N", title=None, scale=alt.Scale(domain=list(cols.values()), range=colors),
                        legend=alt.Legend(orient="top", direction="horizontal") if len(cols) > 1 else None),
    )
    hover = alt.selection_point(fields=[x], nearest=True, on="pointerover", empty=False)
    rule = alt.Chart(wide).mark_rule(color="gray").encode(
        x=f"{x}:Q",
        opacity=alt.condition(hover, alt.value(0.5), alt.value(0)),
        tooltip=[alt.Tooltip(f"{x}:Q", title=x_title, format=".2f")]
        + [alt.Tooltip(f"{c}:Q", format=".3g") for c in cols.values()],
    ).add_params(hover)
    pts = base.mark_point(size=64, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))).transform_filter(hover)
    return (base.mark_line(strokeWidth=2) + rule + pts).properties(height=240)


def paths_chart(tracks: pd.DataFrame, L: str, unit: str) -> alt.Chart:
    """Every cell's path, moved so it starts at the centre (shows overall direction)."""
    xs, ys = f"x_{L}", f"y_{L}"
    long_ids = tracks.groupby("id").size()
    t = tracks[tracks["id"].isin(long_ids[long_ids >= 5].index)].sort_values(["id", "frame"]).copy()
    t["dx"] = t[xs] - t.groupby("id")[xs].transform("first")
    t["dy"] = t[ys] - t.groupby("id")[ys].transform("first")
    lim = float(np.nanmax(np.abs(t[["dx", "dy"]].to_numpy()))) * 1.05 if len(t) else 1
    sel = alt.selection_point(fields=["id"], on="pointerover", nearest=True, empty=False)
    sc = alt.Scale(domain=[-lim, lim], nice=False)
    lines = alt.Chart(t).mark_line(strokeWidth=1.5).encode(
        x=alt.X("dx:Q", title=f"Left ← → right ({unit})", scale=sc),
        y=alt.Y("dy:Q", title=f"Up ← → down ({unit})", scale=alt.Scale(domain=[-lim, lim], nice=False, reverse=True)),
        detail="id:N", order="frame:Q",
        color=alt.condition(sel, alt.value("#eb6834"), alt.value(SERIES)),
        opacity=alt.condition(sel, alt.value(1), alt.value(0.35)),
        tooltip=[alt.Tooltip("id:N", title="Cell")],
    ).add_params(sel)
    origin = alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_point(
        shape="cross", size=120, color="gray").encode(x="x:Q", y="y:Q")
    return (lines + origin).properties(height=360, width=360)


def time_label(summary: dict, seconds: float) -> str:
    if summary.get("mode") == "cells":
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.1f} s"


# ---------------------------------------------------------------- sidebar: settings
with st.sidebar:
    st.header("Settings")
    subject = st.segmented_control(
        "What are you filming?", ["Animals or people", "Cells (microscope)"],
        default="Animals or people", required=True)
    cells_mode = subject.startswith("Cells")

    if cells_mode:
        look = st.segmented_control(
            "How do the cells look?", ["Auto", "Bright", "Dark"], default="Auto", required=True,
            help="Bright: cells are lighter than the background (fluorescence, stained nuclei). "
                 "Dark: cells are darker. Auto: anything that differs from the background, "
                 "such as phase contrast.")
        strict = st.slider("Detection", -0.5, 0.5, 0.0, 0.05,
                           help="Move left to find fainter cells, right to ignore faint specks.")
        st.caption("← finds fainter cells · stricter →")
        divisions = st.toggle("Detect cell division", True)
        split = st.toggle("Separate touching cells", True)
        with st.expander("Microscope calibration"):
            st.caption("Leave at 0 to read these from the file. Without them, results are in "
                       "pixels and frames.")
            px = st.number_input("Pixel size (µm per pixel)", 0.0, 100.0, 0.0, 0.01, format="%.4f")
            dt_min = st.number_input("Time between frames (minutes)", 0.0, 1440.0, 0.0, 0.5)
            diam = st.number_input("Typical cell diameter (pixels, 0 = automatic)", 0.0, 500.0, 0.0, 1.0)
            channel = st.number_input("Channel (for multi-channel files)", 0, 10, 0)
        with st.expander("More options"):
            contact = st.slider("Contact distance (cell diameters)", 0.5, 4.0, 1.5, 0.25,
                                help="Two cells closer than this count as touching.")
            make_video = st.toggle("Make annotated video", True)
        settings = cell_settings(
            cell_appearance=look.lower(), threshold_adjust=strict, detect_divisions=divisions,
            split_touching=split, pixel_size_um=px, frame_interval_s=dt_min * 60,
            cell_diameter_px=diam, channel=int(channel), contact_bl=contact, annotate=make_video)
    else:
        det_label = st.segmented_control(
            "What should it look for?", ["Anything that moves", "Animals or people (AI)"],
            default="Anything that moves", required=True,
            help="'Anything that moves' works for any creature, including ants and insects, "
                 "but needs a camera that stays still. The AI option recognises people and "
                 "common animals and copes with a moving camera, but not insects or fish.")
        detector = "motion" if det_label.startswith("Anything") else "yolo"
        classes, sensitivity, min_area, conf = [], 25.0, 15, 0.3
        if detector == "yolo":
            classes = st.multiselect("Which kinds?", YOLO_CLASSES, default=["bird"])
            conf = st.slider("How sure the AI must be", 0.1, 0.9, 0.3, 0.05,
                             help="Lower finds more, but makes more mistakes.")
            st.caption("The first run downloads a small AI model (about 6 MB).")
        else:
            sens = st.slider("Motion sensitivity", 5, 60, 45, 1,
                             help="Higher picks up fainter movement but also more noise "
                                  "(leaves, ripples).")
            sensitivity = 70.0 - sens
            min_area = st.slider("Smallest object size (pixels)", 2, 200, 15,
                                 help="Ignore moving specks smaller than this.")
        with st.expander("More options"):
            contact = st.slider("Interaction distance (body lengths)", 0.5, 6.0, 2.0, 0.5,
                                help="Two individuals closer than this count as interacting.")
            start = st.number_input("Start at second", 0.0, step=1.0)
            max_s = st.number_input("Only analyze this many seconds (0 = all)", 0.0, step=5.0)
            width = st.select_slider("Processing size (width in pixels)",
                                     [480, 640, 960, 1280, 1920], 960,
                                     help="Smaller is faster; larger finds smaller objects.")
            make_video = st.toggle("Make annotated video", True)
        settings = Settings(detector=detector, classes=classes, conf=conf, sensitivity=sensitivity,
                            min_area=min_area, contact_bl=contact, start_s=start,
                            max_seconds=max_s, resize_width=width, annotate=make_video)

# ---------------------------------------------------------------- main
st.title("Behavior tracker")
tab_add, tab_view, tab_search = st.tabs([":material/upload: Add videos", ":material/insights: Results",
                                         ":material/search: Find a behavior"])

with tab_add:
    if cells_mode:
        st.write("Drop in microscope time-lapses (TIFF stacks, .nd2, .czi) or videos, check the "
                 "cell settings on the left, then press **Analyze**.")
    else:
        st.write("Drop in one or more videos, check the settings on the left, then press **Analyze**.")
    files = st.file_uploader("Videos or microscope files", type=FILE_TYPES,
                             accept_multiple_files=True, label_visibility="collapsed")
    if st.button("Analyze", type="primary", icon=":material/play_arrow:", disabled=not files):
        for up in files:
            path = UPLOADS / up.name
            path.write_bytes(up.getbuffer())
            out = RESULTS / path.stem
            if out.exists():
                shutil.rmtree(out)
            st.write(f"**{up.name}**")
            bar = st.progress(0.0, text="Starting…")
            try:
                summ = run(path, out, settings,
                           progress=lambda stage, f: bar.progress(min(f, 1.0), text=f"{stage}… {f:.0%}"))
            except (SystemExit, Exception) as e:  # noqa: BLE001
                bar.empty()
                st.error(str(e), title=f"Could not analyze {up.name}")
                continue
            bar.progress(1.0, text="Done")
            if summ["individuals_tracked"] == 0:
                st.warning("Try moving the Detection slider left, changing how the cells look, or "
                           "setting the cell diameter." if cells_mode else
                           "Try raising motion sensitivity or lowering the smallest object size, "
                           "or switch to the AI option.", title="Nothing was found to track")
            elif cells_mode:
                c = summ.get("cells", {})
                st.success(f"About {summ['typical_visible']} cells per frame, "
                           f"{c.get('divisions', 0)} divisions. Open the **Results** tab.",
                           title="Done")
            else:
                st.success(f"Tracked about {summ['typical_visible']} individuals, found "
                           f"{summ['behavior_segments']} behavior periods and {summ['interactions']} "
                           "interactions. Open the **Results** tab.", title="Done")
            st.session_state["selected"] = path.stem

with tab_view:
    names = analyses()
    if not names:
        st.info("No results yet. Add a video in the first tab.")
    else:
        sel = st.session_state.get("selected")
        name = st.selectbox("Video", names, index=names.index(sel) if sel in names else 0)
        r = get(name)
        s, fr = r["summary"], r["frames"]
        is_cells = s.get("mode") == "cells"
        vid = r["dir"] / "annotated.mp4"

        if is_cells:
            cs = s.get("cells", {})
            L = "um" if cs.get("calibrated") else "px"
            unit = "µm" if L == "um" else "px"
            if not cs.get("calibrated"):
                st.warning("The file has no pixel size, so distances are in pixels. Set it under "
                           "Microscope calibration and analyze again for microns.",
                           title="Not calibrated")
            if not s.get("time_known"):
                st.warning("The file has no time between frames, so 1 frame is counted as 1 second. "
                           "Set it under Microscope calibration and analyze again.",
                           title="Frame timing unknown")
            with st.container(horizontal=True):
                st.metric("Cells (start → end)", f"{cs.get('cells_at_start', '–')} → {cs.get('cells_at_end', '–')}",
                          border=True)
                st.metric("Divisions", cs.get("divisions", 0), border=True)
                st.metric(f"Median speed ({unit}/min)", cs.get(f"median_speed_{L}_per_min", "–"), border=True)
                st.metric("Median straightness", cs.get("median_straightness", "–"), border=True,
                          help="Distance from start to end divided by the path walked. "
                               "1 = a straight line, near 0 = wandering.")
                dbl = cs.get("population_doubling_time_h")
                st.metric("Doubling time", f"{dbl} h" if dbl else "–", border=True,
                          help="How long the cell count takes to double, from the counts over time.")

            left, right = st.columns([3, 2])
            with left:
                if vid.exists():
                    st.video(str(vid), alt="Time-lapse with each tracked cell circled and numbered")
                    st.caption("Daughter cells share their parent's colour. A white ring marks a "
                               "cell that has just divided.")
                else:
                    st.caption("No annotated video for this run.")
            with right:
                st.markdown("**Direction of migration**")
                fx, fy = cs.get("mean_fmi_x", 0) or 0, cs.get("mean_fmi_y", 0) or 0
                st.caption("Forward migration index: share of each path spent heading that way. "
                           "Near 0 means no preferred direction; above about 0.1 is a clear bias.")
                st.markdown(f"Left/right: **{fx:+.2f}** {'(toward the right)' if fx > 0.1 else '(toward the left)' if fx < -0.1 else '(no bias)'}  \n"
                            f"Up/down: **{fy:+.2f}** {'(toward the bottom)' if fy > 0.1 else '(toward the top)' if fy < -0.1 else '(no bias)'}")
                alpha = cs.get("msd_alpha")
                if alpha:
                    kind = ("directed (crawling with purpose)" if alpha > 1.4 else
                            "random (like diffusion)" if alpha > 0.7 else "confined (staying put)")
                    st.markdown("**How cells spread out**")
                    st.markdown(f"MSD exponent **{alpha:.2f}**, which looks **{kind}**.",
                                help="From the mean squared displacement curve below. "
                                     "About 1 = random walk, about 2 = straight-line movement, below 1 = held in place.")

            if not fr.empty:
                a, b = st.columns(2)
                with a:
                    st.markdown("**Paths from a common start**")
                    st.caption("Each line is one cell, shifted to start at the centre. Hover to pick out one cell.")
                    st.altair_chart(paths_chart(r["tracks"], L, unit), width="stretch",
                                    alt="Cell paths drawn from a shared starting point")
                with b:
                    st.markdown("**Number of cells over time**")
                    st.altair_chart(line_chart(fr.assign(time_min=fr["time_s"] / 60), "time_min",
                                               {"n_visible": "Cells"}, "Time (minutes)", "Cells", [SERIES]),
                                    width="stretch", alt="Cell count over time")
                    if not r["msd"].empty:
                        st.markdown("**Mean squared displacement**")
                        st.caption("How far cells get from where they were, versus time. Log scales.")
                        st.altair_chart(line_chart(r["msd"], "lag_min", {f"msd_{L}2": f"MSD ({unit}²)"},
                                                   "Time lag (minutes)", f"MSD ({unit}²)", [SERIES], log=True),
                                        width="stretch", alt="Mean squared displacement versus time lag")
            st.subheader("Data")
            with st.expander(f"Cells ({len(r['cells'])})", expanded=True):
                st.dataframe(r["cells"], hide_index=True, alt="One row per tracked cell")
            with st.expander(f"Divisions ({r['lineage']['parent_id'].nunique() if len(r['lineage']) else 0})"):
                st.dataframe(r["lineage"], hide_index=True, alt="Parent and daughter cells for each division")
            with st.expander(f"Cell contacts ({len(r['interactions'])})"):
                st.dataframe(r["interactions"], hide_index=True, alt="Pairs of cells that touched")
            with st.expander("Every cell, every frame (first 1,000 rows)"):
                st.dataframe(r["tracks"].head(1000), hide_index=True, alt="Per-frame cell positions")
        else:
            with st.container(horizontal=True):
                st.metric("Typical number in view", s.get("typical_visible", s["max_visible_at_once"]),
                          help=f"Most at once: {s['max_visible_at_once']}", border=True)
                st.metric("Track IDs", s["individuals_tracked"], border=True,
                          help="An individual that is lost and found again gets a new ID, "
                               "so this can be higher than the real number of individuals.")
                st.metric("Interactions", s["interactions"], border=True)
                st.metric("Seconds analyzed", s["analyzed_seconds"], border=True)

            left, right = st.columns([3, 2])
            with left:
                if vid.exists():
                    st.video(str(vid), alt="Video with each tracked individual boxed and numbered")
                else:
                    st.caption("No annotated video for this run.")
            with right:
                st.subheader("Time in each behavior")
                tin = s.get("seconds_in_each_behavior", {})
                for k in COLORS:
                    if k in tin:
                        pct = tin[k] / max(s["analyzed_seconds"], 1e-9)
                        st.markdown(
                            f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;"
                            f"background:{COLORS[k]};margin-right:6px'></span>**{label(k)}**: "
                            f"{tin[k]:.1f} s ({pct:.0%})", unsafe_allow_html=True,
                            help=BEHAVIOR_INFO[k][1])

            if not fr.empty:
                st.subheader("Behavior over time")
                st.altair_chart(timeline_chart(fr, s["fps"], False), width="stretch",
                                alt="Timeline of group behavior")
                a, b = st.columns(2)
                with a:
                    st.markdown("**How coordinated is the group?**")
                    st.caption("1 = all heading the same way (alignment) or all circling together (circling).")
                    st.altair_chart(line_chart(fr, "time_s", {"polarization": "Alignment", "rotation": "Circling"},
                                               "Time (seconds)", "Score (0–1)", [SERIES, "#eb6834"], (0, 1)),
                                    width="stretch", alt="Alignment and circling scores over time")
                with b:
                    st.markdown("**How fast are they moving?**")
                    st.caption("Average speed, in body lengths per second.")
                    st.altair_chart(line_chart(fr, "time_s", {"mean_speed_bl_s": "Average speed"},
                                               "Time (seconds)", "Body lengths / s", [SERIES]),
                                    width="stretch", alt="Average speed over time")

            st.subheader("Data")
            with st.expander(f"Behavior periods ({len(r['behaviors'])})", expanded=True):
                st.dataframe(r["behaviors"], hide_index=True, alt="Behavior periods")
            with st.expander(f"Interactions ({len(r['interactions'])})"):
                st.dataframe(r["interactions"], hide_index=True, alt="Close encounters between pairs")
            with st.expander("Every individual, every frame (first 1,000 rows)"):
                st.dataframe(r["tracks"].head(1000), hide_index=True, alt="Per-frame positions")

        with st.container(horizontal=True, vertical_alignment="center"):
            st.download_button("Download all data (.zip)", zip_folder(r["dir"]), icon=":material/download:",
                               file_name=f"{name}_behavior_data.zip", mime="application/zip")
            st.caption(f"Files are also saved in `app_data/results/{name}/`.")

with tab_search:
    names = analyses()
    if not names:
        st.info("No results yet. Add a video in the first tab.")
    else:
        opts = list(BEHAVIOR_INFO)[:4] + ["interaction", "division"]
        names_map = {"interaction": "Two individuals interacting / cells touching",
                     "division": "Cell division"}
        with st.container(horizontal=True, vertical_alignment="bottom"):
            beh = st.selectbox("Behavior", opts, format_func=lambda b: names_map.get(b, label(b)))
            min_s = st.number_input("At least this many seconds", 0.0, value=0.0 if beh == "division" else 1.0,
                                    step=0.5, disabled=beh == "division")
        rows = []
        for n in names:
            r = get(n)
            fps = r["summary"]["fps"]
            if beh == "division":
                lin = r["lineage"]
                if lin.empty:
                    continue
                for p, g in lin.groupby("parent_id"):
                    f0 = int(g["division_frame"].iloc[0])
                    rows.append({"video": n, "parent_id": p, "daughters": ", ".join(map(str, g["daughter_id"])),
                                 "start_frame": max(0, f0 - 4), "end_frame": f0 + 4,
                                 "when": time_label(r["summary"], f0 / fps)})
                continue
            df = r["interactions"] if beh == "interaction" else r["behaviors"]
            if df.empty:
                continue
            if beh != "interaction":
                df = df[df["behavior"] == beh]
            df = df[df["duration_s"] >= min_s]
            for rec in df.to_dict("records"):
                rows.append({"video": n, "when": time_label(r["summary"], rec["start_s"]),
                             "lasts": time_label(r["summary"], rec["duration_s"]), **rec})
        m = pd.DataFrame(rows)
        if len(m):
            m = m.sort_values(["video", "start_frame"]).reset_index(drop=True)
        st.write(f"Found **{len(m)}** matches across **{m['video'].nunique() if len(m) else 0}** video(s).")
        if len(m):
            if beh == "division":
                show = ["video", "when", "parent_id", "daughters"]
            elif beh == "interaction":
                show = ["video", "when", "lasts", "id_a", "id_b", "min_dist_bl", "alignment",
                        "approached_by", "leader"]
            else:
                show = ["video", "when", "lasts", "mean_individuals", "mean_polarization", "mean_rotation"]
            pick = st.dataframe(m[show], hide_index=True, on_select="rerun", selection_mode="single-row",
                                alt="Matching moments")
            st.download_button("Download this list (.csv)", m.to_csv(index=False).encode(),
                               file_name=f"matches_{beh}.csv", icon=":material/download:")
            chosen = pick.selection.rows[0] if pick.selection.rows else 0
            row = m.iloc[chosen]
            st.markdown(f"**Playing match {chosen + 1}:** {row['video']}, {row['when']}  "
                        "_(click a row above to pick another)_")
            r = get(row["video"])
            summ = r["summary"]
            vid = r["dir"] / "annotated.mp4"
            # position in the annotated video, which may play faster than real time
            pfps = playback_fps(summ["fps"])
            first = int(summ.get("start_s", 0) * summ["fps"])
            t0 = max(0.0, (int(row["start_frame"]) - first) / pfps)
            t1 = (int(row["end_frame"]) - first + 1) / pfps
            vcol, bcol = st.columns([3, 2])
            if vid.exists():
                vcol.video(str(vid), start_time=t0, end_time=max(t1, t0 + 1), loop=True,
                           alt="The selected moment in the annotated video")
            src = source_path(summ)
            with bcol:
                if not src.exists():
                    st.caption("The original file is no longer available, so clips can't be cut.")
                else:
                    st.caption("Cut this moment out of the original (with a little extra either side).")
                    clip = r["dir"] / "clips" / f"{beh}_{int(row['start_frame'])}-{int(row['end_frame'])}.mp4"
                    if st.button("Make a clip of this match", icon=":material/content_cut:"):
                        clip.parent.mkdir(exist_ok=True)
                        extract_clip(src, clip, int(row["start_frame"]), int(row["end_frame"]), 0.5,
                                     channel=summ.get("settings", {}).get("channel", 0))
                    if clip.exists():
                        st.download_button("Download clip", clip.read_bytes(), icon=":material/download:",
                                           file_name=f"{row['video']}_{clip.name}", mime="video/mp4")
