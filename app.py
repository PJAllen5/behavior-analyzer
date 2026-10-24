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

from behavetrack.pipeline import Settings, cell_settings, extract_clip, neuron_settings, playback_fps, run

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
    for f in ("frames", "behaviors", "interactions", "tracks", "cells", "msd", "lineage",
              "neurons", "events", "bursts", "traces_dff", "correlations",
              "neurite_frames", "growth_cones", "sholl"):
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
            if f.is_file() and f.suffix in {".csv", ".json", ".png"}:
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


def raster_chart(ev: pd.DataFrame, bursts: pd.DataFrame, n: int) -> alt.Chart:
    """One tick per firing event, one row per neuron; network bursts shaded."""
    row = max(2, min(14, int(420 / max(n, 1))))
    y = alt.Y("neuron_id:Q", title="Neuron", scale=alt.Scale(domain=[0.5, n + 0.5], nice=False, reverse=True),
              axis=alt.Axis(tickMinStep=1, format="d"))
    ticks = alt.Chart(ev).mark_tick(orient="vertical", thickness=1.5, size=row, color=SERIES).encode(
        x=alt.X("onset_s:Q", title="Time (seconds)", scale=alt.Scale(nice=False)), y=y,
        tooltip=[alt.Tooltip("neuron_id:Q", title="Neuron"), alt.Tooltip("onset_s:Q", title="Starts at (s)"),
                 alt.Tooltip("amplitude_dff:Q", title="Size (ΔF/F)", format=".2f"),
                 alt.Tooltip("half_width_s:Q", title="Half-width (s)")])
    layers = [ticks]
    if not bursts.empty:
        b = bursts.assign(end_s=bursts["start_s"] + bursts["duration_s"])
        layers.insert(0, alt.Chart(b).mark_rect(color="#eb6834", opacity=0.18).encode(
            x="start_s:Q", x2="end_s:Q",
            tooltip=[alt.Tooltip("burst:Q", title="Network burst"), alt.Tooltip("start_s:Q", title="Starts at (s)"),
                     alt.Tooltip("neurons:Q", title="Neurons taking part")]))
    return alt.layer(*layers).properties(height=max(160, min(460, row * n + 20)))


def traces_chart(tr: pd.DataFrame, ids: list[int]) -> alt.Chart:
    """ΔF/F of a few neurons, one small panel each."""
    stride = max(1, int(np.ceil(len(tr) / 3000)))
    cols = {f"neuron_{i}": f"Neuron {i}" for i in ids if f"neuron_{i}" in tr}
    d = tr.iloc[::stride][["time_s", *cols]].rename(columns=cols).melt("time_s", var_name="Neuron", value_name="dff")
    return alt.Chart(d).mark_line(strokeWidth=1.2, color=SERIES).encode(
        x=alt.X("time_s:Q", title="Time (seconds)", scale=alt.Scale(nice=False)),
        y=alt.Y("dff:Q", title="ΔF/F"),
        tooltip=["Neuron", alt.Tooltip("time_s:Q", title="Time (s)"), alt.Tooltip("dff:Q", title="ΔF/F", format=".3f")],
    ).properties(height=70).facet(row=alt.Row("Neuron:N", title=None, sort=list(cols.values()),
                                              header=alt.Header(labelAngle=0, labelAlign="left")),
                                  spacing=6).resolve_scale(y="independent")


def per_neuron_chart(df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str) -> alt.Chart:
    """One line per neuron; hover to pick one out."""
    sel = alt.selection_point(fields=["neuron_id"], on="pointerover", nearest=True, empty=False)
    return alt.Chart(df).mark_line(strokeWidth=1.8, point=alt.OverlayMarkDef(size=18)).encode(
        x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(nice=False)),
        y=alt.Y(f"{y}:Q", title=y_title),
        detail="neuron_id:N",
        color=alt.condition(sel, alt.value("#eb6834"), alt.value(SERIES)),
        opacity=alt.condition(sel, alt.value(1), alt.value(0.45)),
        tooltip=[alt.Tooltip("neuron_id:N", title="Neuron"), alt.Tooltip(f"{x}:Q", title=x_title),
                 alt.Tooltip(f"{y}:Q", title=y_title, format=".1f")],
    ).add_params(sel).properties(height=280)


def time_label(summary: dict, seconds: float) -> str:
    if summary.get("mode") == "cells":
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.1f} s"


def show_activity(r: dict):
    s, ns, fr = r["summary"], r["summary"].get("neurons", {}), r["frames"]
    if not s.get("time_known"):
        st.warning("The file has no frame rate, so 1 frame is counted as 1 second. Set it under "
                   "Microscope calibration and analyze again.", title="Frame timing unknown")
    n = ns.get("neurons_found", 0)
    with st.container(horizontal=True):
        st.metric("Neurons", n, border=True,
                  help="Found by their flicker, or by brightness if you chose 'All visible'.")
        st.metric("Firing", f"{ns.get('active_neurons', 0)} of {n}", border=True,
                  help="Neurons with at least one firing event.")
        st.metric("Events per minute", ns.get("mean_events_per_min", "–"), border=True,
                  help="Average over the neurons that fire.")
        st.metric("Network bursts", ns.get("network_bursts", 0), border=True,
                  help=f"Moments when at least {ns.get('burst_needs_neurons', '–')} neurons start firing "
                       f"together. {ns.get('bursts_per_min', 0)} per minute; average gap "
                       f"{ns.get('mean_interburst_interval_s') or '–'} s.")
        sync = ns.get("mean_pairwise_correlation")
        st.metric("Synchrony", f"{sync:.2f}" if sync is not None else "–", border=True,
                  help="Average correlation between the activity of every pair of firing neurons. "
                       "0 = independent, 1 = identical.")
    if not n:
        st.info("No neurons were found in this movie.")
        return
    vid, rois = r["dir"] / "annotated.mp4", r["dir"] / "rois.png"
    left, right = st.columns([3, 2])
    with left:
        if vid.exists():
            st.video(str(vid), alt="Movie with each neuron outlined; outlines light up while it fires")
            st.caption("Each neuron is outlined. Its outline turns yellow while it fires.")
        else:
            st.caption("No annotated video for this run.")
    with right:
        if rois.exists():
            st.image(str(rois), caption="Left: average image. Right: activity map (bright = pixels that "
                                        "flicker together). Numbers are neuron IDs.")
        frac = ns.get("fraction_events_in_bursts")
        if frac is not None:
            st.markdown(f"**{frac:.0%}** of firing events happen during network bursts. "
                        f"Median event size **{ns.get('median_amplitude_dff')} ΔF/F**, lasting about "
                        f"**{ns.get('median_half_width_s')} s** at half height.")
        if ns.get("max_drift_px"):
            st.caption(f"Drift corrected: up to {ns['max_drift_px']} pixels.")
    ev, bu = r["events"], r["bursts"]
    if not ev.empty:
        st.markdown("**Who fires when**")
        st.caption("Each tick is one firing event; each row is one neuron. Shaded bands are network bursts.")
        st.altair_chart(raster_chart(ev, bu, n), width="stretch",
                        alt="Raster plot of firing events per neuron over time")
    if not fr.empty:
        st.markdown("**Share of neurons firing**")
        st.altair_chart(line_chart(fr, "time_s", {"fraction_firing": "Share firing"}, "Time (seconds)",
                                   "Share of neurons", [SERIES], (0, 1)),
                        width="stretch", alt="Share of neurons firing over time")
    tr, nn = r["traces_dff"], r["neurons"]
    if not tr.empty and not nn.empty:
        st.markdown("**Activity of single neurons (ΔF/F)**")
        top = nn.sort_values("events", ascending=False)["neuron_id"].head(5).tolist()
        pick = st.multiselect("Neurons to show", nn["neuron_id"].tolist(), default=top,
                              max_selections=12, key=f"tr_{r['dir'].name}")
        if pick:
            st.altair_chart(traces_chart(tr, pick), width="stretch", alt="ΔF/F traces of chosen neurons")
    st.subheader("Data")
    with st.expander(f"Neurons ({len(nn)})", expanded=True):
        st.dataframe(nn, hide_index=True, alt="One row per neuron")
    with st.expander(f"Firing events ({len(ev)})"):
        st.dataframe(ev, hide_index=True, alt="One row per firing event")
    with st.expander(f"Network bursts ({len(bu)})"):
        st.dataframe(bu, hide_index=True, alt="One row per network burst")
    with st.expander("Pairs of neurons: distance and correlation"):
        st.dataframe(r["correlations"], hide_index=True, alt="Correlation between each pair of neurons")


def show_neurites(r: dict):
    s, ns = r["summary"], r["summary"].get("neurons", {})
    L = "um" if ns.get("calibrated") else "px"
    unit = "µm" if L == "um" else "px"
    if not ns.get("calibrated"):
        st.warning("The file has no pixel size, so lengths are in pixels. Set it under Microscope "
                   "calibration and analyze again for microns.", title="Not calibrated")
    if not s.get("time_known") and ns.get("frames", 0) > 1:
        st.warning("The file has no time between frames, so growth rates are per frame-hour and not "
                   "meaningful. Set it under Microscope calibration and analyze again.",
                   title="Frame timing unknown")
    with st.container(horizontal=True):
        st.metric("Neurons", ns.get("neurons_found", 0), border=True)
        st.metric(f"Neurite length per neuron ({unit})",
                  f"{ns.get(f'mean_length_per_neuron_{L}_start', '–')} → {ns.get(f'mean_length_per_neuron_{L}_end', '–')}",
                  border=True, help="Total length of all neurites per neuron, first frame → last frame.")
        st.metric(f"Longest neurite ({unit})", ns.get(f"mean_longest_neurite_{L}", "–"), border=True,
                  help="Average over neurons of the longest path from the cell body to a tip, in the last frame.")
        st.metric("Branch points per neuron", ns.get("mean_branch_points", "–"), border=True)
        g = ns.get(f"mean_growth_rate_{L}_per_h")
        st.metric(f"Growth rate ({unit}/h)", g if g is not None else "–", border=True,
                  help="How fast each neuron's total neurite length grows, averaged over neurons.")
    if not ns.get("neurons_found"):
        st.info("No neurons were found in this file.")
        return
    vid, pic = r["dir"] / "annotated.mp4", r["dir"] / "neurites.png"
    left, right = st.columns([3, 2])
    with left:
        if vid.exists():
            st.video(str(vid), alt="Time-lapse with traced neurites coloured by neuron")
            st.caption("Neurites are coloured by the neuron they belong to; gray pieces aren't joined to "
                       "any cell body. White circles are growth cones.")
        elif pic.exists():
            st.image(str(pic), caption="Last frame, with traced neurites coloured by neuron.")
    with right:
        if vid.exists() and pic.exists():
            st.image(str(pic), caption="Last frame, with traced neurites coloured by neuron.")
        st.markdown(f"**Growth cones:** {ns.get('growth_cones_followed', 0)} followed, "
                    f"{ns.get('growth_cones_extending', 0)} extending, "
                    f"{ns.get('growth_cones_retracting', 0)} pulling back. Average extension "
                    f"**{ns.get(f'mean_cone_extension_{L}_per_h') or '–'} {unit}/h**.")
        st.markdown(f"Primary neurites per neuron: **{ns.get('mean_primary_neurites', '–')}**, "
                    f"tips per neuron: **{ns.get('mean_tips', '–')}**.")
    nf, sh = r["neurite_frames"], r["sholl"]
    timed = bool(s.get("time_known"))
    xcol, xt = ("time_h", "Time (hours)") if timed else ("frame", "Frame")
    a, b = st.columns(2)
    if not nf.empty and nf["frame"].nunique() > 1:
        with a:
            st.markdown("**Neurite length over time**")
            st.caption("One line per neuron. Hover to pick one out.")
            st.altair_chart(per_neuron_chart(nf, xcol, f"total_length_{L}", xt, f"Total neurite length ({unit})"),
                            width="stretch", alt="Total neurite length of each neuron over time")
        with b:
            st.markdown("**Longest neurite over time**")
            st.caption("Path from the cell body to the farthest tip.")
            st.altair_chart(per_neuron_chart(nf, xcol, f"longest_neurite_{L}", xt, f"Longest neurite ({unit})"),
                            width="stretch", alt="Longest neurite of each neuron over time")
    if not sh.empty:
        with (b if nf.empty or nf["frame"].nunique() <= 1 else a):
            st.markdown("**Sholl analysis (last frame)**")
            st.caption("How many neurites cross a ring at each distance from the cell body. "
                       "More crossings = more branched.")
            dcol = f"distance_from_soma_edge_{L}"
            mean = sh.groupby(dcol, as_index=False)["crossings"].mean()
            st.altair_chart(line_chart(mean, dcol, {"crossings": "Average crossings"},
                                       f"Distance from cell body ({unit})", "Crossings", [SERIES]),
                            width="stretch", alt="Average Sholl crossings versus distance from the cell body")
    st.subheader("Data")
    with st.expander(f"Neurons ({len(r['neurons'])})", expanded=True):
        st.dataframe(r["neurons"], hide_index=True, alt="One row per neuron")
    with st.expander(f"Growth cones ({len(r['growth_cones'])})"):
        st.dataframe(r["growth_cones"], hide_index=True, alt="One row per growth cone")
    with st.expander("Every neuron, every frame"):
        st.dataframe(nf, hide_index=True, alt="Neurite measures per neuron per frame")
    with st.expander("Sholl analysis"):
        st.dataframe(sh, hide_index=True, alt="Sholl crossings per neuron per ring")


# ---------------------------------------------------------------- sidebar: settings
with st.sidebar:
    st.header("Settings")
    subject = st.segmented_control(
        "What are you filming?", ["Animals or people", "Cells (microscope)", "Neurons (microscope)"],
        default="Animals or people", required=True)
    cells_mode = subject.startswith("Cells")
    neuron_mode = subject.startswith("Neurons")

    if neuron_mode:
        what = st.segmented_control(
            "What should it measure?", ["Firing activity", "Neurite growth"], default="Firing activity",
            required=True,
            help="Firing activity: calcium or voltage imaging (GCaMP, Fluo-4, …). Finds each neuron and "
                 "measures when it fires, how strongly, and which neurons fire together. "
                 "Neurite growth: traces axons and dendrites and measures their length, branching "
                 "and growth over a time-lapse.")
        if what == "Firing activity":
            find_by = st.segmented_control(
                "Which neurons?", ["Ones that fire", "All visible"], default="Ones that fire", required=True,
                help="Ones that fire: finds neurons by their flicker. All visible: finds bright cell "
                     "bodies, so neurons that stay silent are counted too (needs a dye that shows "
                     "resting neurons).")
            strict = st.slider("Detection", -0.5, 0.5, 0.0, 0.05,
                               help="Move left to find fainter neurons, right to ignore weak ones.")
            st.caption("← finds fainter neurons · stricter →")
            ev_sd = st.slider("Firing threshold (noise levels)", 2.0, 8.0, 3.5, 0.5,
                              help="A firing event must rise this many times the neuron's own noise "
                                   "above its resting level. Lower finds smaller events but more false ones.")
            drift = st.toggle("Correct drift", True,
                              help="Line every frame up with the start of the movie, in case the dish moved.")
            with st.expander("Microscope calibration"):
                st.caption("Leave at 0 to read these from the file.")
                px = st.number_input("Pixel size (µm per pixel)", 0.0, 100.0, 0.0, 0.01, format="%.4f")
                rate = st.number_input("Frame rate (frames per second)", 0.0, 10000.0, 0.0, 1.0)
                diam = st.number_input("Typical neuron diameter (pixels, 0 = automatic)", 0.0, 500.0, 0.0, 1.0)
                channel = st.number_input("Channel (for multi-channel files)", 0, 10, 0)
            with st.expander("More options"):
                neuropil = st.slider("Background haze to subtract", 0.0, 1.0, 0.7, 0.05,
                                     help="Light from out-of-focus cells and processes (neuropil) leaks into "
                                          "each neuron. This much of the haze around each neuron is subtracted. "
                                          "0.7 is the usual choice.")
                min_ev = st.number_input("Smallest event (ΔF/F)", 0.0, 5.0, 0.05, 0.01,
                                         help="Ignore events smaller than this. Use 0 for voltage dyes, "
                                              "whose signals are small.")
                burst_frac = st.slider("Network burst: share of active neurons", 0.1, 0.9, 0.3, 0.05,
                                       help="At least this share of the active neurons must start firing "
                                            "together to count as a network burst.")
                burst_win = st.number_input("Network burst: within this many seconds", 0.05, 5.0, 0.5, 0.05)
                base_win = st.number_input("Resting level window (seconds)", 5.0, 600.0, 60.0, 5.0,
                                           help="The resting level follows changes slower than this, such as "
                                                "fading (bleaching).")
                make_video = st.toggle("Make annotated video", True)
            settings = neuron_settings(
                neural_analysis="activity", find_neurons_by="activity" if find_by == "Ones that fire" else "brightness",
                threshold_adjust=strict, event_threshold_sd=ev_sd, correct_drift=drift,
                pixel_size_um=px, frame_interval_s=1 / rate if rate else 0.0, cell_diameter_px=diam,
                channel=int(channel), neuropil_factor=neuropil, min_event_dff=min_ev,
                burst_min_fraction=burst_frac, burst_window_s=burst_win, baseline_window_s=base_win,
                annotate=make_video)
        else:
            look = st.segmented_control(
                "How do the neurites look?", ["Bright", "Dark", "Auto"], default="Bright", required=True,
                help="Bright: lighter than the background (fluorescent labels such as GFP, βIII-tubulin, MAP2). "
                     "Dark: darker than the background (brightfield). Auto: anything that differs "
                     "from the background, such as phase contrast.")
            strict = st.slider("Detection", -0.5, 0.5, 0.0, 0.05,
                               help="Move left to trace fainter neurites, right to ignore faint lines and noise.")
            st.caption("← traces fainter neurites · stricter →")
            width = st.number_input("Neurite thickness (pixels)", 1.0, 30.0, 3.0, 0.5,
                                    help="Typical width of a neurite in the image.")
            with st.expander("Microscope calibration"):
                st.caption("Leave at 0 to read these from the file. Without them, results are in "
                           "pixels and frames.")
                px = st.number_input("Pixel size (µm per pixel)", 0.0, 100.0, 0.0, 0.01, format="%.4f")
                dt_min = st.number_input("Time between frames (minutes)", 0.0, 1440.0, 0.0, 0.5)
                diam = st.number_input("Cell body diameter (pixels, 0 = automatic)", 0.0, 500.0, 0.0, 1.0)
                channel = st.number_input("Channel (for multi-channel files)", 0, 10, 0)
            with st.expander("More options"):
                step = st.number_input("Analyze every Nth frame", 1, 100, 1,
                                       help="Higher is faster for long time-lapses.")
                sholl = st.number_input("Sholl ring spacing (µm)", 1.0, 200.0, 10.0, 1.0,
                                        help="Rings around each cell body for Sholl analysis (pixels if "
                                             "the file has no pixel size).")
                make_video = st.toggle("Make annotated video", True)
            settings = neuron_settings(
                neural_analysis="neurites", cell_appearance=look.lower(), threshold_adjust=strict,
                neurite_width_px=width, pixel_size_um=px, frame_interval_s=dt_min * 60,
                cell_diameter_px=diam, channel=int(channel), frame_step=int(step),
                sholl_step_um=sholl, annotate=make_video)
    elif cells_mode:
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
    if neuron_mode:
        st.write("Drop in calcium-imaging movies or neuron time-lapses (TIFF stacks, .nd2, .czi) or "
                 "videos, check the neuron settings on the left, then press **Analyze**.")
    elif cells_mode:
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
            if neuron_mode:
                ns = summ.get("neurons", {})
                if not ns.get("neurons_found"):
                    st.warning("Try moving the Detection slider left or setting the neuron diameter."
                               if settings.neural_analysis == "activity" else
                               "Try moving the Detection slider left, changing how the neurites look, "
                               "or setting the cell body diameter.", title="No neurons were found")
                elif settings.neural_analysis == "activity":
                    st.success(f"{ns['neurons_found']} neurons, {ns.get('active_neurons', 0)} firing, "
                               f"{ns.get('network_bursts', 0)} network bursts. Open the **Results** tab.",
                               title="Done")
                else:
                    st.success(f"{ns['neurons_found']} neurons traced. Open the **Results** tab.", title="Done")
            elif summ["individuals_tracked"] == 0:
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

        if s.get("mode") == "neurons":
            (show_activity if s.get("neural_analysis") == "activity" else show_neurites)(r)
        elif is_cells:
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
        opts = list(BEHAVIOR_INFO)[:4] + ["interaction", "division", "network_burst"]
        names_map = {"interaction": "Two individuals interacting / cells touching",
                     "division": "Cell division", "network_burst": "Network burst (neurons firing together)"}
        with st.container(horizontal=True, vertical_alignment="bottom"):
            beh = st.selectbox("Behavior", opts, format_func=lambda b: names_map.get(b, label(b)))
            min_s = st.number_input("At least this many seconds", 0.0, value=0.0 if beh in ("division", "network_burst") else 1.0,
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
            if beh == "network_burst":
                for rec in r["bursts"].to_dict("records"):
                    if rec["duration_s"] >= min_s:
                        rows.append({"video": n, "when": time_label(r["summary"], rec["start_s"]),
                                     "lasts": time_label(r["summary"], rec["duration_s"]), **rec})
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
            elif beh == "network_burst":
                show = ["video", "when", "lasts", "neurons", "fraction_of_active", "mean_amplitude_dff"]
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
