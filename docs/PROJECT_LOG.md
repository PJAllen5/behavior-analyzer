# Project log

A record of every design step and decision in Behavior Tracker (`behavetrack`), with the
reasoning and the evidence behind it. Newest work is at the bottom of the timeline. Add an
entry when you make a decision someone might later ask "why?" about (see the end of this file).

Repositories: [behavior-analyzer](https://github.com/PJAllen5/behavior-analyzer) (this one) and
[complexity-engine](https://github.com/PJAllen5/complexity-engine).

## Timeline

| Date | Step | Where |
|---|---|---|
| 2026-10-22 | First prototype: track moving things in video and export behavior data | before the repository existed |
| 2026-10-22 | Browser app so videos can be analyzed without typing commands | before the repository existed |
| 2026-10-22 | Project moved to a permanent folder on the Desktop | — |
| 2026-10-22 | Cells mode: microscope files, cell finding, division and lineage | before the repository existed |
| 2026-10-23 | Put on GitHub (private), MIT license | first commit, `6ee25f0` |
| 2026-10-23 | Neurons mode: calcium-imaging activity and neurite growth (built in VS Code) | PR #1 |
| 2026-10-23 | Developer setup: one source of settings, installable package, tests, VS Code | PR #2 |
| 2026-10-23 | The app only accepts connections from this computer | PR #3 |
| 2026-10-24 | Connected to complexity-engine: export command, four behavior profiles | PR #4; complexity-engine PR #71 |
| 2026-10-24 | Both repositories made public; MIT license added to complexity-engine | — |
| 2026-10-27 | Data checks and experiment details, with cited typical values | PR #5 |
| 2026-10-27 | Download script for real benchmark data | PR #6 |
| 2026-10-27 | First test on real data (Neurofinder): the built-in neuron finder scored F1 0.21 | — |
| 2026-10-27 | Cellpose added as a neuron finder (F1 0.54); benchmark scoring command | PR #7 |
| 2026-10-27 | Drift correction fixed after it misbehaved on real data | PR #8 |
| 2026-10-28 | This log | — |

---

## 1. First prototype (2026-10-22)

**Goal.** Analyze videos of collective behavior (birds flocking, ants moving, people interacting)
and write data files describing the movement and interactions of each individual.

**Decisions**

- **Python with OpenCV, NumPy, SciPy and pandas.** Widely used, free, runs on a Mac without special
  hardware.
- **Default detector: motion (background subtraction, OpenCV's MOG2).** Works on any creature,
  including insects, with no trained model. *Alternative kept as an option:* a YOLO object detector
  (`--detector yolo`) for people and common animals, which also copes with moving cameras.
  *Trade-off:* the motion detector needs a still camera, and animals that stop moving fade out.
- **Our own tracker** (constant-velocity prediction plus optimal matching on distance). Simple,
  dependency-free, and works the same for every detector.
- **Distances in body lengths**, measured from the typical object size. The same thresholds then
  work for ants, birds and people at any zoom.
- **Group behavior labels from two order parameters:** polarization (how aligned headings are) and
  rotation (how much the group circles its centre). Labels: moving together, circling (milling),
  scattered, resting, too few. Thresholds 0.65 and 0.45.
- **Interactions** are pairs closer than 2 body lengths, with duration, closest distance, alignment,
  who approached, and who led.
- **Output files:** `tracks.csv`, `frames.csv`, `behaviors.csv`, `interactions.csv`,
  `summary.json`, and an annotated video.
- **A synthetic test video with known behaviors** (swarming, then flocking, then milling, then
  resting), so results could be checked against a known answer.

**Evidence.** The first test video revealed that the simulated birds merged into one blob, which
was a flaw in the test video, not the tracker; adding spacing between birds fixed it. The milling
phase was first labelled "flocking" because the simulated birds travelled the circle as one tight
group; giving each bird its own speed made true milling. After that, all four behaviors were found
in order and all 40 birds were tracked.

## 2. Browser app (2026-10-22)

**Decision.** A Streamlit app that runs locally: drag in videos, see results, search for a
behavior and cut clips. *Why Streamlit:* a browser interface in pure Python, with file upload,
video playback and charts built in.

**Details decided along the way**

- Videos are written as H.264 so browsers can play them (OpenCV's default format can't be played).
- The first frames are skipped by the motion detector while it learns the background. It had been
  reporting 54 birds in a 40-bird video.
- "Typical number in view" is shown rather than "individuals tracked", since an animal that is lost
  and found again gets a new ID.
- Chart colors follow a tested colorblind-safe palette.
- A double-click launcher (`Start Behavior Tracker.command`) for people who don't use a terminal.

## 3. Cells mode (2026-10-22)

**Goal.** Track cells in microscope time-lapses, including cell division.

**Decisions**

- **Read microscope formats directly:** TIFF stacks (including ImageJ and OME-TIFF), Nikon `.nd2`,
  Zeiss `.czi`, and folders of images, taking pixel size and frame interval from the file.
- **Find cells by appearance, not motion**, since cells can sit still: background subtraction,
  automatic threshold (Otsu), and watershed to split touching cells.
- **Detect division** when one tracked cell becomes two smaller, nearby cells, and record the family
  tree (`lineage.csv`).
- **Cell measures used in the field:** speed, straightness, forward migration index, mean squared
  displacement and its exponent, doubling time.
- **A synthetic time-lapse with an answer key** (`tools/make_test_cells.py`) and a checker
  (`tools/check_cells.py`).

**Evidence.** 99% of cells found with no false detections; speed within 1% of the truth; 18 of 21
divisions found. A bug that dropped divisions in the first frames (the parent was filtered out as
too short) was found by the checker and fixed.

## 4. Storage and GitHub (2026-10-22 to 10-23)

- The project was first created in a temporary folder and moved to the Desktop to keep it.
- **Repository contents:** code, launcher, tools, README. **Kept out:** installed libraries,
  uploaded videos, results, and test media (which the tools regenerate).
- **Private repository first, MIT license.** *Why MIT:* lets others use and change the code as
  long as the credit stays; the most common open-source choice.

## 5. Neurons mode (2026-10-23, PR #1)

Built in a separate VS Code session and merged. Two analyses: **firing activity** in
calcium-imaging movies (drift correction, neuron finding, ΔF/F, events, network bursts,
synchrony) and **neurite growth** (tracing, cell bodies, length, branching, Sholl analysis, growth
cones), each with a synthetic answer-key movie.

## 6. Developer setup (2026-10-23, PR #2)

**Problem.** Default settings were written separately in three places (the settings class, the
command line and the app), and they had started to drift apart. In cells mode, `--contact-bl` was
being ignored.

**Decisions**

- **One source of defaults** (`behavetrack/settings.py`). The command line and the app read from it;
  a test fails if either disagrees.
- **Installable package** (`pyproject.toml`) with the `behavetrack` command and `behavetrack app`.
  The app moved into the package.
- **Two test tiers:** quick tests (seconds) while editing; the full suite (about a minute) before
  pushing, including accuracy against every answer key.
- **VS Code run buttons and tasks.**

**Problem found.** The Desktop is synced to iCloud Drive, which uploaded the 1.6 GB library folder
and marked its files hidden, and Python 3.14 then refused to load the installed package.
**Decision:** keep libraries in `.venv.nosync` (which iCloud skips) with `.venv` as a shortcut to
it. The launcher sets this up automatically. iCloud also left duplicate files ending in " 2"; they
were moved to the Trash. The working copy for day-to-day use is now `~/behavior-tracker`, outside
iCloud.

## 7. Privacy: local-only app (2026-10-23, PR #3)

Streamlit listens on the whole network by default, so anyone on the same Wi-Fi could have opened
the app and seen uploaded videos. **Decision:** bind to `localhost` only. A test checks it.

## 8. Connecting to complexity-engine (2026-10-24, PR #4 and complexity-engine PR #71)

**Question.** How to use the existing complexity-engine repository (complexity, cybernetics and
information-theory analyses, with a cited bibliography) to inform what the tracker measures.

**Options.** Merge the two repositories into one, or connect them.
**Decision: connect them.** *Why:* the engine is a general, well-tested project with its own
history; the tracker produces data and the engine analyzes it. Literature stays in one place.

**What was built**

- complexity-engine became installable, with `complexity-analysis` and `complexity-network`
  commands.
- **Four behavior profiles** in the engine, each with interpretation text written for its field and
  cited (15 papers added): collective behavior, neural activity, animal social networks, neuronal
  functional networks.
- `behavetrack export-complexity` writes the engine's input files and can run it; the app gained a
  "Run complexity analysis" button.
- **Each video is its own network** in network exports, because track IDs in different videos are
  different individuals.
- Neurite-growth results are not exported: too few time points for entropy or DFA.

**Checked against sources.** One draft claim was corrected: schooling fish swarm when slow and
either mill or school as they speed up, so faster does not simply mean more aligned.

## 9. Repositories made public (2026-10-24)

- Both repositories' full history was scanned for passwords, keys, tokens and local paths: none.
- MIT license added to complexity-engine, matching the tracker.
- The "private repository" install notes were removed.

## 10. Data checks and experiment details (2026-10-27, PR #5)

**Question.** How to make sure the data captured is relevant to its field.

**Decisions**

- **Plausibility checks against published values, each cited:** cell speed (World Cell Race),
  cell and neuron size, doubling time, calcium-indicator kinetics versus frame rate, plus setup
  checks (calibration, identity breaks, movement between frames, track length). 18 checks at the
  time; 19 after the drift check was added in PR #8.
- **Checks advise; they never block or change results.**
- **Thresholds, explanations and sources live in one file** (`behavetrack/reference/checks.yaml`),
  so a domain expert can adjust them without touching code. A test makes every citation match the
  bibliography (`reference/references.bib`).
- **Experiment details** follow two reporting standards: REMBI (biological images) and MIACME (cell
  migration). Some checks use them: the indicator sets the frame rate needed, and the kind of cells
  decides which published speeds apply.
- **No neurite growth-rate range**, because no reliable published figure was found. Neurite checks
  are about setup only.

**Evidence.** On the test data the checks flagged two real flaws in the test files themselves
(simulated cells dividing every 7 hours; a neuron movie too short for its resting-level window).

## 11. Real benchmark data (2026-10-27, PR #6)

**Decision.** Test on public datasets with expert answer keys, rather than only on our own
synthetic movies: the Cell Tracking Challenge (cells), Neurofinder (calcium imaging) and
idtracker.ai (animal groups).

- `tools/fetch_benchmarks.py` lists 35 datasets, shows sizes and free space, **asks before
  downloading**, resumes interrupted downloads, and unpacks safely.
- Each dataset gets an `ABOUT.txt` with its source, citation, answer key and the exact analysis
  command, including calibration and experiment details from the source.
- idtracker.ai videos are only on Google Drive, so they are downloaded by hand.
- The Cell Tracking Challenge data server was not responding during testing; the script reports
  this and resumes later.

## 12. First real-data test (2026-10-27)

Neurofinder 00.00: a two-photon GCaMP6s movie from an awake mouse's whisker cortex (vS1), 3,024
frames at 7 Hz, 0.87 µm pixels, 330 hand-labelled neurons.

**Result.** The built-in neuron finder found 29% of the labelled neurons, and only 16% of what it
found was labelled (F1 0.21). Across four settings the best was F1 0.24. Neurons were split into
small pieces (5.3 µm against 8.0 µm labels), and drift correction reported shifts of 53 pixels in
a movie that was already motion-corrected.

**Lesson.** All the data checks passed on this run. Plausibility checks catch setup mistakes;
accuracy needs answer keys. Both are needed.

## 13. Cellpose neuron finder and scoring (2026-10-27, PR #7)

**Options considered:** Suite2p, CaImAn, Cellpose. **Decision: Cellpose.** *Why:* widely used and
published, and **BSD-licensed**, compatible with this project's MIT license. Suite2p (GPL-3) and
CaImAn (GPL-2) are not.

**Model choice.** Cellpose-SAM (about 1.2 GB) rather than the older 27 MB model, chosen for
accuracy; it is downloaded once to `~/.cellpose/models`, never into the repository.

**How it is used.** Cellpose outlines cell bodies on the enhanced mean image (background removed,
contrast stretched) and estimates cell size itself. In the experiment, a size hint scored lower,
and the enhanced mean beat the plain mean and the maximum projection. The rest of the analysis is
unchanged. The built-in finder stays the default; Cellpose is an option and an install extra.

**Scoring.** `behavetrack score` reports recall, precision and F1 against Neurofinder's labels by
matching centres within 5 pixels. It is not Neurofinder's official evaluation, so it is for
comparing our own runs.

| Neurofinder 00.00 | Recall | Precision | F1 |
|---|---|---|---|
| Built-in, original settings | 29% | 16% | 0.21 |
| Built-in, best settings | 18% | 37% | 0.24 |
| Cellpose, without drift correction | 45% | 67% | 0.54 |

## 14. Drift correction fix (2026-10-27, PR #8)

Done in a separate session. **Cause:** 86 mostly blank frames (a gap in the recording) produced
wild shift estimates that the old filter accepted. **Fix:** estimates are kept only when their
correlation peak is strong and they stay close to recent frames; rejected frames take their
neighbours' drift; a real sudden jump is accepted once enough strong frames agree. A new data check
warns when a correction looks wrong.

| Cellpose on Neurofinder 00.00 | Largest shift | F1 |
|---|---|---|
| Old drift correction | 53 px | 0.43 |
| New drift correction | 1.6 px | 0.53 |
| No drift correction | 0 | 0.52 |

Real drift on the test movie is still measured correctly (4.25 px against 4.24 px built in).

---

## Known limits

- **Neuron finding is still far from perfect** (F1 about 0.53). Some labels come from a nuclear
  marker that may not be visible in calcium images at all.
- **Nearly every outlined neuron counts as firing** on the real recording. This may be genuine for
  an awake mouse over seven minutes, but event detection has not been checked against known spike
  times.
- **Cells mode has only been tested on synthetic data.** The Cell Tracking Challenge server was
  down when we tried.
- **Animal tracking has only been tested on synthetic data.** The idtracker.ai videos need a manual
  download.
- **Neurite branch and tip counts** are the least exact neurite measures.
- **Our F1 is centre-matching only** and not comparable with published Neurofinder scores.
- **No automated tests run on GitHub**; tests run on the developer's Mac.

## Open questions and next steps

- Score the cells mode on the Cell Tracking Challenge once its server responds.
- Check event detection against a dataset with known spike times.
- Add Cell Tracking Challenge scoring to `behavetrack score`.
- Consider running the test suite on GitHub Actions (free for public repositories).
- Standard file exports: NWB for neurons, the CMSO biotracks format for cells.
- A catalog of every output column with its definition, units and source.

## How to keep this log

When a pull request makes a decision someone might later ask "why?" about, add a row to the
timeline and an entry with:

- **Goal or problem**: what prompted it.
- **Options considered**, if there was a real choice.
- **Decision and why.**
- **Evidence**: test results, benchmark scores, or what a check showed.

Keep numbers tied to a dataset and command, so they can be reproduced.
