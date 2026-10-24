# Behavior Tracker

Track animals, people, or cells in a video or microscope time-lapse, label what the
group is doing, and export data files describing movement, interactions and (for
cells) division. For neurons it measures firing activity in calcium imaging and
neurite (axon and dendrite) growth. It can also search your videos for a behavior and
cut out the matching clips.

## Getting started

You need a Mac with Python 3 (from [python.org](https://www.python.org/downloads/) if it
isn't installed). Download this project (green **Code** button → **Download ZIP**, then unzip,
or `git clone`), then double-click **`Start Behavior Tracker.command`**. The first launch
installs what it needs, which takes a few minutes.

On Windows or Linux, do the command-line setup below, then run `behavetrack app`.

## Using the app (no typing)

Double-click **`Start Behavior Tracker.command`**. It opens in your web browser
(it runs only on your computer; nothing is uploaded to the internet). The first
launch on a new Mac sets itself up, which takes a few minutes.

1. **Add videos**: choose **Animals or people**, **Cells (microscope)** or **Neurons (microscope)** at the top left,
   drag videos or microscope files onto the page, check the settings, press **Analyze**.
2. **Results**: the annotated video, time spent in each behavior, charts over time,
   the data tables, and a button to download every data file as a .zip.
3. **Find a behavior**: pick a behavior (or cell division, or a network burst) to list every time it happens
   across all your videos. Click a row to play that moment, and cut it into its own clip.

Everything is saved in `app_data/` (`uploads/` for your videos, `results/` for the data).
To stop the app, close the Terminal window that opened with it.

If macOS says the file "can't be opened because it is from an unidentified developer",
right-click it, choose **Open**, then **Open** again.

## Cells mode

For microscope time-lapses of living cells.

**Files it reads:** TIFF stacks (including ImageJ and OME-TIFF), Nikon `.nd2`, Zeiss `.czi`,
regular videos, or (command line) a folder of numbered images. For multi-channel files, pick
the channel; z-stacks are flattened (brightest point through the depth).

**Calibration:** pixel size and time between frames are read from the file when present. If
not, enter them under *Microscope calibration*; otherwise results are in pixels and frames.

**Settings:**
- *How do the cells look?* **Bright** for fluorescence or stained nuclei, **Dark** for cells
  darker than the background, **Auto** for anything that stands out (phase contrast).
- *Detection*: move left to find fainter cells, right to ignore specks.
- *Typical cell diameter*: set it if cells are missed or split in half.
- *Detect cell division* and *Separate touching cells*: on by default.

**Extra files it writes:**

| File | One row per… | What's in it |
|---|---|---|
| `cells.csv` | cell | parent, generation, duration, path length, net displacement, straightness, mean/max speed, direction, forward migration index (x, y), MSD exponent, mean area, fate (divided / left view / lost / present at end) |
| `lineage.csv` | division | parent ID, daughter ID, when |
| `msd.csv` | time lag | mean squared displacement: how far cells get from where they were |

`tracks.csv` also gains columns in real units (µm, µm/min, µm²), and `summary.json`
gains population results: median speed and straightness, migration bias, MSD exponent,
diffusion coefficient, cells at start and end, divisions, and doubling time.

**What the measures mean:**
- *Straightness*: start-to-end distance ÷ path walked. 1 = straight line, near 0 = wandering.
- *Forward migration index (FMI)*: how much of each path heads in one direction. A clear
  non-zero mean (above about 0.1) suggests migration toward something, such as a chemical gradient.
- *MSD exponent*: about 1 = random movement, about 2 = straight-line crawling, below 1 = held in place.

**Accuracy on the built-in test** (`tools/make_test_cells.py`, 25 → 46 cells over 6 h):
99% of cells found with no false detections, speed within 1% of the truth, 18 of 21
divisions identified correctly. Check it yourself with `tools/check_cells.py`.

**Limits:** cells that touch a neighbour while dividing can be missed; very crowded
or overlapping cells (confluent sheets) are hard to separate; this is not a replacement for
specialised segmentation tools such as Cellpose on difficult images.

## Neurons mode

For neurons, in two kinds of recording. Pick one under *What should it measure?*.
It reads the same files as cells mode, and calibration works the same way.

### Firing activity (calcium imaging)

For calcium-imaging (GCaMP, Fluo-4, …) or voltage-imaging movies of neurons in culture,
slices or other flat preparations.

**What it does:** lines every frame up with the start of the movie (stage drift), finds each
neuron, measures its brightness in every frame minus the haze around it (neuropil), turns that
into ΔF/F (change relative to the neuron's resting level, which follows slow fading), and
finds firing events and network bursts.

**Settings:**
- *Which neurons?* **Ones that fire** finds neurons by their flicker (pixels that brighten
  together). **All visible** finds bright cell bodies, so neurons that never fire are counted
  too; this needs a dye that shows resting neurons.
- *Detection*: move left to find fainter neurons.
- *Firing threshold*: an event must rise this many times the neuron's own noise above rest (default 3.5).
- *Correct drift*: on by default.
- *More options*: share of background haze to subtract (0.7), smallest event (0.05 ΔF/F; use 0
  for voltage dyes), what counts as a network burst (30% of firing neurons starting within 0.5 s),
  and how slowly the resting level may change (60 s).

**Files it writes:**

| File | One row per… | What's in it |
|---|---|---|
| `neurons.csv` | neuron | position, area, resting brightness, noise, events, events per minute, mean and largest ΔF/F, half-width, interval between events and its regularity (CV), bursts joined |
| `events.csv` | firing event | neuron, onset, peak, rise time, duration, half-width, size (ΔF/F), signal-to-noise, in a burst or not |
| `bursts.csv` | network burst | start, peak, duration, neurons taking part, share of firing neurons, mean size |
| `traces_dff.csv` | frame | ΔF/F of every neuron (one column each) |
| `traces_raw.csv` | frame | raw brightness and neuropil of every neuron |
| `correlations.csv` | pair of neurons | distance apart and how alike their activity is |
| `frames.csv` | frame | neurons firing, share firing, mean ΔF/F, in a burst or not |
| `rois.png` | | map of the neurons with their ID numbers |

`summary.json` gains: neurons found and firing, events per minute, median event size and
half-width, network bursts (count, per minute, gap between them and its regularity, neurons per
burst), share of events inside bursts, synchrony (mean correlation between firing neurons) and
the largest drift corrected.

**Accuracy on the built-in test** (`tools/make_test_neurons.py`: 40 neurons, 10 of them silent,
90 s at 20 frames/s, with neuropil, bleaching, drift and noise): all 30 firing neurons found with
no false ones (all 40 with *All visible*), 99% of firing events found with 0.4% false, 8 of 8
network bursts found. Check it yourself with `tools/check_neurons.py`.

**Limits:** drift correction handles sliding, not warping or z-movement, so very wobbly
recordings (e.g. awake animals) need motion correction first. Events closer together than about
0.2 s merge into one. Densely packed or overlapping neurons can be merged. ΔF/F is relative to
raw brightness, so a large camera offset makes it smaller. For large in-vivo datasets use a
specialised tool such as Suite2p or CaImAn.

### Neurite growth

For fluorescent (GFP, βIII-tubulin, MAP2, …), brightfield or phase-contrast images of neurons in
culture: a single image or a time-lapse.

**What it does:** finds cell bodies, traces neurites as thin lines, gives each neurite to the
cell body it connects to, and measures length, branching and tips. Across a time-lapse each
neuron keeps its ID, and growth cones are followed to see whether they extend or pull back.

**Settings:**
- *How do the neurites look?* **Bright** (fluorescence), **Dark** (brightfield) or **Auto** (phase contrast).
- *Detection*: move left to trace fainter neurites.
- *Neurite thickness*: typical width in pixels (default 3).
- *Calibration*: pixel size, time between frames, cell body diameter (automatic by default), channel.
- *More options*: analyze every Nth frame, Sholl ring spacing (10 µm).

**Files it writes:**

| File | One row per… | What's in it |
|---|---|---|
| `neurons.csv` | neuron | position, cell body area, final total neurite length, longest neurite, primary neurites, branch points, tips, growth rate (total and longest), mean growth-cone extension rate, Sholl peak |
| `neurite_frames.csv` | neuron per frame | total length, longest neurite, primary neurites, branch points, tips |
| `growth_cones.csv` | growth cone | neuron, how long it was followed, net extension, extension rate, share of time extending or pulling back, behavior (extending / retracting / paused) |
| `growth_cone_tracks.csv` | growth cone per frame | position and distance from the cell body along the neurite |
| `sholl.csv` | neuron per ring | neurite crossings at each distance from the cell body (last frame) |
| `frames.csv` | frame | neurons, total length, tips, branch points, length not joined to a cell body |
| `neurites.png` | | the last frame with traced neurites coloured by neuron |

**What the measures mean:**
- *Longest neurite*: the path along the neurite from the cell body to the farthest tip.
- *Primary neurites*: neurites leaving the cell body. *Tips* are growth cones.
- *Sholl analysis*: how many neurites cross rings drawn around the cell body. More crossings
  farther out means a larger, more branched arbor.

**Accuracy on the built-in test** (`tools/make_test_neurites.py`: 8 neurons, 25 frames over 4 h,
growing, branching and retracting neurites, debris and uneven lighting): 8 of 8 cell bodies found,
total length within 1% overall (median 2.8% per neuron), longest neurite within 3.3%, primary
neurites exact, growth rate 44.7 vs 44.3 µm/h. Check it yourself with `tools/check_neurites.py`.

**Limits:** where neurites from different neurons cross or touch, the shared stretch is split
between them and counts as a branch point; branches shorter than about 3 µm aren't counted yet;
neurites that leave the image are measured only up to the edge. Dense networks (mature cultures)
are hard to assign to single neurons; very sparse labelling works best.

## Complexity analysis (complexity-engine)

[complexity-engine](https://github.com/PJAllen5/complexity-engine) measures how regular,
persistent and networked a signal or network is (sample entropy, spectral entropy, DFA,
mutual information, clustering, modularity, path length), and explains the results with
profiles grounded in a cited literature base. It has four profiles written for this tracker's
output:

| Exported file | From | complexity-engine profile | What it asks |
|---|---|---|---|
| `collective_timeseries.csv` | animals, cells | `collective_behavior` | Does the group's coordination fluctuate randomly, or in long bouts and switches between states? |
| `neural_timeseries.csv` | neuron activity | `neural_activity` | Is each neuron's activity irregular, or structured over long time scales? |
| `contact_network.csv` | animals, cells | `animal_social_network` | Are contacts spread evenly, or concentrated in subgroups and central individuals? |
| `functional_network.csv` | neuron activity | `neuronal_network` | Is coordinated firing spread across the population, or organized into modules and hubs? |

Neurite-growth results are skipped: their time-lapses have too few frames for these measures.

**In the app:** open a result in **Results** and press **Run complexity analysis** at the
bottom. It shows the key numbers and offers the full report, with interpretation and citations,
as an HTML file. Results are saved in that result's `complexity/` folder.

**On the command line:**

```bash
behavetrack export-complexity results/video1:control results/video2:treated --run
```

Each results folder is one recording. Add `:CONDITION` to group recordings for comparison
(otherwise each is its own condition). Without `--run`, it writes the files and prints the
`complexity-analysis` / `complexity-network` commands to run yourself. In the network files each
video is its own network, since track IDs in different videos are different individuals.
For neurons, `--min-correlation` (default 0.3) sets which pairs count as connected.

**Installing complexity-engine.** Install it into this project straight from GitHub:

```bash
.venv/bin/pip install -e ".[complexity]"
```

Or, if you have a local copy of complexity-engine that you're editing:

```bash
.venv/bin/pip install -e /path/to/complexity-engine
```

## Using the command line

One-time setup (the app launcher does this for you):

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[ai]"
```

This installs the `behavetrack` command. Run `source .venv/bin/activate` once per terminal
to use it by name as below, or type the full path `.venv/bin/behavetrack`.
`python -m behavetrack` also works.

```bash
behavetrack app                      # open the browser app
behavetrack analyze --help           # every option, with its default
```

### Analyze one video

```bash
behavetrack analyze my_video.mp4
```

Results go to `my_video_behavior/`:

| File | One row per… | What's in it |
|---|---|---|
| `tracks.csv` | object per frame | ID, position, size, velocity, speed, heading, turn rate, acceleration |
| `frames.csv` | frame | count, group centre, spread, nearest-neighbour distance, polarization, rotation, behavior label, aggregating/dispersing |
| `behaviors.csv` | stretch of behavior | behavior, start/end time, duration, averages |
| `interactions.csv` | close encounter | the two IDs, start/end, duration, closest distance, alignment, who approached, who led |
| `summary.json` | video | totals, seconds spent in each behavior, settings used |
| `annotated.mp4` | | the video with IDs, trails and live metrics drawn on |

### Analyze a microscope time-lapse

```bash
behavetrack analyze my_cells.tif --mode cells
behavetrack analyze my_cells.nd2 --mode cells --cells-look bright --channel 1
behavetrack analyze movie.mp4 --mode cells --pixel-size 0.65 --frame-interval 300
```

### Analyze neurons

```bash
behavetrack analyze gcamp.tif --mode neurons
behavetrack analyze gcamp.tif --mode neurons --find-by brightness --frame-interval 0.05
behavetrack analyze outgrowth.nd2 --mode neurons --analysis neurites --channel 1
behavetrack filter recordings/ --mode neurons --behavior network_burst --clips
```

Run `behavetrack analyze --help` for every option and its default.

### Search many videos for a behavior

```bash
behavetrack filter my_videos/ --behavior collective_motion --min-seconds 2 --clips
```

Behaviors: `collective_motion`, `milling`, `disordered`, `resting`, `interaction`, and
`network_burst` for neuron recordings.
Writes `behavior_results/matches_<behavior>.csv` and, with `--clips`, one video per match.
Videos already analyzed are reused; add `--rerun` to redo them.

## What the labels mean

Distances are in **body lengths (BL)**, the typical object size, so the same
settings work for ants, birds and people.

- **collective_motion**: most individuals head the same way (polarization ≥ 0.65). Flocks, schools, marching ants, a crowd walking together.
- **milling**: individuals circle a shared centre (rotation ≥ 0.45). Fish tornadoes, ant mills, starling vortices.
- **disordered**: moving, but in many directions. Swarming, foraging, a busy plaza.
- **resting**: under 30% of individuals moving.
- **too_few**: fewer than 3 visible.
- **interaction**: two individuals within 2 BL of each other (`--contact-bl` to change).

## Choosing a detector

- `--detector motion` (default): finds anything that moves. No AI model needed and works on any animal, including ants and insects.
  Needs a still camera. Objects that stop moving fade out after a while, and objects that touch can merge into one.
- `--detector yolo --classes person` (or `bird`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `zebra`, …): an AI model that recognises objects.
  Works with a moving camera and on objects standing still. It does not know insects or fish. The first run downloads the model file (about 6 MB).

## Tuning tips

- Missing small or faint things: `--min-area 5 --sensitivity 15`
- Too much noise from leaves or water: `--min-area 40 --sensitivity 40`
- Long or 4K video: `--start 60 --max-seconds 30`, or `--resize-width 640` for speed
- Test video with known behaviors: `.venv/bin/python tools/make_test_video.py samples/test_flock.mp4`
- Test cell time-lapse with an answer key: `.venv/bin/python tools/make_test_cells.py samples/test_cells`
- Test calcium-imaging movie with an answer key: `.venv/bin/python tools/make_test_neurons.py samples/test_neurons`
- Test neurite time-lapse with an answer key: `.venv/bin/python tools/make_test_neurites.py samples/test_neurites`

## Developing

```bash
.venv/bin/pip install -e ".[ai,dev]"      # once: the code, the AI detector, and test tools
.venv/bin/python -m pytest -m "not slow"   # quick tests, a few seconds
.venv/bin/python -m pytest                 # everything, about a minute
```

**Code layout.** All analysis lives in the `behavetrack/` package. The browser app
(`behavetrack/app.py`) and the command line (`behavetrack/cli.py`) are thin front ends that
collect settings and call `pipeline.run()`. Put new features in the package so both can use them.

| File | What it does |
|---|---|
| `settings.py` | every setting and its default, per mode (the single source of defaults) |
| `pipeline.py` | `run()`: reads a file, runs the right analysis, writes the results |
| `sources.py` | reads videos, TIFF/ND2/CZI stacks and image folders |
| `detect.py`, `track.py`, `analyze.py` | animals: finding, tracking, group behavior, interactions |
| `cells.py` | cells: migration, division and lineage measures |
| `activity.py`, `neurites.py` | neurons: firing activity, neurite growth |
| `export.py` | files for complexity-engine, and running it |
| `app.py`, `cli.py` | the browser app and the command line |

**Adding or changing a setting.**
1. Add it, with its default, to `Settings` in `settings.py` (or to the mode's preset there).
2. Command line: add the option in `cli.py` with no default, and map it in `_OPTION_FIELDS`
   (or `_SWITCHES` for an on/off flag). Use the preset's value in the help text.
3. App: add the widget in `app.py`, with the preset's value (`A`, `C`, `NA` or `NN`) as its
   default, and pass it to `with_changes(...)`.

Never type a default number into `cli.py` or `app.py`. `tests/test_settings.py` checks that the
command line and the app produce exactly the `settings.py` defaults in every mode.

**Tests** (`tests/`): settings agree everywhere; every mode stays accurate against its answer
key; every app results page loads; the commands work; microscope files are read correctly.
Test movies are generated automatically and cached in `tests/.data/`. Tests marked `slow`
run full analyses.

**VS Code.** Open the project folder. Its settings select `.venv` and find the tests (beaker
icon). The Run and Debug panel has buttons for the app, each mode's test file, a file you choose,
and the quick tests. Run *Terminal → Run Task → Make test data* once first.

**iCloud Drive.** If the project is in an iCloud-synced folder (such as Desktop or Documents),
keep the libraries in `.venv.nosync`, with `.venv` as a shortcut to it (the launcher sets
this up). Otherwise iCloud tries to upload them and marks them hidden, and Python then can't
find the installed `behavetrack` command.

