# Behavior Tracker

Track animals, people, or cells in a video or microscope time-lapse, label what the
group is doing, and export data files describing movement, interactions and (for
cells) division. It can also search your videos for a behavior and cut out the
matching clips.

## Getting started

You need a Mac with Python 3 (from [python.org](https://www.python.org/downloads/) if it
isn't installed). Download this project (green **Code** button → **Download ZIP**, then unzip,
or `git clone`), then double-click **`Start Behavior Tracker.command`**. The first launch
installs what it needs, which takes a few minutes.

On Windows or Linux, run the command-line setup below, then
`.venv/bin/python -m streamlit run app.py` (on Windows: `.venv\Scripts\python -m streamlit run app.py`).

## Using the app (no typing)

Double-click **`Start Behavior Tracker.command`**. It opens in your web browser
(it runs only on your computer; nothing is uploaded to the internet). The first
launch on a new Mac sets itself up, which takes a few minutes.

1. **Add videos**: choose **Animals or people** or **Cells (microscope)** at the top left,
   drag videos or microscope files onto the page, check the settings, press **Analyze**.
2. **Results**: the annotated video, time spent in each behavior, charts over time,
   the data tables, and a button to download every data file as a .zip.
3. **Find a behavior**: pick a behavior (or cell division) to list every time it happens
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

## Using the command line

One-time setup (the app launcher does this for you):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Analyze one video

```bash
.venv/bin/python -m behavetrack analyze my_video.mp4
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
.venv/bin/python -m behavetrack analyze my_cells.tif --mode cells
.venv/bin/python -m behavetrack analyze my_cells.nd2 --mode cells --cells-look bright --channel 1
.venv/bin/python -m behavetrack analyze movie.mp4 --mode cells --pixel-size 0.65 --frame-interval 300
```

Run `.venv/bin/python -m behavetrack analyze --help` for every option.

### Search many videos for a behavior

```bash
.venv/bin/python -m behavetrack filter my_videos/ --behavior collective_motion --min-seconds 2 --clips
```

Behaviors: `collective_motion`, `milling`, `disordered`, `resting`, `interaction`.
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
