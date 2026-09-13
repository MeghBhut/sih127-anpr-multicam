# ANPR — multi-camera vehicle detection and tracking

**SIH26127.** Prototype: pretrained models, 2–3 test videos acting as cameras, rows in a
database, one map PNG. No fine-tuning, no web dashboard.

> **Rule for the whole system: a wrong link is worse than a missing link.**
> A missing trajectory is incomplete data. A wrong one puts an innocent car's number on
> someone else's route. Every filter exists to reject.

The full contract between the six of us is [`docs/build_instructions.md`](docs/build_instructions.md).
Read it before writing a line. [`docs/decisions.md`](docs/decisions.md) records where the
scaffold interprets that contract — read it before touching `linker.py` or `config.py`.
[`docs/pipeline.html`](docs/pipeline.html) is the same thing as one diagram: open it in a
browser, or share it with the team.

---

## Setup — once per machine

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Two things pip cannot fetch. Both go inside `anpr/`, both are gitignored.

**1. PaddleOCR source** — the plate OCR weights load through it:

```bash
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git anpr/PaddleOCR
```

**2. The OCR weights** — download `model.safetensors` and `en_dict.txt` from
[Awiros/anpr-ocr](https://huggingface.co/Awiros/anpr-ocr) and drop both directly in `anpr/`.

YOLO weights need no setup: `yolo26s.pt` downloads itself on the first run.

Check it worked:

```bash
python -c "import anpr.main; print('ready')"
```

## Run it on video

**Step 1 — put the clips in `anpr/data/videos/`.** Any format OpenCV reads. Name them
`c1.mp4`, `c2.mp4`, `c3.mp4` and you can skip naming them on the command line.

**Step 2 — tell `config.py` when each clip started recording.** This is the one step
people skip, and skipping it means **no link will ever be correct.** The linker decides
"same vehicle" by asking whether enough time passed to drive between two cameras, so the
three clips have to sit on one shared clock:

```python
# anpr/config.py
RECORDING_START = {
    "C1": 1757739600.0,     # unix seconds when C1 started recording
    "C2": 1757739600.0,
    "C3": 1757739660.0,
}
```

To get a unix timestamp from a normal time:

```bash
python -c "import datetime;print(datetime.datetime(2026,9,13,17,30,0).timestamp())"
```

**Step 3 — set the real camera positions and travel times** in `config.CAMERAS` and
`config.CAMERA_GRAPH`. The defaults are invented Ahmedabad coordinates.

**Step 4 — run:**

```bash
python -m anpr.main
```

or point it at specific files:

```bash
python -m anpr.main --videos C1=anpr/data/videos/c1.mp4 C2=anpr/data/videos/c2.mp4
```

Add `--verbose` to see why each pair of sightings was linked or rejected. Add `--append`
to keep the previous run's rows instead of starting clean.

## What you get

```
sightings   : 12
  clean     : 9
  partial   : 2
  missing   : 1
id switches : 0
frames saved: 43 -> anpr/out/frames
links       : 4
clones      : 0
map         : anpr/out/map.png
```

| Where | What |
|---|---|
| `anpr/out/anpr.db` | SQLite: cameras, sightings, links. Open with any SQLite viewer. |
| `anpr/out/frames/` | Annotated screenshots. Green box = plate locked, orange = still reading, grey = no plate, with per-character confidence underneath. |
| `anpr/out/crops/` | The best photo of each vehicle — the evidence image for its row. |
| `anpr/out/crops/fail/` | Crops the OCR could not read, labelled by reason. Feeds the risk slide and any future fine-tuning. |
| `anpr/out/map.png` | Camera dots with a line per route. Solid = plate match, dashed = inferred. |

Every run starts from a clean database, so the same command twice gives the same answer.

## If something goes wrong

| Message | Fix |
|---|---|
| `plate detector not installed` | `pip install "open-image-models[onnx]"` |
| `PaddleOCR not found` | Setup step 1 above |
| `could not open source: ...` | Wrong video path. That camera is skipped; the others still run. |
| `no RECORDING_START in config` | Step 2. The run works, but the links will be meaningless. |
| `detector needs ultralytics` | `pip install ultralytics`. If `yolo26s.pt` will not download, `pip install -U ultralytics`. |
| Everything reads as `no plate` | Check the OCR weights are in `anpr/`, and look in `out/crops/fail/` to see what it is rejecting. |

## Tests

```bash
python -m unittest discover -s tests -t .
python tests/demo_sample_run.py
```

## Layout

```
anpr/
  camera.py        # person 1 — frames tagged with cam_id + unix time
  detector.py      # person 2 — YOLO + BoT-SORT, best crop per track
  plate_reader.py  # person 3 — crop -> chars + per-character confidence
  database.py      # person 4 — cameras, sightings, links
  linker.py        # person 4 — which sightings are the same vehicle
  visualizer.py    # person 5 — annotated frames + map PNG
  plate_logic.py   # person 6 — format rules, voting, link scoring
  config.py        # person 6 — every threshold
  main.py          # person 6 — the loop that joins it all
  data/videos/     # test footage (gitignored — share over Drive)
  out/frames/      # annotated screenshots
  out/crops/       # best crop per sighting, failures in out/crops/fail/
  out/map.png
docs/              # build instructions, task list, guides, diagrams
```

## Ground rules

1. Function signatures in `docs/build_instructions.md` are frozen. Nobody changes one
   without telling the others.
2. Frames and crops are **BGR numpy arrays** everywhere (OpenCV default). Do not convert
   to RGB inside your module.
3. Anything unknown is `None` — never `""`, never `"unknown"`. `plate=None` means not
   read; `plate=""` means a bug.
4. Timestamps are **unix seconds (float)**, not frame numbers. The linker needs wall-clock time.
5. Every threshold lives in `anpr/config.py`. No magic numbers inside modules.
6. Only Megh merges to `main`. PRs need one review from whoever owns the module you touch.

## Working on it

```bash
git checkout main && git pull
git checkout -b p3/plate-reader        # p<your number>/<what you're doing>
# ...work...
git add -A && git commit -m "plate_reader: warp the plate flat before OCR"
git push -u origin p3/plate-reader
```

Then open a PR against `main` and tag the owner of anything you touched.

## Schedule

| Day | Goal |
|---|---|
| 1 | ~~Signatures frozen, stubs committed~~ done |
| 2 | ~~`main.py` runs end to end~~ done |
| 3 | Real detector + tracker on one video (P2), OCR returning chars (P3) |
| 4 | Voting wired in, sightings landing in the DB |
| 5 | Linker producing links across videos; map PNG drawn |
| 6 | Screenshots, demo video, deck assembled; limits table finalised |

If you finish early, help person 3. Plate reading quality decides how good every
screenshot looks.

## Known limits (person 5 collects these)

- Auto-rickshaws are not a COCO class — detected as car or motorcycle, which then
  confuses the linker's same-type filter
- Painted truck plates and night glare still read poorly. Two-line bike plates are
  handled: the OCR is Awiros ANPR (PP-OCRv5), trained on Indian plates, 96.9% on
  dual-row. Failing crops land in `out/crops/fail/` labelled by reason.
- Thresholds and score weights are untuned placeholders until the labelled pairs exist
- Routes between cameras are inferred, not observed — drawn dashed
- Embedding-only links are low confidence and never used alone for identification.
  Person 2's embeddings are not built yet, so a sighting with no plate produces no
  links at all — deliberate, see `docs/decisions.md`
