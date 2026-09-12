# ANPR — multi-camera vehicle detection and tracking

**SIH26127.** Prototype: pretrained models, 2–3 test videos acting as cameras, rows in a
database, one map PNG. No fine-tuning, no web dashboard.

> **Rule for the whole system: a wrong link is worse than a missing link.**
> A missing trajectory is incomplete data. A wrong one puts an innocent car's number on
> someone else's route. Every filter exists to reject.

The full contract between the six of us is [`docs/build_instructions.md`](docs/build_instructions.md).
Read it before writing a line. [`docs/decisions.md`](docs/decisions.md) records where the
scaffold interprets that contract — read it before touching `linker.py` or `config.py`.

---

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m anpr.main
```

Point it at real footage:

```bash
python -m anpr.main --videos C1=anpr/data/videos/c1.mp4 C2=anpr/data/videos/c2.mp4
```

Every module is currently a **stub** returning fake data that matches its signature, so the
chain runs end to end from day 1. Replace the bodies, never the signatures.

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
git add -A && git commit -m "plate_reader: real OCR via fast-alpr"
git push -u origin p3/plate-reader
```

Then open a PR against `main` and tag the owner of anything you touched.

## Schedule

| Day | Goal |
|---|---|
| 1 | Signatures frozen, stubs committed, footage recorded (P1), DB schema up (P4) |
| 2 | `main.py` runs end to end on stubs |
| 3 | Real detector + tracker on one video (P2), OCR returning chars (P3) |
| 4 | Voting wired in, sightings landing in the DB |
| 5 | Linker producing links across videos; map PNG drawn |
| 6 | Screenshots, demo video, deck assembled; limits table finalised |

If you finish early, help person 3. Plate reading quality decides how good every
screenshot looks.

## Known limits (person 5 collects these)

- Auto-rickshaws are not a COCO class — currently detected as car or motorcycle
- Two-line bike plates and painted truck plates read poorly with the pretrained OCR
- Thresholds and score weights are untuned placeholders until the labelled pairs exist
- Routes between cameras are inferred, not observed — drawn dashed
- Embedding-only links are low confidence and never used alone for identification
