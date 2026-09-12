## Repo layout

```
anpr/
  camera.py        # person 1
  detector.py      # person 2
  plate_reader.py  # person 3
  database.py      # person 4
  linker.py        # person 4
  visualizer.py    # person 5
  plate_logic.py   # person 6
  config.py        # person 6
  main.py          # person 6
  data/videos/     # test footage
  out/frames/      # annotated screenshots
  out/crops/       # best crop per sighting
  out/map.png
```

---

## Person 1 — camera.py 

Reads video / webcam / RTSP and hands out frames tagged with a camera ID and a real timestamp.
Also responsible for recording the test footage.

### Build

```python
class Camera:
    def __init__(self, cam_id: str, source: str, skip: int = 2, start_time: float | None = None):
        """
        cam_id     : "C1"
        source     : file path, webcam index as str ("0"), or rtsp:// url
        skip       : process 1 of every (skip+1) frames
        start_time : unix time the recording started; for a live source use time.time()
        """

    def frames(self):
        """Generator. Yields one dict per processed frame:
           {"cam_id": "C1", "t": 1726123456.78, "frame_no": 412, "frame": np.ndarray}
           t = start_time + frame_no / fps   for a file source
        """

    def release(self):
        ...
```

### Tasks
- [ ] `cv2.VideoCapture` wrapper that works for file, webcam and RTSP
- [ ] Frame skipping; read fps from the file, fall back to 25 if missing
- [ ] Compute `t` from `start_time + frame_no / fps` so two videos share one clock
- [ ] Graceful stop at end of file; reconnect attempt for RTSP
- [ ] Record 3 short videos: the **same 5–10 vehicles** passing 2–3 spots, note the real start time of each
- [ ] Record one night clip if possible (feeds the limits slide)

---

## Person 2 — detector.py

Detects and tracks vehicles, describes them, and keeps the single best crop per track.

### Build

```python
class Detector:
    def __init__(self, model_path: str = "yolo26s.pt", conf: float = 0.4):
        """Loads YOLO26 and BoT-SORT. Use gmc_method: none (fixed cameras)."""

    def update(self, frame: np.ndarray, cam_id: str) -> list[dict]:
        """One frame in. Returns LIVE tracks currently visible:
           {"track_id": 17, "box": (x1,y1,x2,y2), "type": "car",
            "colour": "white", "crop": np.ndarray, "quality": 0.72}
           quality in 0..1 = normalised(crop area) * normalised(sharpness)
        """

    def finished_tracks(self) -> list[dict]:
        """Tracks that have left the frame since the last call. Returns:
           {"track_id": 17, "cam_id": "C1", "t_in": float, "t_out": float,
            "type": "car", "colour": "white", "best_crop": np.ndarray,
            "direction": "north", "embedding": np.ndarray | None}
        """
```

### Tasks
- [ ] `model.track(frame, persist=True, tracker="botsort.yaml")` loop
- [ ] Edit the tracker yaml: `gmc_method: none`
- [ ] Map COCO classes → car / bike / bus / truck. **Auto-rickshaws come out as car or motorcycle — record this as a known limit**
- [ ] Colour: HSV histogram of the middle 50% of the crop → white/black/silver/red/blue/other
- [ ] Sharpness: `cv2.Laplacian(gray, cv2.CV_64F).var()`
- [ ] `quality` score; keep the highest-quality crop per track, replace only when beaten
- [ ] Direction from first vs last box centre of the track
- [ ] Emit a track as finished after it is unseen for N frames (`config.TRACK_TIMEOUT`)
- [ ] Optional, only if time: `embed(crop) -> np.ndarray` using pretrained FastReID or DINOv2


---

## Person 3 — plate_reader.py

Turns a vehicle crop into characters with per-character confidence. This is the identity key;
its quality decides how good every screenshot looks.

### Build

```python
def read_plate(crop: np.ndarray) -> dict | None:
    """Vehicle crop in. Returns
         {"chars": ['G','J','0','1','A','B','1','2','3','4'],
          "confs": [0.98, 0.97, ...],        # same length as chars
          "plate_box": (x1,y1,x2,y2)}        # relative to the crop
       Returns None if no plate is found or the crop fails the quality gate.
       Unreadable slot: char '?' with conf 0.0.
    """

def passes_gate(crop: np.ndarray) -> bool:
    """Cheap pre-check before running OCR: min crop size and min sharpness."""
```

### Tasks
- [ ] `pip install fast-alpr[onnx-gpu]` (bundles plate detector + fast-plate-ocr)
- [ ] Use the pretrained `cct-s-v2-global-model`; keep PaddleOCR as a fallback only if it reads Indian plates badly
- [ ] Make sure you return **per-character** probabilities, not one score for the whole plate — voting needs them
- [ ] `passes_gate`: reject crops below `config.MIN_PLATE_PX` or `config.MIN_SHARPNESS`
- [ ] Perspective-warp the plate crop flat before OCR if corners are available
- [ ] Test on 50+ Indian plate crops; write down failure categories (two-line bike plates, painted truck plates, night glare) — these go on the risk slide
- [ ] Save failing crops to `out/crops/fail/` for the future fine-tuning set


---

## Person 4 — database.py + linker.py

Stores sightings and decides which sightings across cameras are the same vehicle.

### Build — database.py

```python
def init_db(path: str) -> None
def save_camera(cam_id: str, lat: float, lon: float, name: str) -> None
def save_sighting(s: dict) -> int          # returns sighting_id
def get_by_plate(plate: str) -> list[dict]
def get_candidates(s: dict, window: float) -> list[dict]
def save_link(a_id: int, b_id: int, score: float, method: str) -> None
def get_links() -> list[dict]
```

Sighting dict:
```python
{"cam_id": "C1", "t_in": float, "t_out": float,
 "plate": "GJ01AB1234" | None, "conf": [0.9, ...] | None,
 "type": "car", "colour": "white", "direction": "north",
 "embedding": np.ndarray | None, "crop_path": "out/crops/17.jpg"}
```

### Build — linker.py

```python
CAMERA_GRAPH = {("C1","C2"): (20.0, 180.0), ...}   # min/max seconds between cameras

def plausible(a: dict, b: dict, slack: float = 5.0) -> bool
def plate_distance(a: str, b: str) -> float
def link_score(a: dict, b: dict) -> tuple[float, str]   # (score, method)
def link_all() -> list[dict]                            # {"a":12,"b":47,"score":0.91,"method":"fuzzy"}
def find_cloned_plates() -> list[dict]
```

### Tasks
- [ ] SQLite (prototype) with tables cameras, sightings, links; Postgres later
- [ ] Hardcode `CAMERA_GRAPH` for the 3 test cameras with realistic travel windows
- [ ] `plate_distance`: weighted edit distance — `'?'` vs anything = 0.1, confusable pair (0/O, 1/I, 8/B, 5/S, 2/Z) = 0.3, other substitution = 1.0
- [ ] `link_score` order: (1) travel time plausible, (2) same vehicle type, (3) **never merge two clean plates that differ** — each returns 0. Then score = 0.6·plate_sim + 0.3·embedding_cos + 0.1·colour. No plate on one side → 0.75·embedding + 0.25·colour, method `inferred`
- [ ] Thresholds in config: `LINK_FUZZY=0.75`, `LINK_INFERRED=0.85` — placeholders until tuned
- [ ] `find_cloned_plates`: same plate, two cameras, time gap below the graph minimum → alert
- [ ] Hand-label ~50 pairs from the test videos (same / different) and sweep the threshold; aim for precision ≥ 0.98. Save the numbers for the deck

---

## Person 5 — visualizer.py

Draws annotated frames and the camera map. Also owns the deck.

### Build

```python
def draw_frame(frame: np.ndarray, tracks: list[dict]) -> np.ndarray:
    """Boxes, track id, vehicle type, plate text (or 'reading...' / 'no plate')."""

def draw_map(cameras: list[dict], links: list[dict], out_path: str = "out/map.png") -> str:
    """Camera dots on a map, a line per link. Solid = exact/fuzzy, dashed = inferred.
       Returns the path to the written PNG."""
```

### Tasks
- [ ] `cv2.rectangle` + `cv2.putText`; colour the box green when the plate is locked, amber while reading, grey when no plate
- [ ] Show per-character confidence under the plate text (this is the evidence story judges like)
- [ ] Save annotated frames to `out/frames/`
- [ ] Map with Folium (or matplotlib over a screenshot of the area) — camera dots plus link lines
- [ ] Assemble the deck: screenshots, DB rows, map PNG, the old person-tracking screenshot labelled "same pipeline, identity key swapped"
- [ ] Record a 60–90 s demo video; test the link in incognito before submitting
- [ ] Maintain the "Built / In progress / Planned" table and collect every limit the others report

---

## Person 6 — main.py + config.py + plate_logic.py

Joins everything, owns the OCR budget, and turns many noisy reads into one plate.

### Build — plate_logic.py

```python
def expected_types(length: int) -> str
def fix_by_format(chars: list[str], confs: list[float]) -> tuple[str, list[float]]
def vote(reads: list[dict]) -> tuple[str, list[float], bool]
```

`reads` = `[{"chars": [...], "confs": [...], "quality": 0.8}, ...]` — one entry per OCR'd frame of the same track.
Returns `("GJ01AB1234", [0.91, ...], locked)`.

Voting steps:
1. Vote on plate **length**, weighted by quality; drop reads of other lengths.
2. Run `fix_by_format` on each surviving read.
3. Per position, accumulate `confidence × quality` per character; take the max.
4. `locked` = 3+ reads agreed, no `'?'` left, every position above `config.MIN_CHAR_CONF`.

Format rules: positions have expected types (LL DD LL DDDD). A digit in a letter slot is
swapped via the confusion table (8→B, 0→O, 1→I) and its confidence drops to 0.6×.
First two letters must be a real state code, else halve their confidence.

### Build — main.py flow

```
for each camera:
    for frame in camera.frames():
        tracks = detector.update(frame, cam_id)
        for t in tracks:
            if track_locked(t): continue
            if ocr_budget_used(t) >= config.MAX_OCR_PER_TRACK: continue
            if not plate_reader.passes_gate(t["crop"]): continue
            r = plate_reader.read_plate(t["crop"])
            if r: reads[t["track_id"]].append({**r, "quality": t["quality"]})
            plate, conf, locked = plate_logic.vote(reads[t["track_id"]])
        frame_out = visualizer.draw_frame(frame, tracks)
    for ft in detector.finished_tracks():
        database.save_sighting(build_sighting(ft, reads[ft["track_id"]]))

linker.link_all()
linker.find_cloned_plates()
visualizer.draw_map(cameras, database.get_links())
```

### Tasks
- [ ] `config.py` with every threshold: `SKIP`, `CONF`, `TRACK_TIMEOUT`, `MIN_PLATE_PX`, `MIN_SHARPNESS`, `MAX_OCR_PER_TRACK`, `MIN_CHAR_CONF`, `LOCK_MIN_READS`, `LINK_FUZZY`, `LINK_INFERRED`
- [ ] Write `plate_logic.py` yourself and be able to rewrite it on a whiteboard
- [ ] OCR budget: early stop once locked, hard cap of frames per track
- [ ] ID-switch guard: if one track produces two clean, clearly different plates, split the track instead of voting
- [ ] Run the full chain on 3 videos; keep the run reproducible with one command
- [ ] Own merges to `main` and unblock whoever is stuck