# Decisions

Choices made while scaffolding that are **not** in `build_instructions.md`. The contract
still wins on signatures; this file records where the scaffold interprets it, and why.
Add a row when you make a call the others would be surprised by.

---

## 1. `linker.py` delegates its maths to `plate_logic.py`

**Owner affected:** person 4 (linker), person 6 (plate_logic)

The contract gives person 4 the signatures `plate_distance(a, b)` and `link_score(a, b)`.
Person 6 had already written both in `plate_logic.py`. Two copies of an edit distance and a
weighted score would drift apart by day 4, and the version on the slide would not be the
version in the DB.

So `anpr/linker.py` keeps the contract's signatures and forwards the maths:

```python
def plate_distance(a, b):        return plate_logic.plate_distance(a, b)
def link_score(a, b):            return plate_logic.link_score(_as_point(a), _as_point(b), CAMERA_GRAPH)
```

**Person 4 still owns:** `link_all` (candidate search), `find_cloned_plates`,
`CAMERA_GRAPH`, and both thresholds. Tuning happens in `config.py`, not by editing
scoring code.

**Person 6 still owns:** what "similar" means — the weights, the confusion table, the
hard rejects.

If P4 wants the maths back, move it wholesale and delete it from `plate_logic.py`. Do not
fork it.

### `_as_point`

`database` stores a sighting as `{"cam_id", "t_in", "t_out", "embedding", ...}`.
`plate_logic` predates the schema and expects a flatter `{"cam", "t", "emb", ...}`.
`linker._as_point` is the one adapter between them, so neither side has to change:

```python
{"cam": s["cam_id"], "t": s["t_in"], "emb": s.get("embedding"), ...}
```

Note it uses `t_in` — a vehicle's arrival time at the camera, which is what the travel
window is measured against. Not `t_out`.

---

## 2. Camera coordinates and travel windows are placeholders

**Owner affected:** person 1

`config.CAMERAS` holds three invented Ahmedabad lat/lons and `config.CAMERA_GRAPH` holds
invented travel windows (20–180 s and friends). They exist so the linker and the map have
something to run against before the footage exists.

**Person 1 replaces both after recording**, using the real spots and the walk/drive time
actually observed between them. Until that happens:

- every link the system produces is meaningless
- `find_cloned_plates` will over-fire, because a too-large minimum makes normal travel look
  impossible
- the map PNG points at the wrong part of the city

Do not tune `LINK_FUZZY` or `LINK_INFERRED` against a fake graph. Real graph first,
then the labelled pairs, then the thresholds.

---

## 3. A missing embedding is *no evidence*, not *zero similarity*

**Owner affected:** person 2 (embeddings are the last, optional task), person 4

`Detector.finished_tracks()` may return `embedding: None` — the contract marks it
"optional, only if time". The original `link_score` did:

```python
emb_sim = np.dot(a["emb"], b["emb"]) / (norm(a) * norm(b) + 1e-9)   # crashes on None
```

The first fix substituted a zero vector in `_as_point`. That silences the crash and is
**worse than the crash**: a zero vector scores `emb_sim = 0.5`, so a missing signal enters
the sum as a confident-looking half-match. Evidence we never gathered would have been
pushing links over the threshold.

The real fix is in `plate_logic.py`:

```python
def embedding_cos(ea, eb) -> float | None:
    """None when either side has no embedding, or the vector is degenerate."""
```

and `link_score` branches on what evidence actually exists:

| plate | embedding | score | method |
|---|---|---|---|
| both | yes | `0.6·plate + 0.3·emb + 0.1·colour` | `exact` / `fuzzy` |
| both | no | `(0.6·plate + 0.1·colour) / 0.7` | `fuzzy_noembed` |
| one missing | yes | `0.75·emb + 0.25·colour` | `inferred` |
| one missing | no | **0.0** | `insufficient_evidence` |

Two things to understand about that table:

**`fuzzy_noembed` renormalises over 0.7** rather than leaving a 0.3 hole. A good plate
match should not be dragged under the threshold by a signal nobody collected. It clears
`LINK_FUZZY`, same as `fuzzy`.

**The last row returns 0 and never links.** With no plate on one side and no embedding,
all that remains is vehicle type, travel time and colour — and every white car on the
road matches that. Linking there is exactly the failure the project rule names: *a wrong
link is worse than a missing link*. Until person 2 lands embeddings, a sighting with no
plate produces **no links at all**. That is the correct behaviour, not a bug. It also
means an "inferred" link is only as good as the embedding behind it.

A degenerate (all-zero) embedding is treated as missing, for the same reason.

### While we are here: `conf=None`

The hard reject "never merge two clean plates that differ" needs per-character
confidences to decide what "clean" means. `_is_clean` returns `False` when `conf` is
`None`, so the reject cannot fire — the pair falls through to the fuzzy score, where two
genuinely different plates land near 0.14 and miss the threshold anyway. Belt and braces,
but do not rely on it: **always store `conf` alongside a plate.**

---

## 4. What is not in git

**Test footage** (`anpr/data/videos/`) — too big, and it is our own recording. Share over
Drive; note each clip's real start time in the sheet, because `Camera(start_time=...)` is
what puts two videos on one clock.

**Model weights** (`*.pt`, `*.onnx`) — ultralytics downloads `yolo26s.pt` into the working
directory on first run, and `fast-alpr` pulls its own model on first call, so every machine
gets them from pip. If we ever fine-tune, that checkpoint is shared by hand like the
footage.

**Run output** (`anpr/out/`) — regenerated by one command, so it stays out of git.
Deck screenshots are the exception: `git add -f` the specific ones person 5 wants to keep.
