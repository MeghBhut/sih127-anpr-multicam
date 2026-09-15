# Decisions

Choices made while scaffolding that are **not** in `build_instructions.md`. The contract
still wins on signatures; this file records where the scaffold interprets it, and why.
Add a row when you make a call the others would be surprised by.

---

## 1. linker.py is person 4's; plate_logic.py kept only the voting

**Owner affected:** person 4 (linker), person 6 (plate_logic)

The scaffold had `linker.py` forwarding `plate_distance` and `link_score` to
`plate_logic.py`, so there was one copy of the maths. Person 4 then delivered a
full `linker.py` with its own implementation, and that is the one in the repo:
it is better than the placeholder (bidirectional camera graph, `'?'`-aware
usability checks, a hard ceiling on evidence-free scores) and person 4 owns
linking.

So `plate_logic.link_score`, `plate_logic.plausible` and
`plate_logic.embedding_cos` are **no longer called by the pipeline**. They are
left in place because they document the reasoning and person 6 can rebuild them
on a whiteboard, but the live scoring is `anpr/linker.py`. Do not fix a scoring
bug in `plate_logic.py` and expect the run to change.

Still live from `plate_logic.py`:

- `vote`, `fix_by_format`, `expected_types` -- the whole voting path
- `plate_distance` -- used by `main.is_id_switch`

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

> Note: this was fixed in `plate_logic.link_score`, which [[section 1]] retired.
> Person 4's `linker._cosine_similarity` reaches the same conclusion independently
> and returns `None` for a missing vector. The lesson still applies, and their
> `_conservative_fallback` partly reintroduces it -- see section 4.

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

## 4. plate_status is written by main.py, and two safety rules depend on it

**Owner affected:** person 4 (linker), person 6 (main)

Person 4's `database.py` stores a `plate_status` on every sighting, and
`linker.py` gates **both** of its last-line safety rules on it:

- `link_score` -> "never merge two clean plates that differ" needs `_is_clean`
- `find_cloned_plates` -> only considers sightings with status `clean`

`save_sighting` does not compute the status; it defaults to `uncertain` for any
plate it is handed. So until something classified the plate, both rules were
switched off. Two vehicles whose plates differ by one real character
(`GJ01AB1234` vs `GJ01AB1284`) linked at score 0.82, and a plate seen on two
cameras 3 seconds apart raised no clone alert.

Voting is the only place that knows how good a read was, so `main.classify_plate`
does it:

| status | meaning |
|---|---|
| `missing` | nothing was read |
| `unreadable` | read, but more than `PARTIAL_MAX_UNKNOWN_FRACTION` is `'?'` |
| `partial` | some `'?'`, enough known characters to still compare |
| `clean` | locked, no `'?'`, every character above `CLEAN_MIN_CHAR_CONF` |
| `uncertain` | read in full, but not confidently enough to be `clean` |

`plate_quality` goes with it: known-character fraction times mean confidence.
Person 4's linker weights the plate term by it, so it must not flatter a read.

Two consequences worth knowing:

**A partial plate is now kept, not discarded.** `build_sighting` used to null out
any plate containing `'?'`. The linker scores `'?'` against any character at 0.1,
so a partial read is real evidence and throwing it away lost links. Only a plate
that was never read becomes `None`.

**One hole is still open.** The clean-plate reject needs *both* sides clean. When
one side lands just under `CLEAN_MIN_CHAR_CONF`, two genuinely different plates
can still link -- measured at 0.768 against a 0.75 threshold. The cause is on
person 4's side: `plate_similarity` normalises the edit distance by plate length,
so one wrong character in ten still scores 0.9 similar, and when embeddings are
absent `_conservative_fallback` is substituted into the embedding slot, which
counts colour twice. Person 4 owns that fix.

---

## 5. ID-switch guard

**Owner affected:** person 2 (tracker), person 6 (main)

A tracker sometimes hands one `track_id` to two vehicles. Voting across that
boundary produces a plate belonging to neither. `main.is_id_switch` watches for
two reads that are each clean and at least `config.ID_SWITCH_MIN_DISTANCE` apart;
when it sees one it starts a new read group, and the finished track is written as
one sighting per group (`C2_17_0.jpg`, `C2_17_1.jpg`). Segment times come from the
reads themselves, not the tracker's `t_in`/`t_out`.

This is a safety net, not a fix. If it fires often, the tracker needs attention.
`python -m anpr.main` prints the count.

---

## 6. Detector.update() takes the frame's timestamp — a frozen signature changed

**Owner affected:** everyone

The contract froze `Detector.update(frame, cam_id)`. That signature has no way to receive
the frame's time, so the detector stamped every track with `time.time()` — the moment the
laptop processed the frame, not the moment the vehicle was filmed. Clips are processed one
after another, so every C1 sighting landed seconds before every C2 sighting regardless of
the real footage, and the travel-time windows were being compared against processing
order. **No link could have been correct.**

The signature is now:

```python
def update(self, frame: np.ndarray, cam_id: str, t: float) -> list[dict]
```

`t` comes straight from `camera.frames()`. Rule 4 (timestamps are unix seconds) was
impossible to honour without it.

Two more things had to change for the clock to be real end to end:

- **`Detector.flush(cam_id)`** — a vehicle still on screen when a clip ends never hit the
  unseen-frames timeout, so it stayed open forever and never became a sighting. `main.py`
  flushes after each camera.
- **`config.RECORDING_START`** — `main.py` used to pass `time.time()` as every camera's
  start time, so even with the detector fixed all three clips looked simultaneous. Person 1
  fills in the real recording times; until then the run warns and links stay meaningless.

---

## 7. The OCR stack is not the one in the contract, and that is the right call

**Owner affected:** person 3

The contract specified fast-alpr with `cct-s-v2-global-model`, with PaddleOCR only as a
fallback "if it reads Indian plates badly". Person 3 went to
[Awiros ANPR OCR](https://huggingface.co/Awiros/anpr-ocr) (PP-OCRv5 / SVTR_HGNet) as the
primary, keeping fast-alpr's *detector* (open-image-models) for finding the plate.

Checked, and it is a better fit than what the contract named:

| | contract | what we run |
|---|---|---|
| trained for | plates worldwide | Indian plates, all state codes |
| dual-row bike plates | not handled | 96.9% |
| overall | not stated | 98.4% |

**This retires a known limit.** "Two-line bike plates read poorly" was on the feasibility
slide; it is no longer true. Painted truck plates and night glare still are.

The cost is a heavier install: PaddleOCR has to be cloned and the weights fetched from
Hugging Face by hand. Both steps are in the README, and both paths are gitignored.

### Preprocessing follows the model author's script, not ppocr's

Plates are padded out to 320px wide before the model sees them, and there are two
different conventions for how:

* `ppocr/data/imaug/rec_img_aug.py` normalises first, then pads with `0.0` (mid grey)
* `test.py` shipped with the Awiros weights pads with black pixels first, then normalises,
  which puts `-1.0` in the padding

We follow **the author's `test.py`**. Their 98.4% was measured with it, and the weights are
theirs. This was briefly changed to the ppocr convention on the reasoning that the model
was trained through ppocr's pipeline; that was wrong, and reading the shipped script
settled it. If anyone wants to revisit it, measure both on real crops rather than
reasoning about which ought to be right.

The one deliberate deviation from `test.py` is `max(1, ...)` on the resize width: a very
tall, narrow plate box rounds to zero and `cv2.resize` raises, which used to end the run.

### A recovered character is a guess

`_ctc_decode_with_gaps` spots suspiciously wide gaps between characters and recovers what
CTC's collapse step dropped. That is real and useful, but the recovered character was
entering the vote looking exactly like one the model actually read. It is now capped at
`config.OCR_RECOVERED_MAX_CONF` (0.45), below `MIN_CHAR_CONF` (0.5), so a guess still
contributes to the vote but can never be the thing that locks a plate. The per-character
tags (`kept` / `low_conf` / `recovered` / `unresolved`) are returned in `read_plate`'s
dict instead of being discarded.

---

## 8. read_plate returns None, it does not raise

**Owner affected:** person 3, person 6

`read_plate` runs once per frame per track. It used to let exceptions escape, so a single
odd crop — a zero-width plate box was enough to crash `cv2.resize` — ended the run for all
three videos at once. The contract's promise was always "returns None when there is no
plate"; failing to read counts as not reading. It now catches, logs, and returns `None`.

The same reasoning applies to imports. `plate_reader` and `detector` both loaded their
heavy packages at module level, so a missing OCR install stopped `main.py` from importing
and nobody could run the camera, database or linker either. Both load on first use now and
raise a message naming the pip command. Every module in the package imports with no
third-party model packages installed.

---

## 9. torch and paddle fight over OpenMP on Windows

**Owner affected:** anyone running the pipeline

The detector runs on torch (via ultralytics) and the OCR runs on paddle. Both ship their
own copy of `libiomp5md.dll`, and on Windows whichever loads first wins. If paddle loads
first, torch dies:

```
OSError: [WinError 127] The specified procedure could not be found.
Error loading "...	orch\lib\shm.dll" or one of its dependencies.
```

Reproduced exactly:

| order | result |
|---|---|
| `import torch` then `import paddle` | both fine |
| `import paddle` then `import torch` | **torch fails** |

In a normal run the detector loads torch first anyway, so the pipeline works by luck.
`plate_reader._load_awiros_model` now imports torch immediately before paddle so the order
is guaranteed rather than accidental. If you ever see that `shm.dll` error, something
loaded paddle first.

Related: do not `pip install paddleocr`. The PyPI package does not ship the `ppocr` module
we need, and it downgrades numpy and installs a second OpenCV, which is what broke this
environment in the first place. Get `ppocr` from the PaddleOCR source, as the README says.

---

## 10. What is not in git

**Test footage** (`anpr/data/videos/`) — too big, and it is our own recording. Share over
Drive; note each clip's real start time in the sheet, because `Camera(start_time=...)` is
what puts two videos on one clock.

**Model weights** — ultralytics downloads `yolo26s.pt` itself on first run. The Awiros OCR
weights (`anpr/model.safetensors`, `anpr/en_dict.txt`) are fetched by hand from Hugging
Face, and the PaddleOCR checkout (`anpr/PaddleOCR/`) is cloned once — see the README.
All three are gitignored. If we ever fine-tune, that checkpoint is shared by hand like
the footage.

**Run output** (`anpr/out/`) — regenerated by one command, so it stays out of git.
Deck screenshots are the exception: `git add -f` the specific ones person 5 wants to keep.
