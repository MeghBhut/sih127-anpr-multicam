"""Every threshold in the system. No magic numbers inside modules.

Owner: person 6 (Megh). Changing a value here is fine; changing a *name*
means telling everyone, because modules import these by name.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
VIDEO_DIR = DATA_DIR / "videos"
OUT_DIR = BASE / "out"
FRAME_DIR = OUT_DIR / "frames"
CROP_DIR = OUT_DIR / "crops"
FAIL_CROP_DIR = CROP_DIR / "fail"
MAP_PNG = OUT_DIR / "map.png"
DB_PATH = OUT_DIR / "anpr.db"

for _d in (VIDEO_DIR, FRAME_DIR, CROP_DIR, FAIL_CROP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- camera
SKIP = 2                 # process 1 of every (SKIP + 1) frames
FPS_FALLBACK = 25.0      # used when the file reports no fps

# ---------------------------------------------------------------- detector
MODEL_PATH = "yolo26s.pt"
CONF = 0.4               # YOLO confidence floor
TRACKER_YAML = "botsort.yaml"
TRACK_TIMEOUT = 30       # frames unseen before a track counts as finished

# ---------------------------------------------------------------- plate reader
MIN_PLATE_PX = 60        # reject crops narrower than this before OCR
MIN_SHARPNESS = 50.0     # cv2.Laplacian(...).var() floor

# fast-alpr's underlying plate detector (open-image-models). Larger image
# size = more accurate, slower. Available sizes: 256/384/416/512/640, plus
# a higher-accuracy 608 variant. Start at 384; drop to 256 if you need more
# speed on CPU/edge hardware, go to 608 if small/distant plates are being missed.
PLATE_DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"

# None = use the model's own default threshold (0.25 for the yolo-v9-t
# family). Raise this if the detector reports false-positive "plates";
# lower it if it's missing real ones.
PLATE_DETECTOR_CONF_THRESH = None

# Path to the cloned PaddleOCR repo. None = auto-detect next to
# plate_reader.py, or auto-clone there if not found anywhere.
PADDLEOCR_DIR = None

# Awiros-ANPR-OCR weights + character dictionary.
# Download both from: https://huggingface.co/surendran0m07/anpr-ocr
AWIROS_WEIGHTS_PATH = BASE / "model.safetensors"
AWIROS_DICT_PATH = BASE / "en_dict.txt"

# "gpu" or "cpu". Falls back to cpu automatically at load time if CUDA
# isn't available, even if this says "gpu".
OCR_DEVICE = "gpu"

# Minimum non-blank probability required inside a CTC decode gap before a
# recovered character is accepted (the HR38AB2421 digit-drop fix).
# Lower = more aggressive recovery, higher risk of inserting a wrong char.
# Higher = safer, more '?' placeholders left on genuinely ambiguous reads.
OCR_GAP_PROB_THRESHOLD = 0.15

# A kept CTC character below this confidence gets downgraded to '?' at
# 0.0 confidence rather than passed downstream looking as reliable as a
# high-confidence read.
OCR_MIN_CHAR_CONFIDENCE = 0.30

# Timestep gap must be wider than (typical_spacing * this) before it's
# treated as a possible dropped character (the HR38AB2421 fix).
OCR_GAP_WIDTH_MULTIPLIER = 1.8

# If a plate crop's width/height ratio is below this, it's treated as
# likely dual-row (two-line) rather than single-row, and gap-scanning is
# skipped for it — a two-line plate's row-transition reads as an
# unusually wide gap the same way a dropped character would, and the two
# aren't currently distinguishable. Single-row plates are typically wide
# (ratio > ~3), true dual-row plates are closer to square (ratio ~1-2).
DUAL_ROW_ASPECT_RATIO_THRESHOLD = 2.0

# ---------------------------------------------------------------- voting
MAX_OCR_PER_TRACK = 12   # OCR budget: hard cap of frames per track
MIN_CHAR_CONF = 0.5      # every position must beat this to lock
LOCK_MIN_READS = 3       # this many agreeing reads before a plate is locked

# ID-switch guard: if one track produces two reads that are each clean and
# this far apart, the tracker has swapped vehicles mid-track. Vote over the
# two halves separately instead of merging them into one nonsense plate.
ID_SWITCH_MIN_DISTANCE = 1.0

# ---------------------------------------------------------------- linker
# Accept thresholds. Placeholders until threshold_eval.py has been run on
# ~50 hand-labelled pairs; target precision >= 0.98.
LINK_FUZZY = 0.75        # exact / fuzzy: plate evidence on both sides
LINK_INFERRED = 0.85     # stricter: no usable plate on at least one side

TRAVEL_TIME_SLACK = 5.0  # seconds of slack on the travel-time window

# What counts as a usable plate at all.
MIN_KNOWN_PLATE_FRACTION = 0.4   # fraction of characters that must not be '?'
PLATE_QUALITY_THRESHOLD = 0.35   # declared by person 4; not yet read by linker.py

# Score adjustments.
TYPE_MISMATCH_PENALTY = 0.5      # declared by person 4; not yet read by linker.py
DIRECTION_MISMATCH_PENALTY = 0.15

# Cloned-plate alerts.
CLONE_TOLERANCE = 0.0            # extra seconds allowed before alerting
CLONE_MIN_PLATE_QUALITY = 0.6    # only alert on reasonably reliable plates

# Score weights, usable plate on both sides.
WEIGHT_PLATE = 0.60
WEIGHT_EMBEDDING = 0.30
WEIGHT_COLOUR = 0.10

# Score weights, no usable plate on one side.
WEIGHT_EMBEDDING_ONLY = 0.75
WEIGHT_COLOUR_ONLY = 0.25

# How a voted plate is classified when the sighting is written. Person 4's
# linker gates BOTH the "never merge two clean plates that differ" rule and
# cloned-plate detection on plate_status == "clean", so these decide whether
# those two safety nets are live at all.
CLEAN_MIN_CHAR_CONF = 0.8        # every character must beat this to be "clean"
PARTIAL_MAX_UNKNOWN_FRACTION = 0.5   # more '?' than this and it is "unreadable"

# ---------------------------------------------------------------- cameras
# cam_id -> (lat, lon, human name). Fill in the real spots after recording.
CAMERAS = {
    "C1": (23.0225, 72.5714, "Gate A"),
    "C2": (23.0300, 72.5800, "Crossing B"),
    "C3": (23.0380, 72.5900, "Exit C"),
}

# (from, to) -> (min_seconds, max_seconds) realistic travel window.
# Both directions are listed: linker.plausible() looks up the pair in
# chronological order, so a missing reverse entry silently kills every link
# in that direction.
#
# WARNING: anpr/linker.py currently defines its own copy of this dict at
# module level and does NOT read this one. The two are identical today.
# Person 4 needs to delete that copy and use config.CAMERA_GRAPH, or person
# 1's real measurements will land here and be silently ignored.
CAMERA_GRAPH = {
    ("C1", "C2"): (20.0, 180.0),
    ("C2", "C1"): (20.0, 180.0),
    ("C2", "C3"): (15.0, 150.0),
    ("C3", "C2"): (15.0, 150.0),
    ("C1", "C3"): (40.0, 300.0),
    ("C3", "C1"): (40.0, 300.0),
}
