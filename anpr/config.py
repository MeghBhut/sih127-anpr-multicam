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

# ---------------------------------------------------------------- voting
MAX_OCR_PER_TRACK = 12   # OCR budget: hard cap of frames per track
MIN_CHAR_CONF = 0.5      # every position must beat this to lock
LOCK_MIN_READS = 3       # this many agreeing reads before a plate is locked

# ---------------------------------------------------------------- linker
LINK_FUZZY = 0.75        # placeholder until tuned on labelled pairs
LINK_INFERRED = 0.85     # stricter: no plate on at least one side
LINK_SLACK = 5.0         # seconds of slack on the travel-time window

# ---------------------------------------------------------------- cameras
# cam_id -> (lat, lon, human name). Fill in the real spots after recording.
CAMERAS = {
    "C1": (23.0225, 72.5714, "Gate A"),
    "C2": (23.0300, 72.5800, "Crossing B"),
    "C3": (23.0380, 72.5900, "Exit C"),
}

# (from, to) -> (min_seconds, max_seconds) realistic travel window
CAMERA_GRAPH = {
    ("C1", "C2"): (20.0, 180.0),
    ("C2", "C3"): (25.0, 200.0),
    ("C1", "C3"): (50.0, 400.0),
}
