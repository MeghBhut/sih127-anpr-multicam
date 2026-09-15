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
ROAD_MAP_PNG = OUT_DIR / "tracker_map.png"   # the schematic tracker map
ROUTE_DIR = OUT_DIR / "routes"               # one map per tracked vehicle
JOURNEY_DIR = OUT_DIR / "journeys"    # one evidence card per linked vehicle
MONTAGE_DIR = OUT_DIR / "montage"     # all cameras at one moment
MONTAGE_COUNT = 6                     # how many moments to render
DB_PATH = OUT_DIR / "anpr.db"

for _d in (VIDEO_DIR, FRAME_DIR, CROP_DIR, FAIL_CROP_DIR, JOURNEY_DIR,
           MONTAGE_DIR, ROUTE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- camera
SKIP = 2                 # process 1 of every (SKIP + 1) frames
FPS_FALLBACK = 25.0      # used when the file reports no fps

# ---------------------------------------------------------------- detector
MODEL_PATH = "yolo26s.pt"
CONF = 0.4               # YOLO confidence floor
TRACKER_YAML = "botsort.yaml"
TRACK_TIMEOUT = 30       # frames unseen before a track counts as finished

# quality = normalised(crop area) * normalised(sharpness), each capped at 1.0.
# A crop at or above these values scores full marks on that half.
QUALITY_MAX_AREA = 120000          # ~350x350 px
QUALITY_MAX_SHARPNESS = 500.0      # cv2.Laplacian(...).var()

# A track whose box centre moves less than this (pixels, first to last) is
# called "stationary" rather than given a compass direction.
DIRECTION_MIN_DISPLACEMENT_PX = 40

# ---------------------------------------------------------------- plate reader
# Two different gates, deliberately named apart. MIN_VEHICLE_PX is checked on
# the whole vehicle crop before OCR runs at all; MIN_PLATE_PX is checked on the
# detected plate box afterwards. A car is always bigger than 60px, so applying
# the plate threshold to the vehicle made the gate do nothing.
MIN_VEHICLE_PX = 80      # smallest vehicle crop worth running the detector on
MIN_PLATE_PX = 60        # smallest detected plate box worth running OCR on
MIN_SHARPNESS = 50.0     # cv2.Laplacian(...).var() floor on the vehicle crop

# Crops that fail the gate or come back unreadable are kept for the future
# fine-tuning set and the risk slide. Capped so a long run cannot fill the disk.
SAVE_FAIL_CROPS = True
MAX_FAIL_CROPS = 200

# fast-alpr's underlying plate detector (open-image-models). Larger image
# size = more accurate, slower. Available sizes: 256/384/416/512/640, plus
# a higher-accuracy 608 variant. Start at 384; drop to 256 if you need more
# speed on CPU/edge hardware, go to 608 if small/distant plates are being missed.
PLATE_DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"

# None = use the model's own default threshold (0.25 for the yolo-v9-t
# family). Raise this if the detector reports false-positive "plates";
# lower it if it's missing real ones.
#
# Measured on 300 frames of c1.mp4: 1076 raw detections, of which only 205
# were even plate-shaped. Raising this to 0.45 leaves 146 and to 0.60 leaves
# 115 -- but that trades away real plates too, so tune it against footage
# rather than guessing.
PLATE_DETECTOR_CONF_THRESH = None

# Shape sanity on whatever the detector calls a plate. These are physical
# facts about number plates, not tuning knobs: a plate is never taller than
# it is wide (a dual-row plate is roughly square, ~1.0-2.0), and it never
# fills half the vehicle it is bolted to. Without these the detector happily
# reports a tarpaulin or a windscreen as a plate at 0.69 confidence, and each
# one spends part of that vehicle's OCR budget producing nothing.
MIN_PLATE_ASPECT = 1.0          # width / height
MAX_PLATE_AREA_FRACTION = 0.5   # of the vehicle crop

# Path to the cloned PaddleOCR repo. None = auto-detect next to
# plate_reader.py, or auto-clone there if not found anywhere.
PADDLEOCR_DIR = None

# Awiros-ANPR-OCR weights + character dictionary.
# Download both from: https://huggingface.co/Awiros/anpr-ocr  (the official repo)
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

# A gap-recovered character is a guess from the model's runner-up probabilities,
# not something it actually read. Cap it below MIN_CHAR_CONF so a guess can
# never be what locks a plate. Raise it if recovery proves reliable on the real
# footage; the read still counts, it just cannot carry the decision alone.
OCR_RECOVERED_MAX_CONF = 0.45

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

# Map rendering. Camera pins are drawn over real street tiles fetched once from
# OpenStreetMap and cached, so the map still renders offline afterwards and
# still makes sense when the prototype grows from 2 cameras to 5 or 6 spread
# across different streets. If the fetch fails the map falls back to a plain
# plot rather than failing the run.
MAP_DIR = DATA_DIR / "map_cache"
MAP_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
MAP_PADDING = 0.25        # fraction of the camera spread added around the edges
MAP_MAX_TILES = 4         # per axis; more tiles = more detail, slower first fetch
MAP_OFFLINE = False       # True skips the network entirely

# The geographic map (out/map.png) is off. It drew camera dots on a street
# view with nothing about vehicles, and out/tracker_map.png replaces it.
# Set True if you ever want the real-coordinates version back.
DRAW_GEO_MAP = False

# How often to write an annotated frame to out/frames/. 1 = every processed
# frame (thousands of files), 0 = never. These are the deck screenshots.
SAVE_FRAME_EVERY = 10

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

# ---------------------------------------------------------- schematic map
# The tracker map is a SCHEMATIC, not a geographic one: you draw the streets
# yourself as polylines and drop cameras onto them. Two reasons that beats a
# real map for this prototype -- it needs no network at demo time, and a
# hand-drawn layout of the two streets we actually filmed reads instantly,
# where a satellite view of Ahmedabad does not.
#
# ROADS: name -> list of (x, y) points, in an abstract canvas space. Where two
# roads share a point (within JUNCTION_TOLERANCE px) they become a junction,
# so a vehicle's route between cameras follows real roads rather than a
# straight line. Adding a third street with two more cameras is four lines
# here and no code change.
# Two streets meeting at a right angle, a camera at the start of each.
# street_1 runs left-to-right and ends at the corner; street_2 starts at the
# bottom and runs up INTO that same corner, so the two roads share the point
# (760, 160) and the junction is real. A vehicle goes C1 -> corner -> C2.
ROADS = {
    "street_1": [(120, 160), (760, 160)],    # horizontal, camera at its left end
    "street_2": [(760, 540), (760, 160)],    # vertical, camera at its bottom end
}

# cam_id -> (road name, how far along it, 0.0 = start, 1.0 = end)
#
# Both cameras sit at 0.0, the start of their own road, which puts them at the
# two far ends with the corner between them. Only list cameras you actually
# filmed: one with no clip still draws as "0 seen", which on a slide reads as
# more cameras than you have.
CAMERA_PLACEMENT = {
    "C1": ("street_1", 0.0),
    "C2": ("street_2", 0.0),
}

JUNCTION_TOLERANCE = 14      # px; roads closer than this are treated as joined

# One route card per identified vehicle, capped so a busy run does not write
# hundreds of images. Vehicles seen at more cameras are drawn first, then the
# ones whose plate was read most cleanly.
MAX_ROUTE_CARDS = 40

# ---------------------------------------------------------------- cameras
# Unix time each clip started recording. The linker compares sightings across
# cameras on one shared clock, so these are what make two videos comparable.
# None = fall back to "now", which is only correct for a live source.
#
# PERSON 1: fill these in from the real recording times. Until you do, every
# clip is treated as starting at the moment the run began, so all three look
# simultaneous and no travel-time window can ever match.
# Both clips are treated as starting at the same instant. The exact value does
# not matter while they share it -- only the difference between cameras does.
# Replace with the real recording times when person 1 has them.
_ASSUMED_START = 1789315000.0

RECORDING_START = {
    "C1": _ASSUMED_START,
    "C2": _ASSUMED_START,
    "C3": None,
}

# cam_id -> (lat, lon, human name). Fill in the real spots after recording.
CAMERAS = {
    "C1": (23.0225, 72.5714, "Gate A"),
    "C2": (23.0300, 72.5800, "Crossing B"),
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
