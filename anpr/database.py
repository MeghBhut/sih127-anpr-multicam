"""
Vehicle identity resolution layer -- persistence.

This module owns the SQLite prototype database for the ANPR project. It is
deliberately simple (plain SQL over sqlite3, no ORM) so that a later
migration to PostgreSQL only requires swapping the connection layer and,
at most, adapting a couple of SQLite-specific pragmas.

Design notes
------------
* Every sighting is stored even when its plate is unreadable. Plate
  reliability is tracked explicitly via ``plate_status`` and
  ``plate_quality`` rather than being inferred from the plate string alone,
  so ``linker.py`` can make informed, conservative decisions.
* NumPy arrays (embeddings) and per-character confidence lists are not
  natively storable in SQLite, so they are JSON-encoded on write and
  JSON-decoded (with validation) on read. Any malformed data is treated as
  "missing" rather than raising, so one bad row cannot crash the pipeline.
* All SQL is parameterized. All connections are context-managed.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

PLATE_STATUSES = {"clean", "partial", "uncertain", "unreadable", "missing"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cam_id TEXT UNIQUE NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    name TEXT
);

CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cam_id TEXT NOT NULL,
    t_in REAL NOT NULL,
    t_out REAL NOT NULL,

    plate TEXT,
    plate_conf TEXT,

    plate_quality REAL,
    plate_status TEXT,

    vehicle_type TEXT,
    colour TEXT,
    direction TEXT,

    embedding TEXT,

    crop_path TEXT,

    created_at REAL
);

CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    a_id INTEGER NOT NULL,
    b_id INTEGER NOT NULL,
    score REAL NOT NULL,
    method TEXT NOT NULL,
    created_at REAL,
    UNIQUE(a_id, b_id)
);

CREATE INDEX IF NOT EXISTS idx_sightings_plate ON sightings(plate);
CREATE INDEX IF NOT EXISTS idx_sightings_cam_time ON sightings(cam_id, t_in, t_out);
"""

# The module keeps a single "current" database path so callers can use the
# functional API (`init_db`, `save_sighting`, ...) without threading a
# connection object through every call, mirroring the required interface.
_DB_PATH: Optional[str] = None


def init_db(path: str) -> None:
    """Create the database file (if needed) and required tables/indices.

    Safe to call multiple times; existing tables/data are preserved.
    """
    global _DB_PATH
    _DB_PATH = path
    with _connect(path) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()
    logger.debug("Initialized database at %s", path)


@contextmanager
def _connect(path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    """Context-managed SQLite connection with row factory set to dict-like."""
    db_path = path or _DB_PATH
    if not db_path:
        raise RuntimeError("database not initialized: call init_db(path) first")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _encode_json(value: Any) -> Optional[str]:
    """Safely JSON-encode a value, returning None on failure or None input."""
    if value is None:
        return None
    try:
        # Convert numpy arrays / numpy scalars to plain Python types.
        if hasattr(value, "tolist"):
            value = value.tolist()
        return json.dumps(value)
    except (TypeError, ValueError) as exc:
        logger.warning("Failed to JSON-encode value: %s", exc)
        return None


def _decode_json(raw: Optional[str]) -> Optional[Any]:
    """Safely JSON-decode a value, returning None on any failure."""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Failed to JSON-decode stored value: %s", exc)
        return None


def encode_embedding(embedding: Any) -> Optional[str]:
    """Encode an embedding (list/np.ndarray/None) as a JSON string.

    Returns None if the embedding is missing or invalid, rather than
    raising, so a single bad embedding cannot crash ingestion.
    """
    if embedding is None:
        return None
    try:
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        vec = [float(x) for x in embedding]
        if not vec:
            return None
        return json.dumps(vec)
    except (TypeError, ValueError) as exc:
        logger.warning("Invalid embedding, storing as None: %s", exc)
        return None


def decode_embedding(raw: Optional[str]) -> Optional[list]:
    """Decode a stored embedding, validating it is a non-empty numeric list."""
    data = _decode_json(raw)
    if data is None:
        return None
    if not isinstance(data, list) or not data:
        return None
    try:
        return [float(x) for x in data]
    except (TypeError, ValueError):
        return None


def normalize_plate(plate: Optional[str]) -> Optional[str]:
    """Normalize a plate string for comparison/storage.

    - uppercase
    - strip spaces, hyphens, and other punctuation
    - preserve '?' as an explicit "unknown character" marker
    - never invents missing characters

    Returns None for None/empty input.
    """
    if plate is None:
        return None
    cleaned = []
    for ch in str(plate).upper():
        if ch.isalnum() or ch == "?":
            cleaned.append(ch)
        # spaces, hyphens, punctuation are dropped silently
    result = "".join(cleaned)
    return result or None


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

def save_camera(cam_id: str, lat: float, lon: float, name: str) -> None:
    """Insert or update a camera record, keyed by cam_id."""
    if not cam_id:
        raise ValueError("cam_id is required")
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO cameras (cam_id, lat, lon, name)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cam_id) DO UPDATE SET
                lat = excluded.lat,
                lon = excluded.lon,
                name = excluded.name
            """,
            (cam_id, lat, lon, name),
        )
        conn.commit()


def get_camera(cam_id: str) -> Optional[dict]:
    """Fetch a single camera by cam_id, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM cameras WHERE cam_id = ?", (cam_id,)
        ).fetchone()
        return dict(row) if row else None


def get_cameras() -> list[dict]:
    """Return all known cameras."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM cameras").fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Sightings
# ---------------------------------------------------------------------------

def _row_to_sighting(row: sqlite3.Row) -> dict:
    """Convert a DB row into the sighting dict shape used by linker.py."""
    d = dict(row)
    d["plate_conf"] = _decode_json(d.get("plate_conf"))
    d["embedding"] = decode_embedding(d.get("embedding"))
    return d


def save_sighting(s: dict) -> int:
    """Persist a sighting and return its new row id.

    Accepts a dict shaped like the merged output of Person 2's tracker and
    Person 3/6's plate reader. Missing/uncertain plate data is preserved
    rather than discarded -- a sighting with no readable plate is still
    stored and can later be matched on appearance alone.

    Expected (all optional except cam_id/t_in/t_out) keys:
        cam_id, t_in, t_out, plate, plate_conf, plate_quality,
        plate_status, vehicle_type / type, colour, direction,
        embedding, crop_path / best_crop_path
    """
    if not s.get("cam_id"):
        raise ValueError("sighting requires cam_id")
    if s.get("t_in") is None or s.get("t_out") is None:
        raise ValueError("sighting requires t_in and t_out")

    plate = normalize_plate(s.get("plate"))
    plate_status = s.get("plate_status")
    if plate_status not in PLATE_STATUSES:
        # Infer a reasonable default if the caller didn't classify it.
        plate_status = "missing" if plate is None else "uncertain"

    plate_conf = s.get("plate_conf")
    plate_conf_json = _encode_json(plate_conf) if plate_conf is not None else None

    embedding_json = encode_embedding(s.get("embedding"))

    vehicle_type = s.get("vehicle_type") or s.get("type")
    crop_path = s.get("crop_path") or s.get("best_crop_path")

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO sightings (
                cam_id, t_in, t_out, plate, plate_conf, plate_quality,
                plate_status, vehicle_type, colour, direction, embedding,
                crop_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                s["cam_id"],
                float(s["t_in"]),
                float(s["t_out"]),
                plate,
                plate_conf_json,
                s.get("plate_quality"),
                plate_status,
                vehicle_type,
                s.get("colour"),
                s.get("direction"),
                embedding_json,
                crop_path,
                time.time(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_sighting(sighting_id: int) -> Optional[dict]:
    """Fetch a single sighting by id, or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM sightings WHERE id = ?", (sighting_id,)
        ).fetchone()
        return _row_to_sighting(row) if row else None


def get_by_plate(plate: str) -> list[dict]:
    """Return sightings with an exact normalized-plate match.

    This is an exact lookup only -- it will not find partial or uncertain
    plates that merely resemble ``plate``. Fuzzy/partial matching across
    uncertain observations is the responsibility of linker.py, which has
    access to the weighted plate-distance logic and confidence data needed
    to make that judgment safely.
    """
    norm = normalize_plate(plate)
    if not norm:
        return []
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sightings WHERE plate = ?", (norm,)
        ).fetchall()
        return [_row_to_sighting(r) for r in rows]


def get_all_sightings() -> list[dict]:
    """Return every sighting in the database (used by link_all())."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM sightings").fetchall()
        return [_row_to_sighting(r) for r in rows]


def get_candidates(s: dict, window: float) -> list[dict]:
    """Return plausible candidate sightings to compare against ``s``.

    This performs cheap SQL-level filtering only, to avoid an O(n^2)
    all-pairs comparison in linker.py:

    * different camera (a vehicle cannot be re-seen by the same camera as
      a "link" in the cross-camera sense; same-camera re-identification is
      out of scope here)
    * temporal overlap with ``s`` widened by ``window`` seconds on both
      sides (covers "before" and "after" cases without assuming a direction
      of travel between cameras)

    Direction/type compatibility and travel-time plausibility per specific
    camera pair are checked by linker.py, which has access to the camera
    graph and per-pair min/max travel times.
    """
    if "cam_id" not in s or "t_in" not in s or "t_out" not in s:
        return []

    lo = float(s["t_in"]) - float(window)
    hi = float(s["t_out"]) + float(window)

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM sightings
            WHERE cam_id != ?
              AND t_in <= ?
              AND t_out >= ?
            """,
            (s["cam_id"], hi, lo),
        ).fetchall()
        candidates = [_row_to_sighting(r) for r in rows]

    # Exclude the sighting itself if it happens to already be in the DB
    # (e.g. caller passed a dict loaded via get_sighting()).
    self_id = s.get("id")
    if self_id is not None:
        candidates = [c for c in candidates if c.get("id") != self_id]
    return candidates


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

def save_link(a_id: int, b_id: int, score: float, method: str) -> None:
    """Persist a link between two sightings, avoiding (a,b)/(b,a) duplicates."""
    lo, hi = (a_id, b_id) if a_id <= b_id else (b_id, a_id)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO links (a_id, b_id, score, method, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(a_id, b_id) DO UPDATE SET
                score = excluded.score,
                method = excluded.method,
                created_at = excluded.created_at
            """,
            (lo, hi, float(score), method, time.time()),
        )
        conn.commit()


def get_links() -> list[dict]:
    """Return all stored links."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM links").fetchall()
        return [dict(r) for r in rows]
