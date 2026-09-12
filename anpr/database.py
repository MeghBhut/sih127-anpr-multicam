"""Owner: person 4. Sightings and links, SQLite for the prototype.

STUB — an in-memory list stands in for the tables so main.py runs day 2.
Signatures are frozen; swap the bodies for real sqlite3 calls.

Sighting dict:
{"cam_id": "C1", "t_in": float, "t_out": float,
 "plate": "GJ01AB1234" | None, "conf": [0.9, ...] | None,
 "type": "car", "colour": "white", "direction": "north",
 "embedding": np.ndarray | None, "crop_path": "out/crops/17.jpg"}
"""

from . import config

_SIGHTINGS: list[dict] = []
_LINKS: list[dict] = []
_CAMERAS: list[dict] = []


def init_db(path: str = str(config.DB_PATH)) -> None:
    """Create tables cameras, sightings, links if they do not exist."""
    _SIGHTINGS.clear()
    _LINKS.clear()
    _CAMERAS.clear()


def save_camera(cam_id: str, lat: float, lon: float, name: str) -> None:
    _CAMERAS.append({"cam_id": cam_id, "lat": lat, "lon": lon, "name": name})


def get_cameras() -> list[dict]:
    return list(_CAMERAS)


def save_sighting(s: dict) -> int:
    """Returns sighting_id."""
    s = dict(s)
    s["id"] = len(_SIGHTINGS) + 1
    _SIGHTINGS.append(s)
    return s["id"]


def get_sightings() -> list[dict]:
    return list(_SIGHTINGS)


def get_by_plate(plate: str) -> list[dict]:
    return [s for s in _SIGHTINGS if s.get("plate") == plate]


def get_candidates(s: dict, window: float) -> list[dict]:
    """Sightings on other cameras within `window` seconds after s."""
    return [o for o in _SIGHTINGS
            if o["cam_id"] != s["cam_id"] and 0 < o["t_in"] - s["t_out"] <= window]


def save_link(a_id: int, b_id: int, score: float, method: str) -> None:
    _LINKS.append({"a": a_id, "b": b_id, "score": score, "method": method})


def get_links() -> list[dict]:
    return list(_LINKS)
