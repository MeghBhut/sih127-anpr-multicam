"""Owner: person 5 / person 6. The street layout, saved outside the code.

config.ROADS and config.CAMERA_PLACEMENT are the defaults. Once someone draws
a layout in the dashboard it is written to data/layout.json and that wins, so
changing where the cameras sit never means editing Python.

    load()   -> {"roads": {...}, "cameras": {...}}   json if present, else config
    save(d)  -> writes data/layout.json
    reset()  -> deletes it, going back to the config defaults
"""

import json
import logging

from . import config

logger = logging.getLogger(__name__)

LAYOUT_PATH = config.DATA_DIR / "layout.json"


def defaults() -> dict:
    """The layout as config.py declares it."""
    return {
        "roads": {name: [list(p) for p in points]
                  for name, points in config.ROADS.items()},
        "cameras": {cam: [road, frac]
                    for cam, (road, frac) in config.CAMERA_PLACEMENT.items()},
    }


def load() -> dict:
    """The layout in use: the saved one if there is one, else config's."""
    if not LAYOUT_PATH.is_file():
        return defaults()
    try:
        data = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("layout.json unreadable (%s); using config defaults", exc)
        return defaults()

    roads, cameras = data.get("roads"), data.get("cameras")
    if not isinstance(roads, dict) or not isinstance(cameras, dict):
        logger.warning("layout.json has no roads/cameras; using config defaults")
        return defaults()
    return {"roads": roads, "cameras": cameras}


def save(data: dict) -> dict:
    """Validate and write a layout. Raises ValueError on anything unusable.

    Validation is not ceremony: a road with one point cannot be drawn, and a
    camera on a road that does not exist silently vanishes from every map.
    Better to refuse the save than to write a layout that renders wrong.
    """
    roads = data.get("roads") or {}
    cameras = data.get("cameras") or {}

    if not isinstance(roads, dict) or not roads:
        raise ValueError("at least one road is needed")

    clean_roads = {}
    for name, points in roads.items():
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"road '{name}' needs at least 2 points")
        pts = []
        for p in points:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError(f"road '{name}' has a malformed point")
            pts.append([float(p[0]), float(p[1])])
        clean_roads[str(name)] = pts

    clean_cams = {}
    for cam, place in cameras.items():
        if not isinstance(place, (list, tuple)) or len(place) != 2:
            raise ValueError(f"camera '{cam}' needs [road, fraction]")
        road, frac = place
        if road not in clean_roads:
            raise ValueError(f"camera '{cam}' sits on unknown road '{road}'")
        clean_cams[str(cam)] = [str(road), max(0.0, min(1.0, float(frac)))]

    out = {"roads": clean_roads, "cameras": clean_cams}
    LAYOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAYOUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def reset() -> dict:
    """Forget the drawn layout and fall back to config.py."""
    if LAYOUT_PATH.is_file():
        LAYOUT_PATH.unlink()
    return defaults()


def as_network_args() -> tuple[dict, dict]:
    """The layout in the shape RoadNetwork wants."""
    data = load()
    roads = {name: [tuple(p) for p in points]
             for name, points in data["roads"].items()}
    cameras = {cam: (place[0], float(place[1]))
               for cam, place in data["cameras"].items()}
    return roads, cameras
