"""Owner: person 5. Annotated frames and the camera map. Owns every screenshot.

STUB — draws nothing real yet, but returns the right types so main.py runs.
"""

import numpy as np

from . import config


def draw_frame(frame: np.ndarray, tracks: list[dict]) -> np.ndarray:
    """Boxes, track id, vehicle type, plate text (or 'reading...' / 'no plate').
    Green box = plate locked, amber = still reading, grey = no plate.
    """
    return frame.copy()


def draw_map(cameras: list[dict], links: list[dict],
             out_path: str = str(config.MAP_PNG)) -> str:
    """Camera dots on a map, one line per link. Solid = exact/fuzzy, dashed = inferred.
    Returns the path to the written PNG.
    """
    return out_path
