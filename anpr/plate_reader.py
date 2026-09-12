"""Owner: person 3. Vehicle crop in, characters + per-character confidence out.

STUB — returns fake data matching the frozen signature.
Anything unknown is None. An unreadable slot is '?' with conf 0.0.
"""

import numpy as np

from . import config


def passes_gate(crop: np.ndarray) -> bool:
    """Cheap pre-check before running OCR: min crop size and min sharpness."""
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    return min(h, w) >= config.MIN_PLATE_PX


def read_plate(crop: np.ndarray) -> dict | None:
    """Returns
         {"chars": ['G','J','0','1','A','B','1','2','3','4'],
          "confs": [0.98, 0.97, ...],        # same length as chars
          "plate_box": (x1,y1,x2,y2)}        # relative to the crop
       None if no plate is found or the crop fails the quality gate.
    """
    if not passes_gate(crop):
        return None
    return {
        "chars": list("GJ01AB1234"),
        "confs": [0.9] * 10,
        "plate_box": (10, 220, 200, 260),
    }
