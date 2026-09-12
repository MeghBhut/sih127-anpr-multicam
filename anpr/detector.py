"""Owner: person 2. Detect + track vehicles, describe them, keep the best crop.

STUB — returns fake data matching the frozen signature.
Crops are BGR numpy arrays. Do not convert to RGB in here.
"""

import time

import numpy as np

from . import config


class Detector:
    def __init__(self, model_path: str = config.MODEL_PATH, conf: float = config.CONF):
        """Loads YOLO26 and BoT-SORT. Use gmc_method: none (fixed cameras)."""
        self.model_path = model_path
        self.conf = conf
        self._finished: list[dict] = []
        self._calls = 0

    def update(self, frame: np.ndarray, cam_id: str) -> list[dict]:
        """One frame in. LIVE tracks currently visible out:
           {"track_id": 17, "box": (x1,y1,x2,y2), "type": "car",
            "colour": "white", "crop": np.ndarray, "quality": 0.72}
           quality in 0..1 = normalised(crop area) * normalised(sharpness)
        """
        track = {
            "track_id": 17,
            "box": (100, 200, 400, 500),
            "type": "car",
            "colour": "white",
            "crop": np.zeros((300, 300, 3), dtype=np.uint8),
            "quality": 0.72,
        }
        self._calls += 1
        if self._calls % 5 == 0:                       # fake a vehicle leaving the frame
            now = time.time()
            self._finished.append({
                "track_id": 17,
                "cam_id": cam_id,
                "t_in": now - 4.0,
                "t_out": now,
                "type": "car",
                "colour": "white",
                "best_crop": track["crop"],
                "direction": "north",
                "embedding": None,
            })
        return [track]

    def finished_tracks(self) -> list[dict]:
        """Tracks that have left the frame since the last call:
           {"track_id": 17, "cam_id": "C1", "t_in": float, "t_out": float,
            "type": "car", "colour": "white", "best_crop": np.ndarray,
            "direction": "north", "embedding": np.ndarray | None}
        """
        out, self._finished = self._finished, []
        return out
