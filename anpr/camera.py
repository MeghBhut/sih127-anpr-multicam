"""Owner: person 1. Frames tagged with a camera id and a real wall-clock time.

STUB — returns fake data matching the frozen signature so main.py runs end to end.
Replace the body, never the signature.
"""

import time

import numpy as np

from . import config


class Camera:
    def __init__(self, cam_id: str, source: str, skip: int = config.SKIP,
                 start_time: float | None = None):
        """
        cam_id     : "C1"
        source     : file path, webcam index as str ("0"), or rtsp:// url
        skip       : process 1 of every (skip+1) frames
        start_time : unix time the recording started; live source -> time.time()
        """
        self.cam_id = cam_id
        self.source = source
        self.skip = skip
        self.start_time = start_time if start_time is not None else time.time()
        self.fps = config.FPS_FALLBACK
        self.cap = None

    def frames(self):
        """Generator. One dict per processed frame:
           {"cam_id": "C1", "t": 1726123456.78, "frame_no": 412, "frame": np.ndarray}
           t = start_time + frame_no / fps   for a file source
        """
        for frame_no in range(0, 30, self.skip + 1):
            yield {
                "cam_id": self.cam_id,
                "t": self.start_time + frame_no / self.fps,
                "frame_no": frame_no,
                "frame": np.zeros((720, 1280, 3), dtype=np.uint8),   # BGR
            }

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
