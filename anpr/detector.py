"""Vehicle detection, tracking, description, and best-crop selection."""

from pathlib import Path
import re

import cv2
import numpy as np

from . import config

VEHICLE_TYPES = {"car": "car", "motorcycle": "bike", "bus": "bus", "truck": "truck"}


def _tracker_config() -> str:
    path = Path(__file__).with_name("botsort_fixed_cam.yaml")
    if path.exists():
        return str(path)
    import ultralytics
    source = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml"
    if not source.exists():
        return "botsort.yaml"
    text = re.sub(r"^gmc_method:.*$", "gmc_method: none", source.read_text(), flags=re.MULTILINE)
    if "gmc_method: none" not in text:
        text += "\ngmc_method: none\n"
    path.write_text(text)
    return str(path)


def _colour(crop: np.ndarray) -> str:
    h, w = crop.shape[:2]
    middle = crop[h // 4:h - h // 4, w // 4:w - w // 4]
    if middle.size == 0:
        return "other"
    hsv = cv2.cvtColor(middle, cv2.COLOR_BGR2HSV)
    hue, saturation, value = (float(np.median(hsv[..., c])) for c in range(3))
    if saturation < 30:
        return "white" if value > 180 else "black" if value < 60 else "silver"
    if hue < 10 or hue > 170:
        return "red"
    return "blue" if 100 <= hue <= 130 else "other"


def _quality(crop: np.ndarray) -> float:
    area = crop.shape[0] * crop.shape[1]
    sharpness = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return float(min(area / config.QUALITY_MAX_AREA, 1.0) *
                 min(sharpness / config.QUALITY_MAX_SHARPNESS, 1.0))


def _direction(first: tuple[int, int, int, int], last: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = first
    X1, Y1, X2, Y2 = last
    dx, dy = (X1 + X2 - x1 - x2) / 2, (Y1 + Y2 - y1 - y2) / 2
    if max(abs(dx), abs(dy)) < config.DIRECTION_MIN_DISPLACEMENT_PX:
        return "stationary"
    if abs(dy) >= abs(dx):
        return "south" if dy > 0 else "north"
    return "east" if dx > 0 else "west"


class Detector:
    def __init__(self, model_path: str = config.MODEL_PATH, conf: float = config.CONF):
        self.model_path = model_path
        self.conf = conf
        self.tracker = _tracker_config()
        self.models: dict[str, object] = {}
        self.state: dict[tuple[str, int], dict] = {}
        self.last_seen: dict[tuple[str, int], int] = {}
        self.finished: list[dict] = []
        self.frame_number: dict[str, int] = {}

    def _model_for(self, cam_id: str):
        """One model per camera: model.track(persist=True) keeps its tracker
        state on the instance, so sharing one would mix track ids."""
        if cam_id not in self.models:
            try:
                from ultralytics import YOLO
            except ImportError as exc:      # keep the rest of the pipeline usable
                raise RuntimeError(
                    "detector needs ultralytics: pip install ultralytics") from exc
            self.models[cam_id] = YOLO(self.model_path)
        return self.models[cam_id]

    def update(self, frame: np.ndarray, cam_id: str, t: float) -> list[dict]:
        """Return live tracks: track_id, box, type, colour, crop, quality.

        `t` is the frame's own unix timestamp from camera.py, NOT the time this
        frame happens to be processed. The linker compares sightings across
        cameras on that clock, so using time.time() here would time every
        vehicle by when the laptop got round to it.
        """
        self.frame_number[cam_id] = self.frame_number.get(cam_id, 0) + 1
        model = self._model_for(cam_id)
        class_ids = [i for i, name in model.names.items() if name in VEHICLE_TYPES]
        results = model.track(frame, persist=True, tracker=self.tracker, conf=self.conf,
                              classes=class_ids, verbose=False)
        live, seen = [], set()
        boxes = results[0].boxes if results else None
        if boxes is not None and boxes.id is not None:
            rows = zip(boxes.xyxy.cpu().tolist(), boxes.id.cpu().tolist(), boxes.cls.cpu().tolist())
            for box, raw_id, raw_class in rows:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2].copy()
                track_id = int(raw_id)
                key = (cam_id, track_id)
                vehicle_type = VEHICLE_TYPES[model.names[int(raw_class)]]
                colour, quality = _colour(crop), _quality(crop)
                if key not in self.state:
                    self.state[key] = {"t_in": t, "first_box": (x1, y1, x2, y2),
                                       "last_box": (x1, y1, x2, y2), "best_crop": crop,
                                       "best_quality": quality, "type_votes": {}, "colour_votes": {},
                                       "t_last": t}
                state = self.state[key]
                state["last_box"] = (x1, y1, x2, y2)
                state["t_last"] = t
                state["type_votes"][vehicle_type] = state["type_votes"].get(vehicle_type, 0) + 1
                state["colour_votes"][colour] = state["colour_votes"].get(colour, 0) + 1
                if quality > state["best_quality"]:
                    state["best_crop"], state["best_quality"] = crop, quality
                self.last_seen[key] = self.frame_number[cam_id]
                seen.add(key)
                live.append({"track_id": track_id, "box": (x1, y1, x2, y2), "type": vehicle_type,
                             "colour": colour, "crop": crop, "quality": quality})
        for key, last_frame in list(self.last_seen.items()):
            if key[0] == cam_id and key not in seen and self.frame_number[cam_id] - last_frame >= config.TRACK_TIMEOUT:
                self._finish(key)
        return live

    def _finish(self, key: tuple[str, int]) -> None:
        state = self.state.pop(key, None)
        self.last_seen.pop(key, None)
        if state is None:
            return
        cam_id, track_id = key
        self.finished.append({"track_id": track_id, "cam_id": cam_id, "t_in": state["t_in"],
                              "t_out": state["t_last"],
                              "type": max(state["type_votes"], key=state["type_votes"].get),
                              "colour": max(state["colour_votes"], key=state["colour_votes"].get),
                              "best_crop": state["best_crop"],
                              "direction": _direction(state["first_box"], state["last_box"]),
                              "embedding": None})

    def flush(self, cam_id: str | None = None) -> None:
        """Finish every track still open, for one camera or all of them.

        Call this when a clip ends. Without it, any vehicle still on screen in
        the last frame stays open forever and never becomes a sighting -- the
        timeout that normally closes a track only runs from inside update().
        """
        for key in [k for k in list(self.state) if cam_id is None or k[0] == cam_id]:
            self._finish(key)

    def finished_tracks(self) -> list[dict]:
        finished, self.finished = self.finished, []
        return finished
