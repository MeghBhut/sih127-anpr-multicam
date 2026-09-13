import time
import logging
import cv2

logger = logging.getLogger(__name__)

DEFAULT_FPS = 25.0
RTSP_RECONNECT_DELAY = 2.0      
RTSP_MAX_RECONNECT_ATTEMPTS = 5  


def _source_is_rtsp(source: str) -> bool:
    return isinstance(source, str) and source.lower().startswith("rtsp://")


def _source_is_webcam_index(source: str) -> bool:
    return isinstance(source, str) and source.isdigit()


class Camera:
    def __init__(self, cam_id: str, source: str, skip: int = 2, start_time: float | None = None):
        self.cam_id = cam_id
        self.source = source
        self.skip = max(0, skip)
        self.is_rtsp = _source_is_rtsp(source)
        self.is_live = self.is_rtsp or _source_is_webcam_index(source)

        self.start_time = start_time if start_time is not None else time.time()

        self._cap = None
        self._fps = DEFAULT_FPS
        self._frame_no = 0          
        self._reconnect_attempts = 0

        self._open()

    def _open(self):

        cap_source = int(self.source) if _source_is_webcam_index(self.source) else self.source
        cap = cv2.VideoCapture(cap_source)

        if not cap.isOpened():
            cap.release()
            raise IOError(f"[{self.cam_id}] could not open source: {self.source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps != fps: 
            logger.warning("[%s] fps not reported by source, falling back to %.1f", self.cam_id, DEFAULT_FPS)
            fps = DEFAULT_FPS
        self._fps = fps

        self._cap = cap
        self._reconnect_attempts = 0
        logger.info("[%s] opened source=%s fps=%.2f skip=%d", self.cam_id, self.source, self._fps, self.skip)

    def _try_reconnect(self) -> bool:
        
        if not self.is_rtsp:
            return False

        if self._cap is not None:
            self._cap.release()

        while self._reconnect_attempts < RTSP_MAX_RECONNECT_ATTEMPTS:
            self._reconnect_attempts += 1
            logger.warning(
                "[%s] RTSP stream dropped, reconnect attempt %d/%d",
                self.cam_id, self._reconnect_attempts, RTSP_MAX_RECONNECT_ATTEMPTS,
            )
            time.sleep(RTSP_RECONNECT_DELAY)
            try:
                self._open()
                return True
            except IOError:
                continue

        logger.error("[%s] giving up on RTSP reconnect after %d attempts", self.cam_id, RTSP_MAX_RECONNECT_ATTEMPTS)
        return False

    def _compute_t(self, frame_no: int) -> float:
        return self.start_time + frame_no / self._fps

    def frames(self):
        stride = self.skip + 1

        while True:
            ok, frame = self._cap.read()

            if not ok:
                if self.is_rtsp:
                    if self._try_reconnect():
                        continue
                    else:
                        break
                else:
                    logger.info("[%s] end of stream", self.cam_id)
                    break

            current_frame_no = self._frame_no
            self._frame_no += 1

            if current_frame_no % stride != 0:
                continue

            yield {
                "cam_id": self.cam_id,
                "t": self._compute_t(current_frame_no),
                "frame_no": current_frame_no,
                "frame": frame,
            }

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("[%s] released", self.cam_id)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Camera smoke test / recorder")
    parser.add_argument("source", help="file path, webcam index, or rtsp:// url")
    parser.add_argument("--cam-id", default="C1")
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--record", help="output path to save a raw recording, e.g. data/videos/c1.mp4")
    parser.add_argument("--seconds", type=float, default=10.0, help="how long to record for")
    args = parser.parse_args()

    start = time.time()
    print(f"[{args.cam_id}] real start_time = {start}")

    cam = Camera(args.cam_id, args.source, skip=args.skip, start_time=start)

    writer = None
    try:
        for item in cam.frames():
            frame = item["frame"]

            if args.record:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record, fourcc, cam._fps, (w, h))
                writer.write(frame)

            if item["t"] - start >= args.seconds:
                break
    finally:
        if writer is not None:
            writer.release()
        cam.release()