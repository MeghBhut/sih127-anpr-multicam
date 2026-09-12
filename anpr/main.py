"""Owner: person 6 (Megh). Joins everything and owns the OCR budget.

Run from the repo root:
    python -m anpr.main
"""

import argparse
import time
from collections import defaultdict

from . import config, database, linker, plate_logic, plate_reader, visualizer
from .camera import Camera
from .detector import Detector


def build_sighting(ft: dict, reads: list[dict]) -> dict:
    """Finished track + its OCR reads -> one row for the sightings table."""
    plate, conf, locked = (None, None, False)
    if reads:
        plate, conf, locked = plate_logic.vote(reads, config.MIN_CHAR_CONF)
        if not plate or "?" in plate:
            plate, conf = None, None          # unknown is None, never "" and never "unknown"
    return {
        "cam_id": ft["cam_id"],
        "t_in": ft["t_in"],
        "t_out": ft["t_out"],
        "plate": plate,
        "conf": conf,
        "type": ft.get("type"),
        "colour": ft.get("colour"),
        "direction": ft.get("direction"),
        "embedding": ft.get("embedding"),
        "crop_path": str(config.CROP_DIR / f"{ft['cam_id']}_{ft['track_id']}.jpg"),
        "locked": locked,
    }


def run(sources: dict[str, str]) -> None:
    """sources: {"C1": "path/to/c1.mp4", ...}"""
    database.init_db()
    for cam_id, (lat, lon, name) in config.CAMERAS.items():
        database.save_camera(cam_id, lat, lon, name)

    detector = Detector()
    reads: dict[int, list[dict]] = defaultdict(list)

    for cam_id, source in sources.items():
        camera = Camera(cam_id, source, start_time=time.time())
        locked_tracks: set[int] = set()

        for f in camera.frames():
            tracks = detector.update(f["frame"], cam_id)

            for t in tracks:
                tid = t["track_id"]
                if tid in locked_tracks:
                    continue                                    # OCR budget: stop when locked
                if len(reads[tid]) >= config.MAX_OCR_PER_TRACK:
                    continue                                    # hard cap per track
                if not plate_reader.passes_gate(t["crop"]):
                    continue

                r = plate_reader.read_plate(t["crop"])
                if r:
                    reads[tid].append({**r, "quality": t["quality"]})

                plate, conf, locked = plate_logic.vote(reads[tid], config.MIN_CHAR_CONF)
                t["plate"], t["plate_conf"], t["locked"] = plate or None, conf, locked
                if locked:
                    locked_tracks.add(tid)

            visualizer.draw_frame(f["frame"], tracks)

            for ft in detector.finished_tracks():
                database.save_sighting(build_sighting(ft, reads[ft["track_id"]]))

        camera.release()

    for ft in detector.finished_tracks():                       # drain the last tracks
        database.save_sighting(build_sighting(ft, reads[ft["track_id"]]))

    links = linker.link_all()
    clones = linker.find_cloned_plates()
    map_path = visualizer.draw_map(database.get_cameras(), database.get_links())

    print(f"sightings : {len(database.get_sightings())}")
    print(f"links     : {len(links)}")
    print(f"clones    : {len(clones)}")
    print(f"map       : {map_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="ANPR prototype, multi-camera run")
    p.add_argument("--videos", nargs="*", default=[],
                   help="cam_id=path pairs, e.g. C1=data/videos/c1.mp4")
    args = p.parse_args()

    sources = dict(v.split("=", 1) for v in args.videos) if args.videos else {
        cam_id: str(config.VIDEO_DIR / f"{cam_id.lower()}.mp4") for cam_id in config.CAMERAS
    }
    run(sources)


if __name__ == "__main__":
    main()
