"""Owner: person 6 (Megh). Joins everything and owns the OCR budget.

Run from the repo root:
    python -m anpr.main
    python -m anpr.main --videos C1=anpr/data/videos/c1.mp4 C2=anpr/data/videos/c2.mp4
"""

import argparse
import logging
import time
from collections import defaultdict

from . import config, database, linker, plate_logic, plate_reader, visualizer
from .camera import Camera
from .detector import Detector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plate reliability
# ---------------------------------------------------------------------------
# database.py stores a plate_status alongside every plate, and linker.py gates
# BOTH of its last-line safety rules on it:
#
#   * "never merge two clean plates that differ"  (link_score)
#   * cloned-plate alerts                          (find_cloned_plates)
#
# Neither fires unless something classifies the plate. Voting is the only place
# that knows how good a read was, so classifying it is person 6's job.

def classify_plate(plate: str | None, confs: list[float] | None,
                   locked: bool) -> tuple[str, float]:
    """Voted plate -> (plate_status, plate_quality) for the sightings table.

    status is one of database.PLATE_STATUSES:
        missing     nothing was read at all
        unreadable  read, but mostly '?'
        partial     some characters unknown, enough known to compare
        clean       locked, no '?', every character above CLEAN_MIN_CHAR_CONF
        uncertain   read in full but not confidently enough to be "clean"

    quality is 0..1 = fraction of known characters * mean confidence. It is
    what linker.py weights a plate match by, so it must not flatter a read.
    """
    if not plate:
        return "missing", 0.0

    known = [c for c in plate if c != "?"]
    known_fraction = len(known) / len(plate)
    valid_confs = [float(c) for c in (confs or []) if isinstance(c, (int, float))]
    mean_conf = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0
    quality = max(0.0, min(1.0, known_fraction * mean_conf))

    if known_fraction == 0.0:
        return "unreadable", 0.0
    if (1.0 - known_fraction) > config.PARTIAL_MAX_UNKNOWN_FRACTION:
        return "unreadable", quality
    if "?" in plate:
        return "partial", quality

    if locked and valid_confs and min(valid_confs) >= config.CLEAN_MIN_CHAR_CONF:
        return "clean", quality
    return "uncertain", quality


def _read_is_clean(read: dict) -> bool:
    """A single OCR read good enough to be used as ID-switch evidence."""
    chars, confs = read.get("chars") or [], read.get("confs") or []
    if not chars or "?" in chars:
        return False
    return min(confs) >= config.CLEAN_MIN_CHAR_CONF


def is_id_switch(group: list[dict], new_read: dict) -> bool:
    """True when `new_read` cannot belong to the same vehicle as `group`.

    The tracker occasionally hands one track_id to two different vehicles.
    Voting across that boundary produces a plate belonging to neither, so the
    caller starts a new group instead. Only clean reads on both sides count as
    evidence -- a blurry read that disagrees is just a blurry read.
    """
    if not _read_is_clean(new_read):
        return False
    new_plate = "".join(new_read["chars"])
    for earlier in group:
        if not _read_is_clean(earlier):
            continue
        if plate_logic.plate_distance("".join(earlier["chars"]), new_plate) >= \
                config.ID_SWITCH_MIN_DISTANCE:
            return True
    return False


# ---------------------------------------------------------------------------
# Sighting rows
# ---------------------------------------------------------------------------

def build_sighting(ft: dict, group: list[dict], suffix: str = "") -> dict:
    """One finished track (or one segment of it) -> one sightings row.

    A plate with some '?' in it is kept, not discarded: linker.py scores '?'
    against any character at 0.1, so a partial read is real evidence. Only a
    plate that was never read at all becomes None.
    """
    plate, confs, locked = None, None, False
    if group:
        plate, confs, locked = plate_logic.vote(group, config.MIN_CHAR_CONF)
        plate = plate or None

    status, quality = classify_plate(plate, confs, locked)
    if status in ("missing", "unreadable"):
        plate, confs = None, None       # unknown is None, never "" or "unknown"

    # A split segment is timed by its own reads; a whole track by the tracker.
    read_times = [r["t"] for r in group if r.get("t") is not None]
    t_in = min(read_times) if read_times and suffix else ft["t_in"]
    t_out = max(read_times) if read_times and suffix else ft["t_out"]

    return {
        "cam_id": ft["cam_id"],
        "t_in": t_in,
        "t_out": t_out,
        "plate": plate,
        # database.py stores this column as plate_conf; the frozen contract
        # calls the key "conf". Send the name the schema actually reads.
        "plate_conf": confs,
        "plate_status": status,
        "plate_quality": quality,
        # save_sighting() accepts "type" or "vehicle_type"; rows read back
        # always come out as "vehicle_type".
        "type": ft.get("type"),
        "colour": ft.get("colour"),
        "direction": ft.get("direction"),
        "embedding": ft.get("embedding"),
        "crop_path": str(config.CROP_DIR / f"{ft['cam_id']}_{ft['track_id']}{suffix}.jpg"),
    }


def save_track(ft: dict, groups: list[list[dict]]) -> list[int]:
    """Write one row per segment of a finished track. Returns the new row ids."""
    if not groups:
        groups = [[]]
    ids = []
    for i, group in enumerate(groups):
        suffix = "" if len(groups) == 1 else f"_{i}"
        ids.append(database.save_sighting(build_sighting(ft, group, suffix)))
    return ids


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run(sources: dict[str, str], fresh: bool = True) -> dict:
    """sources: {"C1": "path/to/c1.mp4", ...}

    fresh=True starts from an empty database. init_db keeps whatever is already
    there, so without this a second run doubles every sighting and links run 1's
    rows to run 2's. The contract asks for one reproducible command, so the
    default is to start clean.
    """
    if fresh and config.DB_PATH.exists():
        config.DB_PATH.unlink()
    database.init_db(str(config.DB_PATH))
    for cam_id, (lat, lon, name) in config.CAMERAS.items():
        database.save_camera(cam_id, lat, lon, name)

    detector = Detector()
    # (cam_id, track_id) -> list of read-groups. Track ids restart on every
    # camera, so a bare id would merge C1's track 3 with C2's track 3.
    groups: dict[tuple[str, int], list[list[dict]]] = defaultdict(lambda: [[]])
    switches = 0

    for cam_id, source in sources.items():
        # Real recording start, so two clips share one clock. None only makes
        # sense for a live source; for files it makes every clip look
        # simultaneous and no travel-time window can match.
        start_time = config.RECORDING_START.get(cam_id)
        if start_time is None:
            start_time = time.time()
            logger.warning(
                "[%s] no RECORDING_START in config; timing this clip from now. "
                "Cross-camera links will be meaningless until person 1 fills it in.",
                cam_id)
        try:
            camera = Camera(cam_id, source, skip=config.SKIP, start_time=start_time)
        except (IOError, OSError) as exc:
            # One unreadable source must not take the other cameras down.
            logger.error("[%s] skipped: %s", cam_id, exc)
            continue

        locked_tracks: set[int] = set()
        try:
            for f in camera.frames():
                tracks = detector.update(f["frame"], cam_id, f["t"])

                for t in tracks:
                    tid = (cam_id, t["track_id"])
                    current = groups[tid][-1]

                    if tid in locked_tracks:
                        continue                                # budget: stop when locked
                    if len(current) >= config.MAX_OCR_PER_TRACK:
                        continue                                # hard cap per segment
                    if not plate_reader.passes_gate(t["crop"]):
                        continue

                    r = plate_reader.read_plate(t["crop"])
                    if r:
                        read = {**r, "quality": t["quality"], "t": f["t"]}
                        if is_id_switch(current, read):
                            logger.warning(
                                "[%s] track %s: ID switch, starting a new segment", cam_id, tid)
                            switches += 1
                            groups[tid].append([read])
                            locked_tracks.discard(tid)
                        else:
                            current.append(read)
                        current = groups[tid][-1]

                    plate, conf, locked = plate_logic.vote(current, config.MIN_CHAR_CONF)
                    t["plate"], t["plate_conf"], t["locked"] = plate or None, conf, locked
                    if locked:
                        locked_tracks.add(tid)

                visualizer.draw_frame(f["frame"], tracks)

                for ft in detector.finished_tracks():
                    save_track(ft, groups.pop((ft["cam_id"], ft["track_id"]), [[]]))
        finally:
            camera.release()

        # Close any track still on screen in the last frame, or it is lost.
        detector.flush(cam_id)
        for ft in detector.finished_tracks():
            save_track(ft, groups.pop((ft["cam_id"], ft["track_id"]), [[]]))

    links = linker.link_all()
    clones = linker.find_cloned_plates()
    map_path = visualizer.draw_map(database.get_cameras(), database.get_links())

    sightings = database.get_all_sightings()
    by_status: dict[str, int] = defaultdict(int)
    for s in sightings:
        by_status[s.get("plate_status") or "?"] += 1

    summary = {
        "sightings": len(sightings),
        "plates": dict(by_status),
        "id_switches": switches,
        "links": len(links),
        "clones": len(clones),
        "map": map_path,
    }

    print(f"sightings   : {summary['sightings']}")
    for status in ("clean", "partial", "uncertain", "unreadable", "missing"):
        if by_status.get(status):
            print(f"  {status:<10}: {by_status[status]}")
    print(f"id switches : {switches}")
    print(f"links       : {summary['links']}")
    print(f"clones      : {summary['clones']}")
    print(f"map         : {map_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="ANPR prototype, multi-camera run")
    p.add_argument("--videos", nargs="*", default=[],
                   help="cam_id=path pairs, e.g. C1=anpr/data/videos/c1.mp4")
    p.add_argument("--verbose", action="store_true", help="show per-pair link decisions")
    p.add_argument("--append", action="store_true",
                   help="add to the existing database instead of starting clean")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    sources = dict(v.split("=", 1) for v in args.videos) if args.videos else {
        cam_id: str(config.VIDEO_DIR / f"{cam_id.lower()}.mp4") for cam_id in config.CAMERAS
    }
    run(sources, fresh=not args.append)


if __name__ == "__main__":
    main()
