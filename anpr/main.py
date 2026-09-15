"""Owner: person 6 (Megh). Joins everything and owns the OCR budget.

Run from the repo root:
    python -m anpr.main
    python -m anpr.main --videos C1=anpr/data/videos/c1.mp4 C2=anpr/data/videos/c2.mp4
"""

import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path

import cv2

from . import (config, database, linker, plate_logic, plate_reader,
               roadmap, visualizer)
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


def build_routes(links: list[dict], sightings: list[dict]) -> list[str]:
    """One schematic map per identified vehicle: this plate went this way.

    Every vehicle whose plate was read gets a card, not only the ones that
    linked across cameras. A vehicle seen at a single camera is a real result
    -- the system identified it and knows where and when -- and its card shows
    exactly that, with the path extending on its own the moment a second
    camera sees the same plate.

    Links matter for ordering, not for inclusion: a vehicle the linker joined
    across two cameras is the more interesting card, so those are drawn first.
    """
    by_id = {s["id"]: s for s in sightings}

    linked_ids: set[int] = set()
    for link in links:
        if link["a_id"] in by_id and link["b_id"] in by_id:
            linked_ids.add(link["a_id"])
            linked_ids.add(link["b_id"])

    # Group every readable sighting by its plate. Two cameras reading the same
    # plate belong on one card even if the linker did not join them.
    groups: dict[str, list[dict]] = {}
    for s in sightings:
        plate = s.get("plate")
        if not plate or s.get("plate_status") in (None, "missing", "unreadable"):
            continue
        groups.setdefault(plate, []).append(s)

    def rank(item):
        plate, rows = item
        cameras = len({r["cam_id"] for r in rows})
        was_linked = any(r["id"] in linked_ids for r in rows)
        best_quality = max((r.get("plate_quality") or 0.0) for r in rows)
        return (-int(was_linked), -cameras, -best_quality)

    out = []
    written: set[Path] = set()
    for plate, rows in sorted(groups.items(), key=rank)[:config.MAX_ROUTE_CARDS]:
        ordered = sorted(rows, key=lambda r: r["t_in"])
        safe = plate.replace("?", "x").replace("/", "_")
        path = config.ROUTE_DIR / f"{safe}.jpg"
        try:
            out.append(roadmap.draw_route(plate, ordered, str(path)))
            written.add(path.resolve())
        except Exception as exc:
            logger.warning("route map failed for %s: %s", plate, exc)

    # Sweep leftovers from earlier runs only AFTER the new set exists. Cards
    # are named after the plate, so one from a previous run would otherwise
    # sit among these looking like part of it -- but clearing the folder up
    # front left nothing to look at while a run was going, and nothing at all
    # if it failed.
    for stale in config.ROUTE_DIR.glob("*.jpg"):
        if stale.resolve() not in written:
            try:
                stale.unlink()
            except OSError as exc:
                logger.debug("could not remove stale card %s: %s", stale, exc)
    return out


def build_journeys(links: list[dict], sightings: list[dict]) -> list[str]:
    """One photographic evidence card per identified vehicle.

    The route card in out/routes/ says WHERE a vehicle went; this says what
    the cameras actually saw -- the crop, the plate, and the per-character
    confidence behind the read. Both are built for every identified vehicle,
    so the two folders line up one to one.

    Previously these were built only from links, so a run that found no
    cross-camera match produced nothing at all, even with 27 vehicles
    identified and photographed.
    """
    by_id = {s["id"]: s for s in sightings}

    # Which pair of sightings the linker actually joined, so a card can show
    # the score. Keyed both ways round, since a card orders by time.
    link_of: dict[tuple[int, int], dict] = {}
    for link in links:
        link_of[(link["a_id"], link["b_id"])] = link
        link_of[(link["b_id"], link["a_id"])] = link

    groups: dict[str, list[dict]] = {}
    for s in sightings:
        if not s.get("plate") or s.get("plate_status") in (None, "missing", "unreadable"):
            continue
        groups.setdefault(s["plate"], []).append(s)

    def rank(item):
        _, rows = item
        return (-len({r["cam_id"] for r in rows}),
                -max((r.get("plate_quality") or 0.0) for r in rows))

    out, written = [], set()
    for plate, rows in sorted(groups.items(), key=rank)[:config.MAX_ROUTE_CARDS]:
        ordered = sorted(rows, key=lambda r: r["t_in"])
        link = None
        for a, b in zip(ordered, ordered[1:]):
            link = link_of.get((a["id"], b["id"])) or link

        safe = plate.replace("?", "x").replace("/", "_")
        path = config.JOURNEY_DIR / f"{safe}.jpg"
        try:
            out.append(visualizer.draw_journey(plate, ordered, link, str(path)))
            written.add(path.resolve())
        except Exception as exc:
            logger.warning("evidence card failed for %s: %s", plate, exc)

    for stale in config.JOURNEY_DIR.glob("*.jpg"):
        if stale.resolve() not in written:
            try:
                stale.unlink()
            except OSError as exc:
                logger.debug("could not remove stale card %s: %s", stale, exc)
    return out


def build_montages(frame_index: list[tuple[str, str, float]]) -> list[str]:
    """All cameras at the same moment, a handful of moments across the run."""
    if not frame_index or config.MONTAGE_COUNT < 1:
        return []
    times = sorted(t for _, _, t in frame_index)
    lo, hi = times[0], times[-1]
    if hi <= lo:
        picks = [lo]
    else:
        n = config.MONTAGE_COUNT
        picks = [lo + (hi - lo) * i / (n - 1) for i in range(n)] if n > 1 else [lo]

    out = []
    for i, t in enumerate(picks):
        path = config.MONTAGE_DIR / f"moment_{i:02d}.jpg"
        try:
            out.append(visualizer.draw_montage(frame_index, t, str(path)))
        except Exception as exc:
            logger.warning("montage failed at t=%s: %s", t, exc)
    return out


def build_map_links(links: list[dict], sightings: list[dict]) -> list[dict]:
    """Database links join two *sightings*; the map draws lines between
    *cameras*. Resolve one to the other here, where the database lives, and
    collapse repeats so one line per camera pair per method reaches the map.
    """
    cam_of = {s["id"]: s["cam_id"] for s in sightings}
    counts: dict[tuple[str, str, str], int] = {}
    for link in links:
        a, b = cam_of.get(link["a_id"]), cam_of.get(link["b_id"])
        if not a or not b or a == b:
            continue
        method = link.get("method") or "inferred"
        key = (a, b) if a <= b else (b, a)
        counts[(key[0], key[1], method)] = counts.get((key[0], key[1], method), 0) + 1
    return [{"from": a, "to": b, "type": m, "count": n}
            for (a, b, m), n in sorted(counts.items())]


def save_track(ft: dict, groups: list[list[dict]]) -> list[int]:
    """Write one row per segment of a finished track. Returns the new row ids.

    Also writes the vehicle's best crop to out/crops/. Every sighting row
    stores a crop_path, and until now nothing ever created the file it named,
    so the database pointed at photos that did not exist. That crop is the
    evidence image for the vehicle.
    """
    if not groups:
        groups = [[]]
    ids = []
    for i, group in enumerate(groups):
        suffix = "" if len(groups) == 1 else f"_{i}"
        sighting = build_sighting(ft, group, suffix)

        best_crop = ft.get("best_crop")
        if best_crop is not None and getattr(best_crop, "size", 0):
            try:
                config.CROP_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(sighting["crop_path"], best_crop)
            except Exception as exc:
                logger.warning("could not write crop %s: %s", sighting["crop_path"], exc)

        ids.append(database.save_sighting(sighting))
    return ids


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run(sources: dict[str, str], fresh: bool = True,
        max_seconds: float | None = None) -> dict:
    """sources: {"C1": "path/to/c1.mp4", ...}

    max_seconds stops every camera after that many seconds of VIDEO time, so
    clips of different lengths cover the same window. Two clips filmed at once
    but 31s and 66s long otherwise give C2 35 extra seconds that C1 could never
    have witnessed, and every vehicle in that tail is unmatchable by
    construction.

    fresh=True starts from an empty database. init_db keeps whatever is already
    there, so without this a second run doubles every sighting and links run 1's
    rows to run 2's. The contract asks for one reproducible command, so the
    default is to start clean.
    """
    # Start clean, but keep the previous run until this one has replaced it.
    # Deleting up front meant an interrupted run left nothing at all -- the
    # old results were gone and the new ones never arrived.
    previous = config.DB_PATH.with_suffix(".prev.db")
    if fresh and config.DB_PATH.exists():
        previous.unlink(missing_ok=True)
        config.DB_PATH.rename(previous)
    database.init_db(str(config.DB_PATH))
    for cam_id, (lat, lon, name) in config.CAMERAS.items():
        database.save_camera(cam_id, lat, lon, name)

    detector = Detector()
    # (cam_id, track_id) -> list of read-groups. Track ids restart on every
    # camera, so a bare id would merge C1's track 3 with C2's track 3.
    groups: dict[tuple[str, int], list[list[dict]]] = defaultdict(lambda: [[]])
    # last vote per track, so a locked or budget-spent track still draws right
    display: dict[tuple[str, int], tuple] = {}
    switches = 0
    frames_saved = 0
    frame_index: list[tuple[str, str, float]] = []   # (path, cam_id, video time)

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

        locked_tracks: set[tuple[str, int]] = set()
        try:
            for f in camera.frames():
                if max_seconds is not None and (f["t"] - start_time) >= max_seconds:
                    logger.info("[%s] reached the %.0fs limit, stopping", cam_id, max_seconds)
                    break

                tracks = detector.update(f["frame"], cam_id, f["t"])

                for t in tracks:
                    tid = (cam_id, t["track_id"])
                    current = groups[tid][-1]

                    budget_left = (tid not in locked_tracks
                                   and len(current) < config.MAX_OCR_PER_TRACK)
                    if budget_left and plate_reader.passes_gate(t["crop"]):
                        r = plate_reader.read_plate(t["crop"])
                        if r:
                            read = {**r, "quality": t["quality"], "t": f["t"]}
                            if is_id_switch(current, read):
                                logger.warning(
                                    "[%s] track %s: ID switch, starting a new segment",
                                    cam_id, t["track_id"])
                                switches += 1
                                groups[tid].append([read])
                                locked_tracks.discard(tid)
                            else:
                                current.append(read)
                            current = groups[tid][-1]

                        plate, conf, locked = plate_logic.vote(current, config.MIN_CHAR_CONF)
                        display[tid] = (plate or None, conf, locked)
                        if locked:
                            locked_tracks.add(tid)

                    # Set these on EVERY track every frame, not only when OCR
                    # ran. A locked track skips the block above, and without
                    # this its box would go back to grey "no plate".
                    t["plate"], t["plate_conf"], t["locked"] = display.get(
                        tid, (None, None, False))

                annotated = visualizer.draw_frame(f["frame"], tracks)
                if config.SAVE_FRAME_EVERY and f["frame_no"] % config.SAVE_FRAME_EVERY == 0:
                    path = visualizer.save_frame(annotated, cam_id, f["frame_no"])
                    frame_index.append((path, cam_id, f["t"]))
                    frames_saved += 1

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

    sightings = database.get_all_sightings()
    # This run produced rows, so the previous database is safe to drop.
    if fresh and previous.exists():
        previous.unlink(missing_ok=True)

    map_links = build_map_links(database.get_links(), sightings)

    # Show every time as "t+12.3s" from the first sighting rather than as a
    # raw unix number, which means nothing to anyone reading a slide.
    if sightings:
        visualizer.set_clock_base(min(s["t_in"] for s in sightings))

    journeys = build_journeys(database.get_links(), sightings)
    montages = build_montages(frame_index)

    # The tracker map: the street layout, who passed each camera, and one
    # route per identified vehicle. This is THE map. The geographic one
    # (visualizer.draw_map, out/map.png) showed camera dots on a satellite
    # view and nothing about vehicles; it is kept for the frozen contract but
    # is off by default, because three maps in out/ is two too many.
    tracker_map = roadmap.draw_tracker_map(sightings, database.get_links())
    routes = build_routes(database.get_links(), sightings)

    map_path = None
    if getattr(config, "DRAW_GEO_MAP", False):
        map_path = visualizer.draw_map(database.get_cameras(), map_links)
    by_status: dict[str, int] = defaultdict(int)
    for s in sightings:
        by_status[s.get("plate_status") or "?"] += 1

    summary = {
        "sightings": len(sightings),
        "plates": dict(by_status),
        "id_switches": switches,
        "frames_saved": frames_saved,
        "journeys": len(journeys),
        "montages": len(montages),
        "routes": len(routes),
        "tracker_map": tracker_map,
        "links": len(links),
        "clones": len(clones),
        "map": map_path,
    }

    print(f"sightings   : {summary['sightings']}")
    for status in ("clean", "partial", "uncertain", "unreadable", "missing"):
        if by_status.get(status):
            print(f"  {status:<10}: {by_status[status]}")
    print(f"id switches : {switches}")
    print(f"frames saved: {frames_saved} -> {config.FRAME_DIR}")
    print(f"montages    : {len(montages)} -> {config.MONTAGE_DIR}")
    print(f"journeys    : {len(journeys)} -> {config.JOURNEY_DIR}")
    print(f"routes      : {len(routes)} -> {config.ROUTE_DIR}")
    print(f"tracker map : {tracker_map}")
    if map_path:
        print(f"geo map     : {map_path}")
    print(f"links       : {summary['links']}")
    print(f"clones      : {summary['clones']}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="ANPR prototype, multi-camera run")
    p.add_argument("--videos", nargs="*", default=[],
                   help="cam_id=path pairs, e.g. C1=anpr/data/videos/c1.mp4")
    p.add_argument("--verbose", action="store_true", help="show per-pair link decisions")
    p.add_argument("--append", action="store_true",
                   help="add to the existing database instead of starting clean")
    p.add_argument("--seconds", type=float, default=None, metavar="N",
                   help="stop each camera after N seconds of video, so clips of "
                        "different lengths cover the same window")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    sources = dict(v.split("=", 1) for v in args.videos) if args.videos else {
        cam_id: str(config.VIDEO_DIR / f"{cam_id.lower()}.mp4") for cam_id in config.CAMERAS
    }

    # Drop cameras whose clip simply is not there. A missing file is a normal
    # state while the team is still filming, not an error worth a red traceback.
    # Live sources (rtsp://, a webcam index) are never checked.
    present = {}
    for cam_id, src in sources.items():
        looks_like_a_file = not (src.startswith("rtsp://") or src.isdigit())
        if looks_like_a_file and not Path(src).exists():
            print(f"{cam_id}: no clip at {src}, skipping")
            continue
        present[cam_id] = src

    if not present:
        p.error("no video sources found. Put clips in anpr/data/videos/ "
                "or pass --videos C1=path.mp4")

    run(present, fresh=not args.append, max_seconds=args.seconds)


if __name__ == "__main__":
    main()
