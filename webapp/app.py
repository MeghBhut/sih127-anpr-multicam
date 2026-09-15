"""The dashboard server.

    .venv\\Scripts\\python.exe -m webapp.app

Draw the streets, drop cameras on them, point each camera at a clip, press
Run. The pipeline goes off in a background thread and the page fills in with
every vehicle it identified: where it was seen, what the camera saw, and the
path it took.

Deliberately small: one Flask file, no database beyond the one the pipeline
already writes, no build step. It runs on the demo laptop with no network.
"""

import json
import logging
import threading
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from anpr import config, database, layout, main as pipeline, roadmap

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

# What a run actually needs, and what each one does if it is missing.
REQUIRED = [
    ("ultralytics", "vehicle detection and tracking"),
    ("open_image_models", "finding the plate on a vehicle"),
    ("paddle", "reading the characters"),
    ("safetensors", "loading the OCR weights"),
]


def missing_dependencies():
    """Which of the pipeline's packages this interpreter cannot import.

    read_plate() and the detector both turn their own failures into "no
    result", which is right for one bad frame and badly wrong for a missing
    install: every plate comes back unread and the dashboard shows zero
    vehicles, which reads as "the footage was poor" rather than "you started
    this with the wrong Python". Check once, up front, and say so.
    """
    import importlib.util
    return [(mod, why) for mod, why in REQUIRED
            if importlib.util.find_spec(mod) is None]

# One run at a time. The pipeline writes to a single database and output tree,
# so two at once would interleave and produce nonsense.
_run_lock = threading.Lock()
_state = {
    "status": "idle",          # idle | running | done | failed
    "message": "",
    "summary": None,
    "error": None,
}


# ---------------------------------------------------------------------------
# Pages and static output
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent / "templates", "index.html")


@app.get("/out/<path:relpath>")
def output_file(relpath):
    """Serve anything the pipeline wrote, and nothing else.

    resolve() both sides and compare: without it, "../../config.py" would
    walk straight out of the output tree.
    """
    root = config.OUT_DIR.resolve()
    target = (root / relpath).resolve()
    if not str(target).startswith(str(root)) or not target.is_file():
        return jsonify({"error": "not found"}), 404
    return send_file(target)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@app.get("/api/layout")
def get_layout():
    data = layout.load()
    data["saved"] = layout.LAYOUT_PATH.is_file()
    return jsonify(data)


@app.post("/api/layout")
def post_layout():
    try:
        saved = layout.save(request.get_json(force=True) or {})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({**saved, "saved": True})


@app.post("/api/layout/reset")
def reset_layout():
    return jsonify({**layout.reset(), "saved": False})


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------

@app.get("/api/clips")
def list_clips():
    """Every video sitting in data/videos, with what we can cheaply learn."""
    import cv2

    clips = []
    for path in sorted(config.VIDEO_DIR.iterdir()):
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        info = {"name": path.name, "size_mb": round(path.stat().st_size / 1e6, 1)}
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            info["fps"] = round(fps, 2)
            info["seconds"] = round(frames / fps, 1) if fps else None
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        clips.append(info)
    return jsonify({"clips": clips, "dir": str(config.VIDEO_DIR)})


@app.post("/api/clips")
def upload_clip():
    """Accept a video from the browser and drop it in data/videos."""
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "no file"}), 400

    name = Path(file.filename).name          # strip any path the browser sent
    if Path(name).suffix.lower() not in VIDEO_EXTS:
        return jsonify({"error": f"not a video: {name}"}), 400

    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    file.save(str(config.VIDEO_DIR / name))
    return jsonify({"name": name})


# ---------------------------------------------------------------------------
# Running the pipeline
# ---------------------------------------------------------------------------

def _run(sources, seconds, starts):
    try:
        _state.update(status="running", message="loading models", error=None)

        # Recording times are what put two clips on one clock, and only the
        # DIFFERENCES between cameras matter. The dashboard sends an offset in
        # seconds per camera (C1 = 0, C2 = 40 means C2 started 40s later),
        # which is far easier to supply than a unix timestamp and carries
        # exactly the same information.
        base = 1_700_000_000.0
        for cam_id in sources:
            offset = (starts or {}).get(cam_id)
            config.RECORDING_START[cam_id] = base + float(offset or 0.0)

        _state["message"] = f"processing {len(sources)} clip(s)"
        summary = pipeline.run(sources, fresh=True, max_seconds=seconds)
        _state.update(status="done", message="finished", summary=summary)
    except Exception as exc:
        logger.exception("pipeline failed")
        _state.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      message=traceback.format_exc(limit=3))
    finally:
        _run_lock.release()


@app.post("/api/run")
def start_run():
    body = request.get_json(force=True) or {}
    sources = body.get("sources") or {}
    if not sources:
        return jsonify({"error": "assign a clip to at least one camera"}), 400

    resolved = {}
    for cam_id, name in sources.items():
        path = config.VIDEO_DIR / Path(str(name)).name
        if not path.is_file():
            return jsonify({"error": f"no clip named {name}"}), 400
        resolved[cam_id] = str(path)

    gaps = missing_dependencies()
    if gaps:
        names = ", ".join(m for m, _ in gaps)
        return jsonify({
            "error": f"this Python is missing: {names}. "
                     "Start the dashboard with the project venv: "
                     ".venv\\Scripts\\python.exe -m webapp.app",
            "missing": [{"module": m, "needed_for": why} for m, why in gaps],
        }), 503

    if not _run_lock.acquire(blocking=False):
        return jsonify({"error": "a run is already going"}), 409

    seconds = body.get("seconds")
    threading.Thread(
        target=_run,
        args=(resolved, float(seconds) if seconds else None, body.get("starts")),
        daemon=True,
    ).start()
    return jsonify({"status": "running", "cameras": list(resolved)})


@app.get("/api/status")
def run_status():
    gaps = missing_dependencies()
    return jsonify({**_state,
                    "missing": [{"module": m, "needed_for": why} for m, why in gaps]})


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _card(kind, plate):
    """URL for a card image, or None when it was never written."""
    safe = plate.replace("?", "x").replace("/", "_")
    folder = config.ROUTE_DIR if kind == "route" else config.JOURNEY_DIR
    return f"/out/{folder.name}/{safe}.jpg" if (folder / f"{safe}.jpg").is_file() else None


@app.get("/api/results")
def results():
    """Everything the last run produced, as one payload the page can render."""
    if not config.DB_PATH.is_file():
        return jsonify({"vehicles": [], "cameras": [], "sightings": 0,
                        "links": 0, "ran": False})

    database.init_db(str(config.DB_PATH))
    rows = database.get_all_sightings()
    links = database.get_links()

    linked_ids = set()
    for link in links:
        linked_ids.update((link["a_id"], link["b_id"]))

    # Group by plate: one entry per identified vehicle, however many cameras
    # saw it. Unread sightings are counted but have no identity to show.
    groups: dict[str, list[dict]] = {}
    unread = 0
    for r in rows:
        if r.get("plate") and r.get("plate_status") not in (None, "missing", "unreadable"):
            groups.setdefault(r["plate"], []).append(r)
        else:
            unread += 1

    base = min((r["t_in"] for r in rows), default=0.0)
    vehicles = []
    for plate, group in groups.items():
        group = sorted(group, key=lambda r: r["t_in"])
        vehicles.append({
            "plate": plate,
            "status": group[0].get("plate_status"),
            "quality": round(max(r.get("plate_quality") or 0 for r in group), 2),
            "type": group[0].get("vehicle_type"),
            "colour": group[0].get("colour"),
            "linked": any(r["id"] in linked_ids for r in group),
            "cameras": [r["cam_id"] for r in group],
            "hops": [{
                "cam_id": r["cam_id"],
                "t": round(r["t_in"] - base, 1),
                "crop": f"/out/crops/{Path(r['crop_path']).name}"
                        if Path(r["crop_path"]).is_file() else None,
                "conf": r.get("plate_conf"),
                "direction": r.get("direction"),
            } for r in group],
            "route_img": _card("route", plate),
            "journey_img": _card("journey", plate),
        })

    # Most interesting first: tracked across cameras, then seen most, then
    # read best. A judge scrolling this should hit the good ones immediately.
    vehicles.sort(key=lambda v: (not v["linked"], -len(set(v["cameras"])), -v["quality"]))

    # The ones with no readable plate belong in the list too. Leaving them out
    # meant the page said "29 vehicles seen" above a list of 5, which reads as
    # a bug rather than as "24 of them could not be identified".
    for r in rows:
        if r.get("plate") and r.get("plate_status") not in (None, "missing", "unreadable"):
            continue
        vehicles.append({
            "plate": None,
            "status": r.get("plate_status") or "missing",
            "quality": 0.0,
            "type": r.get("vehicle_type"),
            "colour": r.get("colour"),
            "linked": False,
            "cameras": [r["cam_id"]],
            "hops": [{
                "cam_id": r["cam_id"],
                "t": round(r["t_in"] - base, 1),
                "crop": f"/out/crops/{Path(r['crop_path']).name}"
                        if Path(r["crop_path"]).is_file() else None,
                "conf": None,
                "direction": r.get("direction"),
            }],
            "route_img": None,
            "journey_img": None,
        })

    per_cam: dict[str, int] = {}
    for r in rows:
        per_cam[r["cam_id"]] = per_cam.get(r["cam_id"], 0) + 1

    net = roadmap.RoadNetwork()
    cameras = [{"cam_id": c, "point": net.camera_point(c), "seen": per_cam.get(c, 0)}
               for c in sorted(net.cam_node)]

    montages = sorted(p.name for p in config.MONTAGE_DIR.glob("*.jpg"))
    frames = sorted(p.name for p in config.FRAME_DIR.glob("*.jpg"))

    return jsonify({
        "ran": True,
        "sightings": len(rows),
        "unread": unread,
        "links": len(links),
        "vehicles": vehicles,
        "cameras": cameras,
        "tracker_map": "/out/tracker_map.png" if config.ROAD_MAP_PNG.is_file() else None,
        "montages": [f"/out/montage/{n}" for n in montages],
        "frames": [f"/out/frames/{n}" for n in frames[:60]],
    })


@app.get("/api/route-preview")
def route_preview():
    """Where a path between two cameras runs, for drawing on the page."""
    net = roadmap.RoadNetwork()
    a, b = request.args.get("a"), request.args.get("b")
    return jsonify({"points": net.route(a, b)})


@app.get("/api/selftest")
def selftest():
    """Read one known-good crop inside this process.

    The pipeline can report zero plates for two very different reasons: the
    footage genuinely has none, or OCR is broken in this process. This tells
    them apart without waiting for a whole run.
    """
    import cv2
    from anpr import plate_reader as pr

    path = request.args.get("crop")
    if not path:
        return jsonify({"error": "pass ?crop=<path to a jpg>"}), 400
    img = cv2.imread(path)
    if img is None:
        return jsonify({"error": f"could not read {path}"}), 400

    out = {"crop": path, "shape": list(img.shape)}
    try:
        out["passes_gate"] = pr.passes_gate(img)
        box = pr._localize_plate(img)
        out["plate_box"] = list(box) if box else None
        r = pr.read_plate(img)
        out["read"] = "".join(r["chars"]) if r else None
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return jsonify(out)


def tailscale_ip():
    """This machine's tailnet address, if Tailscale is up. None otherwise."""
    import subprocess
    for exe in (r"C:\\Program Files\\Tailscale\\tailscale.exe", "tailscale"):
        try:
            out = subprocess.run([exe, "ip", "-4"], capture_output=True,
                                 text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        lines = (out.stdout or "").strip().splitlines()
        if lines and lines[0].strip().startswith("100."):
            return lines[0].strip()
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ANPR dashboard")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to bind; the default reaches this machine only")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--tailscale", action="store_true",
                        help="bind this machine's tailnet address, so teammates "
                             "on the tailnet can reach it and the local network "
                             "cannot")
    args = parser.parse_args()

    host = args.host
    if args.tailscale:
        host = tailscale_ip()
        if not host:
            parser.error("no Tailscale address found. Is Tailscale running and logged in?")

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    gaps = missing_dependencies()
    if gaps:
        print("\n  WARNING: this Python cannot run the pipeline.")
        for mod, why in gaps:
            print(f"    missing {mod:20s} ({why})")
        print("\n  Start it with the project venv instead:")
        print(r"    .venv\\Scripts\\python.exe -m webapp.app" + "\n")

    print(f"\n  ANPR dashboard  ->  http://{host}:{args.port}")
    if host not in ("127.0.0.1", "localhost"):
        print("  anything that can route to that address can reach this")
    print("")
    app.run(host=host, port=args.port, debug=False, threaded=True)
