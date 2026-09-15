"""Owner: person 5. Annotated frames and the camera map.

Reads the shapes the rest of the pipeline actually produces:
  * a live track from detector.update(), with the plate fields main.py adds
  * cameras from database.get_cameras()      -> cam_id, lat, lon, name
  * links   from main.build_map_links()      -> from, to, type
"""

import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")          # no display on the demo machine
import matplotlib.pyplot as plt

from . import config

# Box colour by plate state.
COLOUR_LOCKED = (0, 255, 0)        # green  - plate locked
COLOUR_READING = (0, 165, 255)     # orange - still reading
COLOUR_NO_PLATE = (180, 180, 180)  # grey   - nothing read


def _free_row(taken, x, y, w, h):
    """Nudge a label down until it stops covering one already drawn.

    With a dozen vehicles on screen the labels landed on top of each other and
    none of them could be read. Boxes are kept in `taken` and each new label
    slides down past whatever it collides with.
    """
    step = h + 10
    for _ in range(14):
        box = (x, y - h, x + w, y + 6)
        clash = any(not (box[2] < t[0] or box[0] > t[2] or
                         box[3] < t[1] or box[1] > t[3]) for t in taken)
        if not clash:
            taken.append(box)
            return y
        y += step
    taken.append((x, y - h, x + w, y + 6))
    return y


def draw_text_with_background(
    frame,
    text,
    position,
    font_scale,
    text_colour,
    thickness,
    taken=None
):
    """
    Draw readable text with a black background behind it.

    Pass `taken` (a list) to keep labels from overlapping each other.
    """

    x, y = position

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Find the size of the text
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    # Keep the label inside the image. A box near the right edge would
    # otherwise push its text off-screen and the label would be unreadable.
    frame_width = frame.shape[1]
    if text_width + 10 <= frame_width:
        x = min(x, frame_width - text_width - 5)
    x = max(x, 5)

    if taken is not None:
        y = _free_row(taken, x, y, text_width + 10, text_height + baseline + 8)
        y = min(y, frame.shape[0] - 4)

    # Draw a black rectangle behind the text
    cv2.rectangle(
        frame,
        (x - 5, y - text_height - baseline - 8),
        (x + text_width + 5, y + 5),
        (0, 0, 0),
        -1
    )

    # Draw the actual text
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        text_colour,
        thickness,
        cv2.LINE_AA
    )


def _plate_state(track: dict) -> str:
    """Which of the three states this track is in.

    main.py sets "locked" (bool) and "plate" on every live track, every frame.
    Note this is NOT the database's plate_status ("clean" / "uncertain" / ...),
    which describes how good a finished read was. This is just what to paint.
    """
    if track.get("locked") and track.get("plate"):
        return "locked"
    if track.get("plate"):
        return "reading"
    return "no_plate"


def draw_frame(frame: np.ndarray, tracks: list[dict]) -> np.ndarray:
    """
    Draw large, readable vehicle annotations.

    Returns a new image; the frame passed in is not modified, so main.py can
    still hand the original to the detector or save it untouched.
    """

    frame = frame.copy()
    taken: list[tuple[int, int, int, int]] = []   # label boxes already drawn

    # Get image dimensions
    image_height, image_width = frame.shape[:2]

    # Font sizes based on image width
    vehicle_font_scale = max(image_width / 1200, 1.0)
    plate_font_scale = max(image_width / 1400, 0.9)
    confidence_font_scale = max(image_width / 1900, 0.65)

    # Text thickness based on image size
    vehicle_thickness = max(round(image_width / 450), 3)
    plate_thickness = max(round(image_width / 500), 3)
    confidence_thickness = max(round(image_width / 700), 2)

    for track in tracks:

        # Extract vehicle details
        track_id = track["track_id"]
        x1, y1, x2, y2 = track["box"]
        vehicle_type = track.get("type")

        # Extract plate details
        plate = track.get("plate")
        plate_confs = track.get("plate_conf")
        state = _plate_state(track)

        # Select colour
        if state == "locked":
            colour = COLOUR_LOCKED

        elif state == "reading":
            colour = COLOUR_READING

        else:
            colour = COLOUR_NO_PLATE

        # Draw the vehicle rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            colour,
            4
        )

        # Vehicle label
        vehicle_label = f"ID: {track_id} | Type: {vehicle_type}"

        vehicle_label_y = max(y1 - 20, 45)

        draw_text_with_background(
            frame,
            vehicle_label,
            (x1, vehicle_label_y),
            vehicle_font_scale,
            colour,
            vehicle_thickness,
            taken
        )

        # Plate label
        if state == "locked":
            plate_label = f"Plate: {plate}"

        elif state == "reading":
            plate_label = f"Plate: {plate} (reading...)"

        else:
            plate_label = "Plate: no plate"

        # Position plate text below the box
        plate_y = y2 + 50

        # If there is not enough space below, place it inside the box
        if plate_y > image_height - 80:
            plate_y = y2 - 20

        draw_text_with_background(
            frame,
            plate_label,
            (x1, plate_y),
            plate_font_scale,
            colour,
            plate_thickness,
            taken
        )

        # Confidence text. Shown while reading too, not only once locked --
        # watching the per-character confidence climb is the evidence story.
        if plate and plate_confs:

            confidence_text = "Conf: " + " ".join(
                f"{confidence * 100:.0f}"
                for confidence in plate_confs
            )

            # Ten per-character numbers are wide. Shrink until the row fits
            # rather than letting it run off the edge.
            fitted_scale = confidence_font_scale
            while fitted_scale > 0.3:
                (needed, _), _ = cv2.getTextSize(
                    confidence_text, cv2.FONT_HERSHEY_SIMPLEX,
                    fitted_scale, confidence_thickness)
                if needed + 20 <= image_width:
                    break
                fitted_scale -= 0.05
            confidence_font_scale = fitted_scale

            confidence_y = plate_y + 35

            # If confidence text goes outside the image,
            # place it above the plate text
            if confidence_y > image_height - 10:
                confidence_y = plate_y - 15

            draw_text_with_background(
                frame,
                confidence_text,
                (x1, confidence_y),
                confidence_font_scale,
                colour,
                confidence_thickness,
                taken
            )

    return frame


def save_frame(frame: np.ndarray, cam_id: str, frame_no: int) -> str:
    """Write one annotated frame to out/frames/. Returns the path written.

    These are the deck screenshots -- without this the run produces none.
    """
    config.FRAME_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FRAME_DIR / f"{cam_id}_{frame_no:06d}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


# ---------------------------------------------------------------------------
# Street map: real tiles, fetched once and cached
# ---------------------------------------------------------------------------
TILE_PX = 256


def _deg2xy(lat, lon, zoom):
    """Lat/lon -> fractional web-mercator tile coordinates."""
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(max(-85.05, min(85.05, lat)))
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _pick_zoom(lat_min, lon_min, lat_max, lon_max, max_tiles):
    """Deepest zoom where the cameras still fit in max_tiles x max_tiles."""
    for zoom in range(19, 1, -1):
        x0, y0 = _deg2xy(lat_max, lon_min, zoom)
        x1, y1 = _deg2xy(lat_min, lon_max, zoom)
        if (abs(x1 - x0) + 1) <= max_tiles and (abs(y1 - y0) + 1) <= max_tiles:
            return zoom
    return 2


def _fetch_tiles(zoom, tx0, ty0, tx1, ty1):
    """Stitch the tile grid into one BGR image. None if it cannot be fetched."""
    cache = config.MAP_DIR / f"{zoom}_{tx0}_{ty0}_{tx1}_{ty1}.png"
    if cache.is_file():
        img = cv2.imread(str(cache))
        if img is not None:
            return img
    if getattr(config, "MAP_OFFLINE", False):
        return None

    import urllib.request
    width = (tx1 - tx0 + 1) * TILE_PX
    height = (ty1 - ty0 + 1) * TILE_PX
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    got = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            url = config.MAP_TILE_URL.format(z=zoom, x=tx, y=ty)
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "SIH26127-ANPR-prototype/0.1"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    buf = np.frombuffer(resp.read(), dtype=np.uint8)
                tile = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception as exc:
                logger.warning("map tile %s failed: %s", url, exc)
                continue
            if tile is None:
                continue
            px, py = (tx - tx0) * TILE_PX, (ty - ty0) * TILE_PX
            canvas[py:py + TILE_PX, px:px + TILE_PX] = tile
            got += 1

    if got == 0:
        return None
    config.MAP_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache), canvas)
    return canvas


def _plain_map(cameras, links, out_path):
    """No tiles available: fall back to the plain plot so a run never fails."""
    plt.figure(figsize=(10, 7))
    for cam in cameras:
        plt.scatter(cam["lon"], cam["lat"], s=120, color="blue", zorder=3)
        plt.text(cam["lon"], cam["lat"], f"  {cam['cam_id']}", fontsize=10, zorder=4)
    pos = {c["cam_id"]: (c["lon"], c["lat"]) for c in cameras}
    for link in links:
        if link["from"] not in pos or link["to"] not in pos:
            continue
        (x1, y1), (x2, y2) = pos[link["from"]], pos[link["to"]]
        style = "--" if link.get("type") == "inferred" else "-"
        plt.plot([x1, x2], [y1, y2], linestyle=style, color="black", zorder=2)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Camera Connection Map (no street tiles)")
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def draw_map(cameras: list[dict], links: list[dict], out_path: str = None) -> str:
    """Camera pins and vehicle routes over a real street map.

    cameras: rows from database.get_cameras() -- cam_id, lat, lon, name
    links:   from main.build_map_links() -- {"from", "to", "type", "count"}

    Tiles are fetched once from OpenStreetMap and cached under
    data/map_cache/, so later runs render offline. If the fetch fails the map
    still renders, just without streets -- a demo must never die on the map.

    Scales as cameras are added: the zoom and framing are computed from
    whatever is in config.CAMERAS, so 2 cameras on one road and 6 spread
    across a district both come out sensibly framed.
    """
    if out_path is None:
        out_path = str(config.MAP_PNG)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cams = [c for c in cameras if c.get("lat") is not None and c.get("lon") is not None]
    if not cams:
        logger.warning("no cameras with coordinates; map not drawn")
        return out_path

    lats = [c["lat"] for c in cams]
    lons = [c["lon"] for c in cams]
    pad_lat = max((max(lats) - min(lats)) * config.MAP_PADDING, 0.0015)
    pad_lon = max((max(lons) - min(lons)) * config.MAP_PADDING, 0.0015)
    lat_min, lat_max = min(lats) - pad_lat, max(lats) + pad_lat
    lon_min, lon_max = min(lons) - pad_lon, max(lons) + pad_lon

    zoom = _pick_zoom(lat_min, lon_min, lat_max, lon_max, config.MAP_MAX_TILES)
    x0f, y0f = _deg2xy(lat_max, lon_min, zoom)
    x1f, y1f = _deg2xy(lat_min, lon_max, zoom)
    tx0, ty0, tx1, ty1 = int(x0f), int(y0f), int(x1f), int(y1f)

    tiles = _fetch_tiles(zoom, tx0, ty0, tx1, ty1)
    if tiles is None:
        logger.warning("street tiles unavailable, drawing the plain map instead")
        return _plain_map(cams, links, out_path)

    def to_px(lat, lon):
        x, y = _deg2xy(lat, lon, zoom)
        return int((x - tx0) * TILE_PX), int((y - ty0) * TILE_PX)

    # Crop to the cameras plus padding so the frame is tight, not a whole tile.
    px0, py0 = to_px(lat_max, lon_min)
    px1, py1 = to_px(lat_min, lon_max)
    px0, py0 = max(0, px0), max(0, py0)
    px1 = min(tiles.shape[1], max(px1, px0 + 320))
    py1 = min(tiles.shape[0], max(py1, py0 + 320))
    img = tiles[py0:py1, px0:px1].copy()

    # Fade the map so the overlay reads clearly on top of it.
    img = cv2.addWeighted(img, 0.75, np.full_like(img, 255), 0.25, 0)

    def pin(lat, lon):
        x, y = to_px(lat, lon)
        return x - px0, y - py0

    # ---- routes, drawn under the pins ------------------------------------
    taken: list[tuple[int, int, int, int]] = []
    pos = {c["cam_id"]: pin(c["lat"], c["lon"]) for c in cams}
    for link in links:
        a, b = link.get("from"), link.get("to")
        if a not in pos or b not in pos:
            continue
        p1, p2 = pos[a], pos[b]
        inferred = link.get("type") == "inferred"
        colour = (120, 120, 120) if inferred else (30, 30, 200)
        if inferred:
            _dashed_line(img, p1, p2, colour, 3)
        else:
            cv2.line(img, p1, p2, colour, 3, cv2.LINE_AA)

        count = link.get("count")
        if count:
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            label = f"{count} vehicle" + ("s" if count != 1 else "")
            draw_text_with_background(img, label, (mx - 30, my - 8), 0.55, colour, 2, taken)

    # ---- camera pins ------------------------------------------------------
    for cam in cams:
        x, y = pos[cam["cam_id"]]
        cv2.circle(img, (x, y), 11, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 11, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), 5, (30, 30, 200), -1, cv2.LINE_AA)
        name = cam.get("name") or ""
        label = f"{cam['cam_id']}" + (f"  {name}" if name else "")
        draw_text_with_background(img, label, (x + 16, y + 6), 0.62, (20, 20, 20), 2, taken)

    img = _map_legend(img, links)
    cv2.imwrite(out_path, img)
    return out_path


def _dashed_line(img, p1, p2, colour, thickness, dash=12, gap=9):
    x1, y1 = p1; x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    steps = int(dist // (dash + gap)) + 1
    for i in range(steps):
        a = (i * (dash + gap)) / dist
        b = min((i * (dash + gap) + dash) / dist, 1.0)
        cv2.line(img,
                 (int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a)),
                 (int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b)),
                 colour, thickness, cv2.LINE_AA)


def _map_legend(img, links):
    """Title strip on top, legend strip underneath."""
    h, w = img.shape[:2]
    top = np.full((52, w, 3), 255, dtype=np.uint8)
    cv2.putText(top, "Vehicle routes between cameras", (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)

    bottom = np.full((46, w, 3), 255, dtype=np.uint8)
    cv2.line(bottom, (16, 23), (56, 23), (30, 30, 200), 3, cv2.LINE_AA)
    cv2.putText(bottom, "plate match", (64, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (20, 20, 20), 1, cv2.LINE_AA)
    _dashed_line(bottom, (210, 23), (250, 23), (120, 120, 120), 3)
    cv2.putText(bottom, "inferred, route not observed", (258, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    if not links:
        cv2.putText(bottom, "no links in this run", (w - 230, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 60, 60), 1, cv2.LINE_AA)
    return np.vstack([top, img, bottom])


# ---------------------------------------------------------------------------
# Judge-facing output: one vehicle's journey, and both cameras at once
# ---------------------------------------------------------------------------
def _panel(crop, width, height, caption, sub):
    """One camera's evidence photo, letterboxed, with a caption bar."""
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    if crop is not None and crop.size:
        ch, cw = crop.shape[:2]
        scale = min(width / cw, (height - 54) / ch)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cv2.resize(crop, (nw, nh))
        ox, oy = (width - nw) // 2, (height - 54 - nh) // 2
        panel[oy:oy + nh, ox:ox + nw] = resized
    else:
        cv2.putText(panel, "no crop saved", (16, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2, cv2.LINE_AA)

    cv2.rectangle(panel, (0, height - 54), (width, height), (25, 25, 25), -1)
    cv2.putText(panel, caption, (14, height - 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, sub, (14, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return panel


def draw_journey(plate: str, rows: list[dict], link: dict | None,
                 out_path: str) -> str:
    """The photographic evidence for one vehicle.

    One sighting gives one panel: the photo, the plate, the camera and the
    time. Two or more give the comparison -- the photo at each camera with
    the travel time between them and how the linker matched them.

    This is the counterpart to the route card in roadmap.py: that one shows
    WHERE the vehicle went, this one shows WHAT the system actually saw. A
    judge who doubts the claim looks at the photographs.
    """
    W, H = 420, 300
    rows = sorted(rows, key=lambda r: r.get("t_in", 0))[:3]
    if not rows:
        raise ValueError("no sightings to draw")

    panels = []
    for r in rows:
        panels.append(_panel(_read_crop(r), W, H,
                             f"{r['cam_id']}  {_clock(r.get('t_in', 0))}",
                             f"{r.get('vehicle_type') or '?'} - {r.get('colour') or '?'}"))

    pieces = [panels[0]]
    for i in range(1, len(panels)):
        travel = abs(rows[i].get("t_in", 0) - rows[i - 1].get("t_out", 0))
        pieces.append(_gap(H, travel, link))
        pieces.append(panels[i])
    body = np.hstack(pieces)
    w = body.shape[1]

    header = np.full((80, w, 3), 255, dtype=np.uint8)
    cv2.putText(header, plate, (16, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.25, (20, 20, 20), 3, cv2.LINE_AA)

    if len(rows) >= 2:
        note = " -> ".join(r["cam_id"] for r in rows)
    else:
        note = f"seen at {rows[0]['cam_id']} only"
    cv2.putText(header, note, (16, 72), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (110, 110, 110), 1, cv2.LINE_AA)

    status = rows[0].get("plate_status") or "?"
    (sw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(header, status, (w - sw - 20, 46), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (90, 90, 90), 1, cv2.LINE_AA)

    out = np.vstack([header, body, _conf_strip(rows[0], w)])
    cv2.imwrite(out_path, out)
    return out_path


def _gap(height, travel, link):
    """The arrow between two panels: how long the vehicle took, and why we
    believe it is the same vehicle."""
    gap = np.full((height, 150, 3), 245, dtype=np.uint8)
    cv2.arrowedLine(gap, (18, height // 2 - 30), (132, height // 2 - 30),
                    (30, 30, 200), 3, cv2.LINE_AA, tipLength=0.22)
    cv2.putText(gap, f"{travel:.0f}s", (52, height // 2 - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, (30, 30, 200), 2, cv2.LINE_AA)
    if link:
        cv2.putText(gap, link.get("method", "?"), (14, height // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(gap, f"score {link.get('score', 0):.2f}", (14, height // 2 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 60, 60), 1, cv2.LINE_AA)
    else:
        # Same plate at both cameras, but the linker did not join them --
        # usually the travel time fell outside CAMERA_GRAPH. Say so rather
        # than implying a confirmed match.
        cv2.putText(gap, "same plate,", (14, height // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(gap, "not linked", (14, height // 2 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (110, 110, 110), 1, cv2.LINE_AA)
    return gap


def _conf_strip(row, width):
    """Per-character confidence under the plate: the evidence for the read."""
    strip = np.full((54, width, 3), 250, dtype=np.uint8)
    confs = row.get("plate_conf")
    plate = row.get("plate") or ""
    if not confs or not plate:
        return strip[:1]

    cv2.putText(strip, "per-character confidence", (16, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1, cv2.LINE_AA)
    pairs = []
    for ch, c in zip(plate, confs):
        try:
            pairs.append((ch, float(c)))
        except (TypeError, ValueError):
            continue
    if not pairs:
        return strip[:1]

    # Share the width between however many characters there are, so a ten
    # character plate is not cut off after the eighth.
    step = max(24, min(46, (width - 32) // len(pairs)))
    scale = 0.6 if step >= 40 else 0.5
    x = 16
    for ch, pct in pairs:
        colour = (40, 150, 40) if pct >= 0.8 else (40, 140, 220) if pct >= 0.5 \
            else (60, 60, 210)
        cv2.putText(strip, ch, (x, 44), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(strip, f"{pct * 100:.0f}", (x + int(step * 0.38), 44),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, colour, 1, cv2.LINE_AA)
        x += step
    return strip


def _read_crop(sighting):
    path = sighting.get("crop_path")
    if not path:
        return None
    img = cv2.imread(str(path))
    return img


def _clock(t):
    """Seconds since the clips started, which is what the judges care about."""
    if _CLOCK_BASE is not None:
        return f"t+{t - _CLOCK_BASE:.1f}s"
    # "1789315000.6" tells a reader nothing; show a clock time instead.
    import datetime
    return datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")


_CLOCK_BASE = None


def set_clock_base(t):
    """Show times relative to the first sighting rather than as unix seconds."""
    global _CLOCK_BASE
    _CLOCK_BASE = t


def draw_montage(frames: list[tuple[str, str, float]], t: float, out_path: str) -> str:
    """Every camera at one moment, side by side.

    frames: (path, cam_id, video_time) for annotated frames already written.
    For the given moment, each camera contributes the frame nearest that time,
    so judges see the cameras as simultaneous -- which they are, even though
    the clips were processed one after another.
    """
    by_cam: dict[str, tuple[float, str]] = {}
    for path, cam_id, ft in frames:
        d = abs(ft - t)
        if cam_id not in by_cam or d < by_cam[cam_id][0]:
            by_cam[cam_id] = (d, path)
    if not by_cam:
        return out_path

    tiles = []
    for cam_id in sorted(by_cam):
        img = cv2.imread(by_cam[cam_id][1])
        if img is None:
            continue
        img = cv2.resize(img, (640, 360))
        cv2.rectangle(img, (0, 0), (640, 34), (25, 25, 25), -1)
        cv2.putText(img, f"{cam_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.78, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (639, 359), (25, 25, 25), 2)
        tiles.append(img)
    if not tiles:
        return out_path

    row = np.hstack(tiles)
    bar = np.full((48, row.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(bar, f"All cameras at {_clock(t)}", (14, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, np.vstack([bar, row]))
    return out_path
