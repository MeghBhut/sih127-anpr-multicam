"""Owner: person 5. Annotated frames and the camera map.

Reads the shapes the rest of the pipeline actually produces:
  * a live track from detector.update(), with the plate fields main.py adds
  * cameras from database.get_cameras()      -> cam_id, lat, lon, name
  * links   from main.build_map_links()      -> from, to, type
"""

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


def draw_text_with_background(
    frame,
    text,
    position,
    font_scale,
    text_colour,
    thickness
):
    """
    Draw readable text with a black background behind it.
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
            vehicle_thickness
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
            plate_thickness
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
                confidence_thickness
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


def draw_map(
    cameras: list[dict],
    links: list[dict],
    out_path: str = None
) -> str:
    """
    Draw camera locations and links between cameras.

    cameras: rows from database.get_cameras()
        [{"cam_id": "C1", "lat": 23.0225, "lon": 72.5714, "name": "Gate A"}, ...]

    links: camera-to-camera lines from main.build_map_links()
        [{"from": "C1", "to": "C2", "type": "exact", "count": 3}, ...]

    A link row in the database joins two *sightings*, not two cameras, so the
    sighting-to-camera lookup happens in main.py where the database lives.

    Solid line = exact or fuzzy (plate evidence).
    Dashed line = inferred (appearance only) -- the route was never observed.
    """

    if out_path is None:
        out_path = str(config.MAP_PNG)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))

    # Draw camera dots and labels
    for camera in cameras:
        camera_id = camera["cam_id"]
        latitude = camera["lat"]
        longitude = camera["lon"]
        label = camera.get("name") or camera_id

        plt.scatter(
            longitude,
            latitude,
            s=120,
            color="blue",
            zorder=3
        )

        plt.text(
            longitude,
            latitude,
            f"  {camera_id} ({label})",
            fontsize=10,
            zorder=4
        )

    # Create a quick lookup dictionary
    camera_positions = {
        camera["cam_id"]: (
            camera["lon"],
            camera["lat"]
        )
        for camera in cameras
    }

    # Draw links between cameras
    drawn_styles = set()
    for link in links:
        from_camera = link["from"]
        to_camera = link["to"]
        link_type = link.get("type", "inferred")

        if from_camera not in camera_positions:
            continue

        if to_camera not in camera_positions:
            continue

        x1, y1 = camera_positions[from_camera]
        x2, y2 = camera_positions[to_camera]

        if link_type == "inferred":
            line_style = "--"
            legend_label = "inferred (appearance only)"
        else:
            line_style = "-"
            legend_label = "plate match"

        plt.plot(
            [x1, x2],
            [y1, y2],
            linestyle=line_style,
            color="black",
            linewidth=1.8,
            zorder=2,
            label=legend_label if legend_label not in drawn_styles else None
        )
        drawn_styles.add(legend_label)

        # How many vehicles took this route
        count = link.get("count")
        if count:
            plt.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                f" {count}",
                fontsize=9,
                color="black",
                zorder=4
            )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Camera Connection Map")
    plt.grid(True)
    if drawn_styles:
        plt.legend(loc="best", fontsize=9)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path
