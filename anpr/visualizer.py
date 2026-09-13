import cv2
import numpy as np


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


def draw_frame(frame: np.ndarray, tracks: list[dict]) -> np.ndarray:
    """
    Draw large, readable vehicle annotations.
    """

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
        vehicle_type = track["type"]

        # Extract plate details
        plate = track.get("plate")
        plate_status = track.get("plate_status", "no_plate")
        plate_confs = track.get("plate_confs")

        # Select colour
        if plate_status == "locked":
            colour = (0, 255, 0)          # Green

        elif plate_status == "reading":
            colour = (0, 165, 255)        # Orange

        else:
            colour = (180, 180, 180)      # Light grey

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
        if plate_status == "locked" and plate:
            plate_label = f"Plate: {plate}"

        elif plate_status == "reading":
            plate_label = "Plate: reading..."

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

        # Confidence text
        if plate_status == "locked" and plate and plate_confs:

            confidence_text = "Conf: " + " ".join(
                f"{confidence * 100:.0f}%"
                for confidence in plate_confs
            )

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
import matplotlib.pyplot as plt


def draw_map(
    cameras: list[dict],
    links: list[dict],
    out_path: str = "out/map.png"
) -> str:
    """
    Draw camera locations and links between cameras.

    cameras example:
    [
        {"camera_id": "C1", "lat": 23.0225, "lon": 72.5714},
        {"camera_id": "C2", "lat": 23.0300, "lon": 72.5800}
    ]

    links example:
    [
        {
            "from": "C1",
            "to": "C2",
            "type": "exact"
        }
    ]
    """

    plt.figure(figsize=(10, 7))

    # Draw camera dots and labels
    for camera in cameras:
        camera_id = camera["camera_id"]
        latitude = camera["lat"]
        longitude = camera["lon"]

        plt.scatter(
            longitude,
            latitude,
            s=100,
            color="blue"
        )

        plt.text(
            longitude,
            latitude,
            camera_id,
            fontsize=10
        )

    # Create a quick lookup dictionary
    camera_positions = {
        camera["camera_id"]: (
            camera["lon"],
            camera["lat"]
        )
        for camera in cameras
    }

    # Draw links between cameras
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

        if link_type in ["exact", "fuzzy"]:
            line_style = "-"
        else:
            line_style = "--"

        plt.plot(
            [x1, x2],
            [y1, y2],
            linestyle=line_style,
            color="black"
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Camera Connection Map")
    plt.grid(True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path