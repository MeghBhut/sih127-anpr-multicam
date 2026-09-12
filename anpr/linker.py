"""Owner: person 4. Which sightings across cameras are the same vehicle.

A wrong link is worse than a missing link. Every filter here exists to reject.

STUB — the scoring maths already lives in plate_logic.py (person 6); this module
wraps it so there is one implementation, not two. Person 4 owns the candidate
search, the graph and the thresholds.
"""

import numpy as np

from . import config, database, plate_logic

CAMERA_GRAPH = config.CAMERA_GRAPH


def plausible(a: dict, b: dict, slack: float = config.LINK_SLACK) -> bool:
    """Hard filter: could the vehicle physically get from a's camera to b's in that time?"""
    return plate_logic.plausible(_as_point(a), _as_point(b), CAMERA_GRAPH, slack)


def plate_distance(a: str, b: str) -> float:
    """Weighted edit distance: '?' vs anything 0.1, confusable pair 0.3, else 1.0."""
    return plate_logic.plate_distance(a, b)


def link_score(a: dict, b: dict) -> tuple[float, str]:
    """(score 0..1, method). See plate_logic.link_score for the order of checks."""
    return plate_logic.link_score(_as_point(a), _as_point(b), CAMERA_GRAPH)


def link_all() -> list[dict]:
    """[{"a": 12, "b": 47, "score": 0.91, "method": "fuzzy"}, ...]"""
    out = []
    sightings = database.get_sightings()
    for i, a in enumerate(sightings):
        for b in sightings[i + 1:]:
            score, method = link_score(a, b)
            threshold = config.LINK_INFERRED if method == "inferred" else config.LINK_FUZZY
            if score >= threshold:
                database.save_link(a["id"], b["id"], score, method)
                out.append({"a": a["id"], "b": b["id"], "score": score, "method": method})
    return out


def find_cloned_plates() -> list[dict]:
    """Same plate on two cameras with a gap below the graph minimum -> alert."""
    alerts = []
    sightings = [s for s in database.get_sightings() if s.get("plate")]
    for i, a in enumerate(sightings):
        for b in sightings[i + 1:]:
            if a["plate"] != b["plate"] or a["cam_id"] == b["cam_id"]:
                continue
            win = CAMERA_GRAPH.get((a["cam_id"], b["cam_id"]))
            if win and abs(b["t_in"] - a["t_out"]) < win[0]:
                alerts.append({"plate": a["plate"], "a": a["id"], "b": b["id"]})
    return alerts


def _as_point(s: dict) -> dict:
    """Sighting row -> the flat dict plate_logic expects."""
    return {
        "plate": s.get("plate"),
        "conf": s.get("conf") or [0.0],
        "cam": s["cam_id"],
        "t": s["t_in"],
        "type": s.get("type"),
        "colour": s.get("colour"),
        # person 2's embedding is optional; a zero vector scores a neutral 0.5
        "emb": s.get("embedding") if s.get("embedding") is not None else np.zeros(512),
    }
