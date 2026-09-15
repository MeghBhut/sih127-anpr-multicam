"""
Vehicle identity resolution layer -- cross-camera linking.

Given sightings persisted by ``database.py``, this module decides whether
two sightings from different cameras plausibly represent the same physical
vehicle. It is designed to be conservative: when evidence is weak or
missing, it prefers "no link" over a false merge, because false merges are
far more damaging to an ANPR pipeline than missed links.

The public API is:

    CAMERA_GRAPH        -- (cam_a, cam_b) -> (min_seconds, max_seconds)
    plausible(a, b, slack=5.0) -> bool
    plate_distance(a, b) -> float
    link_score(a, b) -> (score, method)
    link_all() -> list[dict]
    find_cloned_plates() -> list[dict]
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from . import database

try:
    from . import config
except ImportError:  # pragma: no cover - config.py should always exist
    config = None  # type: ignore

logger = logging.getLogger(__name__)


def _cfg(name: str, default: float) -> float:
    """Read a threshold from config.py if present, else use the default.

    Keeps linker.py functional even if Person 6's config.py doesn't define
    every constant this module wants, per the "integrate, don't redesign"
    requirement.
    """
    return float(getattr(config, name, default)) if config is not None else default


LINK_FUZZY: float = _cfg("LINK_FUZZY", 0.75)
LINK_INFERRED: float = _cfg("LINK_INFERRED", 0.85)

MIN_KNOWN_PLATE_FRACTION: float = _cfg("MIN_KNOWN_PLATE_FRACTION", 0.4)
PLATE_QUALITY_THRESHOLD: float = _cfg("PLATE_QUALITY_THRESHOLD", 0.35)
TRAVEL_TIME_SLACK: float = _cfg("TRAVEL_TIME_SLACK", 5.0)
TYPE_MISMATCH_PENALTY: float = _cfg("TYPE_MISMATCH_PENALTY", 0.5)
DIRECTION_MISMATCH_PENALTY: float = _cfg("DIRECTION_MISMATCH_PENALTY", 0.15)
CLONE_TOLERANCE: float = _cfg("CLONE_TOLERANCE", 0.0)
CLONE_MIN_PLATE_QUALITY: float = _cfg("CLONE_MIN_PLATE_QUALITY", 0.6)

WEIGHT_PLATE: float = _cfg("WEIGHT_PLATE", 0.60)
WEIGHT_EMBEDDING: float = _cfg("WEIGHT_EMBEDDING", 0.30)
WEIGHT_COLOUR: float = _cfg("WEIGHT_COLOUR", 0.10)

WEIGHT_EMBEDDING_ONLY: float = _cfg("WEIGHT_EMBEDDING_ONLY", 0.75)
WEIGHT_COLOUR_ONLY: float = _cfg("WEIGHT_COLOUR_ONLY", 0.25)

# Travel-time windows come from config.py, so tuning them means editing one
# file. This module used to keep its own copy, which meant a change in config
# silently did nothing -- the values below are only a fallback for running
# linker.py without a config present.
#
# Every pair needs BOTH directions. plausible() looks the pair up in
# chronological order, so a missing reverse entry silently kills every link
# that way round.
_FALLBACK_GRAPH: dict[tuple[str, str], tuple[float, float]] = {
    ("C1", "C2"): (20.0, 180.0),
    ("C2", "C1"): (20.0, 180.0),
    ("C2", "C3"): (15.0, 150.0),
    ("C3", "C2"): (15.0, 150.0),
    ("C1", "C3"): (40.0, 300.0),
    ("C3", "C1"): (40.0, 300.0),
}

CAMERA_GRAPH: dict[tuple[str, str], tuple[float, float]] = (
    getattr(config, "CAMERA_GRAPH", None) or _FALLBACK_GRAPH
)

# Vehicle types considered mutually exclusive enough to strongly reject a
# match even under weak plate evidence. Anything not listed here as a
# mismatch is treated as "not clearly incompatible" (detector classes are
# noisy, so we don't want an over-eager type filter).
_STRONG_TYPE_MISMATCHES = {
    frozenset({"car", "bus"}),
    frozenset({"car", "truck"}),
    frozenset({"bike", "bus"}),
    frozenset({"bike", "truck"}),
    frozenset({"bike", "car"}),
    frozenset({"motorcycle", "bus"}),
    frozenset({"motorcycle", "truck"}),
    frozenset({"motorcycle", "car"}),
}

# Confusable-character substitution costs (bidirectional).
_CONFUSABLE_PAIRS = {
    frozenset({"0", "O"}): 0.3,
    frozenset({"1", "I"}): 0.3,
    frozenset({"8", "B"}): 0.3,
    frozenset({"5", "S"}): 0.3,
    frozenset({"2", "Z"}): 0.3,
}

_INDEL_PENALTY = 0.8  # configurable insert/delete cost, kept < 1.0 (substitution ceiling)


# ---------------------------------------------------------------------------
# Travel-time plausibility
# ---------------------------------------------------------------------------

def plausible(a: dict, b: dict, slack: float = 5.0) -> bool:
    """Return True if the timing between sightings ``a`` and ``b`` is
    consistent with a real vehicle travelling between their cameras.

    Looks up the appropriate direction in ``CAMERA_GRAPH``. If no entry
    exists for the camera pair, the pair is treated as *not* plausible --
    we never invent travel-time bounds for an unknown route.
    """
    cam_a, cam_b = a.get("cam_id"), b.get("cam_id")
    if not cam_a or not cam_b or cam_a == cam_b:
        return False

    # Order sightings chronologically so we look up the correct direction.
    first, second = (a, b) if a.get("t_in", 0) <= b.get("t_in", 0) else (b, a)
    route = (first.get("cam_id"), second.get("cam_id"))
    bounds = CAMERA_GRAPH.get(route)
    if bounds is None:
        logger.debug("No camera-graph entry for route %s", route)
        return False

    min_t, max_t = bounds
    gap = float(second.get("t_in", 0)) - float(first.get("t_out", 0))
    return (min_t - slack) <= gap <= (max_t + slack)


def _travel_time_gap(a: dict, b: dict) -> float:
    """Absolute gap between the two sightings' in/out timestamps."""
    first, second = (a, b) if a.get("t_in", 0) <= b.get("t_in", 0) else (b, a)
    return float(second.get("t_in", 0)) - float(first.get("t_out", 0))


# ---------------------------------------------------------------------------
# Weighted plate distance
# ---------------------------------------------------------------------------

def _substitution_cost(x: str, y: str) -> float:
    if x == y:
        return 0.0
    if x == "?" or y == "?":
        return 0.1
    pair_cost = _CONFUSABLE_PAIRS.get(frozenset({x, y}))
    if pair_cost is not None:
        return pair_cost
    return 1.0


def plate_distance(a: Optional[str], b: Optional[str]) -> float:
    """Weighted edit distance between two normalized plate strings.

    Rules:
        '?'  vs anything      -> 0.1  (unknown character, small uncertainty cost)
        0<->O, 1<->I, 8<->B,
        5<->S, 2<->Z          -> 0.3  (known OCR confusions)
        other substitution    -> 1.0
        insert/delete         -> _INDEL_PENALTY (configurable)

    Returns ``math.inf`` if either plate is None/empty, since distance is
    undefined when there is nothing to compare -- callers must handle the
    "missing plate" case separately rather than treating it as a match or
    a maximal-distance mismatch.
    """
    if not a or not b:
        return math.inf

    n, m = len(a), len(b)
    # dp[i][j] = weighted edit distance between a[:i] and b[:j]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * _INDEL_PENALTY
    for j in range(1, m + 1):
        dp[0][j] = j * _INDEL_PENALTY

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = _substitution_cost(a[i - 1], b[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + _INDEL_PENALTY,       # delete from a
                dp[i][j - 1] + _INDEL_PENALTY,       # insert into a
                dp[i - 1][j - 1] + sub_cost,          # substitute/match
            )
    return dp[n][m]


def plate_similarity(a: Optional[str], b: Optional[str]) -> float:
    """Normalized plate similarity in [0.0, 1.0], derived from plate_distance.

    Normalizes by the length of the longer plate, clamps to [0, 1]. Two
    None/empty plates are not "similar" -- callers must route missing
    plates through the inferred-match path instead of calling this.
    """
    if not a or not b:
        return 0.0
    dist = plate_distance(a, b)
    if math.isinf(dist):
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    normalized = dist / max_len
    sim = 1.0 - normalized
    return max(0.0, min(1.0, sim))


def _known_char_fraction(plate: Optional[str]) -> float:
    """Fraction of characters in ``plate`` that are not '?' (unknown)."""
    if not plate:
        return 0.0
    known = sum(1 for ch in plate if ch != "?")
    return known / len(plate) if plate else 0.0


def _plate_quality(sighting: dict) -> float:
    """Resolve an effective plate quality score for a sighting.

    Uses the stored ``plate_quality`` if present; otherwise derives a
    conservative estimate from known-character fraction and mean
    per-character confidence.
    """
    stored = sighting.get("plate_quality")
    if isinstance(stored, (int, float)):
        return max(0.0, min(1.0, float(stored)))

    plate = sighting.get("plate")
    if not plate:
        return 0.0

    known_frac = _known_char_fraction(plate)
    conf = sighting.get("plate_conf")
    mean_conf = None
    if isinstance(conf, list) and conf:
        try:
            valid = [float(c) for c in conf if isinstance(c, (int, float))]
            mean_conf = sum(valid) / len(valid) if valid else None
        except (TypeError, ValueError):
            mean_conf = None

    if mean_conf is not None:
        return max(0.0, min(1.0, 0.5 * known_frac + 0.5 * mean_conf))
    return known_frac


def _is_plate_usable(sighting: dict) -> bool:
    """Whether a sighting's plate carries enough signal to be compared at all."""
    plate = sighting.get("plate")
    status = sighting.get("plate_status")
    if not plate or status in (None, "missing", "unreadable"):
        return False
    if _known_char_fraction(plate) < MIN_KNOWN_PLATE_FRACTION:
        return False
    return True


def _is_clean(sighting: dict) -> bool:
    return sighting.get("plate_status") == "clean" and bool(sighting.get("plate"))


# ---------------------------------------------------------------------------
# Supporting-evidence similarities
# ---------------------------------------------------------------------------

def _cosine_similarity(v1, v2) -> Optional[float]:
    """Cosine similarity for two vectors, safely handling edge cases.

    Returns None (not a number, not zero) when similarity cannot be
    meaningfully computed, so callers can distinguish "no evidence" from
    "evidence says dissimilar".
    """
    if v1 is None or v2 is None:
        return None
    try:
        v1 = [float(x) for x in v1]
        v2 = [float(x) for x in v2]
    except (TypeError, ValueError):
        return None
    if not v1 or not v2 or len(v1) != len(v2):
        return None

    dot = sum(x * y for x, y in zip(v1, v2))
    norm1 = math.sqrt(sum(x * x for x in v1))
    norm2 = math.sqrt(sum(y * y for y in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return None

    sim = dot / (norm1 * norm2)
    # Map cosine similarity from [-1, 1] to [0, 1] for consistent scoring.
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def _colour_similarity(a: dict, b: dict) -> Optional[float]:
    """1.0 same colour, 0.0 different, None if either colour is unknown."""
    ca, cb = a.get("colour"), b.get("colour")
    if not ca or not cb:
        return None
    return 1.0 if str(ca).lower() == str(cb).lower() else 0.0


def _direction_adjustment(a: dict, b: dict) -> float:
    """Small additive adjustment based on direction agreement.

    Returns 0.0 when direction is unknown on either side (ignored),
    a small positive nudge when directions agree, and a penalty when they
    clearly disagree. Never large enough to flip an otherwise-decided
    score on its own.
    """
    da, db = a.get("direction"), b.get("direction")
    if not da or not db:
        return 0.0
    if str(da).lower() == str(db).lower():
        return 0.05
    return -DIRECTION_MISMATCH_PENALTY


def _type_compatible(a: dict, b: dict) -> tuple[bool, bool]:
    """Return (is_strong_mismatch, is_known_pair).

    is_known_pair is False when either type is missing, in which case type
    should not influence the decision at all.
    """
    ta, tb = a.get("vehicle_type"), b.get("vehicle_type")
    if not ta or not tb:
        return False, False
    ta, tb = str(ta).lower(), str(tb).lower()
    if ta == tb:
        return False, True
    return frozenset({ta, tb}) in _STRONG_TYPE_MISMATCHES, True


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def link_score(a: dict, b: dict) -> tuple[float, str]:
    """Score how likely sightings ``a`` and ``b`` are the same vehicle.

    Returns (score, method) where method is one of:
        "exact"     -- strong, reliable plate agreement
        "fuzzy"     -- plate match with OCR/partial uncertainty
        "inferred"  -- no usable plate on one/both sides; appearance-based
        "reject"    -- disqualified before scoring (impossible timing,
                       strong type mismatch, or clean-plate conflict)

    This function performs no I/O and does not consult the camera graph's
    slack default from the caller -- ``plausible()`` is called with the
    default slack; callers needing a custom slack should call
    ``plausible()`` themselves first.
    """
    debug: dict = {"a": a.get("id"), "b": b.get("id")}

    if not plausible(a, b, slack=TRAVEL_TIME_SLACK):
        logger.debug("REJECT %s<->%s reason=impossible travel time", a.get("id"), b.get("id"))
        return 0.0, "reject"

    strong_mismatch, _ = _type_compatible(a, b)
    if strong_mismatch:
        logger.debug("REJECT %s<->%s reason=type mismatch", a.get("id"), b.get("id"))
        return 0.0, "reject"

    plate_a, plate_b = a.get("plate"), b.get("plate")
    usable_a, usable_b = _is_plate_usable(a), _is_plate_usable(b)

    # --- Clean-plate safety rule (mandatory, highest priority) -----------
    if _is_clean(a) and _is_clean(b):
        if plate_a != plate_b:
            logger.debug(
                "REJECT %s<->%s reason=clean plates conflict (%s vs %s)",
                a.get("id"), b.get("id"), plate_a, plate_b,
            )
            return 0.0, "reject"

    embedding_sim = _cosine_similarity(a.get("embedding"), b.get("embedding"))
    colour_sim = _colour_similarity(a, b)

    # --- Both sides have a usable plate -----------------------------------
    if usable_a and usable_b:
        p_sim = plate_similarity(plate_a, plate_b)

        # Weight down plate similarity when quality is low on either side,
        # so a low-quality "coincidental" match doesn't get full credit.
        quality_factor = min(_plate_quality(a), _plate_quality(b))
        quality_factor = max(quality_factor, MIN_KNOWN_PLATE_FRACTION)

        eff_embedding = embedding_sim if embedding_sim is not None else _conservative_fallback(a, b)
        eff_colour = colour_sim if colour_sim is not None else 0.5  # neutral/unavailable

        score = (
            WEIGHT_PLATE * p_sim * (0.5 + 0.5 * quality_factor)
            + WEIGHT_EMBEDDING * eff_embedding
            + WEIGHT_COLOUR * eff_colour
        )
        score += _direction_adjustment(a, b)
        score = max(0.0, min(1.0, score))

        if plate_a == plate_b and _plate_quality(a) >= 0.8 and _plate_quality(b) >= 0.8:
            method = "exact"
        else:
            method = "fuzzy"

        debug.update(
            plate_sim=p_sim, embedding_sim=embedding_sim, colour_sim=colour_sim, score=score,
        )
        logger.debug("SCORE %s method=%s", debug, method)
        return score, method

    # --- Exactly one side missing a usable plate --------------------------
    if usable_a != usable_b:
        if embedding_sim is None:
            # No appearance evidence and no plate evidence on one side:
            # conservative fallback only, capped low.
            score = _conservative_fallback(a, b)
            logger.debug(
                "SCORE %s<->%s method=inferred(no-embedding) score=%.3f",
                a.get("id"), b.get("id"), score,
            )
            return score, "inferred"

        eff_colour = colour_sim if colour_sim is not None else 0.5
        score = WEIGHT_EMBEDDING_ONLY * embedding_sim + WEIGHT_COLOUR_ONLY * eff_colour
        score += _direction_adjustment(a, b)
        score = max(0.0, min(1.0, score))
        logger.debug(
            "SCORE %s<->%s method=inferred score=%.3f embedding_sim=%.3f colour_sim=%s",
            a.get("id"), b.get("id"), score, embedding_sim, colour_sim,
        )
        return score, "inferred"

    # --- Both sides missing a usable plate ---------------------------------
    score = _conservative_fallback(a, b)
    logger.debug(
        "SCORE %s<->%s method=inferred(both-missing) score=%.3f",
        a.get("id"), b.get("id"), score,
    )
    return score, "inferred"


def _conservative_fallback(a: dict, b: dict) -> float:
    """Score used when embeddings are unavailable (or both plates missing).

    Deliberately conservative: never treats absence of contradicting
    evidence as evidence of a match. Only colour agreement plus a
    same-type, same-direction pattern can nudge the score up slightly, and
    the ceiling is kept below LINK_INFERRED so weak evidence alone cannot
    trigger an inferred link.
    """
    colour_sim = _colour_similarity(a, b)
    _, known_type_pair = _type_compatible(a, b)

    score = 0.0
    if colour_sim == 1.0:
        score += 0.3
    if known_type_pair:
        score += 0.1
    score += max(0.0, _direction_adjustment(a, b))
    return max(0.0, min(0.6, score))  # hard ceiling: never confidently "inferred" alone


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def link_all(window: float = 600.0) -> list[dict]:
    """Run the full linking pipeline over all stored sightings.

    1. Load every sighting.
    2. For each sighting, fetch plausible candidates via
       ``database.get_candidates`` (cheap SQL-level filter).
    3. Score each unique pair with ``link_score``.
    4. Accept a link if its score clears the threshold appropriate to its
       method (``LINK_FUZZY`` for exact/fuzzy, ``LINK_INFERRED`` for
       inferred).
    5. Persist accepted links (deduplicated, order-independent) and return
       them with debug metadata compatible with Person 5's visualizer.

    ``window`` bounds how far apart (in seconds) two sightings' time ranges
    may be before even being considered as candidates; it should be at
    least as large as the largest ``CAMERA_GRAPH`` max travel time.
    """
    sightings = database.get_all_sightings()
    seen_pairs: set[tuple[int, int]] = set()
    results: list[dict] = []

    for s in sightings:
        candidates = database.get_candidates(s, window=window)
        for c in candidates:
            id_a, id_b = s.get("id"), c.get("id")
            if id_a is None or id_b is None or id_a == id_b:
                continue
            pair_key = (id_a, id_b) if id_a < id_b else (id_b, id_a)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            score, method = link_score(s, c)
            if method == "reject":
                continue

            threshold = LINK_INFERRED if method == "inferred" else LINK_FUZZY
            if score < threshold:
                logger.debug(
                    "REJECT %s<->%s reason=below threshold (%.3f < %.3f, method=%s)",
                    id_a, id_b, score, threshold, method,
                )
                continue

            database.save_link(pair_key[0], pair_key[1], score, method)
            results.append(
                {
                    "a": pair_key[0],
                    "b": pair_key[1],
                    "score": score,
                    "method": method,
                    "time_gap": _travel_time_gap(s, c),
                }
            )
            logger.debug(
                "LINK %s<->%s score=%.3f method=%s decision=LINK",
                pair_key[0], pair_key[1], score, method,
            )

    return results


# ---------------------------------------------------------------------------
# Cloned-plate detection
# ---------------------------------------------------------------------------

def find_cloned_plates() -> list[dict]:
    """Flag pairs of sightings sharing a plate with an impossible travel time.

    Only sightings with a sufficiently reliable plate (clean status and
    quality above ``CLONE_MIN_PLATE_QUALITY``) are considered, so a badly
    OCR'd partial plate cannot trigger a high-confidence clone alert.
    """
    sightings = database.get_all_sightings()

    reliable_by_plate: dict[str, list[dict]] = {}
    for s in sightings:
        if s.get("plate_status") != "clean":
            continue
        if _plate_quality(s) < CLONE_MIN_PLATE_QUALITY:
            continue
        plate = s.get("plate")
        if not plate:
            continue
        reliable_by_plate.setdefault(plate, []).append(s)

    alerts: list[dict] = []
    for plate, group in reliable_by_plate.items():
        if len(group) < 2:
            continue
        group_sorted = sorted(group, key=lambda x: x.get("t_in", 0))
        for i in range(len(group_sorted)):
            for j in range(i + 1, len(group_sorted)):
                a, b = group_sorted[i], group_sorted[j]
                if a.get("cam_id") == b.get("cam_id"):
                    continue

                first, second = (a, b) if a.get("t_in", 0) <= b.get("t_in", 0) else (b, a)
                route = (first.get("cam_id"), second.get("cam_id"))
                bounds = CAMERA_GRAPH.get(route)
                if bounds is None:
                    continue  # unknown route: cannot judge impossibility
                min_t, _max_t = bounds

                gap = float(second.get("t_in", 0)) - float(first.get("t_out", 0))
                if gap < (min_t - CLONE_TOLERANCE):
                    alerts.append(
                        {
                            "plate": plate,
                            "a_id": first.get("id"),
                            "b_id": second.get("id"),
                            "time_gap": gap,
                            "minimum_time": min_t,
                            "severity": "high" if gap < min_t / 2 else "medium",
                            "reason": "impossible travel time",
                        }
                    )
    return alerts
