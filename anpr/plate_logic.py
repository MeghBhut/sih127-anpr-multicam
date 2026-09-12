"""
The three pieces you should write and understand yourself:
  1. format rules   -> fix_by_format(chars, confs)
  2. voting         -> vote(reads)
  3. linker scoring -> link_score(a, b, graph)

Everything here is pure Python + numpy. No model calls.
"""

import re
import numpy as np

# ----------------------------------------------------------------------
# 1. FORMAT RULES
# ----------------------------------------------------------------------
# Indian plate: 2 letters (state) + 1-2 digits (RTO) + 1-3 letters (series) + 4 digits
# Examples: GJ01AB1234, GJ1AB1234, DL3CAB1234, MH12A1234
# BH series: 2 digits + BH + 4 digits + 1-2 letters  e.g. 22BH1234AB
INDIAN_PLATE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")
BH_PLATE     = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")

STATE_CODES = {"GJ","MH","DL","KA","TN","RJ","UP","MP","HR","PB","WB","AP","TS","KL",
               "OD","BR","JH","CG","UK","HP","GA","AS","CH","DD","DN","LD","PY","AN",
               "SK","TR","MN","ML","MZ","NL","AR","JK","LA"}

# OCR confusions: what a digit looks like as a letter, and reverse
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "4": "A", "6": "G"}
TO_DIGIT  = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "A": "4", "G": "6",
             "D": "0", "Q": "0", "T": "1", "L": "1"}


def expected_types(length: int) -> str:
    """
    Which positions must be Letter (L) or Digit (D) for a plate of this length.
    Standard format only; BH handled separately.
      10 chars: LL DD LLL DDDD? no -> most common: LL DD LL DDDD  (GJ01AB1234)
       9 chars: LL D LL DDDD (GJ1AB1234) or LL DD L DDDD (MH12A1234) -> ambiguous, use "?"
    """
    return {
        10: "LLDDLLDDDD",
        11: "LLDDLLLDDDD",
        9:  "LL?" + "?" + "?DDDD",   # unknown middle, still fix the ends
        8:  "LLD?DDDD",
    }.get(length, "?" * length)


def fix_by_format(chars: list[str], confs: list[float]) -> tuple[str, list[float]]:
    """
    Force each position to the type the format expects, using the confusion table.
    chars: ['G','J','0','1','A','8','1','2','3','4']  (from OCR, may contain '?')
    confs: per-character probabilities from the OCR model
    Returns corrected string and confs (a corrected char keeps 0.6 * its conf).
    """
    plate = "".join(chars)
    if BH_PLATE.match(plate):
        return plate, confs

    types = expected_types(len(chars))
    out, out_conf = [], []
    for c, p, t in zip(chars, confs, types):
        if c == "?":
            out.append("?"); out_conf.append(0.0); continue
        if t == "L" and c.isdigit() and c in TO_LETTER:
            out.append(TO_LETTER[c]); out_conf.append(p * 0.6)
        elif t == "D" and c.isalpha() and c in TO_DIGIT:
            out.append(TO_DIGIT[c]); out_conf.append(p * 0.6)
        else:
            out.append(c); out_conf.append(p)

    fixed = "".join(out)
    # state code sanity: if first two letters aren't a real state, lower their confidence
    if fixed[:2] not in STATE_CODES:
        out_conf[0] *= 0.5; out_conf[1] *= 0.5
    return fixed, out_conf


def is_valid(plate: str) -> bool:
    return bool(INDIAN_PLATE.match(plate) or BH_PLATE.match(plate)) and plate[:2] in STATE_CODES | {plate[:2] if BH_PLATE.match(plate) else ""}


# ----------------------------------------------------------------------
# 2. MULTI-FRAME VOTING
# ----------------------------------------------------------------------
# One "read" = one OCR result on one frame of the same track:
#   read = {"chars": ['G','J',...], "confs": [0.98, 0.95, ...], "quality": 0.8}
# quality = your frame-quality score (size * sharpness), 0..1

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?"


def vote(reads: list[dict], min_char_conf: float = 0.5):
    """
    Returns (plate_string, per_position_conf, locked: bool)
    Steps:
      a) vote on length            -> drop reads of other lengths
      b) apply format rules to each read
      c) per-position weighted vote over the surviving reads
      d) locked = every position agreed with high confidence
    """
    if not reads:
        return "", [], False

    # a) length vote (weighted by quality)
    length_votes = {}
    for r in reads:
        L = len(r["chars"])
        length_votes[L] = length_votes.get(L, 0.0) + r["quality"]
    best_len = max(length_votes, key=length_votes.get)
    reads = [r for r in reads if len(r["chars"]) == best_len]

    # b) format-correct each read
    corrected = []
    for r in reads:
        fixed, confs = fix_by_format(r["chars"], r["confs"])
        corrected.append((fixed, confs, r["quality"]))

    # c) per-position vote: score[pos][char] += conf * quality
    score = np.zeros((best_len, len(CHARSET)))
    for fixed, confs, q in corrected:
        for pos, (c, p) in enumerate(zip(fixed, confs)):
            if c == "?":
                continue
            score[pos, CHARSET.index(c)] += p * q

    plate, out_conf = [], []
    total_weight = sum(q for _, _, q in corrected) or 1.0
    for pos in range(best_len):
        if score[pos].sum() == 0:
            plate.append("?"); out_conf.append(0.0); continue
        best = int(score[pos].argmax())
        plate.append(CHARSET[best])
        out_conf.append(float(score[pos, best] / total_weight))   # 0..1, share of weight

    # d) locked if every position is confident and at least 3 reads agreed
    locked = len(corrected) >= 3 and all(c >= min_char_conf for c in out_conf) and "?" not in plate
    return "".join(plate), out_conf, locked


# Usage inside the tracking loop (pseudo):
#   track.reads.append({"chars":..., "confs":..., "quality":...})
#   plate, conf, locked = vote(track.reads)
#   if locked: track.plate = plate; track.stop_ocr = True


# ----------------------------------------------------------------------
# 3. LINKER SCORING
# ----------------------------------------------------------------------
# sighting = {"plate": "GJ01AB1234" or None, "conf": [..per char..],
#             "cam": "C3", "t": 1726123456.0, "type": "car", "colour": "white",
#             "emb": np.array(512)}
# graph[("C1","C3")] = (min_seconds, max_seconds)   # realistic travel window

# cost of substituting one char for another; confusable pairs are cheap
CHEAP = {("0","O"),("O","0"),("1","I"),("I","1"),("8","B"),("B","8"),
         ("5","S"),("S","5"),("2","Z"),("Z","2"),("D","0"),("0","D")}


def plate_distance(a: str, b: str) -> float:
    """Weighted edit distance. '?' matches anything for 0.1. Confusable swap = 0.3. Else 1."""
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1))
    dp[:, 0] = np.arange(n + 1); dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ca, cb = a[i-1], b[j-1]
            if ca == cb:                    sub = 0.0
            elif "?" in (ca, cb):           sub = 0.1
            elif (ca, cb) in CHEAP:         sub = 0.3
            else:                           sub = 1.0
            dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + sub)
    return float(dp[n, m])


def plausible(a: dict, b: dict, graph: dict, slack: float = 5.0) -> bool:
    """Hard filter: could the vehicle physically get from a.cam to b.cam in that time?"""
    win = graph.get((a["cam"], b["cam"]))
    if win is None:
        return False
    dt = b["t"] - a["t"]
    return (win[0] - slack) <= dt <= (win[1] + slack)


def link_score(a: dict, b: dict, graph: dict) -> tuple[float, str]:
    """
    Returns (score 0..1, method). Link only if score >= threshold (tune on labeled pairs).
    Order of checks:
      1. physics (hard)          -> 0
      2. type mismatch (hard)    -> 0
      3. both plates clean & clearly different (hard) -> 0   # never merge two clean reads
      4. plate similarity        -> strong evidence
      5. embedding + colour      -> supporting / fallback
    """
    if not plausible(a, b, graph):
        return 0.0, "implausible"
    if a["type"] != b["type"]:
        return 0.0, "type_mismatch"

    pa, pb = a["plate"], b["plate"]
    clean_a = pa and min(a["conf"]) >= 0.8
    clean_b = pb and min(b["conf"]) >= 0.8

    if pa and pb:
        d = plate_distance(pa, pb)
        if d == 0:
            return 1.0, "exact"
        if clean_a and clean_b and d >= 1.0:
            return 0.0, "different_vehicle"          # rule 3
        plate_sim = max(0.0, 1.0 - d / 3.0)         # d=0 ->1, d=3 ->0
        method = "fuzzy"
    else:
        plate_sim = None
        method = "inferred"                          # no plate on at least one side

    emb_sim = float(np.dot(a["emb"], b["emb"]) /
                    (np.linalg.norm(a["emb"]) * np.linalg.norm(b["emb"]) + 1e-9))
    emb_sim = (emb_sim + 1) / 2                      # -1..1 -> 0..1
    colour_sim = 1.0 if a["colour"] == b["colour"] else 0.4

    if plate_sim is not None:
        score = 0.6 * plate_sim + 0.3 * emb_sim + 0.1 * colour_sim
    else:
        score = 0.75 * emb_sim + 0.25 * colour_sim
    return score, method


# Thresholds you tune on ~300 hand-labelled pairs (pick precision >= 0.98):
#   fuzzy   -> link if score >= 0.75
#   inferred-> link if score >= 0.85  (stricter, and mark as low-confidence in DB)


# ----------------------------------------------------------------------
# Quick self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    reads = [
        {"chars": list("GJ01AB1234"), "confs": [0.9]*10, "quality": 0.9},
        {"chars": list("GJ01A81234"), "confs": [0.9]*4+[0.9,0.6]+[0.9]*4, "quality": 0.7},
        {"chars": list("GJ01AB234"),  "confs": [0.8]*9,  "quality": 0.5},
        {"chars": list("GJ0?AB1234"), "confs": [0.9,0.9,0.9,0.0]+[0.9]*6, "quality": 0.8},
        {"chars": list("GJ01AB1234"), "confs": [0.95]*10, "quality": 0.95},
    ]
    print(vote(reads))          # -> ('GJ01AB1234', [...], True)
    print(plate_distance("GJ01AB1234", "GJ01A81234"))   # 0.3
    print(plate_distance("GJ01AB1234", "GJ01AB1284"))   # 1.0
