"""
Precision/recall threshold-sweep utility for the linker's scoring function.

Usage
-----
Prepare a JSON file of ~50 hand-labeled sighting pairs, each with two
sighting dicts (as they'd be stored/loaded by database.py) and a label:

    [
      {
        "a": { "cam_id": "C1", "t_in": 0.0, "t_out": 0.0,
                "plate": "GJ01AB1234", "plate_status": "clean",
                "plate_quality": 0.97, "vehicle_type": "car",
                "colour": "white", "embedding": [0.1, 0.2, ...] },
        "b": { ... },
        "label": "same"        # or "different"
      },
      ...
    ]

Then run:

    python tests/threshold_eval.py labeled_pairs.json

This sweeps candidate score thresholds and reports precision/recall/TP/FP/
TN/FN at each, plus the recommended threshold (highest recall subject to
precision >= 0.98). Results are also written to threshold_report.json so
they can be dropped straight into a presentation deck.

Note: because link_score() already applies method-specific thresholds
internally to decide accept/reject in link_all(), this script instead
sweeps directly over the raw ``score`` returned by link_score() (treating
any "reject" as score 0), which is what a threshold-tuning experiment
needs to explore the precision/recall trade-off space.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anpr import linker  # noqa: E402


def evaluate(pairs: list[dict], thresholds: list[float]) -> list[dict]:
    """Score every pair once, then sweep thresholds over the cached scores."""
    scored = []
    for item in pairs:
        score, method = linker.link_score(item["a"], item["b"])
        scored.append({"score": score, "method": method, "label": item["label"]})

    report = []
    for t in thresholds:
        tp = fp = tn = fn = 0
        for s in scored:
            predicted_same = s["score"] >= t
            actually_same = s["label"] == "same"
            if predicted_same and actually_same:
                tp += 1
            elif predicted_same and not actually_same:
                fp += 1
            elif not predicted_same and not actually_same:
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        report.append(
            {
                "threshold": t,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return report


def recommend_threshold(report: list[dict], min_precision: float = 0.98) -> dict | None:
    """Pick the threshold with highest recall among those meeting min_precision."""
    candidates = [r for r in report if r["precision"] >= min_precision]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["recall"])


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python threshold_eval.py <labeled_pairs.json>")
        sys.exit(1)

    labeled_path = Path(sys.argv[1])
    pairs = json.loads(labeled_path.read_text())

    thresholds = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95
    report = evaluate(pairs, thresholds)

    print(f"{'thresh':>7} {'prec':>7} {'recall':>7} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    for r in report:
        print(
            f"{r['threshold']:>7.2f} {r['precision']:>7.3f} {r['recall']:>7.3f} "
            f"{r['tp']:>4} {r['fp']:>4} {r['tn']:>4} {r['fn']:>4}"
        )

    best = recommend_threshold(report)
    out = {"sweep": report, "recommended": best}
    Path("threshold_report.json").write_text(json.dumps(out, indent=2))

    if best:
        print(f"\nRecommended threshold: {best['threshold']} "
              f"(precision={best['precision']:.3f}, recall={best['recall']:.3f})")
    else:
        print("\nNo threshold met the required precision >= 0.98 with this labeled set.")


if __name__ == "__main__":
    main()
