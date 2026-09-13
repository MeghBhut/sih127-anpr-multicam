"""
End-to-end smoke test of the Person-4 pipeline using realistic sample data.

Simulates ~3 cameras (C1, C2, C3) seeing a mix of:
  - the same vehicle with a clean plate at two cameras (should link, "exact")
  - the same vehicle with OCR noise / a partial plate (should link, "fuzzy")
  - a vehicle with a missing plate but strong appearance match (should link, "inferred")
  - a vehicle with a missing plate and weak appearance match (should NOT link)
  - two different vehicles with different clean plates (should NOT link)
  - a cloned-plate scenario (same clean plate, impossible travel time)
  - assorted noise sightings that should just sit in the DB unlinked

Run: python3 demo_sample_run.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anpr import database
from anpr import linker

DB_PATH = "demo.db"


def reset_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    database.init_db(DB_PATH)


def main():
    reset_db()

    database.save_camera("C1", 23.0225, 72.5714, "Ring Road North")
    database.save_camera("C2", 23.0300, 72.5800, "SG Highway Junction")
    database.save_camera("C3", 23.0450, 72.5900, "Airport Road")

    ids = {}

    # 1. Same vehicle, clean plate at C1 -> C2, gap=70s (plausible: 20-180s)
    ids["clean_a"] = database.save_sighting({
        "cam_id": "C1", "t_in": 1000.0, "t_out": 1001.0,
        "plate": "GJ01AB1234", "plate_conf": [0.98] * 10,
        "plate_quality": 0.97, "plate_status": "clean",
        "type": "car", "colour": "white", "direction": "north",
        "embedding": [0.9, 0.1, 0.05, 0.02],
        "crop_path": "out/crops/1.jpg",
    })
    ids["clean_b"] = database.save_sighting({
        "cam_id": "C2", "t_in": 1070.0, "t_out": 1071.0,
        "plate": "GJ01AB1234", "plate_conf": [0.95] * 10,
        "plate_quality": 0.95, "plate_status": "clean",
        "type": "car", "colour": "white", "direction": "north",
        "embedding": [0.89, 0.12, 0.04, 0.03],
        "crop_path": "out/crops/2.jpg",
    })

    # 2. Same vehicle, OCR-confused / partial plate at C2 -> C3, gap=60s (15-150s)
    ids["fuzzy_a"] = database.save_sighting({
        "cam_id": "C2", "t_in": 2000.0, "t_out": 2001.0,
        "plate": "GJ0IAB99Z6", "plate_conf": [0.7] * 10,
        "plate_quality": 0.6, "plate_status": "uncertain",
        "type": "car", "colour": "silver", "direction": "east",
        "embedding": [0.2, 0.8, 0.1, 0.0],
        "crop_path": "out/crops/3.jpg",
    })
    ids["fuzzy_b"] = database.save_sighting({
        "cam_id": "C3", "t_in": 2060.0, "t_out": 2061.0,
        "plate": "GJ01AB9926", "plate_conf": [0.9] * 10,
        "plate_quality": 0.85, "plate_status": "clean",
        "type": "car", "colour": "silver", "direction": "east",
        "embedding": [0.22, 0.78, 0.09, 0.02],
        "crop_path": "out/crops/4.jpg",
    })

    # 3. One side HAS a usable plate, other side's plate is unreadable/missing,
    #    but embedding + colour + type strongly agree -> should link as "inferred".
    ids["inferred_a"] = database.save_sighting({
        "cam_id": "C1", "t_in": 3000.0, "t_out": 3001.0,
        "plate": "KA05EF7788", "plate_conf": [0.9] * 10,
        "plate_quality": 0.9, "plate_status": "clean",
        "type": "bike", "colour": "red", "direction": "north",
        "embedding": [0.1, 0.1, 0.95, 0.05],
        "crop_path": "out/crops/5.jpg",
    })
    ids["inferred_b"] = database.save_sighting({
        "cam_id": "C2", "t_in": 3090.0, "t_out": 3091.0,
        "plate": None, "plate_status": "missing",
        "type": "bike", "colour": "red", "direction": "north",
        "embedding": [0.12, 0.09, 0.96, 0.04],
        "crop_path": "out/crops/6.jpg",
    })

    # 4. Missing plate, weak embedding (different vehicle, no plate captured) - should NOT link
    ids["weak_a"] = database.save_sighting({
        "cam_id": "C1", "t_in": 4000.0, "t_out": 4001.0,
        "plate": None, "plate_status": "missing",
        "type": "car", "colour": "black", "direction": "north",
        "embedding": [0.9, 0.05, 0.02, 0.01],
        "crop_path": "out/crops/7.jpg",
    })
    ids["weak_b"] = database.save_sighting({
        "cam_id": "C2", "t_in": 4070.0, "t_out": 4071.0,
        "plate": None, "plate_status": "missing",
        "type": "car", "colour": "blue", "direction": "north",
        "embedding": [0.05, 0.9, 0.03, 0.02],
        "crop_path": "out/crops/8.jpg",
    })

    # 5. Two different vehicles, both clean plates, plausible timing - should NOT link (score 0)
    ids["diff_a"] = database.save_sighting({
        "cam_id": "C1", "t_in": 5000.0, "t_out": 5001.0,
        "plate": "MH12XY9876", "plate_conf": [0.96] * 10,
        "plate_quality": 0.95, "plate_status": "clean",
        "type": "car", "colour": "white", "direction": "north",
        "embedding": [0.5, 0.5, 0.5, 0.5],
        "crop_path": "out/crops/9.jpg",
    })
    ids["diff_b"] = database.save_sighting({
        "cam_id": "C2", "t_in": 5070.0, "t_out": 5071.0,
        "plate": "GJ01AB1234", "plate_conf": [0.96] * 10,
        "plate_quality": 0.95, "plate_status": "clean",
        "type": "car", "colour": "white", "direction": "north",
        "embedding": [0.51, 0.49, 0.5, 0.5],
        "crop_path": "out/crops/10.jpg",
    })

    # 6. Cloned plate: same clean plate at C1 and C2 only 5s apart (min travel time 20s)
    ids["clone_a"] = database.save_sighting({
        "cam_id": "C1", "t_in": 6000.0, "t_out": 6000.5,
        "plate": "GJ05CD4321", "plate_conf": [0.97] * 10,
        "plate_quality": 0.96, "plate_status": "clean",
        "type": "car", "colour": "grey", "direction": "north",
        "crop_path": "out/crops/11.jpg",
    })
    ids["clone_b"] = database.save_sighting({
        "cam_id": "C2", "t_in": 6005.0, "t_out": 6005.5,
        "plate": "GJ05CD4321", "plate_conf": [0.97] * 10,
        "plate_quality": 0.96, "plate_status": "clean",
        "type": "car", "colour": "grey", "direction": "north",
        "crop_path": "out/crops/12.jpg",
    })

    # 7. Assorted single/noise sightings with no plausible partner
    ids["noise_a"] = database.save_sighting({
        "cam_id": "C3", "t_in": 9000.0, "t_out": 9001.0,
        "plate": "??????????", "plate_status": "unreadable",
        "type": "truck", "colour": "yellow", "direction": "south",
        "crop_path": "out/crops/13.jpg",
    })

    print("=== Saved sightings ===")
    for label, sid in ids.items():
        print(f"  {label:12s} -> id={sid}")

    print("\n=== Running linker.link_all() ===")
    results = linker.link_all(window=300.0)
    for r in results:
        print(f"  LINK a={r['a']} b={r['b']} score={r['score']:.3f} "
              f"method={r['method']} time_gap={r['time_gap']:.1f}s")

    print(f"\nTotal links created: {len(results)}")

    print("\n=== Checking expectations ===")
    id_to_label = {v: k for k, v in ids.items()}
    linked_pairs = {frozenset((r["a"], r["b"])) for r in results}

    checks = [
        ("clean_a", "clean_b", True, "same clean plate should link (exact)"),
        ("fuzzy_a", "fuzzy_b", True, "OCR-confused/partial plate should link (fuzzy)"),
        ("inferred_a", "inferred_b", True, "missing plate + strong embedding should link (inferred)"),
        ("weak_a", "weak_b", False, "missing plate + weak embedding should NOT link"),
        ("diff_a", "diff_b", False, "different clean plates should NEVER link"),
    ]
    all_ok = True
    for a_label, b_label, expected_linked, desc in checks:
        pair = frozenset((ids[a_label], ids[b_label]))
        actually_linked = pair in linked_pairs
        status = "OK" if actually_linked == expected_linked else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  [{status}] {desc} (linked={actually_linked}, expected={expected_linked})")

    print("\n=== Running linker.find_cloned_plates() ===")
    alerts = linker.find_cloned_plates()
    for a in alerts:
        print(f"  ALERT plate={a['plate']} a_id={a['a_id']} b_id={a['b_id']} "
              f"gap={a['time_gap']:.1f}s min={a['minimum_time']:.1f}s severity={a['severity']}")

    clone_ok = len(alerts) == 1 and alerts[0]["plate"] == "GJ05CD4321"
    print(f"\n  [{'OK' if clone_ok else 'FAIL'}] cloned-plate scenario detected exactly once")
    if not clone_ok:
        all_ok = False

    print("\n=== database.get_links() sanity check ===")
    db_links = database.get_links()
    print(f"  {len(db_links)} links persisted in DB (matches link_all() result count: "
          f"{len(db_links) == len(results)})")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
