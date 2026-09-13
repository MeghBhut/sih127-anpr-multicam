"""Unit tests for anpr.linker."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anpr import database  # noqa: E402
from anpr import linker  # noqa: E402


def _sighting(**kwargs) -> dict:
    """Build a sighting dict with sensible defaults for tests."""
    base = {
        "cam_id": "C1",
        "t_in": 0.0,
        "t_out": 1.0,
        "plate": None,
        "plate_conf": None,
        "plate_quality": None,
        "plate_status": "missing",
        "vehicle_type": "car",
        "colour": "white",
        "direction": "north",
        "embedding": None,
        "id": None,
    }
    base.update(kwargs)
    return base


class TestPlateDistance(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(linker.plate_distance("GJ01AB1234", "GJ01AB1234"), 0.0)

    def test_one_i_confusion(self):
        d = linker.plate_distance("GJ01AB1234", "GJ0IAB1234")
        self.assertAlmostEqual(d, 0.3, places=5)

    def test_zero_o_confusion(self):
        d = linker.plate_distance("GJ01AB1234", "GJOAB1234A"[:10])
        # Build a controlled single-substitution case instead of relying on slicing tricks.
        d = linker.plate_distance("AO", "A0")
        self.assertAlmostEqual(d, 0.3, places=5)

    def test_eight_b_confusion(self):
        d = linker.plate_distance("A8", "AB")
        self.assertAlmostEqual(d, 0.3, places=5)

    def test_five_s_confusion(self):
        d = linker.plate_distance("A5", "AS")
        self.assertAlmostEqual(d, 0.3, places=5)

    def test_two_z_confusion(self):
        d = linker.plate_distance("A2", "AZ")
        self.assertAlmostEqual(d, 0.3, places=5)

    def test_partial_plate_small_distance(self):
        d = linker.plate_distance("GJ01AB??34", "GJ01AB1234")
        self.assertAlmostEqual(d, 0.2, places=5)
        sim = linker.plate_similarity("GJ01AB??34", "GJ01AB1234")
        self.assertGreater(sim, 0.9)

    def test_very_different_plates_large_distance(self):
        d = linker.plate_distance("GJ01AB1234", "MH12XY9876")
        self.assertGreater(d, 5.0)
        sim = linker.plate_similarity("GJ01AB1234", "MH12XY9876")
        self.assertLess(sim, 0.3)

    def test_empty_or_none(self):
        self.assertTrue(linker.plate_distance(None, "GJ01AB1234") == float("inf"))
        self.assertTrue(linker.plate_distance("", "GJ01AB1234") == float("inf"))
        self.assertEqual(linker.plate_similarity(None, "GJ01AB1234"), 0.0)


class TestPlausible(unittest.TestCase):
    def test_same_camera_never_plausible(self):
        a = _sighting(cam_id="C1", t_in=0.0, t_out=1.0)
        b = _sighting(cam_id="C1", t_in=10.0, t_out=11.0)
        self.assertFalse(linker.plausible(a, b))

    def test_plausible_travel_time(self):
        a = _sighting(cam_id="C1", t_in=0.0, t_out=0.0)
        b = _sighting(cam_id="C2", t_in=70.0, t_out=71.0)
        self.assertTrue(linker.plausible(a, b))

    def test_impossible_travel_time_too_fast(self):
        a = _sighting(cam_id="C1", t_in=0.0, t_out=0.0)
        b = _sighting(cam_id="C2", t_in=5.0, t_out=6.0)
        self.assertFalse(linker.plausible(a, b, slack=5.0))

    def test_unknown_route_not_plausible(self):
        a = _sighting(cam_id="C1", t_in=0.0, t_out=0.0)
        b = _sighting(cam_id="C9", t_in=70.0, t_out=71.0)
        self.assertFalse(linker.plausible(a, b))


class TestLinkScore(unittest.TestCase):
    def test_same_clean_plate_high_score(self):
        a = _sighting(cam_id="C1", t_in=0.0, t_out=0.0, plate="GJ01AB1234",
                       plate_status="clean", plate_quality=0.97, id=1)
        b = _sighting(cam_id="C2", t_in=70.0, t_out=71.0, plate="GJ01AB1234",
                       plate_status="clean", plate_quality=0.96, id=2)
        score, method = linker.link_score(a, b)
        self.assertGreaterEqual(score, linker.LINK_FUZZY)
        self.assertEqual(method, "exact")

    def test_different_clean_plates_score_zero(self):
        a = _sighting(cam_id="C1", t_in=0.0, t_out=0.0, plate="GJ01AB1234",
                       plate_status="clean", plate_quality=0.97, id=1)
        b = _sighting(cam_id="C2", t_in=70.0, t_out=71.0, plate="GJ01AB5678",
                       plate_status="clean", plate_quality=0.96, id=2)
        score, method = linker.link_score(a, b)
        self.assertEqual(score, 0.0)
        self.assertEqual(method, "reject")

    def test_clean_vs_partial_matching_plate_possible_match(self):
        a = _sighting(cam_id="C1", t_in=0.0, t_out=0.0, plate="GJ01AB1234",
                       plate_status="clean", plate_quality=0.95, id=1)
        b = _sighting(cam_id="C2", t_in=70.0, t_out=71.0, plate="GJ01AB??34",
                       plate_status="partial", plate_quality=0.6, id=2)
        score, method = linker.link_score(a, b)
        self.assertGreaterEqual(score, linker.LINK_FUZZY)
        self.assertEqual(method, "fuzzy")

    def test_missing_plate_strong_embedding_inferred_match(self):
        a = _sighting(cam_id="C1", t_in=0.0, t_out=0.0, plate="GJ01AB1234",
                       plate_status="clean", plate_quality=0.9,
                       embedding=[1.0, 0.0, 0.0], id=1)
        b = _sighting(cam_id="C2", t_in=70.0, t_out=71.0, plate=None,
                       plate_status="missing", embedding=[0.99, 0.01, 0.0], id=2)
        score, method = linker.link_score(a, b)
        self.assertEqual(method, "inferred")
        self.assertGreaterEqual(score, linker.LINK_INFERRED)

    def test_missing_plate_weak_embedding_no_confident_match(self):
        a = _sighting(cam_id="C1", t_in=0.0, t_out=0.0, plate="GJ01AB1234",
                       plate_status="clean", plate_quality=0.9,
                       embedding=[1.0, 0.0, 0.0], colour="white", id=1)
        b = _sighting(cam_id="C2", t_in=70.0, t_out=71.0, plate=None,
                       plate_status="missing", embedding=[0.0, 1.0, 0.0],
                       colour="black", id=2)
        score, method = linker.link_score(a, b)
        self.assertEqual(method, "inferred")
        self.assertLess(score, linker.LINK_INFERRED)

    def test_different_type_strong_mismatch_rejected(self):
        a = _sighting(cam_id="C1", t_in=0.0, t_out=0.0, vehicle_type="car",
                       plate=None, plate_status="missing", id=1)
        b = _sighting(cam_id="C2", t_in=70.0, t_out=71.0, vehicle_type="bus",
                       plate=None, plate_status="missing", id=2)
        score, method = linker.link_score(a, b)
        self.assertEqual(score, 0.0)
        self.assertEqual(method, "reject")

    def test_same_colour_only_never_enough(self):
        a = _sighting(cam_id="C1", t_in=0.0, t_out=0.0, plate=None,
                       plate_status="missing", vehicle_type=None,
                       colour="red", direction=None, id=1)
        b = _sighting(cam_id="C2", t_in=70.0, t_out=71.0, plate=None,
                       plate_status="missing", vehicle_type=None,
                       colour="red", direction=None, id=2)
        score, method = linker.link_score(a, b)
        self.assertLess(score, linker.LINK_INFERRED)


class TestLinkAllAndClones(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        database.init_db(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_link_all_creates_no_duplicate_links(self):
        id_a = database.save_sighting(
            {
                "cam_id": "C1", "t_in": 0.0, "t_out": 0.0,
                "plate": "GJ01AB1234", "plate_status": "clean", "plate_quality": 0.97,
                "type": "car", "colour": "white",
            }
        )
        id_b = database.save_sighting(
            {
                "cam_id": "C2", "t_in": 70.0, "t_out": 71.0,
                "plate": "GJ01AB1234", "plate_status": "clean", "plate_quality": 0.96,
                "type": "car", "colour": "white",
            }
        )
        results = linker.link_all(window=200.0)
        self.assertEqual(len(results), 1)
        pair = results[0]
        self.assertEqual({pair["a"], pair["b"]}, {id_a, id_b})

        links = database.get_links()
        self.assertEqual(len(links), 1)

    def test_clone_alert_on_impossible_travel_time(self):
        database.save_sighting(
            {
                "cam_id": "C1", "t_in": 1000.0, "t_out": 1000.0,
                "plate": "GJ01AB1234", "plate_status": "clean", "plate_quality": 0.95,
            }
        )
        database.save_sighting(
            {
                "cam_id": "C2", "t_in": 1005.0, "t_out": 1006.0,
                "plate": "GJ01AB1234", "plate_status": "clean", "plate_quality": 0.95,
            }
        )
        alerts = linker.find_cloned_plates()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["plate"], "GJ01AB1234")
        self.assertEqual(alerts[0]["reason"], "impossible travel time")

    def test_no_clone_alert_on_plausible_travel_time(self):
        database.save_sighting(
            {
                "cam_id": "C1", "t_in": 1000.0, "t_out": 1000.0,
                "plate": "GJ01AB1234", "plate_status": "clean", "plate_quality": 0.95,
            }
        )
        database.save_sighting(
            {
                "cam_id": "C2", "t_in": 1070.0, "t_out": 1071.0,
                "plate": "GJ01AB1234", "plate_status": "clean", "plate_quality": 0.95,
            }
        )
        alerts = linker.find_cloned_plates()
        self.assertEqual(alerts, [])

    def test_partial_plate_does_not_trigger_clone_alert(self):
        database.save_sighting(
            {
                "cam_id": "C1", "t_in": 1000.0, "t_out": 1000.0,
                "plate": "GJ01AB??34", "plate_status": "partial", "plate_quality": 0.4,
            }
        )
        database.save_sighting(
            {
                "cam_id": "C2", "t_in": 1005.0, "t_out": 1006.0,
                "plate": "GJ01AB??34", "plate_status": "partial", "plate_quality": 0.4,
            }
        )
        alerts = linker.find_cloned_plates()
        self.assertEqual(alerts, [])


if __name__ == "__main__":
    unittest.main()
