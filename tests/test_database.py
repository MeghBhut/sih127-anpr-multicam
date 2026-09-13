"""Unit tests for anpr.database."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anpr import database  # noqa: E402


class TestDatabaseBasics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        database.init_db(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_init_db_idempotent(self):
        database.init_db(self.tmp.name)  # should not raise
        database.init_db(self.tmp.name)

    def test_save_and_get_camera(self):
        database.save_camera("C1", 23.03, 72.58, "Main St")
        cam = database.get_camera("C1")
        self.assertIsNotNone(cam)
        self.assertEqual(cam["cam_id"], "C1")
        self.assertEqual(cam["name"], "Main St")

    def test_save_camera_upsert(self):
        database.save_camera("C1", 1.0, 1.0, "Old")
        database.save_camera("C1", 2.0, 2.0, "New")
        cams = database.get_cameras()
        self.assertEqual(len(cams), 1)
        self.assertEqual(cams[0]["name"], "New")

    def test_save_sighting_clean_plate(self):
        sid = database.save_sighting(
            {
                "cam_id": "C1",
                "t_in": 100.0,
                "t_out": 105.0,
                "plate": "GJ01AB1234",
                "plate_conf": [0.98] * 10,
                "plate_quality": 0.97,
                "plate_status": "clean",
                "type": "car",
                "colour": "white",
                "direction": "north",
                "embedding": [0.1, 0.2, 0.3],
                "crop_path": "out/crops/1.jpg",
            }
        )
        self.assertIsInstance(sid, int)
        s = database.get_sighting(sid)
        self.assertEqual(s["plate"], "GJ01AB1234")
        self.assertEqual(s["plate_status"], "clean")
        self.assertEqual(s["plate_conf"], [0.98] * 10)
        self.assertEqual(s["embedding"], [0.1, 0.2, 0.3])

    def test_save_sighting_missing_plate_is_preserved(self):
        sid = database.save_sighting(
            {
                "cam_id": "C2",
                "t_in": 200.0,
                "t_out": 205.0,
                "plate": None,
                "plate_status": "missing",
                "type": "car",
                "colour": "red",
            }
        )
        s = database.get_sighting(sid)
        self.assertIsNone(s["plate"])
        self.assertEqual(s["plate_status"], "missing")

    def test_save_sighting_partial_plate(self):
        sid = database.save_sighting(
            {
                "cam_id": "C1",
                "t_in": 10.0,
                "t_out": 12.0,
                "plate": "GJ01AB??34",
                "plate_status": "partial",
            }
        )
        s = database.get_sighting(sid)
        self.assertEqual(s["plate"], "GJ01AB??34")
        self.assertEqual(s["plate_status"], "partial")

    def test_save_sighting_requires_cam_id(self):
        with self.assertRaises(ValueError):
            database.save_sighting({"t_in": 1.0, "t_out": 2.0})

    def test_save_sighting_bad_embedding_does_not_crash(self):
        sid = database.save_sighting(
            {
                "cam_id": "C1",
                "t_in": 1.0,
                "t_out": 2.0,
                "embedding": "not-a-vector",
            }
        )
        s = database.get_sighting(sid)
        self.assertIsNone(s["embedding"])

    def test_get_by_plate_exact(self):
        database.save_sighting(
            {"cam_id": "C1", "t_in": 1.0, "t_out": 2.0, "plate": "GJ01AB1234", "plate_status": "clean"}
        )
        database.save_sighting(
            {"cam_id": "C2", "t_in": 3.0, "t_out": 4.0, "plate": "GJ01AB1234", "plate_status": "clean"}
        )
        database.save_sighting(
            {"cam_id": "C3", "t_in": 5.0, "t_out": 6.0, "plate": "MH12XY9876", "plate_status": "clean"}
        )
        results = database.get_by_plate("gj 01 ab 1234")
        self.assertEqual(len(results), 2)

    def test_get_by_plate_no_match(self):
        results = database.get_by_plate("ZZ99ZZ9999")
        self.assertEqual(results, [])

    def test_normalize_plate(self):
        self.assertEqual(database.normalize_plate("gj 01 ab 1234"), "GJ01AB1234")
        self.assertEqual(database.normalize_plate("GJ-01-AB-1234"), "GJ01AB1234")
        self.assertEqual(database.normalize_plate("GJ01AB??34"), "GJ01AB??34")
        self.assertIsNone(database.normalize_plate(None))
        self.assertIsNone(database.normalize_plate(""))

    def test_get_candidates_filters_same_camera_and_window(self):
        database.save_sighting({"cam_id": "C1", "t_in": 100.0, "t_out": 105.0})
        database.save_sighting({"cam_id": "C1", "t_in": 106.0, "t_out": 110.0})  # same cam
        database.save_sighting({"cam_id": "C2", "t_in": 150.0, "t_out": 155.0})  # in window
        database.save_sighting({"cam_id": "C2", "t_in": 5000.0, "t_out": 5005.0})  # out of window

        query = {"cam_id": "C1", "t_in": 100.0, "t_out": 105.0}
        candidates = database.get_candidates(query, window=100.0)
        cams = {c["cam_id"] for c in candidates}
        self.assertNotIn("C1", cams)
        self.assertIn("C2", cams)
        self.assertEqual(len(candidates), 1)

    def test_save_link_and_get_links_dedup(self):
        database.save_link(1, 2, 0.9, "exact")
        database.save_link(2, 1, 0.95, "exact")  # reversed pair, should upsert not duplicate
        links = database.get_links()
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["score"], 0.95)

    def test_encode_decode_embedding_roundtrip(self):
        raw = database.encode_embedding([1.0, 2.0, 3.0])
        self.assertEqual(database.decode_embedding(raw), [1.0, 2.0, 3.0])

    def test_decode_embedding_malformed_json(self):
        self.assertIsNone(database.decode_embedding("{not valid json"))

    def test_encode_embedding_none(self):
        self.assertIsNone(database.encode_embedding(None))


if __name__ == "__main__":
    unittest.main()
