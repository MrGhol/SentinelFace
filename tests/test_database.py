import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.database import FaceDatabase


class TestFaceDatabase(unittest.TestCase):
    def _rand_emb(self) -> np.ndarray:
        emb = np.random.rand(512).astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)
        return emb

    def test_enroll_and_match_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FaceDatabase(Path(tmp))
            emb = self._rand_emb()
            safe = db.enroll("Alice", emb)
            self.assertEqual(safe, "Alice")

            name, sim = db.match(emb, threshold=0.5)
            self.assertEqual(name, "Alice")
            self.assertGreaterEqual(sim, 0.5)

    def test_match_unknown_when_threshold_high(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FaceDatabase(Path(tmp))
            emb1 = self._rand_emb()
            emb2 = self._rand_emb()
            db.enroll("Bob", emb1)

            name, sim = db.match(emb2, threshold=0.999)
            self.assertEqual(name, "UNKNOWN")
            self.assertLess(sim, 0.999)

    def test_enroll_sanitizes_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FaceDatabase(Path(tmp))
            emb = self._rand_emb()
            safe = db.enroll("../../hack", emb)
            self.assertNotIn("/", safe)
            self.assertNotIn("\\", safe)
            self.assertTrue((Path(tmp) / f"{safe}.npy").exists())

    def test_enroll_rejects_reserved_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FaceDatabase(Path(tmp))
            emb = self._rand_emb()
            with self.assertRaises(ValueError):
                db.enroll("CON", emb)


if __name__ == "__main__":
    unittest.main()
