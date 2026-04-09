import unittest
import numpy as np

from config import Config
from core.tracker import FaceTracker, Track

class TestTrackerLogic(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.track_grace_frames = 2
        self.cfg.tracker_iou = 0.4
        self.tracker = FaceTracker(self.cfg)

    def test_graceful_kill_area_jump(self):
        # Create a track with a normal initial box
        emb = np.random.rand(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        self.tracker.update([((100, 100, 50, 50), emb, 0.9, 0.9, None, None)], 640, 480)
        
        self.assertEqual(len(self.tracker.tracks), 1)
        tid = list(self.tracker.tracks.keys())[0]
        track = self.tracker.tracks[tid]
        
        # Predict once to set _prev_area
        track.predict(640, 480)
        self.assertEqual(track.suspect_counter, 0)
        
        # Manually force a massive area jump to trigger the sanity check condition
        track.box = (100, 100, 300, 300) # Area jumped from 2500 -> 90000 
        
        # Calling sanity_ok directly to test the suspect counter logic
        self.assertTrue(track.sanity_ok(640, 480, self.cfg))
        self.assertEqual(track.suspect_counter, 1) # Grace frame 1
        
        self.assertTrue(track.sanity_ok(640, 480, self.cfg))
        self.assertEqual(track.suspect_counter, 2) # Grace frame 2
        
        # Third time is over the limit (grace_frames = 2), should return False and be killed by the caller
        self.assertFalse(track.sanity_ok(640, 480, self.cfg))
        self.assertEqual(track.suspect_counter, 3) 

    def test_reid_matrix_lock(self):
        # A simple smoke test to ensure the lock operates without deadlocking during basic usage
        emb1 = np.random.rand(512).astype(np.float32)
        emb2 = np.random.rand(512).astype(np.float32)
        
        self.tracker.update([((10, 10, 20, 20), emb1, 0.9, 0.9, None, None)], 640, 480)
        
        # Simulate an external thread hitting Re-ID logic while tracker predicts
        def external_reid():
            with self.tracker._reid_lock:
                # Mock read
                matrix = self.tracker._reid_matrix
                return len(matrix)
                
        # This shouldn't deadlock
        res = external_reid()
        self.assertTrue(res >= 0)
        
        # Second update forces re-evaluation
        self.tracker.update([((30, 30, 20, 20), emb2, 0.9, 0.9, None, None)], 640, 480)
        self.assertEqual(len(self.tracker.tracks), 2)

if __name__ == "__main__":
    unittest.main()
