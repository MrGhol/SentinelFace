import json
import os
import tempfile
import unittest

from config import Config, load_config

class TestConfig(unittest.TestCase):
    def test_layering(self):
        # 1. Defaults
        cfg = Config()
        self.assertEqual(cfg.health_memory_min_free_mb, 500.0)
        self.assertEqual(cfg.track_grace_frames, 2)
        
        # 2. JSON Override
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump({"health_memory_min_free_mb": 1000.0, "track_grace_frames": 5}, f)
            temp_path = f.name
            
        try:
            cfg2 = load_config(temp_path)
            self.assertEqual(cfg2.health_memory_min_free_mb, 1000.0)
            self.assertEqual(cfg2.track_grace_frames, 5)
            self.assertEqual(cfg2.similarity_threshold, 0.45) # remain default
            
            # 3. CLI Override overrides JSON
            cli = {"health_memory_min_free_mb": 2000.0, "similarity_threshold": 0.55}
            cfg3 = load_config(temp_path, cli)
            self.assertEqual(cfg3.health_memory_min_free_mb, 2000.0)
            self.assertEqual(cfg3.track_grace_frames, 5) # From JSON
            self.assertEqual(cfg3.similarity_threshold, 0.55) # From CLI
            
        finally:
            os.remove(temp_path)

    def test_validation(self):
        with self.assertRaises(ValueError):
            cli = {"track_grace_frames": -1}
            load_config("nonexistent.json", cli)
            
        with self.assertRaises(ValueError):
            cli = {"similarity_threshold": 1.5}
            load_config("nonexistent.json", cli)

if __name__ == "__main__":
    unittest.main()
