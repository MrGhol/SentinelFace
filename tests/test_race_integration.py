import unittest
import numpy as np
from config import Config
from models.race import RaceClassifier
from models.utils import build_providers

class TestRaceIntegration(unittest.TestCase):
    def test_race_classifier(self):
        cfg = Config()
        providers, cuda_opts = build_providers(cfg)
        
        # Instantiate
        classifier = RaceClassifier(cfg, providers, cuda_opts)
        
        # Predict on random crop (224, 224, 3)
        dummy_crop = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        race, conf = classifier.predict(dummy_crop)
        
        print(f"\n[RaceClassifier Test] Predicted: '{race}' with confidence: {conf:.4f}")
        self.assertIn(race, classifier.RACES)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        
        # Clean up
        classifier.destroy()

if __name__ == "__main__":
    unittest.main()
