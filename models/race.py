import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config
from models.utils import make_session, validate_model_shapes

logger = logging.getLogger("FaceSystem.Race")

class RaceClassifier:
    RACES = ["Black", "East Asian", "Indian", "Latino_Hispanic", "Middle Eastern", "Southeast Asian", "White"]

    def __init__(self, cfg: Config, providers: List, cuda_opts: Optional[Dict]):
        self.cfg = cfg
        self.session = make_session(cfg.race_model, cfg, providers, cuda_opts)
        self.inp_name = self.session.get_inputs()[0].name
        
        # Expected input shape: [1, 3, 224, 224] (batch, channels, height, width)
        validate_model_shapes(self.session, "Race", [("", [1, 3, 224, 224])], 1)
        logger.info("RaceClassifier ready: %s", cfg.race_model)

    def _preprocess(self, crop_224: np.ndarray) -> np.ndarray:
        """Converts face crop to [1, 3, 224, 224] float32 scaled to [0, 1]."""
        if crop_224.shape[:2] != (224, 224):
            crop_224 = cv2.resize(crop_224, (224, 224))
        img = crop_224.astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[np.newaxis]

    def predict(self, crop_224: np.ndarray) -> Tuple[str, float]:
        """Returns (race, probability) or ("?", 0.0) on failure."""
        if crop_224 is None or crop_224.size == 0:
            return "?", 0.0
        
        blob = self._preprocess(crop_224)
        try:
            raw = self.session.run(None, {self.inp_name: blob})[0].flatten()
        except Exception as exc:
            logger.error("Race inference failed: %s", exc)
            return "?", 0.0

        idx = int(np.argmax(raw))
        return self.RACES[idx], float(raw[idx])

    def destroy(self):
        del self.session
        self.session = None
