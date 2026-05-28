import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config
from models.utils import make_session, validate_model_shapes

logger = logging.getLogger("FaceSystem.Expression")

class ExpressionRecognizer:
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def __init__(self, cfg: Config, providers: List, cuda_opts: Optional[Dict]):
        self.cfg = cfg
        self.session = make_session(cfg.facial_expression_model, cfg, providers, cuda_opts)
        self.inp_name = self.session.get_inputs()[0].name
        
        # Expected input shape: [-1, 48, 48, 1]
        validate_model_shapes(self.session, "Expression", [("", [-1, 48, 48, 1])], 1)
        logger.info("ExpressionRecognizer ready: %s", cfg.facial_expression_model)

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Converts 224x224 RGB face crop to 48x48 grayscale, in [0, 255] range."""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        blob = resized.astype(np.float32)[np.newaxis, ..., np.newaxis]
        return blob

    def predict(self, face_crop: np.ndarray) -> Tuple[str, float]:
        """Returns (emotion, probability) or ("?", 0.0) on failure."""
        if face_crop is None or face_crop.size == 0:
            return "?", 0.0
        
        blob = self._preprocess(face_crop)
        try:
            raw = self.session.run(None, {self.inp_name: blob})[0].flatten()
        except Exception as exc:
            logger.error("Expression inference failed: %s", exc)
            return "?", 0.0

        # Convert raw logits → probabilities so conf is in [0, 1] and can be
        # compared against facial_expression_conf_gate (a probability threshold).
        e = np.exp(raw.astype(np.float64) - raw.max())
        probs = (e / (e.sum() + 1e-9)).astype(np.float32)
        idx = int(np.argmax(probs))
        return self.EMOTIONS[idx], float(probs[idx])

    def destroy(self):
        del self.session
        self.session = None
