import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config
from models.utils import make_session, validate_model_shapes

logger = logging.getLogger("FaceSystem.FairFace")

def fairface_crop(rgb_frame: np.ndarray, bbox: np.ndarray,
                  pad: float = 0.4) -> Optional[np.ndarray]:
    """
    Padded bbox crop → 224×224 RGB for FairFace.
    Uses the original frame (NOT the ArcFace 112×112 alignment warp) so
    FairFace gets the forehead/chin/ear context it was trained on.
    """
    h, w = rgb_frame.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    px, py = int(bw * pad), int(bh * pad)
    x1p = max(0, x1 - px);  y1p = max(0, y1 - py)
    x2p = min(w, x2 + px);  y2p = min(h, y2 + py)
    if x2p <= x1p or y2p <= y1p:
        return None
    return cv2.resize(rgb_frame[y1p:y2p, x1p:x2p], (224, 224))


class FairFaceAttributes:
    """FairFace ONNX gender + age estimator."""

    AGE_GROUPS = ["0-2", "3-9", "10-19", "20-29",
                  "30-39", "40-49", "50-59", "60-69", "70+"]

    def __init__(self, cfg: Config, providers: List, cuda_opts: Optional[Dict]):
        self.cfg         = cfg
        self.gender_sess = make_session(cfg.gender_model, cfg, providers, cuda_opts)
        self.age_sess    = make_session(cfg.age_model,    cfg, providers, cuda_opts)
        self.gender_input = self.gender_sess.get_inputs()[0].name
        self.age_input    = self.age_sess.get_inputs()[0].name
        validate_model_shapes(self.gender_sess, "FairFace-Gender",
                               [("", [-1, 3, 224, 224])], 1)
        validate_model_shapes(self.age_sess,    "FairFace-Age",
                               [("", [-1, 3, 224, 224])], 1)
        self._shape_logged = False
        logger.info("FairFace ready — gender classes: %s  age classes: %s",
                    self.gender_sess.get_outputs()[0].shape[-1],
                    self.age_sess.get_outputs()[0].shape[-1])

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        # Cast to float64 before exp: safe against float16→float32 underflow.
        # +1e-9 in denominator guards zero-sum edge case.
        e = np.exp(x.astype(np.float64) - np.max(x))
        return (e / (e.sum() + 1e-9)).astype(np.float32)

    def _preprocess(self, crop_224: np.ndarray) -> np.ndarray:
        img = crop_224.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return ((img - mean) / std).transpose(2, 0, 1)[np.newaxis]

    def estimate(self, crop_224: np.ndarray) -> Tuple[str, str]:
        """Returns (gender, age_group) or ("?", "?") on failure/low confidence."""
        blob = self._preprocess(crop_224)
        try:
            g_raw = self.gender_sess.run(None, {self.gender_input: blob})[0].flatten()
            a_raw = self.age_sess.run(None,    {self.age_input:    blob})[0].flatten()
        except Exception as exc:
            logger.error("FairFace inference: %s", exc)
            return "?", "?"

        g_probs = self._softmax(g_raw)
        a_probs = self._softmax(a_raw)

        if not self._shape_logged:
            logger.info("FairFace first inference — gender softmax: %s  age softmax: %s",
                        np.round(g_probs, 3), np.round(a_probs, 3))
            self._shape_logged = True

        gate = self.cfg.fairface_conf_gate
        if g_probs.max() < gate or a_probs.max() < gate:
            return "?", "?"

        # Gender: use explicit class mapping if configured, else even-index heuristic.
        male_idx = self.cfg.fairface_male_class_indices
        if male_idx:
            safe    = [i for i in male_idx if i < len(g_probs)]
            m_prob  = float(g_probs[safe].sum()) if safe else 0.0
            f_prob  = 1.0 - m_prob
        else:
            m_prob = float(g_probs[1::2].sum())   # odd indices → Male
            f_prob = float(g_probs[0::2].sum())   # even indices → Female
        gender = "Male" if m_prob >= f_prob else "Female"

        age_group = self.AGE_GROUPS[min(int(np.argmax(a_probs)), len(self.AGE_GROUPS) - 1)]
        return gender, age_group

    def destroy(self):
        del self.gender_sess, self.age_sess
        self.gender_sess = self.age_sess = None
