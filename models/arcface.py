import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import Config
from models.utils import make_session, validate_model_shapes

logger = logging.getLogger("FaceSystem.ArcFace")

class ArcFaceONNX:
    def __init__(self, cfg: Config, providers: List, cuda_opts: Optional[Dict]):
        self.session  = make_session(cfg.arcface_model, cfg, providers, cuda_opts)
        self.inp_name = self.session.get_inputs()[0].name
        validate_model_shapes(self.session, "ArcFace", [("", [-1, 3, 112, 112])], 1)
        logger.info("ArcFace ready: %s", cfg.arcface_model)

    def get_embedding(self, aligned_rgb: np.ndarray) -> Optional[np.ndarray]:
        blob = ((np.ascontiguousarray(aligned_rgb).astype(np.float32) - 127.5) / 128.0
                ).transpose(2, 0, 1)[np.newaxis]
        try:
            emb = self.session.run(None, {self.inp_name: blob})[0].flatten()
        except Exception as exc:
            logger.error("ArcFace inference: %s", exc); return None
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-9)

    def destroy(self):
        del self.session; self.session = None
