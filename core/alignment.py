from typing import Optional, Tuple

import cv2
import numpy as np

from config import Config

KPS_LEFT_EYE  = 0
KPS_RIGHT_EYE = 1


def align_face(img: np.ndarray, bbox: np.ndarray,
               kps: np.ndarray) -> Optional[np.ndarray]:
    """5-point similarity-transform → 112×112 RGB (ArcFace input)."""
    h, w = img.shape[:2]
    le = np.clip(kps[KPS_LEFT_EYE].astype(float),  [0, 0], [w-1, h-1])
    re = np.clip(kps[KPS_RIGHT_EYE].astype(float), [0, 0], [w-1, h-1])
    dist = np.hypot(re[0]-le[0], re[1]-le[1])
    if dist < 1.0: return None
    angle  = np.degrees(np.arctan2(re[1]-le[1], re[0]-le[0]))
    center = (le + re) / 2.0
    M = cv2.getRotationMatrix2D(tuple(center), angle, 40.0 / dist)
    M[0, 2] += 56.0 - center[0]; M[1, 2] += 56.0 - center[1]
    warped = cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR)
    if warped is None or warped.size == 0: return None
    if warped.max() == 0 or warped.std() < 1.0: return None
    return warped


def face_quality(aligned_rgb: np.ndarray, bbox: np.ndarray,
                 cfg: Config) -> Tuple[float, bool]:
    if aligned_rgb is None or aligned_rgb.size == 0: return 0.0, False
    if min(int(bbox[2])-int(bbox[0]), int(bbox[3])-int(bbox[1])) < cfg.quality_min_face_px:
        return 0.0, False
    gray = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2GRAY)
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_var < cfg.quality_blur_thresh or gray.mean() < cfg.quality_min_brightness:
        return 0.0, False
    return float(min(blur_var / 500.0, 1.0)), True


def fused_score(sim: float, quality: float, det_conf: float, cfg: Config) -> float:
    return cfg.sim_weight * sim + cfg.quality_weight * quality + cfg.conf_weight * det_conf
