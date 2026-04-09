import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config
from models.utils import make_session, validate_model_shapes

logger = logging.getLogger("FaceSystem.SCRFD")

class SCRFD:
    _CONV_A = {
        8:  {"scores": "448", "bboxes": "451", "kps": "454"},
        16: {"scores": "471", "bboxes": "474", "kps": "477"},
        32: {"scores": "494", "bboxes": "497", "kps": "500"},
    }
    _CONV_B = {
        8:  {"scores": "score_8",  "bboxes": "bbox_8",  "kps": "kps_8"},
        16: {"scores": "score_16", "bboxes": "bbox_16", "kps": "kps_16"},
        32: {"scores": "score_32", "bboxes": "bbox_32", "kps": "kps_32"},
    }
    _HEURISTIC = {
        "scores": ("score", "cls", "conf"),
        "bboxes": ("bbox", "box", "reg"),
        "kps":    ("kps", "landmark", "pts", "keypoint"),
    }

    def __init__(self, cfg: Config, providers: List, cuda_opts: Optional[Dict]):
        self.cfg      = cfg
        self.session  = make_session(cfg.scrfd_model, cfg, providers, cuda_opts)
        self.inp_name = self.session.get_inputs()[0].name
        self.strides  = [8, 16, 32]
        self.n_anch   = 2
        self.STRIDE_OUTPUTS = self._resolve_outputs()
        validate_model_shapes(self.session, "SCRFD",
                               [("", [-1, 3, -1, -1])], 9)
        logger.info("SCRFD ready: %s", cfg.scrfd_model)

    def _resolve_outputs(self) -> Dict[int, Dict[str, str]]:
        available     = {o.name for o in self.session.get_outputs()}
        output_shapes = {o.name: o.shape for o in self.session.get_outputs()}

        for label, conv in (("buffalo_l numeric", self._CONV_A),
                             ("symbolic stride",   self._CONV_B)):
            if all(d[k] in available
                   for d in conv.values() for k in ("scores", "bboxes", "kps")):
                logger.info("SCRFD output convention: %s", label)
                return conv

        logger.warning("SCRFD: heuristic output matching.")
        names = sorted(available)
        if len(names) != 9:
            raise ValueError(f"SCRFD: expected 9 outputs, got {len(names)}")

        candidates: Dict[str, List[str]] = {t: [] for t in ("scores", "bboxes", "kps")}
        for name in names:
            nl = name.lower()
            for typ, kws in self._HEURISTIC.items():
                if any(kw in nl for kw in kws):
                    candidates[typ].append(name); break
        for typ, cands in candidates.items():
            if len(cands) != 3:
                raise ValueError(f"SCRFD heuristic: expected 3 '{typ}', got {cands}")

        def anchor_count(n: str) -> int:
            shape = output_shapes.get(n, [])
            try: return int(shape[1]) if len(shape) >= 2 else int(shape[0])
            except (TypeError, IndexError, ValueError): return 0

        # Sort by anchor count (large→small) to align with stride order (8→16→32)
        for typ in candidates:
            candidates[typ].sort(key=anchor_count, reverse=True)

        result = {stride: {k: candidates[k][i] for k in ("scores", "bboxes", "kps")}
                  for i, stride in enumerate(self.strides)}
        logger.warning("SCRFD heuristic map: %s", result)
        return result

    def detect(self, rgb_img: np.ndarray, conf_thresh: float,
               iou_thresh: float) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        h, w = rgb_img.shape[:2]
        blob, scale, (pad_x, pad_y) = self._preprocess(rgb_img)
        tw, th = self.cfg.scrfd_input_size

        out_names = [self.STRIDE_OUTPUTS[s][k]
                     for s in self.strides for k in ("scores", "bboxes", "kps")]
        try:
            raw     = self.session.run(out_names, {self.inp_name: blob})
            outputs = dict(zip(out_names, raw))
        except Exception as exc:
            logger.error("SCRFD inference: %s", exc); return []

        all_scores, all_bboxes, all_kpss = [], [], []
        for s in self.strides:
            sc  = outputs[self.STRIDE_OUTPUTS[s]["scores"]]
            bx  = outputs[self.STRIDE_OUTPUTS[s]["bboxes"]]
            kp  = outputs[self.STRIDE_OUTPUTS[s]["kps"]]
            if sc.ndim == 3: sc, bx, kp = sc[0], bx[0], kp[0]
            if sc.ndim == 2: sc = sc[:, 1] if sc.shape[1] == 2 else sc.flatten()
            anch = self._anchors(s, (tw, th))
            cx, cy, sa = anch[:, 0], anch[:, 1], anch[:, 2]
            dec_box = np.stack([cx - bx[:, 0]*sa, cy - bx[:, 1]*sa,
                                cx + bx[:, 2]*sa, cy + bx[:, 3]*sa], axis=1)
            kpf = kp.reshape(len(anch), 5, 2)
            dec_kps = np.stack([cx[:, None] + kpf[:, :, 0]*sa[:, None],
                                cy[:, None] + kpf[:, :, 1]*sa[:, None]], axis=2)
            all_scores.append(sc); all_bboxes.append(dec_box); all_kpss.append(dec_kps)

        scores = np.concatenate(all_scores)
        bboxes = np.concatenate(all_bboxes)
        kpss   = np.concatenate(all_kpss)
        keep = scores >= conf_thresh
        if not keep.any(): return []
        scores, bboxes, kpss = scores[keep], bboxes[keep], kpss[keep]
        bboxes[:, 0::2] = np.clip((bboxes[:, 0::2] - pad_x) / scale, 0, w)
        bboxes[:, 1::2] = np.clip((bboxes[:, 1::2] - pad_y) / scale, 0, h)
        kpss[:, :, 0] = (kpss[:, :, 0] - pad_x) / scale
        kpss[:, :, 1] = (kpss[:, :, 1] - pad_y) / scale

        results = []
        for i in self._nms(bboxes, scores, iou_thresh):
            box = bboxes[i].astype(int); kp = kpss[i].astype(int)
            if box[2] > box[0] and box[3] > box[1]:
                results.append((box, kp, float(scores[i])))
        return results

    def _anchors(self, stride: int, input_size: Tuple[int, int]) -> np.ndarray:
        tw, th = input_size
        gx, gy = np.meshgrid(np.arange(tw // stride), np.arange(th // stride))
        cx = np.repeat((gx.flatten() + 0.5) * stride, self.n_anch)
        cy = np.repeat((gy.flatten() + 0.5) * stride, self.n_anch)
        return np.stack([cx, cy, np.full_like(cx, stride)], axis=1).astype(np.float32)

    def _preprocess(self, rgb_img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w   = rgb_img.shape[:2]
        tw, th = self.cfg.scrfd_input_size
        scale  = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        bgr    = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(bgr, (nw, nh))
        dh, dw = th - nh, tw - nw
        top, bot  = dh // 2, dh - dh // 2
        left, rgt = dw // 2, dw - dw // 2
        padded = cv2.copyMakeBorder(resized, top, bot, left, rgt,
                                    cv2.BORDER_CONSTANT, value=0)
        blob = ((padded.astype(np.float32) - self.cfg.scrfd_mean)
                / self.cfg.scrfd_std).transpose(2, 0, 1)[np.newaxis]
        return blob, scale, (left, top)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
        if len(boxes) == 0: return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]; keep = []
        while order.size:
            i = order[0]; keep.append(int(i))
            if order.size == 1: break
            xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0., xx2 - xx1) * np.maximum(0., yy2 - yy1)
            ovr   = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[np.where(ovr <= iou_thresh)[0] + 1]
        return keep

    def destroy(self):
        del self.session; self.session = None
