import logging
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import Config

logger = logging.getLogger("FaceSystem.Tracker")

def _iou_xywh(a: Tuple, b: Tuple) -> float:
    ax, ay, aw, ah = a; ax2, ay2 = ax + aw, ay + ah
    bx, by, bw, bh = b; bx2, by2 = bx + bw, by + bh
    ix = max(0., min(ax2, bx2) - max(ax, bx))
    iy = max(0., min(ay2, by2) - max(ay, by))
    inter = ix * iy
    return 0. if inter == 0. else inter / (aw*ah + bw*bh - inter + 1e-9)


class Track:
    def __init__(self, box: Tuple, emb: Optional[np.ndarray],
                 quality: float, det_conf: float, smoothing_window: int,
                 expr_smooth_window: int = 10, race_smooth_window: int = 10):
        self.box         = box
        self.features    = deque(maxlen=smoothing_window)
        self.emb_sum: Optional[np.ndarray] = None  # lazily sized from first embedding
        self.track_age   = 0
        self.hits        = 1
        self.quality     = quality
        self.det_conf    = det_conf
        self.emb_changed = False

        self.gender            = "?"
        self.person_age        = "?"
        self.gender_votes      = deque(maxlen=20)
        self.age_samples       = deque(maxlen=20)
        self.genderage_settled = False
        self.gate_fail_count   = 0
        self.last_aligned:      Optional[np.ndarray] = None
        self.last_fairface_crop: Optional[np.ndarray] = None

        self.race                 = "?"
        self.race_votes           = deque(maxlen=race_smooth_window)
        self.race_settled         = False
        self.race_gate_fail_count = 0

        self.expression       = "?"
        self.expression_votes = deque(maxlen=expr_smooth_window)
        self.last_expression_crop: Optional[np.ndarray] = None

        self._prev_area: Optional[float] = None
        self.suspect_counter = 0

        if emb is not None:
            if self.emb_sum is None:
                self.emb_sum = np.zeros_like(emb)
            self.features.append(emb); self.emb_sum += emb

    def predict(self, frame_w: int = 99999, frame_h: int = 99999) -> None:
        x, y, w, h = self.box
        fw, fh = float(frame_w), float(frame_h)
        w = max(1.0, min(float(w), fw))
        h = max(1.0, min(float(h), fh))
        x = max(0.0, min(x, fw - w))
        y = max(0.0, min(y, fh - h))
        self.box = (x, y, w, h)

    def update(self, box: Tuple, emb: Optional[np.ndarray], quality: float,
               det_conf: float, aligned: Optional[np.ndarray],
               ff_crop: Optional[np.ndarray], expr_crop: Optional[np.ndarray] = None) -> None:
        self.box = box; self.quality = quality; self.det_conf = det_conf
        self.track_age = 0; self.hits += 1; self.emb_changed = False
        
        # Reset suspect counter on stable update
        self.suspect_counter = 0
        
        if emb is not None:
            if self.emb_sum is None:
                self.emb_sum = np.zeros_like(emb)
            if len(self.features) == self.features.maxlen:
                self.emb_sum -= self.features.popleft()
            self.features.append(emb); self.emb_sum += emb; self.emb_changed = True
        if aligned is not None:
            self.last_aligned = aligned
        if expr_crop is not None:
            self.last_expression_crop = expr_crop
        if not self.genderage_settled:
            if ff_crop is not None:
                self.last_fairface_crop = ff_crop

    def set_fps_hint(self, fps: float) -> None:
        pass

    def sanity_ok(self, frame_w: int, frame_h: int, cfg: Config) -> bool:
        """
        Graceful kill system logic: increment suspect_counter on violations.
        Kill track only when counter exceeds grace limit.
        """
        x, y, w, h = self.box
        if w <= 0 or h <= 0:
            return False
            
        frame_area = float(frame_w * frame_h)
        box_area   = w * h
        is_suspect = False

        if box_area > frame_area * cfg.track_max_area_frac:
            is_suspect = True

        aspect = max(w / h, h / w)
        if aspect > cfg.track_max_aspect_ratio:
            is_suspect = True

        prev_area = self._prev_area
        self._prev_area = float(box_area)
        if prev_area is not None and prev_area > 0 and box_area / prev_area > cfg.track_max_area_jump:
            is_suspect = True

        if is_suspect:
            self.suspect_counter += 1
            if self.suspect_counter > cfg.track_grace_frames:
                logger.debug("Track evicted gracefully (suspect_counter=%d)", self.suspect_counter)
                return False
        else:
            # Reverting back to stable but doing it only strictly via full update to track age in parent class
            pass
            
        return True

    def apply_genderage(self, gender: str, age: str, settle_votes: int,
                        max_gate_fails: int) -> None:
        if self.genderage_settled: return
        if gender == "?" or age == "?":
            self.gate_fail_count += 1
            if self.gate_fail_count >= max_gate_fails:
                if self.race_settled:
                    self.last_fairface_crop = None 
            return
        self.gate_fail_count = 0
        self.gender_votes.append(gender); self.age_samples.append(age)
        self.gender     = max(set(self.gender_votes), key=self.gender_votes.count)
        self.person_age = max(set(self.age_samples),  key=self.age_samples.count)
        if len(self.gender_votes) >= settle_votes:
            self.genderage_settled  = True
            if self.race_settled:
                self.last_fairface_crop = None

    def apply_race(self, race: str, settle_votes: int, max_gate_fails: int) -> None:
        if self.race_settled: return
        if race == "?":
            self.race_gate_fail_count += 1
            if self.race_gate_fail_count >= max_gate_fails:
                if self.genderage_settled:
                    self.last_fairface_crop = None
            return
        self.race_gate_fail_count = 0
        self.race_votes.append(race)
        self.race = max(set(self.race_votes), key=self.race_votes.count)
        if len(self.race_votes) >= settle_votes:
            self.race_settled = True
            if self.genderage_settled:
                self.last_fairface_crop = None

    def tick_age_cleanup(self, max_crop_age: int) -> None:
        if self.track_age > max_crop_age:
            self.last_fairface_crop = None
            self.last_aligned       = None
            self.last_expression_crop = None

    def apply_expression(self, expression: str) -> None:
        if expression == "?":
            return
        self.expression_votes.append(expression)
        self.expression = max(set(self.expression_votes), key=self.expression_votes.count)

    def smoothed_embedding(self) -> np.ndarray:
        if not self.features or self.emb_sum is None:
            return np.zeros(512, dtype=np.float32)  # fallback; dim matches ArcFace default
        if len(self.features) == 1:
            return self.features[0]
        mean = self.emb_sum / len(self.features)
        norm = np.linalg.norm(mean)
        return mean / (norm + 1e-9)


class FaceTracker:
    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self._reid_matrix: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self._reid_tids:   List[int]  = []
        self._reid_dirty:  bool       = True
        self._current_fps: float      = 30.0
        
        # Risk B: Enforce granularity locks during matrix queries mapping identity pools
        self._reid_lock = threading.RLock()

    def reset(self) -> None:
        """Fully resets tracking, ID sequence, and threaded RE-ID buffer."""
        self.tracks.clear()
        self.next_id = 0
        with self._reid_lock:
            self._reid_matrix = np.empty((0, 512), dtype=np.float32)
            self._reid_tids   = []
            self._reid_dirty  = True
        self._current_fps = 30.0

    def _rebuild_reid(self) -> None:
        tids = list(self.tracks.keys())
        with self._reid_lock:
            if not tids:
                self._reid_matrix = np.empty((0, 512), dtype=np.float32)
                self._reid_tids   = []
            else:
                self._reid_matrix = np.stack(
                    [self.tracks[tid].smoothed_embedding() for tid in tids], axis=0)
                self._reid_tids = tids
            self._reid_dirty = False

    def update(self, detections: List[Tuple],
               frame_w: int = 99999, frame_h: int = 99999) -> List[Tuple]:
        insane_tids = []
        for tid, tr in self.tracks.items():
            tr.predict(frame_w, frame_h)
            tr.track_age += 1
            tr.tick_age_cleanup(self.cfg.fairface_max_crop_age)
            if not tr.sanity_ok(frame_w, frame_h, self.cfg):
                insane_tids.append(tid)

        if insane_tids:
            for tid in insane_tids:
                del self.tracks[tid]
            self._reid_dirty = True  

        tids    = list(self.tracks.keys())
        t_boxes = [self.tracks[tid].box for tid in tids]

        structure_changed = False   

        if detections and tids:
            iou_mat = np.zeros((len(detections), len(tids)), np.float32)
            for i, det in enumerate(detections):
                for j, tb in enumerate(t_boxes):
                    iou_mat[i, j] = _iou_xywh(det[0], tb)
            row_ind, col_ind = linear_sum_assignment(-iou_mat)
            matched = set()
            for i, j in zip(row_ind, col_ind):
                if iou_mat[i, j] >= self.cfg.tracker_iou:
                    det_data = detections[i]
                    box, emb, q, dc, aligned, ff_crop = det_data[:6]
                    expr_crop = det_data[6] if len(det_data) > 6 else None
                    self.tracks[tids[j]].update(box, emb, q, dc, aligned, ff_crop, expr_crop)
                    if emb is not None:
                        self._reid_dirty = True
                    matched.add(i)
            for i, det in enumerate(detections):
                if i not in matched:
                    self._spawn(det); structure_changed = True
        elif detections:
            for det in detections: self._spawn(det)
            structure_changed = True

        before = len(self.tracks)
        self.tracks = {tid: tr for tid, tr in self.tracks.items()
                       if tr.track_age <= self.cfg.tracker_max_age}
        if len(self.tracks) != before:
            structure_changed = True

        # Bug #10 fix: do NOT reset next_id mid-session. IDs must be monotonically
        # increasing so that VideoWorker._recog_cache never gets stale hits from
        # recycled IDs. next_id is only reset in FaceTracker.reset().

        if structure_changed or self._reid_dirty:
            self._rebuild_reid()

        return [(tid, tr.box, tr.smoothed_embedding(), tr.quality, tr.det_conf)
                for tid, tr in self.tracks.items()
                if tr.hits >= self.cfg.tracker_min_hits]

    def emb_changed(self, tid: int) -> bool:
        tr = self.tracks.get(tid); return tr.emb_changed if tr else False

    def _spawn(self, det: Tuple) -> None:
        box, emb, qual, dconf, aligned, ff_crop = det[:6]
        expr_crop = det[6] if len(det) > 6 else None

        if emb is not None:
            if self._reid_dirty:
                self._rebuild_reid()
                
            with self._reid_lock:
                if len(self._reid_tids) > 0:
                    sims     = self._reid_matrix @ emb
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    if best_sim >= self.cfg.reid_threshold:
                        best_tid = self._reid_tids[best_idx]
                        self.tracks[best_tid].update(box, emb, qual, dconf, aligned, ff_crop, expr_crop)
                        self._reid_dirty = True
                        logger.debug("Re-ID: det → track %d (sim=%.3f)", best_tid, best_sim)
                        return

        if len(self.tracks) >= self.cfg.max_active_tracks:
            victim = next((tid for tid, tr in self.tracks.items()
                           if tr.hits < self.cfg.tracker_min_hits), None)
            if victim is not None:
                del self.tracks[victim]; self._reid_dirty = True
            else:
                return

        embed = emb if qual >= self.cfg.min_update_quality else None
        tid   = self.next_id; self.next_id += 1
        tr    = Track(box, embed, qual, dconf, self.cfg.smoothing_window, self.cfg.facial_expression_smooth_window, self.cfg.race_smooth_window)
        tr.last_aligned       = aligned
        tr.last_fairface_crop = ff_crop
        tr.last_expression_crop = expr_crop
        tr.set_fps_hint(self._current_fps)
        self.tracks[tid]  = tr
        self._reid_dirty  = True
