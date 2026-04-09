import logging
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
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
                 quality: float, det_conf: float, smoothing_window: int):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        T = np.eye(8, dtype=np.float32)
        for i in range(4): T[i, i+4] = 1.0
        self.kf.transitionMatrix    = T
        self.kf.processNoiseCov     = np.diag(
            np.array([1., 1., 5., 5., 0.1, 0.1, 0.01, 0.01], dtype=np.float32))
        self.kf.measurementNoiseCov = np.diag(
            np.array([1., 1., 10., 10.], dtype=np.float32))
        self.kf.errorCovPost        = np.eye(8, dtype=np.float32) * 10.0

        x, y, w, h = box
        self.kf.statePost = np.array([x+w/2., y+h/2., w, h, 0., 0., 0., 0.],
                                     dtype=np.float32)
        self.box         = box
        self.features    = deque(maxlen=smoothing_window)
        self.emb_sum     = np.zeros(512, dtype=np.float32)
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

        self._prev_area: Optional[float] = None
        self._prev_vel: Tuple[float, float] = (0.0, 0.0) 
        self.suspect_counter = 0

        if emb is not None:
            self.features.append(emb); self.emb_sum += emb

    def predict(self, frame_w: int = 99999, frame_h: int = 99999) -> None:
        pred = self.kf.predict()
        cx, cy, w, h, vx, vy = pred.flatten()[:6]
        fw, fh = float(frame_w), float(frame_h)
        w = max(1.0, min(float(w), fw))
        h = max(1.0, min(float(h), fh))
        x = max(0.0, min(cx - w / 2.0, fw - w))
        y = max(0.0, min(cy - h / 2.0, fh - h))
        self.box = (x, y, w, h)
        self._prev_vel = (vx, vy)

    def update(self, box: Tuple, emb: Optional[np.ndarray], quality: float,
               det_conf: float, aligned: Optional[np.ndarray],
               ff_crop: Optional[np.ndarray]) -> None:
        x, y, w, h = box
        self.kf.correct(np.array([x+w/2., y+h/2., w, h], dtype=np.float32))
        self.box = box; self.quality = quality; self.det_conf = det_conf
        self.track_age = 0; self.hits += 1; self.emb_changed = False
        
        # Reset suspect counter on stable update
        self.suspect_counter = 0
        
        if emb is not None:
            if len(self.features) == self.features.maxlen:
                self.emb_sum -= self.features.popleft()
            self.features.append(emb); self.emb_sum += emb; self.emb_changed = True
        if not self.genderage_settled:
            if aligned  is not None: self.last_aligned       = aligned
            if ff_crop  is not None: self.last_fairface_crop = ff_crop

    def set_fps_hint(self, fps: float) -> None:
        fps   = max(1.0, fps)
        scale = 30.0 / fps   
        self.kf.processNoiseCov = np.diag(
            np.array([1.*scale, 1.*scale, 5.*scale, 5.*scale,
                      0.1*scale, 0.1*scale, 0.01*scale, 0.01*scale],
                     dtype=np.float32))

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

        prev_vx, prev_vy = self._prev_vel
        curr_vx, curr_vy = self.kf.statePost[4:6]
        prev_mag = np.hypot(prev_vx, prev_vy)
        curr_mag = np.hypot(curr_vx, curr_vy)
        if prev_mag > 0 and curr_mag / prev_mag > cfg.track_max_vel_jump:
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
                self.last_fairface_crop = None 
            return
        self.gate_fail_count = 0
        self.gender_votes.append(gender); self.age_samples.append(age)
        self.gender     = max(set(self.gender_votes), key=self.gender_votes.count)
        self.person_age = max(set(self.age_samples),  key=self.age_samples.count)
        if len(self.gender_votes) >= settle_votes:
            self.genderage_settled  = True
            self.last_aligned       = None
            self.last_fairface_crop = None

    def tick_age_cleanup(self, max_crop_age: int) -> None:
        if not self.genderage_settled and self.track_age > max_crop_age:
            self.last_fairface_crop = None
            self.last_aligned       = None

    def smoothed_embedding(self) -> np.ndarray:
        if not self.features: return np.zeros(512, dtype=np.float32)
        if len(self.features) == 1: return self.features[0]
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
                    box, emb, q, dc, aligned, ff_crop = detections[i]
                    self.tracks[tids[j]].update(box, emb, q, dc, aligned, ff_crop)
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

        if not self.tracks:
            self.next_id = 0

        if structure_changed or self._reid_dirty:
            self._rebuild_reid()

        return [(tid, tr.box, tr.smoothed_embedding(), tr.quality, tr.det_conf)
                for tid, tr in self.tracks.items()
                if tr.hits >= self.cfg.tracker_min_hits]

    def emb_changed(self, tid: int) -> bool:
        tr = self.tracks.get(tid); return tr.emb_changed if tr else False

    def _spawn(self, det: Tuple) -> None:
        box, emb, qual, dconf, aligned, ff_crop = det

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
                        self.tracks[best_tid].update(box, emb, qual, dconf, aligned, ff_crop)
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
        tr    = Track(box, embed, qual, dconf, self.cfg.smoothing_window)
        tr.last_aligned       = aligned
        tr.last_fairface_crop = ff_crop
        tr.set_fps_hint(self._current_fps)
        self.tracks[tid]  = tr
        self._reid_dirty  = True
