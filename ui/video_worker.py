import gc
import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from config import Config
from core.alignment import align_face, face_quality, fused_score
from core.database import FaceDatabase
from core.tracker import FaceTracker
from models.arcface import ArcFaceONNX
from models.fairface import FairFaceAttributes, fairface_crop
from models.scrfd import SCRFD
from state import SystemState
from utils.health import HealthMonitor

logger = logging.getLogger("FaceSystem.VideoWorker")

def _frame_hash(f: np.ndarray) -> int:
    """Fast perceptual hash: downsample centre crop to 8×8 grey, pack to int."""
    h, w = f.shape[:2]
    cy, cx = h // 2, w // 2
    patch  = f[cy-32:cy+32, cx-32:cx+32] if h > 64 and w > 64 else f
    tiny   = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA)
    grey   = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY) if tiny.ndim == 3 else tiny
    return int(np.packbits(grey.flatten() > grey.mean()).tobytes().hex(), 16)

class VideoWorker(QThread):
    frame_ready           = Signal(bytes, int, int, int)
    last_good_det_updated = Signal(tuple)
    worker_error          = Signal(str)
    worker_warning        = Signal(str)
    health_alerts_ready   = Signal(list)

    _GPU_FAIL_THRESHOLD = 5

    def __init__(self, cfg: Config, state: SystemState, providers: List, cuda_opts: Optional[Dict],
                 db: FaceDatabase, tracker: FaceTracker, parent=None):
        super().__init__(parent)
        self.cfg        = cfg
        self.state      = state
        self._providers = providers
        self._cuda_opts = cuda_opts
        self.db         = db
        self.tracker    = tracker
        self.health     = HealthMonitor(cfg, state)
        
        self._source    = None
        self.debug_mode = False
        self.last_good_det: Optional[Tuple] = None
        self._gpu_fail_count = 0
        self._recog_cache: Dict[int, Tuple[str, float, int]] = {}

        self.scrfd:    Optional[SCRFD]              = None
        self.arcface:  Optional[ArcFaceONNX]        = None
        self.fairface: Optional[FairFaceAttributes] = None

    def set_source(self, source) -> None:
        if isinstance(source, str) and source.strip().lstrip("-").isdigit():
            source = int(source)
        self._source = source

    def toggle_debug(self) -> None: self.debug_mode = not self.debug_mode
    def stop(self, timeout_ms: int = 1500) -> None:
        self.state.running = False
        self.requestInterruption()
        if self.isRunning():
            if not self.wait(timeout_ms):
                logger.warning("VideoWorker stop timed out; thread still running.")
                self.worker_warning.emit("Stop taking too long â€” shutting down in background.")

    def _load_models(self, force_cpu: bool = False) -> bool:
        plan = (
            [(["CPUExecutionProvider"], None, "CPU")]
            if force_cpu else
            [(self._providers, self._cuda_opts, "GPU"),
             (["CPUExecutionProvider"], None,   "CPU fallback")]
        )
        for attempt, (providers, cuda_opts, label) in enumerate(plan):
            try:
                logger.info("Loading models on %s…", label)
                self.scrfd    = SCRFD(self.cfg, providers, cuda_opts)
                self.arcface  = ArcFaceONNX(self.cfg, providers, cuda_opts)
                self.fairface = FairFaceAttributes(self.cfg, providers, cuda_opts)
                self._warmup_models()
                if attempt > 0 or force_cpu:
                    self.worker_warning.emit("⚠ GPU unavailable — running on CPU")
                return True
            except Exception as exc:
                err = str(exc).lower()
                is_gpu = any(k in err for k in ("out of memory", "cuda", "cudnn", "onnxruntime"))
                logger.error("Model load failed (%s): %s", label, exc)
                if not force_cpu and attempt == 0 and is_gpu:
                    logger.warning("GPU error — retrying on CPU")
                    self.release_models(); continue
                return False
        return False

    def _warmup_models(self) -> None:
        try:
            self.scrfd.session.run(None, {
                self.scrfd.inp_name: np.zeros((1, 3, 640, 640), np.float32)})
            self.arcface.session.run(None, {
                self.arcface.inp_name: np.zeros((1, 3, 112, 112), np.float32)})
            dummy_ff = np.zeros((1, 3, 224, 224), np.float32)
            self.fairface.gender_sess.run(None, {self.fairface.gender_input: dummy_ff})
            self.fairface.age_sess.run(None,    {self.fairface.age_input:    dummy_ff})
            logger.info("ONNX warm-up complete.")
        except Exception as exc:
            logger.warning("ONNX warm-up failed (non-fatal): %s", exc)

    def _handle_inference_error(self, exc: Exception, context: str) -> None:
        err = str(exc).lower()
        if any(k in err for k in ("out of memory", "cuda", "cudnn")):
            self._gpu_fail_count += 1
            logger.error("%s GPU error (%d/%d): %s",
                         context, self._gpu_fail_count, self._GPU_FAIL_THRESHOLD, exc)
            if self._gpu_fail_count >= self._GPU_FAIL_THRESHOLD:
                logger.warning("GPU threshold reached — reloading on CPU")
                self.release_models()
                if self._load_models(force_cpu=True):
                    self._gpu_fail_count = 0
                    self.worker_warning.emit("⚠ GPU failed repeatedly — switched to CPU")
                else:
                    self.worker_error.emit("Recovery failed: CPU model load error")
                    self.state.running = False
        else:
            logger.error("%s: %s", context, exc)
            if self.scrfd is None:
                self.state.running = False

    def run(self) -> None:
        if self._source is None: return
        self.state.running = True
        if self.scrfd is None:
            if not self._load_models():
                self.worker_error.emit("Model load error — check logs")
                self.state.running = False; return
        try:
            self._main_loop()
        except Exception as exc:
            logger.exception("VideoWorker crashed: %s", exc)
            self.worker_error.emit(f"Worker error: {exc}")
        finally:
            self.state.running = False

    def _main_loop(self) -> None:
        is_camera = isinstance(self._source, int)
        cap = self._open_capture()
        if cap is None: return

        src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
        src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        out_w   = self.cfg.display_width
        out_h   = int(out_w * src_h / src_w)
        d_scale = out_w / src_w

        detect_scale  = self.cfg.detect_scale_initial
        scale_locked  = False
        avg_face_size = deque(maxlen=30)
        frame_num     = 0
        fps_queue     = deque(maxlen=self.cfg.fps_avg_frames)

        self.tracker.reset()
        self._recog_cache    = {}

        cur_det_every    = self.cfg.detect_every_n
        fps_check_window = deque(maxlen=self.cfg.fps_avg_frames)

        src_fps    = cap.get(cv2.CAP_PROP_FPS)
        if not is_camera and (src_fps <= 1 or src_fps > 120): src_fps = 30
        frame_delay = (1.0 / src_fps) if not is_camera else 0.0

        metrics = {"det_ms": 0.0, "emb_ms": 0.0, "total_ms": 0.0, "frames": 0, "faces": 0}
        consecutive_failures = 0
        last_frame_ts        = time.monotonic()
        stall_timeout        = self.cfg.camera_stall_timeout
        frozen_limit         = self.cfg.camera_frozen_frame_limit if is_camera else 0
        backoff_base         = self.cfg.camera_backoff_base
        backoff_cap          = self.cfg.camera_backoff_cap
        frozen_streak        = 0
        last_frame_hash: Optional[int] = None

        def _reconnect(cap: cv2.VideoCapture, attempt: int, reason: str) -> Optional[cv2.VideoCapture]:
            cap.release()
            delay = min(backoff_base * (2 ** (attempt - 1)), backoff_cap)
            logger.warning("Reconnect attempt %d (%s) — sleeping %.1fs", attempt, reason, delay)
            self.worker_warning.emit(f"⚠ {reason} — reconnecting ({attempt}/{self.cfg.camera_reconnect_attempts})…")
            time.sleep(delay)
            new_cap = cv2.VideoCapture(self._source)
            if new_cap.isOpened():
                logger.info("Camera reopened on attempt %d", attempt)
                return new_cap
            new_cap.release()
            return None

        ret, frame = cap.read()
        if not ret:
            cap.release(); self.worker_error.emit("Cannot read first frame"); return
        last_frame_hash = _frame_hash(frame)

        while self.state.running:
            if self.isInterruptionRequested():
                break
            if stall_timeout > 0 and is_camera:
                elapsed = time.monotonic() - last_frame_ts
                if elapsed > stall_timeout:
                    consecutive_failures += 1
                    if consecutive_failures > self.cfg.camera_reconnect_attempts:
                        self.worker_error.emit("Camera stall: reconnect limit reached"); break
                    cap = _reconnect(cap, consecutive_failures, f"Stall {elapsed:.1f}s")
                    if cap is None:
                        self.worker_error.emit("Camera stall reconnect failed"); break
                    last_frame_ts = time.monotonic(); frozen_streak = 0; continue

            ret, next_frame = cap.read()

            if not ret:
                if not is_camera:
                    logger.info("Stream ended."); break
                consecutive_failures += 1
                if consecutive_failures > self.cfg.camera_reconnect_attempts:
                    logger.info("Reconnect limit reached — stopping."); break
                cap = _reconnect(cap, consecutive_failures, "Read failure")
                if cap is None:
                    self.worker_error.emit("Camera reconnect failed"); break
                last_frame_ts = time.monotonic(); frozen_streak = 0; continue

            if frozen_limit > 0:
                h = _frame_hash(next_frame)
                if h == last_frame_hash:
                    frozen_streak += 1
                    if frozen_streak >= frozen_limit:
                        logger.warning("Frozen frame detected (%d identical) — reconnecting", frozen_streak)
                        consecutive_failures += 1
                        if consecutive_failures > self.cfg.camera_reconnect_attempts:
                            self.worker_error.emit("Frozen feed: reconnect limit reached"); break
                        cap = _reconnect(cap, consecutive_failures, "Frozen feed")
                        if cap is None:
                            self.worker_error.emit("Frozen feed reconnect failed"); break
                        last_frame_ts = time.monotonic(); frozen_streak = 0; continue
                else:
                    frozen_streak    = 0
                    last_frame_hash  = h

            consecutive_failures = 0
            last_frame_ts        = time.monotonic()

            frame_num += 1
            t_total    = time.perf_counter()
            frame_disp = cv2.resize(frame, (out_w, out_h))
            rgb_full   = None

            dets_orig: List = []
            if frame_num % cur_det_every == 0:
                t_det     = time.perf_counter()
                small     = cv2.resize(frame, None, fx=detect_scale, fy=detect_scale)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                try:
                    dets_small = self.scrfd.detect(rgb_small,
                                                   self.cfg.conf_threshold,
                                                   self.cfg.iou_threshold)
                except Exception as exc:
                    self._handle_inference_error(exc, "SCRFD detect")
                    dets_small = []
                metrics["det_ms"] = (time.perf_counter() - t_det) * 1000.0
                inv = 1.0 / detect_scale
                for bbox_s, kps_s, score in dets_small:
                    dets_orig.append(((bbox_s * inv).astype(int),
                                      (kps_s * inv).astype(int), score))

            t_emb = time.perf_counter(); face_sizes = []; detections = []
            if dets_orig:
                if rgb_full is None:
                    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for bbox, kps, det_conf in dets_orig:
                    aligned = align_face(rgb_full, bbox, kps)
                    if aligned is None: continue
                    quality, ok = face_quality(aligned, bbox, self.cfg)
                    if not ok: continue
                    emb = None
                    if quality >= self.cfg.min_update_quality:
                        try:
                            emb = self.arcface.get_embedding(aligned)
                        except Exception as exc:
                            self._handle_inference_error(exc, "ArcFace embed")
                    ff_crop = fairface_crop(rgb_full, bbox, self.cfg.fairface_bbox_pad)
                    x1d = int(bbox[0] * d_scale); y1d = int(bbox[1] * d_scale)
                    x2d = int(bbox[2] * d_scale); y2d = int(bbox[3] * d_scale)
                    detections.append(((x1d, y1d, x2d-x1d, y2d-y1d),
                                       emb, quality, det_conf, aligned, ff_crop))
                    if emb is not None:
                        self.last_good_det = (bbox, kps, aligned, emb)
                        self.last_good_det_updated.emit(self.last_good_det)
                    face_sizes.append(min(bbox[2]-bbox[0], bbox[3]-bbox[1]))
            metrics["emb_ms"] = (time.perf_counter() - t_emb) * 1000.0
            metrics["faces"] += len(detections)

            active_tracks = self.tracker.update(detections, frame_w=out_w, frame_h=out_h)
            self.state.active_track_count = len(active_tracks)

            do_fairface = (self.cfg.fairface_every_n <= 1
                           or frame_num % self.cfg.fairface_every_n == 0)
            if do_fairface:
                sv  = self.cfg.genderage_settle_votes
                mgf = self.cfg.fairface_max_gate_fails
                for tid, _b, _e, _q, _dc in active_tracks:
                    tr = self.tracker.tracks.get(tid)
                    if tr and not tr.genderage_settled and tr.last_fairface_crop is not None:
                        try:
                            gender, age_grp = self.fairface.estimate(tr.last_fairface_crop)
                        except Exception as exc:
                            self._handle_inference_error(exc, "FairFace estimate")
                            gender, age_grp = "?", "?"
                        tr.apply_genderage(gender, age_grp, sv, mgf)

            active_ids = set()
            sim_thr    = self.cfg.similarity_threshold
            fused_thr  = self.cfg.fused_threshold

            for tid, box, semb, tqual, dconf in active_tracks:
                active_ids.add(tid)
                if tqual < self.cfg.min_display_quality: continue

                cached    = self._recog_cache.get(tid)
                changed   = self.tracker.emb_changed(tid)
                emb_valid = semb is not None and float(np.linalg.norm(semb)) > 1e-6
                stale     = cached is None or changed or \
                            (frame_num - cached[2]) >= self.cfg.recog_cache_frames
                if stale:
                    name, sim = (self.db.match(semb, sim_thr)
                                 if emb_valid else ("UNKNOWN", 0.0))
                    self._recog_cache[tid] = (name, sim, frame_num)
                else:
                    name, sim = cached[0], cached[1]

                if name != "UNKNOWN" and fused_score(sim, tqual, dconf, self.cfg) < fused_thr:
                    if self.debug_mode:
                        logger.debug("Track %d '%s' rejected by fused gate", tid, name)
                    name = "UNKNOWN"

                tr      = self.tracker.tracks.get(tid)
                gender  = tr.gender     if tr else "?"
                age_str = tr.person_age if tr else "?"

                x  = max(0, int(box[0])); y  = max(0, int(box[1]))
                wb = min(int(box[2]), out_w - x); hb = min(int(box[3]), out_h - y)
                if wb <= 0 or hb <= 0: continue

                color = (0, 200, 0) if name != "UNKNOWN" else (0, 0, 220)
                cv2.rectangle(frame_disp, (x, y), (x+wb, y+hb), color, 2)

                label = f"{name} {sim:.2f}  {gender}  {age_str}"
                if self.debug_mode:
                    settled = tr.genderage_settled if tr else False
                    label  += (f"  ID:{tid} Q:{tqual:.2f} GA:{'✓' if settled else '…'}")

                (tw_, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                lx = max(0, x); ly = max(th_+8, y)
                cv2.rectangle(frame_disp, (lx, ly-th_-6), (lx+tw_+2, ly-2), color, -1)
                cv2.putText(frame_disp, label, (lx+1, ly-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            self._recog_cache = {k: v for k, v in self._recog_cache.items() if k in active_ids}

            if not scale_locked and face_sizes:
                avg_face_size.append(float(np.mean(face_sizes)))
                ha  = float(np.mean(avg_face_size))
                lo  = self.cfg.small_face_thresh_px
                hi  = self.cfg.large_face_thresh_px
                hys = self.cfg.detect_scale_hysteresis
                if ha > hi * (1 + hys):
                    detect_scale = max(self.cfg.detect_scale_min, detect_scale * self.cfg.scale_adjust_factor)
                elif ha < lo * (1 - hys):
                    detect_scale = min(self.cfg.detect_scale_max, detect_scale / self.cfg.scale_adjust_factor)
            if frame_num >= self.cfg.scale_warmup_frames:
                scale_locked = True

            now = time.perf_counter()
            fps_queue.append(now); fps_check_window.append(now)
            current_fps = 0.0
            if len(fps_queue) > 1:
                current_fps = len(fps_queue) / (fps_queue[-1] - fps_queue[0] + 1e-9)
                self.tracker._current_fps = current_fps
            
            if len(fps_check_window) > 10:
                mfps = len(fps_check_window) / (fps_check_window[-1] - fps_check_window[0] + 1e-9)
                if mfps < self.cfg.target_fps * 0.85:
                    cur_det_every = min(cur_det_every + 1, self.cfg.detect_every_n_max)
                elif mfps > self.cfg.target_fps * 1.1 and cur_det_every > self.cfg.detect_every_n:
                    cur_det_every -= 1

            metrics["total_ms"] = (time.perf_counter() - t_total) * 1000.0
            metrics["frames"]  += 1
            
            if self.debug_mode and frame_num % 30 == 0:
                logger.info(
                    "Frame %d | FPS %.1f | Det %.1fms | Emb %.1fms | Total %.1fms | Tracks %d | DetEvery %d",
                    frame_num, current_fps, metrics["det_ms"], metrics["emb_ms"],
                    metrics["total_ms"], len(active_tracks), cur_det_every)
                metrics["faces"] = metrics["frames"] = 0

            if frame_num % 30 == 0:
                alerts = self.health.update(current_fps, metrics["total_ms"])
                if alerts:
                    self.health_alerts_ready.emit(alerts)

            hud = (f"FPS:{current_fps:.1f}  Scale:{detect_scale:.2f}"
                   f"  Tracks:{len(active_tracks)}  Thr:{sim_thr:.2f}"
                   f"  Debug:{'ON' if self.debug_mode else 'off'}")
            for dx, dy, clr in ((1, 1, (0, 0, 0)), (0, 0, (255, 255, 255))):
                cv2.putText(frame_disp, hud, (10+dx, 22+dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2 if dx else 1, cv2.LINE_AA)

            rgb_disp = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            hd, wd, ch = rgb_disp.shape
            self.frame_ready.emit(rgb_disp.tobytes(), wd, hd, ch * wd)

            frame = next_frame

            if frame_delay > 0:
                elapsed = time.perf_counter() - t_total
                if frame_delay > elapsed: time.sleep(frame_delay - elapsed)

        cap.release()

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        is_camera = isinstance(self._source, int)
        attempts  = self.cfg.camera_reconnect_attempts if is_camera else 1
        for attempt in range(1, attempts + 1):
            cap = cv2.VideoCapture(self._source)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret: return cap
                cap.release()
            if attempt < attempts:
                self.worker_warning.emit(f"⚠ Cannot open source — retry {attempt}/{attempts}")
                time.sleep(self.cfg.camera_reconnect_delay)
        self.worker_error.emit(f"Cannot open: {self._source}")
        return None

    def release_models(self) -> None:
        for m in (self.scrfd, self.arcface, self.fairface):
            if m is not None:
                try: m.destroy()
                except Exception: pass
        self.scrfd = self.arcface = self.fairface = None
        gc.collect(); logger.info("ONNX sessions released.")
