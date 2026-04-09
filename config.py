import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

logger = logging.getLogger("FaceSystem.Config")

@dataclass
class Config:
    # ── Model paths ───────────────────────────────────────────────────────────
    scrfd_model:   str = "models/buffalo_l/det_10g.onnx"
    arcface_model: str = "models/buffalo_l/w600k_r50.onnx"
    gender_model:  str = "models/gender.onnx"
    age_model:     str = "models/age.onnx"
    enroll_dir:    str = "enrolled_faces"

    # ── Recognition ───────────────────────────────────────────────────────────
    similarity_threshold: float = 0.45
    fused_threshold:      float = 0.40
    conf_threshold:       float = 0.50
    sim_weight:           float = 0.60
    quality_weight:       float = 0.25
    conf_weight:          float = 0.15

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker_iou:       float = 0.40
    tracker_max_age:   int   = 60
    tracker_min_hits:  int   = 2
    max_active_tracks: int   = 50
    reid_threshold:    float = 0.50
    
    # Track sanity kill conditions
    track_max_area_frac:   float = 0.65  # kill if box > 65% of frame area
    track_max_aspect_ratio: float = 8.0  # kill if w/h or h/w exceeds this
    track_max_area_jump:   float = 2.5   # kill if area grows > 2.5x in one predict
    track_max_vel_jump:    float = 3.0   # kill if velocity jumps > 3x previous
    track_grace_frames:    int   = 2     # frames before killing on violation

    # ── Quality gates ─────────────────────────────────────────────────────────
    quality_blur_thresh:    int   = 100
    quality_min_face_px:    int   = 60
    quality_min_brightness: int   = 20
    min_display_quality:    float = 0.30
    min_update_quality:     float = 0.20

    # ── Smoothing / caching ───────────────────────────────────────────────────
    recog_cache_frames: int = 5
    smoothing_window:   int = 10
    fps_avg_frames:     int = 30

    # ── Adaptive FPS ──────────────────────────────────────────────────────────
    target_fps:         float = 25.0
    detect_every_n:     int   = 1
    detect_every_n_max: int   = 4

    # ── FairFace gender/age ───────────────────────────────────────────────────
    genderage_settle_votes:  int   = 5
    fairface_conf_gate:      float = 0.20
    fairface_bbox_pad:       float = 0.40
    fairface_max_crop_age:   int   = 30   
    fairface_max_gate_fails: int   = 10   
    fairface_male_class_indices: tuple = ()

    # ── SCRFD detector ────────────────────────────────────────────────────────
    scrfd_input_size: Tuple[int, int] = (640, 640)
    scrfd_mean:       float = 127.5
    scrfd_std:        float = 128.0
    iou_threshold:    float = 0.45

    # ── Adaptive detect-scale ─────────────────────────────────────────────────
    detect_scale_initial:    float = 0.50
    detect_scale_min:        float = 0.35
    detect_scale_max:        float = 0.50
    large_face_thresh_px:    int   = 200
    small_face_thresh_px:    int   = 100
    detect_scale_hysteresis: float = 0.10   
    scale_adjust_factor:     float = 0.90
    scale_warmup_frames:     int   = 150

    # ── Camera resilience ─────────────────────────────────────────────────────
    camera_reconnect_attempts: int   = 3
    camera_reconnect_delay:    float = 1.0   
    camera_stall_timeout:      float = 5.0
    camera_frozen_frame_limit: int   = 30
    camera_backoff_base:       float = 0.5
    camera_backoff_cap:        float = 16.0

    # ── Runtime ───────────────────────────────────────────────────────────────
    ort_intra_threads: int  = 2
    ort_inter_threads: int  = 2
    display_width:     int  = 960
    use_gpu:           bool = True

    # ── Health monitoring ─────────────────────────────────────────────────────
    health_fps_low_threshold:    float = 10.0
    health_inf_time_high:        float = 100.0
    health_memory_min_free_mb:   float = 500.0
    health_alert_cooldown:       float = 5.0

    def validate(self) -> None:
        def _range(name: str, val: float, lo: float = None, hi: float = None) -> None:
            if lo is not None and val < lo:
                raise ValueError(f"{name} must be >= {lo}")
            if hi is not None and val > hi:
                raise ValueError(f"{name} must be <= {hi}")

        def _positive_int(name: str, val: int, allow_zero: bool = False) -> None:
            if allow_zero:
                if val < 0:
                    raise ValueError(f"{name} must be >= 0")
            else:
                if val <= 0:
                    raise ValueError(f"{name} must be > 0")

        # Recognition thresholds/weights
        _range("similarity_threshold", self.similarity_threshold, 0.0, 1.0)
        _range("fused_threshold",      self.fused_threshold,      0.0, 1.0)
        _range("conf_threshold",       self.conf_threshold,       0.0, 1.0)
        _range("sim_weight",           self.sim_weight,           0.0, 1.0)
        _range("quality_weight",       self.quality_weight,       0.0, 1.0)
        _range("conf_weight",          self.conf_weight,          0.0, 1.0)

        # Tracker
        _range("tracker_iou",        self.tracker_iou,        0.0, 1.0)
        _positive_int("tracker_max_age",   self.tracker_max_age, allow_zero=True)
        _positive_int("tracker_min_hits",  self.tracker_min_hits)
        _positive_int("max_active_tracks", self.max_active_tracks)
        _range("reid_threshold",     self.reid_threshold,     0.0, 1.0)

        _range("track_max_area_frac",    self.track_max_area_frac,    0.0, 1.0)
        _range("track_max_aspect_ratio", self.track_max_aspect_ratio, 1.0, None)
        _range("track_max_area_jump",    self.track_max_area_jump,    1.0, None)
        _range("track_max_vel_jump",     self.track_max_vel_jump,     1.0, None)
        _positive_int("track_grace_frames", self.track_grace_frames, allow_zero=True)

        # Quality gates
        _positive_int("quality_blur_thresh",    self.quality_blur_thresh, allow_zero=True)
        _positive_int("quality_min_face_px",    self.quality_min_face_px)
        _positive_int("quality_min_brightness", self.quality_min_brightness, allow_zero=True)
        _range("min_display_quality", self.min_display_quality, 0.0, 1.0)
        _range("min_update_quality",  self.min_update_quality,  0.0, 1.0)

        # Smoothing / caching
        _positive_int("recog_cache_frames", self.recog_cache_frames, allow_zero=True)
        _positive_int("smoothing_window",   self.smoothing_window)
        _positive_int("fps_avg_frames",     self.fps_avg_frames)

        # Adaptive FPS
        _range("target_fps", self.target_fps, 1e-6, None)
        _positive_int("detect_every_n",     self.detect_every_n)
        _positive_int("detect_every_n_max", self.detect_every_n_max)
        if self.detect_every_n_max < self.detect_every_n:
            raise ValueError("detect_every_n_max must be >= detect_every_n")

        # FairFace
        _positive_int("genderage_settle_votes",  self.genderage_settle_votes)
        _range("fairface_conf_gate",      self.fairface_conf_gate,      0.0, 1.0)
        _range("fairface_bbox_pad",       self.fairface_bbox_pad,       0.0, None)
        _positive_int("fairface_max_crop_age",   self.fairface_max_crop_age, allow_zero=True)
        _positive_int("fairface_max_gate_fails", self.fairface_max_gate_fails, allow_zero=True)

        # SCRFD detector
        if (not isinstance(self.scrfd_input_size, tuple)
                or len(self.scrfd_input_size) != 2
                or any(int(v) <= 0 for v in self.scrfd_input_size)):
            raise ValueError("scrfd_input_size must be a tuple of two positive ints")
        _range("scrfd_std",     self.scrfd_std,     1e-6, None)
        _range("iou_threshold", self.iou_threshold, 0.0, 1.0)

        # Adaptive detect-scale
        _range("detect_scale_min",     self.detect_scale_min,     1e-6, None)
        _range("detect_scale_max",     self.detect_scale_max,     1e-6, None)
        _range("detect_scale_initial", self.detect_scale_initial, 1e-6, None)
        if not (self.detect_scale_min <= self.detect_scale_initial <= self.detect_scale_max):
            raise ValueError("detect_scale_initial must be within [detect_scale_min, detect_scale_max]")
        _positive_int("large_face_thresh_px", self.large_face_thresh_px)
        _positive_int("small_face_thresh_px", self.small_face_thresh_px)
        _range("detect_scale_hysteresis", self.detect_scale_hysteresis, 0.0, None)
        _range("scale_adjust_factor",     self.scale_adjust_factor,     1e-6, None)
        _positive_int("scale_warmup_frames", self.scale_warmup_frames, allow_zero=True)

        # Camera resilience
        _positive_int("camera_reconnect_attempts", self.camera_reconnect_attempts, allow_zero=True)
        _range("camera_reconnect_delay",    self.camera_reconnect_delay,    0.0, None)
        _range("camera_stall_timeout",      self.camera_stall_timeout,      0.0, None)
        _positive_int("camera_frozen_frame_limit", self.camera_frozen_frame_limit, allow_zero=True)
        _range("camera_backoff_base",       self.camera_backoff_base,       0.0, None)
        _range("camera_backoff_cap",        self.camera_backoff_cap,        0.0, None)
        if self.camera_backoff_cap < self.camera_backoff_base:
            raise ValueError("camera_backoff_cap must be >= camera_backoff_base")

        # Runtime
        _positive_int("ort_intra_threads", self.ort_intra_threads)
        _positive_int("ort_inter_threads", self.ort_inter_threads)
        _positive_int("display_width",     self.display_width)

        # Health monitoring
        _range("health_fps_low_threshold",  self.health_fps_low_threshold,  0.0, None)
        _range("health_inf_time_high",      self.health_inf_time_high,      0.0, None)
        _range("health_memory_min_free_mb", self.health_memory_min_free_mb, 1e-6, None)
        _range("health_alert_cooldown",     self.health_alert_cooldown,     0.0, None)

def load_config(json_path: str = "config.json", cli_overrides: dict = None) -> Config:
    """
    Layered config loader:
    1. Base defaults (from dataclass)
    2. JSON file overrides
    3. CLI argument overrides
    """
    cfg = Config()

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                for k, v in data.items():
                    if hasattr(cfg, k):
                        if k == "scrfd_input_size" and isinstance(v, list):
                            v = tuple(v)
                        setattr(cfg, k, v)
            logger.info(f"Loaded config overrides from {json_path}")
        except Exception as e:
            logger.error(f"Failed to load JSON config {json_path}: {e}")

    if cli_overrides:
        for k, v in cli_overrides.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)

    cfg.validate()
    return cfg
