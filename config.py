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
        if self.track_grace_frames < 0:
            raise ValueError("track_grace_frames must be >= 0")
        if self.health_memory_min_free_mb <= 0:
            raise ValueError("health_memory_min_free_mb must be > 0")
        if self.similarity_threshold < 0.0 or self.similarity_threshold > 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

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
