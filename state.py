from dataclasses import dataclass

@dataclass
class SystemState:
    """
    Shared application state to prevent boolean sprawl and UI/Worker desync.
    Written by VideoWorker, read periodically by GUI/Watchdog.
    """
    running: bool = False
    
    # Performance tracking
    current_fps: float = 0.0
    current_memory_free_mb: float = 0.0
    current_inference_time_ms: float = 0.0
    
    # Active tracks
    active_track_count: int = 0
    total_faces_processed: int = 0
    total_frames_processed: int = 0
