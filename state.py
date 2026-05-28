import threading
from dataclasses import dataclass, field

@dataclass
class SystemState:
    """
    Shared application state to prevent boolean sprawl and UI/Worker desync.
    Written by VideoWorker, read periodically by GUI/Watchdog.
    """
    running: bool = False

    # Performance tracking — written by HealthMonitor (worker thread),
    # read by _poll_state (GUI thread). Protected by _metrics_lock.
    current_fps: float = 0.0
    current_memory_free_mb: float = 0.0
    current_inference_time_ms: float = 0.0

    # Active tracks
    active_track_count: int = 0
    total_faces_processed: int = 0
    total_frames_processed: int = 0

    # Lock protecting the three metric floats above.
    # Single-word fields (running, int counters) are left un-locked:
    # CPython's GIL makes them effectively atomic and this is documented.
    _metrics_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )
