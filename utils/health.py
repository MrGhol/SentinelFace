import logging
import time
from collections import deque

import psutil

from config import Config
from state import SystemState

logger = logging.getLogger("FaceSystem.Health")

class HealthMonitor:
    def __init__(self, cfg: Config, state: SystemState):
        self.cfg = cfg
        self.state = state
        self.fps_queue = deque(maxlen=30)
        self.inf_queue = deque(maxlen=30)
        
        # Independent cooldowns per alert type
        self._last_alert_time = {
            "low_fps": 0.0,
            "high_inf": 0.0,
            "low_mem": 0.0
        }

    def update(self, fps: float, total_ms: float) -> list[str]:
        """
        Records metrics to state and returns any triggered alerts.
        """
        self.fps_queue.append(fps)
        self.inf_queue.append(total_ms)
        
        avg_fps = sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0.0
        avg_inf = sum(self.inf_queue) / len(self.inf_queue) if self.inf_queue else 0.0
        
        mem = psutil.virtual_memory()
        mem_free_mb = mem.available / (1024 * 1024)
        
        # Write to shared state
        self.state.current_fps = float(avg_fps)
        self.state.current_inference_time_ms = float(avg_inf)
        self.state.current_memory_free_mb = float(mem_free_mb)
        
        alerts = []
        now = time.monotonic()
        
        if avg_fps < self.cfg.health_fps_low_threshold:
            if now - self._last_alert_time["low_fps"] > self.cfg.health_alert_cooldown:
                msg = f"Low FPS: {avg_fps:.1f}"
                logger.warning(f"Health alert: {msg}")
                alerts.append(msg)
                self._last_alert_time["low_fps"] = now
                
        if avg_inf > self.cfg.health_inf_time_high:
            if now - self._last_alert_time["high_inf"] > self.cfg.health_alert_cooldown:
                msg = f"High inference time: {avg_inf:.1f}ms"
                logger.warning(f"Health alert: {msg}")
                alerts.append(msg)
                self._last_alert_time["high_inf"] = now
                
        if mem_free_mb < self.cfg.health_memory_min_free_mb:
            if now - self._last_alert_time["low_mem"] > self.cfg.health_alert_cooldown:
                msg = f"Low memory: {mem_free_mb:.0f} MB free"
                logger.warning(f"Health alert: {msg}")
                alerts.append(msg)
                self._last_alert_time["low_mem"] = now
                
        return alerts
