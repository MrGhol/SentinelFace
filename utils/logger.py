import logging
from pathlib import Path

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Dual-sink logging:
      • Console  — INFO and above
      • Rotating file (logs/facerecog.log) — DEBUG and above, 5 × 5 MB
    If the log directory cannot be created, falls back to console-only.
    """
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s – %(message)s")
    root = logging.getLogger()
    
    # Avoid duplicate handlers if setup runs multiple times
    if root.hasHandlers():
        root.handlers.clear()
        
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    try:
        from logging.handlers import RotatingFileHandler
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            Path(log_dir) / "facerecog.log",
            maxBytes=5 * 1024 * 1024,   # 5 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.info("Rotating logs enabled → %s (5 MB × 5 backups)",
                  (Path(log_dir) / "facerecog.log").resolve())
    except Exception as exc:
        root.warning("Rotating log setup failed (console-only): %s", exc)

    return logging.getLogger("FaceSystem")
