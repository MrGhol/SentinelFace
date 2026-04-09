import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from config import Config, load_config
from core.database import FaceDatabase
from core.tracker import FaceTracker
from state import SystemState
from ui.main_window import MainWindow
from ui.video_worker import VideoWorker
from utils.logger import setup_logging


def validate_model_files(cfg: Config) -> list[str]:
    missing = []
    for attr, label in (("scrfd_model",   "SCRFD detection"),
                        ("arcface_model", "ArcFace embedding"),
                        ("gender_model",  "FairFace gender"),
                        ("age_model",     "FairFace age")):
        p = Path(getattr(cfg, attr))
        if not p.exists():
            missing.append(f"  [{label}]\n    {p.resolve()}")
    return missing


def parse_args() -> dict:
    p = argparse.ArgumentParser(description="Face Recognition System v4.8 (Modular)")
    p.add_argument("--config",     default="config.json", help="Path to JSON config overrides")
    p.add_argument("--scrfd",      default=None, dest="scrfd_model")
    p.add_argument("--arcface",    default=None, dest="arcface_model")
    p.add_argument("--gender",     default=None, dest="gender_model")
    p.add_argument("--age",        default=None, dest="age_model")
    p.add_argument("--enroll-dir", default=None, dest="enroll_dir")
    p.add_argument("--threshold",  type=float, default=None, dest="similarity_threshold")
    p.add_argument("--no-gpu",     action="store_true")
    args, _ = p.parse_known_args()
    
    overrides = {k: v for k, v in vars(args).items() if v is not None and k not in ("config", "no_gpu")}
    if args.no_gpu:
        overrides["use_gpu"] = False
        
    return args.config, overrides


def main():
    logger = setup_logging()
    
    app = QApplication(sys.argv)
    
    json_path, cli_overrides = parse_args()
    try:
        cfg = load_config(json_path, cli_overrides)
    except Exception as e:
        logger.error(f"Configuration failed to load: {e}")
        sys.exit(1)

    missing = validate_model_files(cfg)
    if missing:
        msg = "Model files not found:\n\n" + "\n\n".join(missing)
        logger.error(msg)
        # We still show the window so the user sees the error graphically
        err_win = QMessageBox()
        err_win.setIcon(QMessageBox.Critical)
        err_win.setWindowTitle("Missing Model Files")
        err_win.setText(msg)
        err_win.exec()
        sys.exit(1)

    try:
        # Dependency graph instantiation
        from models.utils import build_providers
        providers, cuda_opts = build_providers(cfg)
        
        state = SystemState()
        db = FaceDatabase(Path(cfg.enroll_dir))
        tracker = FaceTracker(cfg)
        worker = VideoWorker(cfg, state, providers, cuda_opts, db, tracker)
        win = MainWindow(cfg, state, db, worker)
        
    except Exception as exc:
        logger.exception("Initialisation failed")
        err_win = QMessageBox()
        err_win.setIcon(QMessageBox.Critical)
        err_win.setWindowTitle("Init Error")
        err_win.setText(f"Initialisation failed:\n\n{exc}")
        err_win.exec()
        sys.exit(1)

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
