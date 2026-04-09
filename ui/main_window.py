import gc
import logging
from typing import Optional, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout,
                               QInputDialog, QLabel, QMessageBox, QPushButton,
                               QStatusBar, QVBoxLayout, QWidget)

from config import Config
from state import SystemState
from ui.video_worker import VideoWorker
from core.database import FaceDatabase

logger = logging.getLogger("FaceSystem.UI")

class MainWindow(QWidget):
    def __init__(self, cfg: Config, state: SystemState, db: FaceDatabase, worker: VideoWorker):
        super().__init__()
        self.setWindowTitle("Face Recognition System v4.8 (Modular)")
        self.setMinimumSize(980, 700)
        
        self.cfg = cfg
        self.state = state
        self.db = db
        self.worker = worker

        # Modern Dark Theme Palette
        self.apply_dark_theme()

        # Layout Setup
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Video Area
        self.video_container = QWidget()
        video_layout = QVBoxLayout(self.video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel("Open a video or camera to begin.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #0b0c10; color: #c5c6c7; font-size: 16px; border-radius: 8px; border: 2px solid #1f2833;")
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label, stretch=1)

        # Side Panel Area
        self.side_panel = QWidget()
        self.side_panel.setFixedWidth(320)
        side_layout = QVBoxLayout(self.side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(15)

        # Controls Group
        control_box = QGroupBox("Controls")
        control_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; border: 1px solid #45a29e; border-radius: 6px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #66fcf1; }")
        control_layout = QVBoxLayout(control_box)
        control_layout.setSpacing(10)

        self.open_btn   = QPushButton("📂 Open Video")
        self.cam_btn    = QPushButton("🎥 Open Camera")
        self.stop_btn   = QPushButton("⏹ Stop")
        self.enroll_btn = QPushButton("➕ Enroll Face")
        self.debug_btn  = QPushButton("🐛 Debug: OFF")
        
        buttons = (self.open_btn, self.cam_btn, self.stop_btn,
                   self.enroll_btn, self.debug_btn)
                   
        btn_style = """
            QPushButton {
                background-color: #1f2833; color: #c5c6c7; border: 1px solid #45a29e;
                border-radius: 4px; padding: 8px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background-color: #2b3a4a; color: #ffffff; }
            QPushButton:pressed { background-color: #45a29e; color: #0b0c10; }
        """
        for b in buttons: 
            b.setStyleSheet(btn_style)
            b.setCursor(Qt.PointingHandCursor)
            control_layout.addWidget(b)
            
        side_layout.addWidget(control_box)

        # Thresholds Group
        thr_box = QGroupBox("Runtime Thresholds")
        thr_box.setStyleSheet(control_box.styleSheet())
        thr_form = QFormLayout(thr_box)
        thr_form.setLabelAlignment(Qt.AlignRight)
        
        self._sim_spin  = self._spin(0.0, 1.0, self.cfg.similarity_threshold, "Similarity threshold")
        self._fuse_spin = self._spin(0.0, 1.0, self.cfg.fused_threshold, "Fused score threshold")
        self._conf_spin = self._spin(0.0, 1.0, self.cfg.conf_threshold, "Detection confidence")
        
        thr_form.addRow("Similarity:", self._sim_spin)
        thr_form.addRow("Fused:",      self._fuse_spin)
        thr_form.addRow("Det Conf:",   self._conf_spin)
        side_layout.addWidget(thr_box)

        # Health Group
        health_box = QGroupBox("System Health")
        health_box.setStyleSheet(control_box.styleSheet())
        health_form = QFormLayout(health_box)
        health_form.setLabelAlignment(Qt.AlignRight)
        
        lbl_style = "font-family: monospace; font-size: 14px; font-weight: bold; color: #45a29e;"
        self.fps_label = QLabel("N/A")
        self.inf_label = QLabel("N/A")
        self.mem_label = QLabel("N/A")
        for lbl in (self.fps_label, self.inf_label, self.mem_label):
            lbl.setStyleSheet(lbl_style)
            
        health_form.addRow("FPS (avg):", self.fps_label)
        health_form.addRow("Inf Time (ms):", self.inf_label)
        health_form.addRow("Free Mem (MB):",  self.mem_label)
        side_layout.addWidget(health_box)

        side_layout.addStretch() # Push everything up

        # Status Bar
        self._status = QStatusBar()
        self._status.setStyleSheet("background-color: #1f2833; color: #66fcf1; padding: 2px;")
        self._status.showMessage("Ready.")
        side_layout.addWidget(self._status)

        # Add to main layout
        main_layout.addWidget(self.video_container, stretch=1)
        main_layout.addWidget(self.side_panel)
        
        # Connect Worker signals
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.last_good_det_updated.connect(self._on_last_good_det)
        self.worker.worker_error.connect(self._on_worker_error)
        self.worker.worker_warning.connect(self._on_worker_warning)
        self.worker.health_alerts_ready.connect(self._on_health_alerts)
        
        # UI Events
        self.open_btn.clicked.connect(self._open_video)
        self.cam_btn.clicked.connect(self._open_camera)
        self.stop_btn.clicked.connect(self._stop)
        self.enroll_btn.clicked.connect(self._enroll)
        self.debug_btn.clicked.connect(self._toggle_debug)
        self._sim_spin.valueChanged.connect(lambda v: setattr(self.cfg, "similarity_threshold", v))
        self._fuse_spin.valueChanged.connect(lambda v: setattr(self.cfg, "fused_threshold", v))
        self._conf_spin.valueChanged.connect(lambda v: setattr(self.cfg, "conf_threshold", v))

        self._last_good_det: Optional[Tuple] = None
        
        # UI State poll timer (reads safely from SystemState without requiring signals for basic metrics)
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_state)
        self._poll_timer.start(500) # Update health UI every 500ms from state

    @staticmethod
    def _spin(lo: float, hi: float, default: float, tip: str) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setStyleSheet("""
            QDoubleSpinBox { background-color: #1f2833; color: #c5c6c7; border: 1px solid #45a29e; border-radius: 4px; padding: 2px; }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { background-color: #2b3a4a; border-radius: 2px; }
        """)
        sb.setRange(lo, hi); sb.setSingleStep(0.05)
        sb.setDecimals(2); sb.setValue(default); sb.setToolTip(tip)
        return sb
        
    def apply_dark_theme(self):
        self.setStyleSheet("background-color: #0b0c10; color: #c5c6c7;")
        # Set app-wide fusion palette dynamically
        app = QApplication.instance()
        if app:
            from PySide6.QtGui import QPalette, QColor
            app.setStyle("Fusion")
            p = QPalette()
            p.setColor(QPalette.Window, QColor(11, 12, 16))
            p.setColor(QPalette.WindowText, QColor(197, 198, 199))
            p.setColor(QPalette.Base, QColor(31, 40, 51))
            p.setColor(QPalette.AlternateBase, QColor(11, 12, 16))
            p.setColor(QPalette.ToolTipBase, QColor(69, 162, 158))
            p.setColor(QPalette.ToolTipText, QColor(11, 12, 16))
            p.setColor(QPalette.Text, QColor(197, 198, 199))
            p.setColor(QPalette.Button, QColor(31, 40, 51))
            p.setColor(QPalette.ButtonText, QColor(197, 198, 199))
            p.setColor(QPalette.BrightText, Qt.red)
            p.setColor(QPalette.Link, QColor(102, 252, 241))
            p.setColor(QPalette.Highlight, QColor(69, 162, 158))
            p.setColor(QPalette.HighlightedText, Qt.black)
            app.setPalette(p)
        
    def _poll_state(self) -> None:
        """Periodically update UI labels with thread-safe reads from SystemState."""
        if self.state.running:
            self.fps_label.setText(f"{self.state.current_fps:.1f}")
            self.inf_label.setText(f"{self.state.current_inference_time_ms:.1f}")
            self.mem_label.setText(f"{self.state.current_memory_free_mb:.0f}")

    def _start_worker(self, source) -> None:
        if self.state.running: 
            self.worker.stop()
        self.worker.set_source(source)
        self.worker.start()
        self._status.showMessage(f"Running: {source}")

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)")
        if path: self._start_worker(path)

    def _open_camera(self) -> None:
        idx, ok = QInputDialog.getInt(self, "Camera", "Camera index:", 0, 0, 9)
        if ok: self._start_worker(idx)

    def _stop(self) -> None:
        if self.state.running: 
            self.worker.stop()
        self.video_label.setText("Stopped.")
        self._status.showMessage("Stopped.")

    def _toggle_debug(self) -> None:
        self.worker.toggle_debug()
        on = self.worker.debug_mode
        self.debug_btn.setText(f"🐛 Debug: {'ON' if on else 'OFF'}")

    def _enroll(self) -> None:
        if self._last_good_det is None:
            QMessageBox.warning(self, "Enroll", "No face captured yet."); return
        _, _, aligned_rgb, emb = self._last_good_det
        if emb is None:
            QMessageBox.warning(self, "Enroll", "Quality too low — try again."); return
        hp, wp = aligned_rgb.shape[:2]
        qimg = QImage(aligned_rgb.tobytes(), wp, hp, 3*wp, QImage.Format_RGB888).copy()
        preview = QLabel(self)
        preview.setPixmap(QPixmap.fromImage(qimg).scaled(224, 224, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        preview.setWindowFlags(Qt.Window)
        preview.setWindowTitle("Enrollment Preview"); preview.show()
        name, ok = QInputDialog.getText(self, "Enrollment", "Name for this face:")
        preview.close()
        if ok and name.strip():
            self.db.enroll(name.strip(), emb)
            QMessageBox.information(self, "Enroll", f"'{name.strip()}' enrolled successfully.")
            self._status.showMessage(f"Enrolled: {name.strip()}")

    def _on_last_good_det(self, det: tuple) -> None: 
        self._last_good_det = det

    def _on_frame(self, data: bytes, w: int, h: int, bpl: int) -> None:
        q_img = QImage(data, w, h, bpl, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(self.video_label.size(),
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_worker_error(self, msg: str) -> None:
        self._status.showMessage(f"⚠ {msg}")

    def _on_worker_warning(self, msg: str) -> None:
        self._status.showMessage(msg)

    def _on_health_alerts(self, msgs: list) -> None:
        if not msgs: return
        alert_text = " | ".join(msgs)
        self._status.showMessage(f"🚨 Health: {alert_text}")
        QMessageBox.warning(self, "System Health Alert", "\n".join(msgs))

    def closeEvent(self, event) -> None:
        if self.state.running: 
            self.worker.stop()
        self.worker.release_models()
        self._poll_timer.stop()
        gc.collect()
        event.accept()
