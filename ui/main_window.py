import gc
import logging
from typing import Optional, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout,
                               QInputDialog, QLabel, QMessageBox, QPushButton,
                               QStatusBar, QVBoxLayout, QWidget, QDialog,
                               QDialogButtonBox, QLineEdit)

from config import Config
from state import SystemState
from ui.video_worker import VideoWorker
from core.database import FaceDatabase

logger = logging.getLogger("FaceSystem.UI")

class RTSPLoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RTSP Connection Credentials")
        self.setFixedWidth(420)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Style sheet matching the premium dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #0b0c10;
                color: #c5c6c7;
            }
            QLabel {
                color: #66fcf1;
                font-weight: bold;
                font-size: 13px;
            }
            QLineEdit {
                background-color: #1f2833;
                color: #ffffff;
                border: 1px solid #45a29e;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #66fcf1;
            }
            QDialogButtonBox {
                margin-top: 10px;
            }
            QPushButton {
                background-color: #1f2833;
                color: #c5c6c7;
                border: 1px solid #45a29e;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2b3a4a;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #45a29e;
                color: #0b0c10;
            }
        """)
        
        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        
        self.url_input = QLineEdit("rtsp://")
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Leave empty if no auth")
        
        self.pass_input = QLineEdit()
        self.pass_input.setEchoMode(QLineEdit.Password)
        self.pass_input.setPlaceholderText("Leave empty if no auth")
        
        form_layout.addRow("RTSP URL:", self.url_input)
        form_layout.addRow("Username:", self.user_input)
        form_layout.addRow("Password:", self.pass_input)
        
        layout.addLayout(form_layout)
        
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        
        layout.addWidget(self.buttons)
        
    def get_credentials(self) -> Tuple[str, str, str]:
        return (
            self.url_input.text().strip(),
            self.user_input.text().strip(),
            self.pass_input.text().strip()
        )

class MainWindow(QWidget):
    def __init__(self, cfg: Config, state: SystemState, db: FaceDatabase, worker: VideoWorker):
        super().__init__()
        self.setWindowTitle("Face Recognition System v4.8 (Modular)")
        self.setMinimumSize(980, 700)
        
        self.cfg = cfg
        self.state = state
        self.db = db
        self.worker = worker

        # Bug #1 fix: store a pending source so _on_worker_finished can
        # start the new source only after the old thread has fully exited.
        self._pending_source = None

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
        self.rtsp_btn   = QPushButton("🌐 Open RTSP")
        self.stop_btn   = QPushButton("⏹ Stop")
        self.enroll_btn = QPushButton("➕ Enroll Face")
        self.debug_btn  = QPushButton("🐛 Debug: OFF")
        
        buttons = (self.open_btn, self.cam_btn, self.rtsp_btn, self.stop_btn,
                   self.enroll_btn, self.debug_btn)
                   
        btn_style = """
            QPushButton {
                background-color: #1f2833; color: #c5c6c7; border: 1px solid #45a29e;
                border-radius: 4px; padding: 8px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background-color: #2b3a4a; color: #ffffff; }
            QPushButton:pressed { background-color: #45a29e; color: #0b0c10; }
            QPushButton:checked { background-color: #45a29e; color: #0b0c10; }
            QPushButton:disabled { background-color: #141a22; color: #555f6b; border-color: #2a3a4a; }
        """
        for b in buttons: 
            b.setStyleSheet(btn_style)
            b.setCursor(Qt.PointingHandCursor)
            control_layout.addWidget(b)
            
        side_layout.addWidget(control_box)

        # Video Orientation Group
        trans_box = QGroupBox("Video Orientation")
        trans_box.setStyleSheet(control_box.styleSheet())
        trans_layout = QVBoxLayout(trans_box)
        trans_layout.setSpacing(10)

        self.fliph_btn = QPushButton("↔ Flip Horizontal")
        self.flipv_btn = QPushButton("↕ Flip Vertical")
        self.rot_btn   = QPushButton("⟳ Rotate: 0°")

        trans_buttons = (self.fliph_btn, self.flipv_btn, self.rot_btn)
        for b in trans_buttons:
            b.setStyleSheet(btn_style)
            b.setCursor(Qt.PointingHandCursor)
            
        self.fliph_btn.setCheckable(True)
        self.flipv_btn.setCheckable(True)
        
        self.fliph_btn.setChecked(getattr(self.cfg, "flip_h", False))
        self.flipv_btn.setChecked(getattr(self.cfg, "flip_v", False))
        self.rot_btn.setText(f"⟳ Rotate: {getattr(self.cfg, 'rotation', 0)}°")
        
        for b in trans_buttons:
            trans_layout.addWidget(b)

        side_layout.addWidget(trans_box)

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
        # Bug #1/#2 fix: react to thread lifecycle signals instead of blocking.
        # started  → re-enable buttons once the thread is actually running.
        # finished → re-enable buttons and launch any pending source.
        self.worker.started.connect(self._on_worker_started)
        self.worker.finished.connect(self._on_worker_finished)
        
        # UI Events
        self.open_btn.clicked.connect(self._open_video)
        self.cam_btn.clicked.connect(self._open_camera)
        self.rtsp_btn.clicked.connect(self._open_rtsp)
        self.stop_btn.clicked.connect(self._stop)
        self.enroll_btn.clicked.connect(self._enroll)
        self.debug_btn.clicked.connect(self._toggle_debug)
        self.fliph_btn.toggled.connect(self._toggle_flip_h)
        self.flipv_btn.toggled.connect(self._toggle_flip_v)
        self.rot_btn.clicked.connect(self._cycle_rotation)
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

    # ── Bug #6 fix: button interlock helper ───────────────────────────────────
    def _set_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable all source-control buttons during thread transitions.
        Prevents click-spamming that compounds the race conditions (Bug #6 fix)."""
        for btn in (self.open_btn, self.cam_btn, self.rtsp_btn, self.stop_btn):
            btn.setEnabled(enabled)

    # ── Bug #1 fix: deferred start via finished signal ────────────────────────
    def _start_worker(self, source) -> None:
        """Request a new source. If the worker is still running, save the source
        as pending and let _on_worker_finished start it once the thread exits.
        This eliminates the race where start() is a no-op on a live thread."""
        self._pending_source = source
        if self.worker.isRunning():
            # Disable buttons immediately so the user can't queue more requests.
            self._set_controls_enabled(False)
            self._status.showMessage("Stopping previous source…")
            self.worker.stop()  # non-blocking: signals thread to exit
            # _on_worker_finished will call _launch_pending() when thread exits.
        else:
            self._launch_pending()

    def _launch_pending(self) -> None:
        """Start the worker with the stored pending source. Only called when
        the worker thread is confirmed to be not running."""
        if self._pending_source is None:
            return
        self.worker.set_source(self._pending_source)
        source_label = str(self._pending_source)
        self._pending_source = None
        # Buttons stay disabled through the brief start window.
        # _on_worker_started will re-enable them once the thread is running.
        self.worker.start()
        self._status.showMessage(f"Running: {source_label}")

    def _on_worker_started(self) -> None:
        """Slot called when the VideoWorker thread begins executing.
        Re-enables buttons so the user can switch sources or stop the stream.
        Buttons are only disabled during the brief stop→start transition window,
        not for the entire duration of the stream."""
        self._set_controls_enabled(True)

    def _on_worker_finished(self) -> None:
        """Slot called when the VideoWorker thread finishes (Bug #1/#2 fix).
        Re-enables buttons and starts any pending source."""
        self._set_controls_enabled(True)
        if self._pending_source is not None:
            self._launch_pending()

    # ─────────────────────────────────────────────────────────────────────────

    def _poll_state(self) -> None:
        """Periodically update UI labels with thread-safe reads from SystemState."""
        if self.state.running:
            # Bug #13 fix: acquire lock so we read a consistent snapshot of all
            # three metric fields that are written by the worker thread.
            with self.state._metrics_lock:
                fps = self.state.current_fps
                inf = self.state.current_inference_time_ms
                mem = self.state.current_memory_free_mb
            self.fps_label.setText(f"{fps:.1f}")
            self.inf_label.setText(f"{inf:.1f}")
            self.mem_label.setText(f"{mem:.0f}")

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)")
        if path: self._start_worker(path)

    def _open_camera(self) -> None:
        idx, ok = QInputDialog.getInt(self, "Camera", "Camera index:", 0, 0, 9)
        if ok: self._start_worker(idx)

    def _open_rtsp(self) -> None:
        dialog = RTSPLoginDialog(self)
        if dialog.exec() == QDialog.Accepted:
            url, user, password = dialog.get_credentials()
            if url:
                import urllib.parse
                # Format URL with credentials
                prefix = "rtsp://"
                if url.lower().startswith("rtsps://"):
                    prefix = "rtsps://"
                elif not url.lower().startswith("rtsp://"):
                    url = prefix + url
                    
                rest = url[len(prefix):]
                if "@" not in rest and (user or password):
                    user_q = urllib.parse.quote(user)
                    pass_q = urllib.parse.quote(password)
                    if user_q and pass_q:
                        url = f"{prefix}{user_q}:{pass_q}@{rest}"
                    elif user_q:
                        url = f"{prefix}{user_q}@{rest}"
                
                self._start_worker(url)

    def _stop(self) -> None:
        if self.worker.isRunning():
            self._set_controls_enabled(False)  # Bug #6: disable until thread exits
            self.worker.stop()
            self._status.showMessage("Stopping…")
        self.video_label.setText("Stopped.")

    def _toggle_flip_h(self, checked: bool) -> None:
        self.worker.set_flip_h(checked)
        self._status.showMessage(f"Horizontal Flip: {'ON' if checked else 'OFF'}")

    def _toggle_flip_v(self, checked: bool) -> None:
        self.worker.set_flip_v(checked)
        self._status.showMessage(f"Vertical Flip: {'ON' if checked else 'OFF'}")

    def _cycle_rotation(self) -> None:
        current = self.worker.rotation
        next_rot = (current + 90) % 360
        self.worker.set_rotation(next_rot)
        self.rot_btn.setText(f"⟳ Rotate: {next_rot}°")
        self._status.showMessage(f"Rotation: {next_rot}°")

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
        raw_name = name.strip()
        if ok and raw_name:
            try:
                safe_name = self.db.enroll(raw_name, emb)
            except ValueError as exc:
                QMessageBox.warning(self, "Enroll", str(exc))
                return
            if safe_name != raw_name:
                QMessageBox.information(
                    self, "Enroll", f"'{raw_name}' enrolled as '{safe_name}'.")
            else:
                QMessageBox.information(self, "Enroll", f"'{safe_name}' enrolled successfully.")
            self._status.showMessage(f"Enrolled: {safe_name}")

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
        """Bug #4 fix: removed QMessageBox.warning() which blocked the event loop
        and caused an inescapable modal dialog loop under sustained low performance.
        Alerts are now shown in the status bar with a brief visual flash."""
        if not msgs:
            return
        alert_text = " | ".join(msgs)
        self._status.showMessage(f"🚨 {alert_text}")
        # Brief colour flash to draw attention without blocking the event loop.
        self._status.setStyleSheet(
            "background-color: #7b1a1a; color: #ff6b6b; padding: 2px;")
        QTimer.singleShot(3000, self._reset_status_style)

    def _reset_status_style(self) -> None:
        self._status.setStyleSheet(
            "background-color: #1f2833; color: #66fcf1; padding: 2px;")

    def closeEvent(self, event) -> None:
        """Bug #2 fix: always stop the worker and wait for full thread exit
        before releasing ONNX models. The original code could call release_models()
        while the thread was mid-inference, causing a segfault/access violation."""
        self._poll_timer.stop()

        # Disconnect the finished signal so _on_worker_finished doesn't fire
        # during shutdown and attempt to re-enable buttons or launch pending sources.
        try:
            self.worker.started.disconnect(self._on_worker_started)
            self.worker.finished.disconnect(self._on_worker_finished)
        except RuntimeError:
            pass  # already disconnected

        # Unconditionally stop — stop() checks isRunning() internally so this
        # is safe even if the thread is already stopped.
        self.worker.stop()

        # Block here until the background thread is fully exited.
        # This is acceptable: the user is closing the app.
        # Without this wait(), release_models() below could race with inference.
        self.worker.wait()

        # Thread is guaranteed stopped — safe to delete ONNX sessions.
        self.worker.release_models()
        gc.collect()
        event.accept()
