# SentinelFace

**Real-Time Face Recognition System (Modular Architecture)**

## Overview

SentinelFace is a robust, real-time face recognition system built with a modular, production-oriented architecture.

Originally developed as a monolithic prototype (`FaceDet.py`), the system has been refactored into a scalable package structure (`SentinelFace/`) designed for maintainability, concurrency safety, and long-term evolution.

The application uses:

- **PySide6** for the GUI
- **ONNX Runtime** for GPU-accelerated inference
- **SCRFD** for detection
- **ArcFace** for embeddings
- **FairFace** for age and gender classification

The UI and ML pipeline are cleanly decoupled to ensure responsiveness under real-time load.

## Installation

### CPU install

    pip install -r requirements-cpu.txt

### GPU install

    pip install -r requirements-gpu.txt

## Notes

GPU inference requires a compatible NVIDIA driver and CUDA runtime for `onnxruntime-gpu`.

If GPU is unavailable at runtime, the app will fall back to CPU automatically.

## Models (Not in Git)

Model binaries are intentionally excluded from this repository. You need these files locally:

- `models/buffalo_l/det_10g.onnx`
- `models/buffalo_l/w600k_r50.onnx`
- `models/gender.onnx`
- `models/age.onnx`

### Download sources

| Model | Source |
| --- | --- |
| InsightFace `buffalo_l` pack | https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download |
| FairFace repository | https://github.com/dchen236/FairFace |
| FairFace pretrained weights | https://drive.google.com/drive/folders/1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu?usp=sharing |

### Notes

FairFace provides PyTorch weights. This project expects ONNX exports for gender and age, saved as `models/gender.onnx` and `models/age.onnx`.

See `models/README.md` for details and placement.

## Architecture Overview

    SentinelFace/
    ├── main.py
    ├── config.py
    ├── state.py
    ├── core/
    │   ├── alignment.py
    │   ├── database.py
    │   └── tracker.py
    ├── models/
    │   ├── scrfd.py
    │   ├── arcface.py
    │   ├── fairface.py
    │   └── utils.py
    ├── ui/
    │   ├── main_window.py
    │   └── video_worker.py
    └── utils/
        ├── health.py
        └── logger.py

The system enforces strict separation of concerns and prohibits upward dependencies.

## Core Components

### 1. Configuration (`config.py`)

Implements a layered configuration system:

- Typed defaults (`@dataclass`)
- JSON overrides
- CLI overrides

Final configuration is validated to prevent invalid runtime thresholds.

This allows deployment tuning without code modification.

### 2. Shared System State (`state.py`)

A `SystemState` dataclass acts as a central telemetry source:

- FPS
- Inference time
- Memory availability
- Active track count

The worker thread updates this state, while the GUI reads it safely via timers.

This eliminates global state sprawl and cross-thread flag issues.

### 3. Concurrency Model

#### VideoWorker (`QThread`)

Responsible for:

- Frame ingestion (OpenCV)
- Health monitoring
- Detection → Alignment → Re-ID → Demographics → Rendering
- Emitting lightweight Qt signals

All ML operations run inside this worker thread.

#### MainWindow

- Consumes worker signals
- Displays frames
- Polls telemetry
- Remains fully responsive

The UI never performs heavy computation.

### 4. Model Adapters (`models/`)

Each ONNX model is wrapped in a dedicated adapter:

- SCRFD — Face detection
- ArcFace — 512D embedding extraction
- FairFace — Age and gender classification

Adapters validate input/output shapes and isolate inference logic from the rest of the system.

### 5. Tracking & Re-Identification (`core/tracker.py`)

The tracking system includes:

- IoU-based association
- Kalman motion prediction
- Cosine similarity Re-ID
- Grace-based track eviction (temporal smoothing)
- Defensive bounding box sanity checks

A granular `RLock` protects the embedding similarity matrix, allowing safe future concurrency without locking the entire tracker.

### 6. Face Database (`core/database.py`)

Handles:

- Enrollment loading
- Embedding storage
- Fast vectorized similarity search

Designed to remain thread-safe and injection-driven, with no globals.

### 7. Health Monitoring (`utils/health.py`)

Monitors:

- FPS drops
- Inference spikes
- Low memory conditions

Each alert type has independent cooldown timers to prevent log flooding.

Memory monitoring uses absolute available MB to avoid false alarms on mid-range systems.

## Resilience Features

- Camera disconnect auto-reconnect with backoff
- Frozen frame detection
- Graceful tracker cleanup
- Rotating logs (5MB × 5 backups)
- GPU fallback safety

The system is designed for long-running real-time deployment.

## Testing

Minimal regression scaffolding exists:

- `test_config.py` — Verifies layered config resolution
- `test_tracker_logic.py` — Verifies grace-kill and locking behavior

These tests protect against architectural regressions during refactoring.

## Design Principles

- Explicit dependency injection
- No hidden globals
- Strict layering
- Defensive programming
- Clearly defined concurrency boundaries
- Extensibility without premature complexity

## Current Status

- **Architecture Level:** Engineered System
- **Deployment Readiness:** Production-ready (single-node)
- **Scalability Path:** Prepared for multi-camera and distributed evolution

## Future Directions

- Multi-camera orchestration
- Centralized embedding service
- Event-driven architecture
- Deployment packaging
- Centralized logging aggregation

## Conclusion

SentinelFace has evolved from a prototype into a structured, resilient real-time vision system.

The modular architecture eliminates monolithic bottlenecks, isolates GUI and ML workloads, and provides a stable foundation for future scaling.

## License

MIT License