SentinelFace
Real-Time Face Recognition System (Modular Architecture)
Overview

SentinelFace is a robust, real-time face recognition system built with a modular, production-oriented architecture.

Originally developed as a monolithic prototype (FaceDet.py), the system has been fully refactored into a scalable package structure (SentinelFace/), designed for maintainability, concurrency safety, and long-term evolution.

The application uses:

PySide6 for the GUI

ONNX Runtime for GPU-accelerated inference

SCRFD for detection

ArcFace for embeddings

FairFace for age & gender classification

The UI and ML pipeline are cleanly decoupled to ensure responsiveness under real-time load.

Architecture Overview
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

Core Components
1. Configuration (config.py)

Implements a layered configuration system:

Typed Defaults (@dataclass)

JSON Overrides

CLI Overrides

Final configuration is validated to prevent invalid runtime thresholds.

This allows deployment tuning without code modification.

2. Shared System State (state.py)

A SystemState dataclass acts as a central telemetry source:

FPS

Inference time

Memory availability

Active track count

The worker thread updates this state, while the GUI reads it safely via timers.

This eliminates global state sprawl and cross-thread flag issues.

3. Concurrency Model
VideoWorker (QThread)

Responsible for:

Frame ingestion (OpenCV)

Health monitoring

Detection → Alignment → Re-ID → Demographics → Rendering

Emitting lightweight Qt signals

All ML operations run inside this worker thread.

MainWindow

Consumes worker signals

Displays frames

Polls telemetry

Remains fully responsive

The UI never performs heavy computation.

4. Model Adapters (models/)

Each ONNX model is wrapped in a dedicated adapter:

SCRFD – Face detection

ArcFace – 512D embedding extraction

FairFace – Age & gender classification

Adapters validate input/output shapes and isolate inference logic from the rest of the system.

5. Tracking & Re-Identification (core/tracker.py)

The tracking system includes:

IoU-based association

Kalman motion prediction

Cosine similarity Re-ID

Grace-based track eviction (temporal smoothing)

Defensive bounding box sanity checks

A granular RLock protects the embedding similarity matrix, allowing safe future concurrency without locking the entire tracker.

6. Face Database (core/database.py)

Handles:

Enrollment loading

Embedding storage

Fast vectorized similarity search

Designed to remain thread-safe and injection-driven (no globals).

7. Health Monitoring (utils/health.py)

Monitors:

FPS drops

Inference spikes

Low memory conditions

Each alert type has independent cooldown timers to prevent log flooding.

Memory monitoring uses absolute available MB to avoid false alarms on mid-range systems.

Resilience Features

Camera disconnect auto-reconnect with backoff

Frozen frame detection

Graceful tracker cleanup

Rotating logs (5MB × 5 backups)

GPU fallback safety

The system is designed for long-running real-time deployment.

Testing

Minimal regression scaffolding exists:

test_config.py – Verifies layered config resolution

test_tracker_logic.py – Verifies grace-kill & locking behavior

These tests protect against architectural regressions during refactoring.

Design Principles

Explicit dependency injection

No hidden globals

Strict layering

Defensive programming

Concurrency boundaries clearly defined

Extensibility without premature complexity

Current Status

Architecture Level: Engineered System
Deployment Readiness: Production-ready (single-node)
Scalability Path: Prepared for multi-camera & distributed evolution

Future Directions

Multi-camera orchestration

Centralized embedding service

Event-driven architecture

Deployment packaging

Centralized logging aggregation

Conclusion

SentinelFace has evolved from a prototype into a structured, resilient real-time vision system.

The modular architecture eliminates monolithic bottlenecks, isolates GUI and ML workloads, and provides a stable foundation for future scaling.