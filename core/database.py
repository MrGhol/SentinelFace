import logging
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("FaceSystem.Database")

_RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

def _sanitize_identity_name(raw_name: str) -> str:
    if raw_name is None:
        raise ValueError("Enrollment name is required.")
    name = str(raw_name).strip()
    if not name:
        raise ValueError("Enrollment name cannot be empty.")

    # Normalize whitespace, then keep only safe filename characters.
    name = re.sub(r"\s+", "_", name)
    name = _SAFE_NAME_RE.sub("_", name)
    name = re.sub(r"_+", "_", name).strip(" ._")

    if not name or name in (".", ".."):
        raise ValueError("Enrollment name is invalid after sanitization.")

    base = name.split(".")[0]
    if base.upper() in _RESERVED_WINDOWS_NAMES:
        raise ValueError(f"Enrollment name '{raw_name}' is reserved on Windows.")

    return name

class FaceDatabase:
    """Thread-safe enrolled identity store backed by .npy files."""

    def __init__(self, enroll_dir: Path):
        self.enroll_dir   = enroll_dir; self.enroll_dir.mkdir(parents=True, exist_ok=True)
        self._lock        = threading.RLock()
        self.identities:  Dict[str, np.ndarray] = {}
        self.name_indices: Dict[str, List[int]] = {}
        self.global_embs: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self._load_all()

    def _rebuild_global(self) -> None:
        if not self.identities:
            self.global_embs = np.empty((0, 512), dtype=np.float32)
            self.name_indices = {}; return
        mats, idx = [], 0; self.name_indices = {}
        for name in sorted(self.identities):
            data = self.identities[name]
            mats.append(data)
            self.name_indices[name] = list(range(idx, idx + len(data)))
            idx += len(data)
        self.global_embs = np.vstack(mats)

    def _load_all(self) -> None:
        t0 = time.time()
        for npy in sorted(self.enroll_dir.glob("*.npy")):
            try:
                data = np.load(npy).astype(np.float32)
                if data.ndim == 1: data = data[np.newaxis]
                if data.ndim != 2 or data.shape[1] != 512:
                    logger.warning("Skipping %s: shape %s", npy, data.shape); continue
                self.identities[npy.stem] = data
                logger.info("  Loaded '%-20s' (%d sample(s))", npy.stem, len(data))
            except Exception as exc:
                logger.warning("Cannot load %s: %s", npy, exc)
        legacy = Path("my_face.npy")
        if legacy.exists() and "me" not in self.identities:
            try:
                data = np.load(legacy).astype(np.float32)
                if data.ndim == 1: data = data[np.newaxis]
                if data.shape[1] == 512:
                    self.identities["me"] = data
                    logger.info("  Loaded 'me' from legacy my_face.npy")
            except Exception as exc:
                logger.warning("Cannot load my_face.npy: %s", exc)
        self._rebuild_global()
        logger.info("DB ready: %d identity/ies, %d embeddings (%.2fs)",
                    len(self.identities), self.global_embs.shape[0], time.time() - t0)

    def enroll(self, name: str, embedding: np.ndarray) -> str:
        safe_name = _sanitize_identity_name(name)

        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb[0]
        if emb.ndim != 1 or emb.shape[0] != 512:
            raise ValueError("Embedding must be a 512-dimensional vector.")
        emb = np.ascontiguousarray(emb[np.newaxis], dtype=np.float32)

        # Ensure final path stays within enrollment directory.
        base_dir = self.enroll_dir.resolve()
        out_path = (self.enroll_dir / f"{safe_name}.npy").resolve()
        if base_dir not in out_path.parents and out_path.parent != base_dir:
            raise ValueError("Unsafe enrollment name (path traversal detected).")

        with self._lock:
            self.identities[safe_name] = (np.vstack([self.identities[safe_name], emb])
                                          if safe_name in self.identities else emb)
            self._rebuild_global()
            np.save(out_path, self.identities[safe_name])
        logger.info("Enrolled '%s' (%d samples)", safe_name,
                    len(self.identities[safe_name]))
        return safe_name

    def match(self, query: np.ndarray, threshold: float) -> Tuple[str, float]:
        with self._lock:
            if self.global_embs.shape[0] == 0: return "UNKNOWN", 0.0
            gembs = self.global_embs; nidx = dict(self.name_indices)
        q = query / (np.linalg.norm(query) + 1e-9)
        sims = gembs @ q
        best_sim, best_name = 0.0, "UNKNOWN"
        for name, indices in nidx.items():
            if not indices: continue
            s = float(np.max(sims[indices]))
            if s > best_sim: best_sim, best_name = s, name
        return (best_name, best_sim) if best_sim >= threshold else ("UNKNOWN", best_sim)
