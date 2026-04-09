import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("FaceSystem.Database")

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

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        emb = np.ascontiguousarray(embedding[np.newaxis], dtype=np.float32)
        with self._lock:
            self.identities[name] = (np.vstack([self.identities[name], emb])
                                     if name in self.identities else emb)
            self._rebuild_global()
            np.save(self.enroll_dir / f"{name}.npy", self.identities[name])
        logger.info("Enrolled '%s' (%d samples)", name,
                    len(self.identities[name]))

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
