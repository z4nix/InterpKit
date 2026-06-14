"""Model sessions: loaded models, keyed and serialized per session.

Each ``Session`` owns one loaded :class:`~interpkit.core.model.Model` and
a single-worker executor. Submitting every job for a session to its own
``max_workers=1`` pool is the GPU-safety mechanism: two ops can never run
concurrently on the same model (forward hooks and the activation cache
are not reentrant), while ops on *different* models may overlap freely.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass
class Session:
    """One loaded model and its private work queue."""

    id: str
    model_id: str
    load_kwargs: dict[str, Any]
    status: str = "loading"  # loading | ready | error
    error: str | None = None
    model: Any = None  # interpkit.core.model.Model once loaded
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="interpkit-gui")
    )
    created_at: float = field(default_factory=time)

    @property
    def key(self) -> tuple:
        return (
            self.model_id,
            self.load_kwargs.get("device"),
            self.load_kwargs.get("dtype"),
            self.load_kwargs.get("device_map"),
        )

    def snapshot(self) -> dict[str, Any]:
        snap: dict[str, Any] = {
            "id": self.id,
            "model_id": self.model_id,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            **{k: self.load_kwargs.get(k) for k in ("device", "dtype", "device_map")},
        }
        if self.model is not None:
            snap["device"] = self.model.device
            snap["dtype"] = str(self.model.dtype)
        return snap


class ModelRegistry:
    """Thread-safe registry of model sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(
        self,
        model_id: str,
        *,
        device: str | None = None,
        dtype: str | None = None,
        device_map: str | None = None,
    ) -> tuple[Session, bool]:
        """Return ``(session, created)``.

        Reuses an existing non-errored session with the same load key —
        loading the same model twice into local memory is never what a
        GUI user wants.
        """
        key = (model_id, device, dtype, device_map)
        with self._lock:
            for sess in self._sessions.values():
                if sess.key == key and sess.status != "error":
                    return sess, False
            sess = Session(
                id=uuid.uuid4().hex[:12],
                model_id=model_id,
                load_kwargs={"device": device, "dtype": dtype, "device_map": device_map},
            )
            self._sessions[sess.id] = sess
            return sess, True

    def add_preloaded(self, model: Any, *, model_id: str = "preloaded") -> Session:
        """Test seam: register an already-loaded Model as a ready session."""
        sess = Session(
            id=uuid.uuid4().hex[:12],
            model_id=model_id,
            load_kwargs={"device": None, "dtype": None, "device_map": None},
            status="ready",
            model=model,
        )
        with self._lock:
            self._sessions[sess.id] = sess
        return sess

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list(self) -> list[Session]:
        with self._lock:
            return list(self._sessions.values())

    def remove(self, session_id: str) -> bool:
        """Unload a session's model and free device memory."""
        with self._lock:
            sess = self._sessions.pop(session_id, None)
        if sess is None:
            return False
        sess.executor.shutdown(wait=False, cancel_futures=True)
        device = sess.model.device if sess.model is not None else None
        sess.model = None
        sess.status = "unloaded"

        import gc

        gc.collect()
        if device is not None:
            import torch

            mps = getattr(torch.backends, "mps", None)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif device == "mps" and mps is not None and mps.is_available():
                torch.mps.empty_cache()
        return True
