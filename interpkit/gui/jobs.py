"""In-memory job bookkeeping for GUI operations.

Every GUI action that touches a model (loading it, running an op) is a
``Job`` executed on its session's single-worker executor. Endpoints only
create and read jobs, so the asyncio event loop never blocks on torch.

Jobs are kept in a bounded in-process store — the GUI is a local,
single-user tool, so there is no persistence layer.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

_MAX_JOBS = 200


def to_jsonable(obj: Any) -> Any:
    """Coerce an op result into JSON-safe data.

    Tensors / numpy arrays become nested lists, numpy scalars become
    Python numbers, and non-finite floats become ``None`` (strict JSON —
    and therefore ``JSON.parse`` in the browser — rejects NaN/Infinity,
    which trace metrics legitimately produce).
    """
    import numpy as np
    import torch

    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if hasattr(obj, "__float__"):
        return to_jsonable(float(obj))
    return str(obj)


@dataclass
class Job:
    """One unit of model work (a load or an op run) and its lifecycle."""

    id: str
    session_id: str
    op: str
    params: dict[str, Any]
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: dict[str, Any] | None = None
    result: Any = None
    error: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def mark_running(self) -> None:
        self.status = "running"
        self.started_at = time.time()

    def mark_done(self, result: Any) -> None:
        self.result = result
        self.status = "done"
        self.finished_at = time.time()

    def mark_error(self, error_type: str, message: str) -> None:
        self.error = {"type": error_type, "message": message}
        self.status = "error"
        self.finished_at = time.time()

    def mark_cancelled(self) -> None:
        self.status = "cancelled"
        self.finished_at = time.time()

    def snapshot(self, *, include_result: bool = True) -> dict[str, Any]:
        """JSON view of the job for the polling endpoints."""
        snap: dict[str, Any] = {
            "id": self.id,
            "session_id": self.session_id,
            "op": self.op,
            "params": self.params,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if include_result:
            snap["result"] = self.result
        return snap


class JobManager:
    """Thread-safe bounded store of jobs, newest last."""

    def __init__(self, max_jobs: int = _MAX_JOBS) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def create(self, *, session_id: str, op: str, params: dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], session_id=session_id, op=op, params=params)
        with self._lock:
            self._jobs[job.id] = job
            # Evict oldest *finished* jobs beyond the cap; never evict
            # queued/running jobs whose futures still reference them.
            while len(self._jobs) > self._max_jobs:
                evicted = False
                for jid, j in self._jobs.items():
                    if j.status in ("done", "error", "cancelled"):
                        del self._jobs[jid]
                        evicted = True
                        break
                if not evicted:
                    break
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, session_id: str | None = None) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if session_id is not None:
            jobs = [j for j in jobs if j.session_id == session_id]
        return jobs

    def request_cancel(self, job_id: str) -> Job | None:
        """Cooperative cancel: queued jobs die before starting; running jobs
        stop at their next progress checkpoint (long-running ops only)."""
        job = self.get(job_id)
        if job is not None and job.status in ("queued", "running"):
            job.cancel_event.set()
        return job
