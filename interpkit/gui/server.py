"""FastAPI app for the interpkit GUI.

Endpoints never touch torch directly: they enqueue work on a session's
single-worker executor and read job state, so the event loop stays
responsive while models load and ops run. The frontend polls
``GET /api/jobs/{id}`` until the job settles.
"""

from __future__ import annotations

import logging
import threading
import webbrowser
from importlib import resources
from importlib.metadata import version as _pkg_version
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from interpkit.core.exceptions import InterpkitError
from interpkit.gui.jobs import Job, JobManager, to_jsonable
from interpkit.gui.ops import OP_REGISTRY, JobCancelled, JobContext, OpSpec, catalog
from interpkit.gui.schemas import CreateSessionRequest, serialize_arch, serialize_support
from interpkit.gui.sessions import ModelRegistry, Session

logger = logging.getLogger(__name__)

# The CLI boundary (cli/main.py run()) treats these as user-facing
# validation failures, not bugs; the GUI mirrors that contract.
_USER_ERRORS = (InterpkitError, ValueError, KeyError, IndexError)


def _error_payload(exc: BaseException) -> tuple[str, str]:
    # KeyError.__str__ wraps the message in quotes; pull args[0].
    msg = exc.args[0] if (isinstance(exc, KeyError) and exc.args) else str(exc)
    return type(exc).__name__, str(msg)


def _submit_op(session: Session, jobs: JobManager, spec: OpSpec, params: BaseModel) -> Job:
    """Queue one op run on the session's serial executor."""
    job = jobs.create(
        session_id=session.id,
        op=spec.name,
        params=to_jsonable(params.model_dump(exclude_defaults=True)),
    )
    ctx = JobContext(job, model_id=session.model_id)

    def _work() -> None:
        if job.cancel_event.is_set():
            job.mark_cancelled()
            return
        job.mark_running()
        try:
            result = spec.run(session.model, params, ctx)
            job.mark_done(to_jsonable(result))
        except JobCancelled:
            job.mark_cancelled()
        except _USER_ERRORS as exc:
            job.mark_error(*_error_payload(exc))
        except Exception as exc:  # internal bug — log the traceback server-side
            logger.exception("GUI op %r failed", spec.name)
            job.mark_error(*_error_payload(exc))

    session.executor.submit(_work)
    return job


def _submit_load(session: Session, jobs: JobManager) -> Job:
    """Queue the model load as the session's first job."""
    job = jobs.create(session_id=session.id, op="load", params={"model_id": session.model_id})

    def _work() -> None:
        if job.cancel_event.is_set():
            job.mark_cancelled()
            session.status = "error"
            session.error = "load cancelled"
            return
        job.mark_running()
        try:
            from interpkit.core.model import load

            # Match the CLI's load contract: never forward dtype=None /
            # device_map=None — load() owns those defaults.
            kwargs: dict[str, Any] = {"device": session.load_kwargs.get("device")}
            if session.load_kwargs.get("dtype") is not None:
                kwargs["dtype"] = session.load_kwargs["dtype"]
            if session.load_kwargs.get("device_map") is not None:
                kwargs["device_map"] = session.load_kwargs["device_map"]
            session.model = load(session.model_id, **kwargs)
            session.status = "ready"
            job.mark_done(_session_detail(session))
        except Exception as exc:
            session.status = "error"
            session.error = str(exc)
            if isinstance(exc, _USER_ERRORS):
                job.mark_error(*_error_payload(exc))
            else:
                logger.exception("GUI model load failed for %r", session.model_id)
                job.mark_error(*_error_payload(exc))

    session.executor.submit(_work)
    return job


def _session_detail(session: Session) -> dict[str, Any]:
    detail = session.snapshot()
    if session.status == "ready" and session.model is not None:
        detail["arch"] = serialize_arch(session.model)
        detail["support"] = serialize_support(session.model, OP_REGISTRY)
    return detail


def create_app(
    registry: ModelRegistry | None = None,
    jobs: JobManager | None = None,
) -> FastAPI:
    """App factory. ``registry``/``jobs`` are injectable for tests."""
    registry = registry if registry is not None else ModelRegistry()
    jobs = jobs if jobs is not None else JobManager()

    app = FastAPI(title="interpkit GUI", version=_pkg_version("interpkit"), docs_url=None, redoc_url=None)
    app.state.registry = registry
    app.state.jobs = jobs

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        import torch

        has_cuda = torch.cuda.is_available()
        has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
        return {
            "version": _pkg_version("interpkit"),
            "devices": {"cpu": True, "cuda": has_cuda, "mps": has_mps},
            "default_device": "cuda" if has_cuda else ("mps" if has_mps else "cpu"),
        }

    @app.get("/api/ops")
    async def ops_catalog() -> dict[str, Any]:
        return catalog()

    @app.post("/api/sessions")
    async def create_session(req: CreateSessionRequest) -> dict[str, Any]:
        session, created = registry.create(
            req.model_id, device=req.device, dtype=req.dtype, device_map=req.device_map
        )
        # A freshly created session enqueues its load as the first job;
        # an existing (deduped) session is returned as-is with no new job.
        job_id = _submit_load(session, jobs).id if created else None
        return {"session": _session_detail(session), "job_id": job_id}

    @app.get("/api/sessions")
    async def list_sessions() -> dict[str, Any]:
        return {"sessions": [s.snapshot() for s in registry.list()]}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        session = registry.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")
        return _session_detail(session)

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        if not registry.remove(session_id):
            raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/ops/{op_name}")
    async def run_op(session_id: str, op_name: str, request: Request) -> dict[str, Any]:
        session = registry.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")
        spec = OP_REGISTRY.get(op_name)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"unknown op {op_name!r}")
        if session.status != "ready":
            raise HTTPException(status_code=409, detail=f"session is {session.status}, not ready")

        body = await request.json()
        try:
            params = spec.params.model_validate(body or {})
        except ValidationError as exc:
            return JSONResponse(status_code=422, content={"detail": exc.errors(include_url=False)})

        job = _submit_op(session, jobs, spec, params)
        return {"job_id": job.id}

    @app.get("/api/jobs")
    async def list_jobs(session: str | None = None) -> dict[str, Any]:
        return {"jobs": [j.snapshot(include_result=False) for j in jobs.list(session)]}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
        return job.snapshot()

    @app.delete("/api/jobs/{job_id}")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        job = jobs.request_cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
        return job.snapshot(include_result=False)

    # ------------------------------------------------------------------
    # Frontend (mounted last so /api/* wins)
    # ------------------------------------------------------------------

    static_dir = resources.files("interpkit.gui") / "static"
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


def serve(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    """Run the GUI server (blocking) and optionally open the browser."""
    import uvicorn
    from rich.console import Console

    app = create_app()
    url = f"http://{host}:{port}"
    Console().print(
        f"\n  [bold green]interpkit GUI[/bold green] running at [bold]{url}[/bold]  (Ctrl+C to stop)\n"
    )
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
