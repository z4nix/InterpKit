"""GUI API end-to-end tests via FastAPI TestClient.

A preloaded gpt2 session is injected through ``ModelRegistry.add_preloaded``
so these tests exercise the full session → job → op → poll path without the
slow HF download in the request flow.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from interpkit.gui.jobs import JobManager
from interpkit.gui.server import create_app
from interpkit.gui.sessions import ModelRegistry


@pytest.fixture()
def gui(gpt2_model):
    """TestClient with a preloaded gpt2 session. Returns (client, session_id)."""
    registry = ModelRegistry()
    jobs = JobManager()
    app = create_app(registry=registry, jobs=jobs)
    client = TestClient(app)
    session = registry.add_preloaded(gpt2_model, model_id="gpt2")
    return client, session.id


def _wait(client, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


# ---------------------------------------------------------------------------
# Static / catalog / health
# ---------------------------------------------------------------------------


def test_health(gui):
    client, _ = gui
    body = client.get("/api/health").json()
    assert "version" in body
    assert body["devices"]["cpu"] is True
    assert body["default_device"] in ("cpu", "cuda", "mps")


def test_catalog_shape(gui):
    client, _ = gui
    body = client.get("/api/ops").json()
    names = {op["name"] for op in body["ops"]}
    assert {"scan", "lens", "dla", "attention"} <= names
    for op in body["ops"]:
        assert "fields" in op and isinstance(op["fields"], list)


def test_index_and_assets_served(gui):
    client, _ = gui
    assert client.get("/").status_code == 200
    assert client.get("/js/main.js").status_code == 200
    assert client.get("/css/app.css").status_code == 200


# ---------------------------------------------------------------------------
# Session lifecycle + arch serialization
# ---------------------------------------------------------------------------


def test_session_detail_arch_is_json_safe(gui):
    client, sid = gui
    detail = client.get(f"/api/sessions/{sid}").json()
    assert detail["status"] == "ready"
    arch = detail["arch"]
    # whitelisted, JSON-safe fields only — no leaked nn.Module reprs
    assert arch["family"] == "causal_lm"
    assert arch["num_layers"] == 12
    assert arch["num_attention_heads"] == 12
    assert isinstance(arch["paths"]["blocks"], list) and arch["paths"]["blocks"]
    # support map covers every op
    assert set(detail["support"]) == {op["name"] for op in client.get("/api/ops").json()["ops"]}
    # arch must round-trip through strict JSON
    import json

    json.dumps(detail)


def test_unknown_session_404(gui):
    client, _ = gui
    assert client.get("/api/sessions/nope").status_code == 404


# ---------------------------------------------------------------------------
# Ops: end-to-end via job polling
# ---------------------------------------------------------------------------


def test_run_lens(gui):
    client, sid = gui
    r = client.post(f"/api/sessions/{sid}/ops/lens", json={"text": "The capital of France is"})
    assert r.status_code == 200
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert len(job["result"]["results"]) == 12
    assert job["result"]["tokens"][0] == "The"


def test_run_dla(gui):
    client, sid = gui
    r = client.post(f"/api/sessions/{sid}/ops/dla", json={"text": "The capital of France is", "top_k": 5})
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert "contributions" in job["result"]
    assert job["result"]["target_token"]


def test_run_scan(gui):
    client, sid = gui
    r = client.post(f"/api/sessions/{sid}/ops/scan", json={"text": "The capital of France is"})
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert "prediction" in job["result"]


def test_bad_params_422(gui):
    client, sid = gui
    # lens requires `text`
    r = client.post(f"/api/sessions/{sid}/ops/lens", json={})
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], list)


def test_unknown_op_404(gui):
    client, sid = gui
    r = client.post(f"/api/sessions/{sid}/ops/nosuchop", json={})
    assert r.status_code == 404


def test_typed_error_surfaced(gui):
    client, sid = gui
    # Unknown module path → KeyError with a "did you mean" hint, surfaced
    # as a settled error job (not an HTTP 500).
    r = client.post(f"/api/sessions/{sid}/ops/ablate", json={"text": "hi", "at": "no.such.module"})
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "error"
    assert job["error"]["type"] == "KeyError"
    assert "not found" in job["error"]["message"].lower()


def test_jobs_history_listed(gui):
    client, sid = gui
    r = client.post(f"/api/sessions/{sid}/ops/lens", json={"text": "hello world"})
    _wait(client, r.json()["job_id"])
    jobs = client.get(f"/api/jobs?session={sid}").json()["jobs"]
    assert any(j["op"] == "lens" for j in jobs)


def test_unsupported_op_greyed_in_support_map(distilbert_model):
    """An encoder-only model marks generation ops unsupported."""
    registry = ModelRegistry()
    app = create_app(registry=registry, jobs=JobManager())
    client = TestClient(app)
    session = registry.add_preloaded(distilbert_model, model_id="distilbert-base-uncased")
    detail = client.get(f"/api/sessions/{session.id}").json()
    # DistilBERT cannot generate → chat/generate unsupported.
    assert detail["support"]["chat"]["supported"] is False
    assert detail["support"]["chat"]["reason"]
