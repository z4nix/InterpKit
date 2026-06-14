"""Overview ops: scan, inspect, report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from interpkit.gui.ops.base import JobContext, OpSpec, ui
from interpkit.gui.schemas import serialize_inspect


class ScanParams(BaseModel):
    text: str = Field(
        ...,
        description="Input text to analyse",
        json_schema_extra=ui(widget="textarea", rows=2, placeholder="The capital of France is"),
    )


def _run_scan(model: Any, p: ScanParams, ctx: JobContext) -> Any:
    return model.scan(p.text)


class InspectParams(BaseModel):
    pass


def _run_inspect(model: Any, p: InspectParams, ctx: JobContext) -> Any:
    return serialize_inspect(model)


class ReportParams(BaseModel):
    text: str = Field(
        ...,
        description="Input text to analyse",
        json_schema_extra=ui(widget="textarea", rows=2, placeholder="The capital of France is"),
    )
    save: str = Field(
        "report.html",
        description="Where to save the standalone HTML report (server-side path)",
        json_schema_extra=ui(widget="path", advanced=True),
    )


def _run_report(model: Any, p: ReportParams, ctx: JobContext) -> Any:
    result = model.report(p.text, save=p.save)
    out = dict(result)
    html_path = Path(out.get("html_path", p.save))
    # Inline the report so the GUI can preview / download it without a
    # file-serving endpoint. Reports are self-contained single files.
    try:
        out["report_html"] = html_path.read_text(encoding="utf-8")
        out["html_path"] = str(html_path.resolve())
    except OSError:
        pass
    return out


SPECS: list[OpSpec] = [
    OpSpec(
        name="scan",
        category="overview",
        title="Scan",
        description="One-click overview: runs DLA, logit lens, attention, and attribution, then surfaces the key findings.",
        params=ScanParams,
        run=_run_scan,
        long_running=True,
        support_key="scan",
    ),
    OpSpec(
        name="inspect",
        category="overview",
        title="Inspect",
        description="The model's detected architecture: family, block stack, special paths, and the full module tree.",
        params=InspectParams,
        run=_run_inspect,
        support_key="inspect",
    ),
    OpSpec(
        name="report",
        category="overview",
        title="Report",
        description="Generate a comprehensive standalone HTML report — prediction, DLA, logit lens, attention, attribution.",
        params=ReportParams,
        run=_run_report,
        long_running=True,
        support_key="report",
    ),
]
