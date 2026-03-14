"""inspect — module tree with types, param counts, output shapes, and detected roles."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interpkit.core.model import Model


def run_inspect(model: "Model") -> None:
    from interpkit.core.render import render_inspect

    render_inspect(model.arch_info)
