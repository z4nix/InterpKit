"""GUI op-registry contract tests — fast, no model loading.

The registry is the single source of truth the server dispatches from and
the frontend builds forms from. These tests guard its shape and, crucially,
that it stays in lock-step with the CLI command surface (the project's
surface-parity rule).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from interpkit.gui.ops import CATEGORIES, OP_REGISTRY, catalog
from interpkit.gui.ops.base import fields_from_model

_CATEGORY_IDS = {cid for cid, _ in CATEGORIES}

# CLI commands that intentionally have no GUI op panel.
_CLI_ONLY = {"gui"}


def _cli_command_names() -> set[str]:
    """Every registered Typer command name (as the user types it)."""
    from interpkit.cli.main import app

    names = set()
    for info in app.registered_commands:
        # Typer uses the explicit name if given, else the function name with
        # underscores normalised to hyphens.
        name = info.name or info.callback.__name__.replace("_", "-")
        names.add(name)
    return names


def test_every_op_has_valid_category():
    for name, spec in OP_REGISTRY.items():
        assert spec.name == name
        assert spec.category in _CATEGORY_IDS, f"{name} has unknown category {spec.category!r}"


def test_every_op_has_title_and_description():
    for name, spec in OP_REGISTRY.items():
        assert spec.title, f"{name} missing title"
        assert spec.description and spec.description.rstrip().endswith((".", "?", "!")), (
            f"{name} description should be a sentence"
        )


def test_fields_are_well_formed():
    valid_types = {"string", "integer", "number", "boolean", "enum"}
    for name, spec in OP_REGISTRY.items():
        fields = fields_from_model(spec.params)
        for field in fields:
            assert field["type"] in valid_types, f"{name}.{field['name']} has bad type {field['type']}"
            if field["type"] == "enum":
                assert field["choices"], f"{name}.{field['name']} enum without choices"
            assert isinstance(field["ui"], dict)


def test_catalog_is_json_serializable():
    payload = catalog()
    # Must round-trip through strict JSON (the wire format to the browser).
    dumped = json.dumps(payload)
    assert json.loads(dumped) == payload
    assert {c["id"] for c in payload["categories"]} == _CATEGORY_IDS
    assert len(payload["ops"]) == len(OP_REGISTRY)


def test_cli_command_parity():
    """Every CLI command (except gui itself) must have a GUI op."""
    cli = _cli_command_names() - _CLI_ONLY
    gui = set(OP_REGISTRY)
    missing = cli - gui
    assert not missing, f"CLI commands without a GUI op panel: {sorted(missing)}"


def test_no_orphan_gui_ops():
    """Every GUI op should map back to a real CLI command (no typos)."""
    cli = _cli_command_names()
    orphans = set(OP_REGISTRY) - cli
    assert not orphans, f"GUI ops with no matching CLI command: {sorted(orphans)}"


def test_support_keys_are_known():
    """Each op's support_key must exist in the library support matrix."""
    from interpkit.core.support_matrix import SUPPORT_MATRIX

    for name, spec in OP_REGISTRY.items():
        if spec.support_key is not None:
            assert spec.support_key in SUPPORT_MATRIX, f"{name}: unknown support_key {spec.support_key!r}"
