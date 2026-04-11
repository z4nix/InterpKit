"""Centralised color palette and design tokens for InterpKit CLI output."""

from __future__ import annotations

# Brand gradient used for GradientRule / GradientText headings.
BRAND_COLORS: list[str] = ["#ebf4f5", "#a3b5d1"]

# Primary accent — module names, key column highlights, callouts, panel borders.
ACCENT: str = "#a3b5d1"
ACCENT_DIM: str = "dim #a3b5d1"

# Semantic role pills (label text, Rich style).
ROLE_PILL: dict[str, tuple[str, str]] = {
    "attention": ("attn", "bold white on #6b7d9e"),
    "mlp": ("mlp", "bold white on purple4"),
    "head": ("head", "bold white on dark_blue"),
    "norm": ("norm", "bold black on yellow"),
    "embed": ("embed", "bold white on dark_green"),
}

# Scan finding section tags.
SECTION_STYLE: dict[str, str] = {
    "prediction": "bold white on grey23",
    "dla": "bold white on purple4",
    "lens": "bold white on dark_blue",
    "attention": "bold white on #6b7d9e",
    "attribution": "bold black on yellow",
}

# Feedback colours.
POSITIVE: str = "green"
NEGATIVE: str = "red"
WARNING: str = "yellow"
