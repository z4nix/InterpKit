"""Centralised color palette and design tokens for InterpKit CLI output."""

from __future__ import annotations

# Brand gradient used for GradientRule / GradientText headings.
# Both endpoints chosen to remain visible on both light and dark terminal backgrounds.
BRAND_COLORS: list[str] = ["#7a99cc", "#5f7fb8"]

# Primary accent — module names, key column highlights, callouts, panel borders.
# Mid-tone blue (~4:1 contrast on both white and #1e1e1e backgrounds, AA pass).
ACCENT: str = "#5f7fb8"

# Lighter accent variant for secondary borders (e.g. nested help panels).
# Plain hex — Rich's `dim` modifier on top of an already-pale color reads as invisible.
ACCENT_DIM: str = "#7a99cc"

# Legible-but-secondary text: data columns that previously used `[dim]` and washed out.
MUTED: str = "#8a9bb8"

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
