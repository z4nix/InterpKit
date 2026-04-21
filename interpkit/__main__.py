"""Entry point so ``python -m interpkit`` invokes the Typer CLI.

Mirrors the ``[project.scripts] interpkit = "interpkit.cli.main:app"``
console script declared in :file:`pyproject.toml`, so users without the
console script on their ``$PATH`` (e.g. just-installed in a fresh
environment, vendored copies, ad-hoc subprocess invocations) can still
reach every CLI command via ``python -m interpkit ...``.
"""

from interpkit.cli.main import app


def main() -> None:
    """Invoke the Typer app — separate function makes patching easier in tests."""
    app()


if __name__ == "__main__":
    main()
