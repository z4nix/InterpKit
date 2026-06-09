"""Entry point so ``python -m interpkit`` invokes the Typer CLI.

Mirrors the ``[project.scripts] interpkit = "interpkit.cli.main:run"``
console script declared in :file:`pyproject.toml`, so users without the
console script on their ``$PATH`` (e.g. just-installed in a fresh
environment, vendored copies, ad-hoc subprocess invocations) can still
reach every CLI command via ``python -m interpkit ...``.
"""

from interpkit.cli.main import run


def main() -> None:
    """Invoke the CLI — separate function makes patching easier in tests.

    Uses ``run`` (not ``app`` directly) so interpkit's fail-loud errors are
    rendered as clean one-line messages instead of tracebacks.
    """
    run()


if __name__ == "__main__":
    main()
