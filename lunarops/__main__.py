"""Entry point for ``python -m lunarops``; runs :func:`lunarops.cli.main`."""

from lunarops.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
