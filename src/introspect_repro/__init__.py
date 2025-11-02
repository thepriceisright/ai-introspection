"""Package initialisation for introspect_repro.

Ensures environment variables defined in the repository's .env file are loaded
before any submodules attempt to read them.
"""

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency should exist
    load_dotenv = None  # type: ignore[assignment]

_ENV_INITIALISED = False


def load_project_env(dotenv_path: str | Path | None = None) -> None:
    """Load variables from the repository's .env file once per process."""
    global _ENV_INITIALISED

    if _ENV_INITIALISED or load_dotenv is None:
        return

    candidate = Path(dotenv_path) if dotenv_path else Path(__file__).resolve().parents[1] / ".env"
    if candidate.exists():
        load_dotenv(candidate, override=False)
    _ENV_INITIALISED = True


load_project_env()

__all__ = ["load_project_env"]
