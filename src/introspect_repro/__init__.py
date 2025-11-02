"""Package initialisation for introspect_repro.

Ensures environment variables defined in the repository's .env file are loaded
before any submodules attempt to read them.
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_ENV_INITIALISED = False
_VENV_ACTIVATED = False
_PROJECT_ROOT: Path | None = None
load_dotenv = None  # lazily imported
dotenv_values = None  # lazily imported


def _find_dotenv(start: Path) -> Path | None:
    """Walk parent directories from `start` looking for a .env file."""
    for parent in (start, *start.parents):
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def load_project_env(dotenv_path: str | Path | None = None) -> None:
    """Load variables from the repository's .env file once per process."""
    global _ENV_INITIALISED

    if _ENV_INITIALISED:
        return

    _ensure_dotenv_import()

    if dotenv_path:
        candidate = Path(dotenv_path)
    else:
        candidate = _find_dotenv(Path(__file__).resolve().parent)

    if candidate and candidate.exists():
        global _PROJECT_ROOT
        _PROJECT_ROOT = candidate.parent
        if load_dotenv is not None:
            load_dotenv(candidate, override=False)
            if dotenv_values is not None:
                for key, value in dotenv_values(candidate).items():
                    if value is None:
                        continue
                    if not os.environ.get(key):
                        os.environ[key] = value
        else:  # Fall back to minimal loader if python-dotenv is absent
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and (key not in os.environ or not os.environ.get(key)):
                    os.environ[key] = value
    _ENV_INITIALISED = True


def activate_local_venv(venv_path: str | Path | None = None) -> None:
    """Ensure the project's `.venv` site-packages are on `sys.path`."""
    global _VENV_ACTIVATED, _PROJECT_ROOT
    if _VENV_ACTIVATED:
        return

    if venv_path:
        venv_dir = Path(venv_path)
    else:
        base = _PROJECT_ROOT or Path(__file__).resolve().parent
        for parent in (base, *base.parents):
            candidate = parent / ".venv"
            if candidate.exists():
                venv_dir = candidate
                break
        else:
            return

    if not venv_dir.exists():
        return

    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_candidates = [
        venv_dir / "lib" / python_version / "site-packages",
        venv_dir / "Lib" / "site-packages",  # Windows
    ]
    for site_dir in site_candidates:
        if site_dir.exists():
            site.addsitedir(site_dir.as_posix())
            os.environ.setdefault("VIRTUAL_ENV", str(venv_dir))
            _VENV_ACTIVATED = True
            break


def _ensure_dotenv_import() -> None:
    global load_dotenv, dotenv_values
    if load_dotenv is not None:
        return
    try:
        from dotenv import dotenv_values as _values  # type: ignore
        from dotenv import load_dotenv as _load  # type: ignore
    except ImportError:  # pragma: no cover - dependency should exist
        load_dotenv = None
        dotenv_values = None
    else:
        load_dotenv = _load
        dotenv_values = _values


activate_local_venv()
load_project_env()

__all__ = ["activate_local_venv", "load_project_env"]
