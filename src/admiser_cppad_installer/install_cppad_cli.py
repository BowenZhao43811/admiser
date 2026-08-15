from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Console entry point for `admiser-install-cppad`."""

    print("==============================================")
    print("  ADMISER CppAD / cppad_py installation helper")
    print("  (tested on Debian/Ubuntu/WSL only)")
    print("==============================================")
    print()

    # locate the shell script relative to this file
    pkg_dir = Path(__file__).resolve().parent
    script_path = pkg_dir / "build_cppad.sh"

    if not script_path.is_file():
        print(f"error: install script not found: {script_path}", file=sys.stderr)
        raise SystemExit(1)

    bash = os.environ.get("BASH", "bash")

    print(f"running install script: {script_path}")
    print("Notes:")
    print("  - the script installs into the virtual environment that is currently active;")
    print("  - environment variables can be set beforehand, e.g.:")
    print("      VENV_DIR=myenv admiser-install-cppad")
    print()

    env = os.environ.copy()

    try:
        subprocess.run([bash, str(script_path)], check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"installation failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
