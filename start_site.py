#!/usr/bin/env python3
"""One-command launcher for the cat/dog CBIR comparison website.

Usage:
    python start_site.py            # setup (once) + run on default port
    python start_site.py [port]     # e.g. python start_site.py 8503

Works on macOS / Linux / Windows. Creates a local .venv, installs
requirements.txt, then starts Streamlit locally.
"""
import os
import sys
import hashlib
import platform
import subprocess
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8502


def py_in_venv() -> Path:
    if os.name == "nt":
        return ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def ensure_venv() -> Path:
    py = py_in_venv()
    if not py.exists():
        print("[setup] creating virtual environment (.venv) ...")
        venv.create(ROOT / ".venv", with_pip=True)
    return py


def ensure_deps(py: Path) -> None:
    # check if streamlit/torch already present (fast path)
    probe = subprocess.run([str(py), "-c",
                            "import streamlit, torch, torchvision, sklearn"],
                           capture_output=True)
    if probe.returncode == 0:
        print("[setup] dependencies already installed. (use --force to reinstall)")
        return
    print("[setup] installing dependencies (streamlit, torch, ...) -- this can take a few minutes ...")
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=False)
    subprocess.run([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    print("[setup] dependencies installed.\n")


def ensure_ssl_fix(py: Path) -> None:
    """python.org framework builds on macOS break SSL; point at certifi's bundle."""
    if not sys.platform.startswith("darwin"):
        return
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        cacert = subprocess.run(
            [str(py), "-c", "import certifi, pathlib; print(pathlib.Path(certifi.where()))"],
            capture_output=True, text=True).stdout.strip()
    except Exception:
        return
    if cacert and Path(cacert).exists():
        os.environ["SSL_CERT_FILE"] = cacert
        print(f"[setup] SSL_CERT_FILE={cacert}")


def main() -> None:
    port = DEFAULT_PORT
    force = "--force" in sys.argv
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    py = ensure_venv()
    if force:
        subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "-r",
                        str(ROOT / "requirements.txt")], check=True)
    ensure_deps(py)
    ensure_ssl_fix(py)

    # token to guard a local port name collision with another running instance
    import random
    port_token = os.environ.get("STREAMLIT_SERVER_PORT_TOKEN") or (
        hashlib.sha1(os.urandom(8)).hexdigest()[:12])

    print(f"\n===== 啟動中 / Starting {ROOT.name} on http://localhost:{port} =====\n")
    cmd = [str(py), "-m", "streamlit", "run", str(ROOT / "app.py"),
           "--server.headless", "true",
           "--server.port", str(port),
           "--server.fileWatcherType", "none",
           "--browser.gatherUsageStats", "false"]
    subprocess.run(cmd)
    sys.exit(0)


if __name__ == "__main__":
    main()