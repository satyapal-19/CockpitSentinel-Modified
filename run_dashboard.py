"""CockpitSentinel Telematics Web Dashboard — Resilient Production Runner.

Ensures reliable, continuous execution:
- Auto-detects and uses virtual environment (.venv)
- Safe against Windows terminal encoding (cp1252/cp437) UnicodeEncodeError
- Resolves models directory (D:\\CockpitSentinel\\models)
- Checks and frees port 8000 if occupied by stale processes
- Auto-opens web browser to dashboard
- Keep-alive supervisor prevents abrupt window closure
"""

from __future__ import annotations

import argparse
import contextlib
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Safe stream configuration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_venv(repo_root: Path) -> None:
    """Respawn under virtual environment if invoked by an external Python."""
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    target_python = venv_python.resolve()

    if current_python != target_python:
        print(f"[*] Switching to project virtual environment: {target_python}")
        cmd = [str(target_python), *sys.argv]
        sys.exit(subprocess.call(cmd))


def _ensure_models_env(repo_root: Path) -> None:
    """Ensure COCKPIT_MODELS_ROOT is set and valid."""
    if not os.environ.get("COCKPIT_MODELS_ROOT"):
        candidates = [
            Path(r"D:\CockpitSentinel\models"),
            Path(r"C:\CockpitSentinel\models"),
            repo_root / "models",
        ]
        for c in candidates:
            if c.exists():
                os.environ["COCKPIT_MODELS_ROOT"] = str(c)
                break


def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _free_port_windows(port: int) -> None:
    """Terminate lingering processes holding the target port on Windows."""
    try:
        cmd = f"netstat -ano | findstr :{port} | findstr LISTENING"
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        pids: set[int] = set()
        for line in out.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and parts[-1].isdigit():
                pid = int(parts[-1])
                if pid > 0 and pid != os.getpid():
                    pids.add(pid)

        for pid in pids:
            print(f"[!] Reclaiming port {port}: terminating stale process (PID {pid})...")
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
    except Exception:
        pass


def _launch_browser(url: str, delay: float = 1.2) -> None:
    def _worker():
        time.sleep(delay)
        with contextlib.suppress(Exception):
            webbrowser.open(url)

    threading.Thread(target=_worker, daemon=True).start()


def main() -> None:
    repo_root = _find_repo_root()
    _ensure_venv(repo_root)
    _ensure_models_env(repo_root)

    # Add src to sys.path
    src_dir = str(repo_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import uvicorn

    from cockpit_sentinel.dashboard.app import create_app

    parser = argparse.ArgumentParser(
        description="CockpitSentinel Resilient Dashboard Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host IP")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--source", default="0", help="Webcam index or video path")
    parser.add_argument("--device", default="auto", help="Inference device: 'auto', 'cpu', 'cuda'")
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open browser automatically"
    )
    parser.add_argument("--no-audio", action="store_true", help="Disable acoustic alerts")
    parser.add_argument("--no-camera", action="store_true", help="Disable camera monitoring")
    args = parser.parse_args()

    # Free port if occupied by stale previous run
    if _is_port_in_use(args.port, args.host):
        _free_port_windows(args.port)
        time.sleep(0.5)

    models_dir = os.environ.get("COCKPIT_MODELS_ROOT", "default/auto")
    url = f"http://{args.host}:{args.port}"
    src_kind = "DirectShow Webcam" if str(args.source).isdigit() else "File"

    print("\n" + "=" * 74)
    print("  COCKPITSENTINEL TELEMATICS DASHBOARD")
    print("=" * 74)
    print(f"[*] Dashboard URL : {url}")
    print(f"[*] Camera Source : {args.source} ({src_kind})")
    print(f"[*] Models Root   : {models_dir}")
    print(f"[*] Compute Device: {args.device}")
    print("[*] Active Detect : Fatigue, PERCLOS, Micro-sleep, Cell Phone, Smoking")
    print("[*] Status        : Keep-alive active (Press Ctrl+C to terminate)")
    print("=" * 74 + "\n")

    if not args.no_browser:
        _launch_browser(url, delay=1.5)

    while True:
        try:
            app = create_app(
                start_camera=not args.no_camera,
                source=args.source,
                device=args.device,
                no_audio=args.no_audio,
            )
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                log_level="info",
                access_log=False,
            )
            print("\n[*] Dashboard shutdown cleanly.")
            break
        except KeyboardInterrupt:
            print("\n[*] Dashboard terminated by user (Ctrl+C). Exiting.")
            break
        except Exception as exc:
            print(f"\n[!] Dashboard server encountered an unexpected error: {exc}")
            import traceback

            traceback.print_exc()
            print("\n[*] Restarting server in 3 seconds (Press Ctrl+C to cancel)...")
            try:
                time.sleep(3.0)
            except KeyboardInterrupt:
                print("\n[*] Exit requested.")
                break


if __name__ == "__main__":
    main()
