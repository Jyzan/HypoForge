#!/usr/bin/env python
"""Launch the local HypoForge M1-M6 progress dashboard."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="HypoForge M1-M6 local dashboard")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/full_pipeline_m2agentic_qwen37plus_live.yaml"),
        help="M1-M6 YAML configuration",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "output/ui_runs"),
        help="Directory containing one persisted folder per UI run",
    )
    parser.add_argument(
        "--env-file",
        default="",
        help="Optional env file; QWEN_* names are mapped automatically",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the page automatically",
    )
    args = parser.parse_args()

    from hypoforge.webapp import build_server, load_runtime_environment

    load_runtime_environment(args.env_file or None)
    static_dir = REPO_ROOT / "hypoforge/web"
    server = build_server(
        host=args.host,
        port=args.port,
        config_path=args.config,
        output_root=args.output_root,
        static_dir=static_dir,
    )
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"HypoForge dashboard: {url}")
    print(f"Runs are persisted under: {Path(args.output_root).resolve()}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
