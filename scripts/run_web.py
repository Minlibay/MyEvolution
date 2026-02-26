"""Run FastAPI + WebSocket server for the simulation UI."""

import sys
from pathlib import Path


def main():
    # Ensure src is on sys.path
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))

    import uvicorn  # noqa

    uvicorn.run(
        "web.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(src_path)],
    )


if __name__ == "__main__":
    main()
