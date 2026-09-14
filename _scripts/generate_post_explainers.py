"""Rebuild both posts' responsive SVG figures.

    python3 _scripts/generate_post_explainers.py

Each generator uses only the Python standard library.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    scripts = Path(__file__).resolve().parent
    for name in ("generate_gpu_visuals.py", "generate_flowmap_visuals.py"):
        runpy.run_path(str(scripts / name), run_name="__main__")
