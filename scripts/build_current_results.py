from __future__ import annotations

import subprocess
import sys
from pathlib import Path


STEPS = [
    "summarize_parallel_deep_gpu_full.py",
    "summarize_parallel_model_suite.py",
    "generate_current_results_figures.py",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for script in STEPS:
        print(f"[current-results] {script}", flush=True)
        subprocess.run([sys.executable, str(root / "scripts" / script)], cwd=root, check=True)


if __name__ == "__main__":
    main()
