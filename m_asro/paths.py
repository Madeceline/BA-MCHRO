"""Repository and experiment artifact locations."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
RUNS_ROOT = EXPERIMENTS_ROOT / "runs"
BENCHMARK_ROOT = EXPERIMENTS_ROOT / "benchmarks"
FIGURE_ROOT = EXPERIMENTS_ROOT / "benchmark_figures"


def artifact_path(value: str | Path) -> Path:
    """Resolve archived paths without rewriting immutable experiment reports."""
    path = Path(value)
    if path.exists():
        return path
    for name in ("runs", "benchmarks", "benchmark_figures", "examples"):
        try:
            suffix = path.relative_to(PROJECT_ROOT / name)
        except ValueError:
            continue
        target = "demo" if name == "examples" else name
        return EXPERIMENTS_ROOT / target / suffix
    return path
