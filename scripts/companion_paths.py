"""Locate RF inputs in this repository or an extracted publication companion."""
from pathlib import Path


def locate_companion(root: Path) -> tuple[Path, Path, Path]:
    root = root.resolve()
    for candidate in (root, root / "reproducibility"):
        for analysis, baseline, transfer in (
            (candidate / "analysis", candidate / "rf_baseline", candidate / "reference_transfer"),
            (candidate / "paper/quadratic_stability", candidate / "paper/rf_baseline",
             candidate / "paper/reference_transfer"),
        ):
            if all(path.is_dir() for path in (analysis, baseline, transfer)):
                return analysis, baseline, transfer
    raise ValueError("Select this repository, the supplement's reproducibility directory, or its containing directory.")
