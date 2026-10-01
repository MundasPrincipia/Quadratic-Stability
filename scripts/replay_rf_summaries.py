"""Redirect the unchanged summary implementation to explicitly supplied inputs."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from companion_paths import locate_companion

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    companion, output = args.companion.resolve(), args.output.resolve()
    if (output.exists() or output == companion or companion in output.parents
            or output in companion.parents or output == ROOT or ROOT in output.parents
            or output in ROOT.parents):
        parser.error("Use a fresh output directory outside the repository and companion.")
    script = ROOT / "paper/quadratic_stability/src/summarize_fair_design.py"
    spec = importlib.util.spec_from_file_location("quadratic_stability_saved_summaries", script)
    if spec is None or spec.loader is None:
        raise ImportError(script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only input directory bindings change; all scientific calculations remain
    # in the byte-identical implementation copied from the complete companion.
    try:
        module.HERE, module.PARENT, module.V17 = locate_companion(companion)
    except ValueError as error:
        parser.error(str(error))
    module.run(module.HERE / "results/fair_design_run01", output)


if __name__ == "__main__":
    main()
