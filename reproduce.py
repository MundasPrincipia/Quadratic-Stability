"""Run the analytic example or replay the saved RF inputs included in this repository."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from scripts.companion_paths import locate_companion

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper/quadratic_stability"


def overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def execute(script: Path, *arguments: object) -> None:
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    # Supported modes never search the author's historical installation.
    env["CZOPERATOR_WORKSPACE"] = str(ROOT / "RAW_WORKSPACE_NOT_INCLUDED")
    subprocess.run([sys.executable, "-B", "-X", "utf8", str(script),
                    *map(str, arguments)], cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("boundary-refinement", "rf-interface", "rf-summaries", "rf-applicability"))
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh output directory outside this repository and external inputs")
    parser.add_argument("--companion", type=Path,
                        help="Optional external RF input directory; defaults to this repository")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or overlaps(output, ROOT):
        parser.error("Use a fresh output directory outside the code repository.")
    if args.mode == "boundary-refinement":
        if args.companion is not None:
            parser.error("The analytic example does not use --companion.")
        execute(PAPER / "src/run_boundary_refinement.py", "--output", output)
        execute(PAPER / "figures/plot_refinement.py", "--results", output,
                "--output", output / "figures")
        return

    companion = args.companion.resolve() if args.companion is not None else ROOT
    if not companion.is_dir():
        parser.error("The RF input directory does not exist.")
    if overlaps(output, companion):
        parser.error("Output must be outside the frozen companion.")
    try:
        analysis, parent, transfer = locate_companion(companion)
    except ValueError as error:
        parser.error(str(error))
    if args.mode in ("rf-interface", "rf-applicability"):
        required = [parent / "data/raw_rebuild_run02" / name for name in
                    ("FLAGS.npz", "S1/SOURCE_INPUTS.npz", "S4/SOURCE_INPUTS.npz")]
        for source, count in (("S1", 13), ("S4", 3)):
            heads = list((parent / "data/raw_rebuild_run02" / source).glob("*/COMMON_HEAD.npz"))
            if len(heads) != count:
                parser.error(f"Incomplete RF companion: expected {count} saved {source} fold heads.")
    else:
        required = [
            parent / "protocol/ANCHORS.csv",
            parent / "results/e3_run01/DESIGN_RESULTS.csv",
            parent / "results/e3_run01/PAIR_RESULTS.csv",
            parent / "legacy_evidence/matrix/design/run_20260905_044629_927159/design.csv",
            transfer / "results/transfer_run01/TRANSFER_RESULTS.csv",
        ]
        results = analysis / "results/fair_design_run01"
        required += [results / name for name in ("DESIGN_RESULTS.csv", "PAIR_RESULTS.csv")]
        required += [results / f"ANCHOR_{i:03d}.npz" for i in range(160)]
    if args.mode == "rf-applicability":
        required += [parent.parent / "checks/rf_applicability/REFERENCE_ARRAYS.json"]
    missing = [p.relative_to(companion).as_posix() for p in required if not p.is_file()]
    if missing:
        parser.error("Incomplete RF companion: missing " + ", ".join(missing[:5]))

    if args.mode == "rf-applicability":
        execute(ROOT / "scripts/replay_rf_applicability.py", "--parent", parent, "--output", output)
    elif args.mode == "rf-interface":
        execute(PAPER / "src/check_rf_interface.py", "--parent", parent, "--output", output)
    else:
        execute(ROOT / "scripts/replay_rf_summaries.py", "--companion", companion,
                "--output", output)


if __name__ == "__main__":
    main()
