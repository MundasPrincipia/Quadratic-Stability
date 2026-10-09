"""Rebuild the omitted E1 arrays from frozen RF inputs and verify their payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent, output = args.parent.resolve(), args.output.resolve()
    if output.exists() or output == parent or parent in output.parents or output in parent.parents:
        parser.error("Choose a new output directory outside the RF inputs.")
    reference_path = parent.parent / "checks/rf_applicability/REFERENCE_ARRAYS.json"
    reference = json.loads(reference_path.read_text(encoding="utf8"))
    # A data-only companion reuses the existing repository implementation.
    code = parent / "src"
    if not (code / "run_e1_applicability.py").is_file():
        code = Path(__file__).resolve().parents[1] / "paper/rf_baseline/src"
    sys.path.insert(0, str(code))
    import run_e1_applicability as e1
    import common
    common.HERE = parent

    # Redirect only input/output locations; execute the frozen numerical routine.
    output.mkdir(parents=True)
    (output / "results").mkdir()
    e1.HERE = output
    e1.DATA = parent / "data/raw_rebuild_run02"
    with e1.threadpool_limits(limits=1):
        e1.run()
    result = output / "results/e1_run01"
    checks = []
    for name, record in reference["files"].items():
        path = result / name
        with zipfile.ZipFile(path) as payload:
            assert set(payload.namelist()) == set(record["array_sha256"]), name
            for key, expected in record["array_sha256"].items():
                assert sha(payload.read(key)) == expected, (name, key)
        checks.append({"path": name, "all_array_payloads_exact": True,
                       "container_bytes_identical": sha(path.read_bytes()) == record["file_sha256"]})
    for name in ("STRATIFIED_APPLICABILITY.csv", "NATURAL_ROWS.csv"):
        assert (result / name).read_bytes() == (parent / "results/e1_run01" / name).read_bytes(), name
    got = json.loads((result / "SUMMARY.json").read_text())
    expected = json.loads((parent / "results/e1_run01/SUMMARY.json").read_text())
    excluded = {"created_at", "input_manifest_sha256"}
    assert {k:v for k,v in got.items() if k not in excluded} == {k:v for k,v in expected.items() if k not in excluded}
    receipt = {"status": "PASS", "files": len(checks), "all_array_payloads_exact": True,
               "summary_CSVs_byte_identical": True, "scientific_summary_fields_identical": True,
               "container_bytes_identical": sum(row["container_bytes_identical"] for row in checks),
               "checks": checks}
    (output / "REPLAY_CHECK.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
    print(json.dumps({k:v for k,v in receipt.items() if k != "checks"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
