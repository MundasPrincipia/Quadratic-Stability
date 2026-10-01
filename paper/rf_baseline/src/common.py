"""Version-local I/O and immutable protocol guards."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import os
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('CZOPERATOR_WORKSPACE',str(HERE.parents[1]))).resolve()
RF = ROOT / 'paper/revision_track_s_decision_semantics_v13_20260905/inherited_v11/inherited_rf'
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def check_lock():
    lock = json.loads((HERE / 'protocol/LOCK.json').read_text())
    for rel, digest in lock['inputs'].items():
        if sha(HERE / rel) != digest:
            raise RuntimeError('locked protocol changed: '+rel)
    return lock


def decode(x):
    return x.decode() if isinstance(x, bytes) else str(x)
