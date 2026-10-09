"""Fresh-destination local raw-IQ or derived-input replay and exact comparison."""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from common import HERE,ROOT,now,sha,write_json

DIRS=['data/raw_rebuild_run02','results/e1_run01','results/e2_run02','results/e3_run01','results/analysis_run03']
META={'created_at','created_utc','input_manifest_sha256','prior_executed_result_sha256'}

def scientific(value):
    if isinstance(value,dict):return {k:scientific(v) for k,v in value.items() if k not in META}
    if isinstance(value,list):return [scientific(v) for v in value]
    return value


def compare(reference,replayed):
    checks=[]
    files=[p for rel in DIRS for p in (reference/rel).rglob('*') if p.is_file() and p.name!='MANIFEST.json']
    files += [reference/'results'/n for n in ['THEORY_REGRESSIONS.json','THEORY_REGRESSIONS_EXTENDED_RUN02.json']]
    for p in sorted(files):
        rel=p.relative_to(reference);q=replayed/rel
        if not q.exists():raise AssertionError('missing replay output '+str(rel))
        byte_equal=sha(p)==sha(q)
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as a,np.load(q,allow_pickle=False) as b:
                assert set(a.files)==set(b.files),str(rel)
                for key in a.files:
                    av,bv=a[key],b[key]
                    same=np.array_equal(av,bv,equal_nan=True) if av.dtype.kind in 'fc' else np.array_equal(av,bv)
                    assert same,(str(rel),key)
            method='all arrays exact (including NaN positions)'
        elif p.suffix=='.json':
            assert scientific(json.loads(p.read_text()))==scientific(json.loads(q.read_text())),str(rel)
            method='scientific JSON exact; timestamps/provenance pointers excluded'
        else:
            assert byte_equal,str(rel)
            method='byte identical'
        checks.append(dict(path=rel.as_posix(),method=method,byte_equal=byte_equal,reference_sha256=sha(p),replay_sha256=sha(q)))
    return checks


def run(destination,workspace,derived):
    destination=destination.resolve()
    destination.mkdir(parents=True,exist_ok=False)
    for rel in ['src','protocol','PAPER_ACCEPTANCE_CONTRACT.md','PROOF_PACKAGE.md']:
        p=HERE/rel;q=destination/rel
        if p.is_dir():shutil.copytree(p,q,ignore=shutil.ignore_patterns('__pycache__'))
        else:q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    env=os.environ.copy();env['CZOPERATOR_WORKSPACE']=str(workspace.resolve());env['PYTHONUTF8']='1'
    env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    commands=[]
    if derived:shutil.copytree(HERE/'data/raw_rebuild_run02',destination/'data/raw_rebuild_run02')
    else:commands.append(['rebuild_raw_iq.py','--output',str(destination/'data/raw_rebuild_run02')])
    commands += [[s] for s in ['run_theory_regressions.py','run_extended_theory_regressions.py',
        'run_e1_applicability.py','run_e2_perturbations.py','run_e3_design.py','summarize_results.py']]
    logdir=destination/'replay_logs';logdir.mkdir()
    history=[]
    for i,command in enumerate(commands):
        cmd=[sys.executable,str(destination/'src'/command[0]),*command[1:]]
        print('REPLAY stage',i+1,'/',len(commands),command[0],flush=True)
        with (logdir/f'{i:02d}_{Path(command[0]).stem}.log').open('x',encoding='utf-8') as log:
            process=subprocess.run(cmd,cwd=destination,env=env,stdout=log,stderr=subprocess.STDOUT)
        history.append(dict(command=command,returncode=process.returncode))
        if process.returncode:raise RuntimeError('replay stage failed; inspect '+str(logdir))
    checks=compare(HERE,destination)
    write_json(destination/'REPLAY_RECEIPT.json',dict(created_at=now(),status='PASS',
        mode='derived-input replay' if derived else 'local published raw-IQ reconstruction and result replay',
        workspace=str(workspace.resolve()),scientific_files_checked=len(checks),
        byte_identical=sum(r['byte_equal'] for r in checks),commands=history,checks=checks))
    print('PASS',len(checks),'scientific files; receipt',destination/'REPLAY_RECEIPT.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--workspace',type=Path,default=ROOT);p.add_argument('--derived',action='store_true')
    a=p.parse_args();run(a.destination,a.workspace,a.derived)
