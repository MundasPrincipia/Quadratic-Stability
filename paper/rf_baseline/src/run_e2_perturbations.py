"""Controlled high-order tasks, canonical global intervals and RF embeddings."""
import csv
import json
import numpy as np
from threadpoolctl import threadpool_limits
from common import HERE,sha,now,write_json,write_csv,check_lock
from controlled_profiles import FAMILIES,solve,witness,canonical_task
from numerical_bounds import envelope

DATA=HERE/'data/raw_rebuild_run02'


def run():
    check_lock();assert json.loads((DATA/'SUMMARY.json').read_text())['status']=='PASS_RAW_IQ_REBUILD_AND_REFIT'
    out=HERE/'results/e2_run02';out.mkdir(exist_ok=False)
    canonical=[]
    for order,kind in FAMILIES:
        for e in range(9):
            for j in range(12):canonical.append(solve(j,e,order,kind))
        print('E2 canonical:',order,kind,flush=True)
    write_csv(out/'CANONICAL_INTERVALS.csv',canonical)
    states=np.stack([witness(r) for r in canonical])
    tasks=np.stack([canonical_task(r['order'],r['perturbation'],r['epsilon']) for r in canonical])
    dvals=np.array([r['delta'] for r in canonical]);canonloss=np.array([r['loss_lower'] for r in canonical])
    with (HERE/'protocol/ANCHORS.csv').open(encoding='utf-8',newline='') as f:anchors=list(csv.DictReader(f))
    logs=[];cache={};heads={}
    for ai,a in enumerate(anchors):
        source,fold=a['source'],a['fold']; key=(source,fold)
        if source not in cache:cache[source]=np.load(DATA/source/'SOURCE_INPUTS.npz')
        if key not in heads:heads[key]=np.load(DATA/source/fold/'COMMON_HEAD.npz')
        src=cache[source];hd=heads[key];row=int(a['array_index']);raw=src['C'][row];n=len(raw);k=n//8
        full=np.array([np.vdot(raw,h@raw).real+b for h,b in zip(hd['class_operators_physical'],hd['raw_intercept'])])
        classes=np.argsort(-full,kind='stable');ia,ib=map(int,classes[:2]);h=hd['class_operators_physical'][ia]-hd['class_operators_physical'][ib]
        ix=np.arange(n).reshape(k,8)[:,:7].reshape(-1);iy=np.arange(n).reshape(k,8)[:,7:].reshape(-1)
        x=raw[ix].copy();x/=np.linalg.norm(x)
        j=h[np.ix_(ix,iy)];j=j-x[:,None]*(x.conj()@j)[None,:]
        left,sing,right=np.linalg.svd(j,full_matrices=False)
        if sing[0]>0:w=left[:,0];v=right.conj().T[:,0];status='head_J_singular_directions'
        else:
            w=np.eye(len(x),dtype=complex)[:,int(np.argmin(abs(x)))];w-=x*np.vdot(x,w);v=np.eye(k,dtype=complex)[:,0];status='coordinate_fallback'
        w-=x*np.vdot(x,w);w/=np.linalg.norm(w);v/=np.linalg.norm(v)
        basis=np.zeros((n,3),complex);basis[ix,0]=x;basis[ix,1]=w;basis[iy,2]=v
        gram=basis.conj().T@basis;defect=np.linalg.norm(gram-np.eye(3),'fro')
        assert defect<1e-10
        # Exact rank factorization of the embedded dense matrix Q H3 Q*: use
        # its small Gram to evaluate every case, with dense checks below.
        inner=states@gram.T
        norms=np.einsum('ni,ij,nj->n',states.conj(),gram,states).real
        rawloss=-np.einsum('ni,nij,nj->n',inner.conj(),tasks,inner).real/norms
        base=gram[:,0]
        basescore=np.einsum('i,nij,j->n',base.conj(),tasks,base).real/gram[0,0].real
        centeredloss=rawloss+basescore
        discrepancy=np.abs(centeredloss-canonloss)
        # Orthogonalization defect has a matrix-perturbation interpretation.
        polar_error=defect/(1+np.sqrt(1-defect))
        h3norm=np.linalg.norm(tasks,axis=(1,2))
        op_error=(2*polar_error+polar_error**2)*h3norm
        defect_bound=op_error*(2*np.sqrt(dvals)+(np.sqrt(2)+2)*dvals+np.sqrt(2)*dvals**1.5+dvals**2)
        floating_error=envelope(3,h3norm.max())
        # The saved binary64 embedding is a numerical implementation check;
        # its roundoff allowance does not replace the canonical interval.
        assert np.all(discrepancy<=defect_bound+floating_error)
        dense_max=0.
        for ci in [i*108+107 for i in range(9)]:
            dense=basis@tasks[ci]@basis.conj().T;psi=basis@states[ci];psi/=np.linalg.norm(psi)
            bx=basis[:,0]/np.linalg.norm(basis[:,0])
            val=-np.vdot(psi,dense@psi).real+np.vdot(bx,dense@bx).real
            dense_max=max(dense_max,abs(val-centeredloss[ci]))
        np.savez_compressed(out/f'ANCHOR_{ai:03d}.npz',basis=basis,gram=gram,
            constructed_loss=centeredloss,canonical_difference=discrepancy,operator_defect_bound=op_error,
            endpoint_defect_bound=defect_bound,embedded_window_25=np.array([r['perturbation_bound'] for r in canonical])+defect_bound<=np.array([r['unperturbed_loss'] for r in canonical])/4,
            unmodified_pair_H=h,unmodified_pair_intercept=hd['raw_intercept'][ia]-hd['raw_intercept'][ib])
        logs.append(dict(anchor=ai,source=source,fold=fold,device=int(a['device']),row_id=int(a['row_id']),
            direction_status=status,candidate=int(hd['classes'][ia]),competitor=int(hd['classes'][ib]),
            gram_defect=float(defect),cells=len(canonical),max_canonical_discrepancy=float(discrepancy.max()),
            max_dense_factorization_error=float(dense_max),all_checks_pass=True))
        if (ai+1)%10==0:print('E2 RF embeddings:',ai+1,'/',len(anchors),flush=True)
    write_csv(out/'ANCHOR_CHECKS.csv',logs)
    write_json(out/'SUMMARY.json',dict(created_at=now(),canonical_cells=len(canonical),anchors=len(anchors),
        embedded_cells=len(canonical)*len(anchors),unresolved=sum(r['status']!='CERTIFIED_GLOBAL' for r in canonical),
        max_relative_bracket_width=max(r['relative_bracket_width'] for r in canonical),
        window_cells=sum(r['window_25'] for r in canonical),input_manifest_sha256=sha(DATA/'MANIFEST.json'),
        scope='nine controlled task/perturbation families, canonical interval proofs; 160 RF direction embeddings are not independent laws'))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
