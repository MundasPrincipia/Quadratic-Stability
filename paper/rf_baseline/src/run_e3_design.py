"""Prespecified paired, difference-head versus raw-score observation design."""
import csv
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from common import HERE,sha,now,write_json,write_csv,check_lock
from numerical_bounds import envelope,gauge_bound,task_norm_bound,score_enclosure,radius_lower,obs_distance,omega

DATA=HERE/'data/raw_rebuild_run02'


def candidate_witness(h,beta,x,u,k,q,c):
    """One fixed residual ray, not a global optimizer. Save only actual failures."""
    d=8-q; y=(c@u[:,q:].T).reshape(-1)
    mass=np.vdot(y,y).real
    if mass<=1e-30:return None
    hx=h@x;hy=h@y
    a0=float(np.vdot(x,hx).real);b0=float(np.vdot(y,hy).real);cross=float(np.vdot(x,hy).real)
    overlap=float(np.vdot(x,y).real);xmass=float(np.vdot(x,x).real)
    err=envelope(len(h),np.linalg.norm(h,'fro')+abs(beta))
    def ray_upper(s):
        gamma=np.sqrt(max(0.,1-s*mass));r=np.sqrt(s)
        denominator=gamma*gamma*xmass+s*mass-2*gamma*r*overlap
        return (gamma*gamma*a0+s*b0-2*gamma*r*cross)/denominator+beta+2*err
    def state(s):
        z=np.sqrt(max(0.,1-s*mass))*x-np.sqrt(s)*y
        return z/np.linalg.norm(z)
    # Every scheduled radius is tested; no failed optimizer is discarded.
    grid=np.r_[np.logspace(-8,np.log10(.25*(1-1e-5)),60),.25*(1-1e-5)]
    previous=0.
    for s in grid:
        if ray_upper(s)<=0:
            left=previous;right=float(s)
            for _ in range(40):
                mid=(left+right)/2
                if ray_upper(mid)<=0:right=mid
                else:left=mid
            # Step inward into the failure set to make verification nonfragile.
            right=min(float(s),right*(1+1e-5)+1e-12)
            psi=state(right); value,lower,upper=score_enclosure(h,beta,psi)
            distance,distance_upper=obs_distance(psi,x,u,k,q)
            if upper<=0 and distance_upper<=.25:
                return psi,distance,distance_upper,value,upper
        previous=float(s)
    return None


def run():
    check_lock(); assert json.loads((DATA/'SUMMARY.json').read_text())['status']=='PASS_RAW_IQ_REBUILD_AND_REFIT'
    out=HERE/'results/e3_run01'; out.mkdir(exist_ok=False)
    with (HERE/'protocol/ANCHORS.csv').open(newline='',encoding='utf-8') as f:anchors=list(csv.DictReader(f))
    pairrows,summary=[],[]; inputcache={}; headcache={}; normcache={}
    for ai,anchor in enumerate(anchors):
        source,fold=anchor['source'],anchor['fold']; key=(source,fold)
        if source not in inputcache:inputcache[source]=np.load(DATA/source/'SOURCE_INPUTS.npz')
        if key not in headcache:headcache[key]=np.load(DATA/source/fold/'COMMON_HEAD.npz')
        src=inputcache[source]; head=headcache[key]; h=head['class_operators_physical']; beta=head['raw_intercept']
        row=int(anchor['array_index']); raw=src['C'][row]; krow=len(raw)//8
        xmat=raw.reshape(krow,8).copy(); xmat[:,5:]=0; x=xmat.reshape(-1); x/=np.linalg.norm(x)
        scores=np.array([score_enclosure(hj,bj,x)[0] for hj,bj in zip(h,beta)])
        order=np.argsort(-scores,kind='stable'); a,b=map(int,order[:2])
        gradients=np.stack([(hj@x).reshape(krow,8)[:,5:] for hj in h])
        difference=gradients[a]-gradients[b]
        _,_,vpair=np.linalg.svd(difference.conj(),full_matrices=True)
        _,_,vraw=np.linalg.svd(gradients.conj().reshape(-1,3),full_matrices=True)
        rotations={'difference':vpair.conj().T,'raw_scores':vraw.conj().T}
        for seed in [17,29,43]:
            rng=np.random.default_rng(seed)
            z=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
            q,r=np.linalg.qr(z); phases=np.diag(r)/np.maximum(abs(np.diag(r)),1e-300)
            rotations[f'random_{seed}']=q*phases.conj()
        schemes=[('baseline',0,np.eye(3,dtype=complex))]+[(name,k,v) for k in [1,2] for name,v in rotations.items()]
        witness_payload={}; basis_payload={}
        for name,added,v in schemes:
            q=5+added; d=8-q; u=np.eye(8,dtype=complex);u[5:,5:]=v
            basis_payload[f'{name}_k{added}']=u
            local=[]
            for competitor in range(len(h)):
                if competitor==a:continue
                hd=h[a]-h[competitor]; bd=float(beta[a]-beta[competitor])
                m,ml,mu=score_enclosure(hd,bd,x)
                nkey=(source,fold,min(a,competitor),max(a,competitor))
                if nkey not in normcache:normcache[nkey]=task_norm_bound(hd)
                mh=normcache[nkey]
                g=(gradients[a]-gradients[competitor])@v[:,added:].conj()
                defect=np.linalg.norm(v.conj().T@v-np.eye(3),'fro')
                extra=envelope(len(hd),np.linalg.norm(hd,'fro'))+2*defect*np.linalg.norm(hd,'fro')
                gauge=gauge_bound(g,d,extra)
                dl=radius_lower(ml,gauge['upper'],mh)
                found=candidate_witness(hd,bd,x,u,krow,q,gauge['direction']) if ml>0 else None
                record=dict(anchor=ai,source=source,fold=fold,device=int(anchor['device']),row_id=int(anchor['row_id']),
                    method=name,k=added,q=q,d=d,candidate=int(head['classes'][a]),competitor=int(head['classes'][competitor]),
                    target_pair=competitor==b,margin=m,margin_lower=ml,M_upper=mh,alpha_lower=gauge['primal'],alpha_upper=gauge['upper'],
                    solver_success=gauge['success'],solver_message=gauge['message'],radius_lower=dl,
                    radius_upper=None if found is None else found[2],censored=found is None,
                    witness_score=None if found is None else found[3],witness_score_upper=None if found is None else found[4],
                    witness_distance=None if found is None else found[1],no_strict_base=ml<=0)
                pairrows.append(record);local.append(record)
                if found is not None:witness_payload[f'{name}_k{added}_competitor{competitor}']=found[0]
            target=next(r for r in local if r['target_pair'])
            available=[r['radius_upper'] for r in local if r['radius_upper'] is not None]
            summary.append(dict(anchor=ai,source=source,fold=fold,device=int(anchor['device']),row_id=int(anchor['row_id']),method=name,k=added,
                candidate=int(head['classes'][a]),runner_up=int(head['classes'][b]),label=int(src['labels'][row]),
                pair_alpha_lower=target['alpha_lower'],pair_alpha_upper=target['alpha_upper'],
                pair_radius_lower=target['radius_lower'],pair_radius_upper=target['radius_upper'],pair_censored=target['censored'],
                multiclass_radius_lower=min(r['radius_lower'] for r in local),multiclass_radius_upper=min(available) if available else None,
                multiclass_censored=not bool(available),solver_failures=sum(not r['solver_success'] for r in local)))
        np.savez_compressed(out/f'ANCHOR_{ai:03d}_WITNESSES.npz',base=x,**witness_payload)
        np.savez_compressed(out/f'ANCHOR_{ai:03d}_BASES.npz',**basis_payload)
        print('E3 anchor',ai+1,'/',len(anchors),source,fold,flush=True)
    write_csv(out/'PAIR_RESULTS.csv',pairrows);write_csv(out/'DESIGN_RESULTS.csv',summary)
    write_json(out/'SUMMARY.json',dict(created_at=now(),anchors=len(anchors),design_cells=len(summary),pair_cells=len(pairrows),
        witnessed=sum(not r['censored'] for r in pairrows),solver_non_success=sum(not r['solver_success'] for r in pairrows),
        scope='fixed boundary anchors; canonical coefficient design, finite certified bounds and explicitly witnessed failures',
        input_manifest_sha256=sha(DATA/'MANIFEST.json')))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
