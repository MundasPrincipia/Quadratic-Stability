"""All natural held-out rows: boundary distance, transfer, and finite brackets."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from common import HERE,sha,now,write_csv,write_json,check_lock
from numerical_bounds import envelope,task_norm_bound,score_enclosure
from controlled_profiles import grid_radius

DATA=HERE/'data/raw_rebuild_run02'


def run():
    check_lock(); assert json.loads((DATA/'SUMMARY.json').read_text())['status']=='PASS_RAW_IQ_REBUILD_AND_REFIT'
    out=HERE/'results/e1_run01';out.mkdir(exist_ok=False)
    flags=np.load(DATA/'FLAGS.npz');radii=np.array([float(grid_radius(j)) for j in range(12)])
    strata=[]; distributions=[]
    for source in ['S1','S4']:
        data=np.load(DATA/source/'SOURCE_INPUTS.npz'); c=data['C'];c=c/np.linalg.norm(c,axis=1)[:,None]
        k=c.shape[1]//8;n=c.shape[1]
        for hp in sorted((DATA/source).glob('*/COMMON_HEAD.npz')):
            head=np.load(hp);ids=head['validation_ids'];h=head['class_operators_physical'];beta=head['raw_intercept'];pairs=head['pairs']
            mh=np.zeros((10,10));hnorm=np.zeros((10,10))
            for a,b in pairs:
                hd=h[a]-h[b];mh[a,b]=mh[b,a]=task_norm_bound(hd);hnorm[a,b]=hnorm[b,a]=np.linalg.norm(hd,'fro')+abs(beta[a]-beta[b])
            full=head['full_scores'];pred=np.argmax(full,axis=1);competitors=np.array([[b for b in range(10) if b!=a] for a in pred])
            rows=np.arange(len(ids))
            for flag in flags.files:
                u=flags[flag];adapt=c[ids].reshape(-1,k,8)@u.conj();s=np.sum(abs(adapt[:,:,7:])**2,axis=(1,2))
                mass=np.sum(abs(adapt[:,:,:7])**2,axis=(1,2));valid=mass>1e-24
                xadapt=adapt.copy();xadapt[:,:,7:]=0;xadapt/=np.sqrt(np.maximum(mass,1e-300))[:,None,None]
                xphys=(xadapt@u.T).reshape(-1,n);tau=np.sqrt(2)*s
                tau_upper=np.nextafter(tau+envelope(n),np.inf)
                hx=np.stack([xphys@hj.T for hj in h],axis=1)
                base=np.einsum('ni,nji->nj',xphys.conj(),hx).real+beta
                gradients=(hx.reshape(len(ids),10,k,8)@u.conj())[:,:,:,7]
                gdiff=gradients[rows,pred,None,:]-gradients[rows[:,None],competitors,:]
                alpha=2**.75*np.linalg.norm(gdiff,axis=2)
                margins=base[rows,pred,None]-base[rows[:,None],competitors]
                mhs=mh[pred[:,None],competitors]
                err=np.array([[envelope(n,hnorm[a,b]) for b in bs] for a,bs in zip(pred,competitors)])
                raw=np.load(hp.parent/f'{flag}_BASELINE.npz')
                # Exact natural-fiber phase witnesses, feasible for every delta.
                pairmap={tuple(p):i for i,p in enumerate(pairs)}
                wl=np.empty_like(margins);wu=np.empty_like(margins)
                for r,a in enumerate(pred):
                    for j,b in enumerate(competitors[r]):
                        ix=pairmap[tuple(sorted((a,b)))];sign=1 if a<b else -1
                        center=sign*raw['pair_center'][r,ix];rad=raw['pair_radius'][r,ix]
                        wl[r,j]=center-rad+err[r,j];wu[r,j]=center+rad-err[r,j]
                outer_l=np.full((len(ids),12,9),np.nan);outer_u=outer_l.copy()
                inner_wl=np.repeat(wl[:,None,:],12,axis=1);inner_wu=np.repeat(wu[:,None,:],12,axis=1)
                applicable=valid[:,None]&(tau_upper[:,None]<=radii[None,:]/2)&(tau_upper[:,None]+radii[None,:]<=.25)
                target=np.argmax(np.where(competitors==np.argsort(-full,axis=1,kind='stable')[:,1,None],1,0),axis=1)
                # Witness parameters (target pair only) are enough to reconstruct
                # the normalized raw state from x, g and the inner radius.
                witness_parameter=np.full((len(ids),12),np.nan)
                for j,delta in enumerate(radii):
                    use=valid&(delta+tau_upper<=.25)
                    rad=delta+tau_upper[use]
                    width=(alpha[use]+2*err[use])*np.sqrt(rad[:,None])+5*mhs[use]*rad[:,None]
                    outer_l[use,j]=margins[use]-err[use]-width
                    outer_u[use,j]=margins[use]+err[use]+width
                    for r in np.flatnonzero(applicable[:,j]):
                        rin=(delta-tau_upper[r])*(1-1e-7)
                        a=pred[r];b=competitors[r,target[r]];g=gdiff[r,target[r]]
                        gn=np.linalg.norm(g)
                        if gn==0:continue
                        y=g/(2**.25*gn);physical=(y[:,None]@u[:,7:].T).reshape(-1)
                        gamma=np.sqrt(1-rin*np.vdot(physical,physical).real)
                        hdiff=h[a]-h[b];bdiff=beta[a]-beta[b]
                        for sign in [-1,1]:
                            psi=gamma*xphys[r]+sign*np.sqrt(rin)*physical
                            val,lo,hi=score_enclosure(hdiff,bdiff,psi)
                            inner_wl[r,j,target[r]]=min(inner_wl[r,j,target[r]],hi)
                            inner_wu[r,j,target[r]]=max(inner_wu[r,j,target[r]],lo)
                        witness_parameter[r,j]=rin
                    for device in np.unique(data['labels'][ids]):
                        mask=data['labels'][ids]==device
                        strata.append(dict(source=source,fold=hp.parent.name,flag=flag,device=int(device),delta=float(delta),
                            denominator=int(mask.sum()),applicable=int(np.sum(applicable[mask,j])),
                            outer_bound_available=int(np.sum(use[mask])),zero_resolved=int(np.sum(~valid[mask]))))
                for r,row in enumerate(ids):
                    distributions.append(dict(source=source,fold=hp.parent.name,flag=flag,device=int(data['labels'][row]),row_id=int(row),
                        residual_mass=float(s[r]),tau=float(tau[r]),resolved_nonzero=bool(valid[r]),
                        natural_full_head_class=int(head['classes'][pred[r]]),label=int(data['labels'][row])))
                path=out/source/hp.parent.name;path.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path/f'{flag}.npz',row_ids=ids,classes=head['classes'],candidate=pred,competitors=competitors,
                    delta=radii,residual_mass=s,tau=tau,tau_upper=tau_upper,tau_over_delta=tau[:,None]/radii,applicable=applicable,
                    base_margins=margins,alpha=alpha,M_upper=mhs,roundoff_envelope=err,
                    outer_lower=outer_l,outer_upper=outer_u,witness_lower=inner_wl,witness_upper=inner_wu,
                    target_pair_column=target,inner_witness_radius=witness_parameter,
                    xphysical=xphys,target_gradient=gdiff[rows,target])
                print('E1',source,hp.parent.name,flag,flush=True)
    write_csv(out/'STRATIFIED_APPLICABILITY.csv',strata);write_csv(out/'NATURAL_ROWS.csv',distributions)
    write_json(out/'SUMMARY.json',dict(created_at=now(),natural_cells=len(distributions),radii=12,
        row_radius_cells=len(distributions)*12,applicable=sum(r['applicable'] for r in strata),
        input_manifest_sha256=sha(DATA/'MANIFEST.json'),
        scope='natural held-out RF states; phase witnesses always feasible; inner boundary witnesses only under transfer conditions'))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
