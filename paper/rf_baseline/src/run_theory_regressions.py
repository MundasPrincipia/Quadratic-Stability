"""Deterministic small-dimensional regressions; proof is separate."""
import numpy as np
from scipy.special import ndtr,ndtri
from common import HERE,write_json,now,check_lock


def run():
    check_lock(); rows=[]
    rng=np.random.default_rng(170929)
    # Exact Schur witness: negative, PSD, semidefinite, zero tangent.
    for lam in [-2.,0.,1.]:
        kr=2.; j=.7; kt=lam+j*j/kr; a=j/kr; kappa=2*a**4
        for delta in [1e-1,1e-2,1e-3,1e-4]:
            t2=delta**2/(1+np.sqrt(1+kappa*delta**2))
            assert abs(2*t2+kappa*t2*t2-delta**2)<1e-16
            y=-a*np.sqrt(t2); w=np.sqrt(t2)
            loss=kt*w*w+2*j*w*y+kr*y*y
            assert abs(loss-lam*t2)<1e-16
            if lam<0: assert abs((-loss)/delta**2+lam/2)<.001
            else: assert np.linalg.eigvalsh([[kt,j],[j,kr]])[0]>-1e-15
            rows.append(dict(test='schur',parameter=lam,delta=delta,pass_=True))
    for _ in range(100):
        n=5; k=1; d=2; tangent=2
        z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); e=(z+z.conj().T)/2
        h=e[0,0].real; g=e[3:,0]; b=e[1:3,0]; kr=e[3:,3:]-h*np.eye(2); j=e[1:3,3:]; kt=e[1:3,1:3]-h*np.eye(2)
        delta=10**rng.uniform(-6,-.61); c=rng.normal(size=2)+1j*rng.normal(size=2); w=rng.normal(size=2)+1j*rng.normal(size=2)
        c=c/(1.5*np.linalg.norm(c)**4)**.25*.6; w=w/np.linalg.norm(w)*.5
        gamma=np.sqrt(1-delta*np.vdot(c,c).real-delta**2*np.vdot(w,w).real)
        psi=np.r_[gamma,delta*w,np.sqrt(delta)*c]
        alpha=2*np.linalg.norm(g)/(1.5)**.25
        bound=alpha*np.sqrt(delta)+(np.sqrt(2)*np.linalg.norm(b)+np.linalg.norm(kr,2))*delta+np.sqrt(2)*np.linalg.norm(j,2)*delta**1.5+.5*np.linalg.norm(kt,2)*delta**2
        assert abs(np.vdot(psi,e@psi).real-h)<=bound+1e-12
    rows.append(dict(test='perturbation_100_draws',parameter=100,delta='',pass_=True))
    for eta in [.01,.05,.2,.49]:
        for radius in [.001,.01,.1]:
            sigma=.3; n=4*sigma*sigma*ndtri(1-eta)**2/radius**2
            tv=2*ndtr(np.sqrt(n)*radius/(2*sigma))-1
            assert abs(tv-(1-2*eta))<1e-14
            rows.append(dict(test='gaussian_tv',parameter=eta,delta=radius,pass_=True))
    a=np.array([[1,1],[-1,1]]); assert np.linalg.matrix_rank(a)==2 and np.linalg.matrix_rank(a[:1]-a[1:])==1
    write_json(HERE/'results/THEORY_REGRESSIONS.json',dict(created_at=now(),status='PASS',
        schur_cases=12,perturbation_cases=100,gaussian_cases=12,common_mode_cases=1,
        zero_tangent='positive residual alone is globally nonnegative',singular_residual='excluded by hypothesis, no inverse taken',rows=rows))


if __name__=='__main__':run()
