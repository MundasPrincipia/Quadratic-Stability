"""Executed edge, leakage and fixed-family regressions; not substitute proofs."""
import json
import numpy as np
from scipy.special import ndtr,ndtri
from common import HERE,now,sha,check_lock,write_json
from controlled_profiles import FAMILIES,canonical_task,perturbation_envelope
import mpmath as mp


def require_positive_definite(a):
    if len(a)==0 or np.linalg.eigvalsh(a)[0]<=0:
        raise ValueError('Schur specialization requires strictly positive residual block')
    return np.linalg.inv(a)


def run():
    check_lock();rng=np.random.default_rng(170929);rows=[]
    old=HERE/'results/THEORY_REGRESSIONS.json'
    # Empty tangent: no minimum eigenvalue is requested for the empty S.
    kr=np.diag([.3,2.]);h=np.diag([0.,.3,2.]);tangent=np.empty((0,0))
    assert tangent.size==0 and np.linalg.eigvalsh(h)[0]==0
    for delta in [1e-6,1e-3,.1,.25]:
        for k in range(10):
            y=rng.normal(size=2)+1j*rng.normal(size=2)
            y=y/np.linalg.norm(y)*np.sqrt(delta/2)
            psi=np.r_[np.sqrt(1-np.vdot(y,y).real),y]
            score=np.vdot(psi,h@psi).real
            assert score>=0 and np.vdot(np.array([1.,0.,0.]),h@np.array([1.,0.,0.]))==0
            rows.append(dict(test='zero_tangent_flat_lower',delta=delta,score=float(score),pass_=True))
    for spectrum in [[0.,1.],[0.,0.],[-.1,1.]]:
        rejected=False
        try:require_positive_definite(np.diag(spectrum))
        except ValueError:rejected=True
        assert rejected
        rows.append(dict(test='singular_or_indefinite_Kr_rejected',spectrum=spectrum,pass_=True))
    # Complex, multidimensional Schur square: check both signs and zero.
    kr=np.array([[2.,.2j],[-.2j,1.5]],complex);inv=require_positive_definite(kr)
    j=np.array([[.4+.1j,.2],[-.2j,.3-.1j]])
    for lam in [-1.,0.,.8]:
        seff=np.diag([lam,lam+1.]);kt=seff+j@inv@j.conj().T
        for delta in [1e-2,1e-3,1e-4]:
            v=np.array([1.,0.]);a=inv@j.conj().T@v
            kappa=2*np.linalg.norm(a)**4
            t2=delta**2/(1+np.sqrt(1+kappa*delta**2));w=np.sqrt(t2)*v;y=-np.sqrt(t2)*a
            loss=(np.vdot(w,kt@w)+2*np.vdot(w,j@y)+np.vdot(y,kr@y)).real
            assert abs(loss-lam*t2)<1e-18
            assert abs(2*t2+kappa*t2*t2-delta**2)<1e-18
            if lam>=0:assert np.linalg.eigvalsh(np.block([[kt,j],[j.conj().T,kr]]))[0]>=-1e-14
            rows.append(dict(test='complex_2d_schur',lambda_min=lam,delta=delta,loss=float(loss),pass_=True))
    # Each prescribed perturbation direction is checked, not just random E.
    for order,kind in FAMILIES:
        h0=canonical_task(order,kind,0.);e=canonical_task(order,kind,1.)-h0
        for delta in [1e-6,1e-3,.1,.25]:
            for eps in [1e-8,1e-4,.1]:
                s=delta/np.sqrt(6);t=delta**2/3
                for phase in [1.,-1.,1j,-1j]:
                    psi=np.array([np.sqrt(1-s-t),np.sqrt(t)*phase,-np.sqrt(s)])
                    difference=abs(eps*np.vdot(psi,e@psi).real-eps*e[0,0].real)
                    bound=float(perturbation_envelope(mp.mpf(delta),mp.mpf(eps),order,kind))
                    assert difference<=bound*(1+1e-12)+1e-25
                rows.append(dict(test='prescribed_perturbation_branch',order=order,kind=kind,delta=delta,epsilon=eps,pass_=True))
    # H is fixed in each family; only the scalar intercept m changes.
    # Explicit d=1 downward endpoints: cross 2sqrt(s(1-s)), residual s,
    # mixed (2/3)^(3/4)delta^(3/2), and tangent delta^2/2.
    margins=np.logspace(-6,-2,17);sigma=.3;eta=.05
    multiplier=4*sigma**2*ndtri(1-eta)**2
    for name,p,expected in [('cross',.5,-4.),('linear',1.,-2.),('mixed',1.5,-4/3),('quadratic',2.,-1.)]:
        if name=='cross':
            s=margins**2/(2*(1+np.sqrt(1-margins**2)));radii=np.sqrt(2)*s
            recovered=2*np.sqrt(s*(1-s))
        elif name=='linear':radii=np.sqrt(2)*margins;recovered=radii/np.sqrt(2)
        elif name=='mixed':
            c=(2/3)**.75;radii=(margins/c)**(2/3);recovered=c*radii**1.5
        else:radii=np.sqrt(2*margins);recovered=radii*radii/2
        assert radii.max()<=.25 and np.allclose(recovered,margins,rtol=1e-13,atol=0)
        ns=multiplier/radii**2
        tv=2*ndtr(np.sqrt(ns)*radii/(2*sigma))-1
        assert np.max(abs(tv-(1-2*eta)))<1e-14
        slopes=np.diff(np.log(ns))/np.diff(np.log(margins))
        assert np.max(abs(slopes-expected))<1e-4
        rows.append(dict(test='fixed_matrix_intercept_only_rate',family=name,p=p,expected_N_exponent=expected,
            margins=margins.tolist(),first_failure_radii=radii.tolist(),N_lower=ns.tolist(),local_log_slopes=slopes.tolist(),pass_=True))
    write_json(HERE/'results/THEORY_REGRESSIONS_EXTENDED_RUN02.json',dict(created_at=now(),
        status='PASS_EXECUTED_DECLARED_REGRESSIONS',prior_executed_result_sha256=sha(old),
        counts=dict(zero_tangent=40,residual_rejection=3,complex_schur=9,prescribed_branch=108,fixed_rate_families=4),
        scope='Deterministic regression controls plus prior 125 executed cases; mathematical proof remains in manuscript.',rows=rows))
    print('PASS: 164 added regression records including all four fixed families')


if __name__=='__main__':run()
