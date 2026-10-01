"""Conservative binary64 envelopes and analytic RF bounds.

IEEE binary64, round-to-nearest CPU arithmetic, no overflow/underflow in the
declared RF range. The generous normwise budgets dominate accumulated dot,
matrix-product and normalization errors. Eigensolver/optimizer success is
never a certificate: residuals and Gram defects are checked separately.
"""
import numpy as np
from scipy.optimize import minimize

EPS=np.finfo(float).eps


def envelope(n,scale=1.):
    t=8192*int(n)**2*EPS
    if t>=.01: raise ValueError('dimension outside roundoff budget')
    return np.nextafter(t/(1-t)*(1+float(scale)),np.inf)


def omega(c,d):
    gram=c.conj().T@c
    return (float(np.vdot(c,c).real)**2+float(np.vdot(gram,gram).real)/d)**.25


def task_norm_bound(h):
    """Shift is arbitrary; checked spectral reconstruction gives an upper."""
    n=len(h); eigen,q=np.linalg.eigh(h)
    shift=(eigen[-1]+eigen[0])/2
    lam=eigen-shift
    a=h-shift*np.eye(n)
    gram=np.linalg.norm(q.conj().T@q-np.eye(n),'fro')
    residual=np.linalg.norm(a-(q*lam)@q.conj().T,'fro')
    err=envelope(n,np.linalg.norm(a,'fro')+abs(shift))
    return float(np.nextafter((1+gram+err)*np.max(abs(lam))+residual+err,np.inf))


def score_enclosure(h,beta,psi):
    norm=np.linalg.norm(psi)
    z=psi/norm
    value=float(np.vdot(z,h@z).real+beta)
    err=envelope(len(h),np.linalg.norm(h,'fro')+abs(beta))
    return value,float(np.nextafter(value-err,-np.inf)),float(np.nextafter(value+err,np.inf))


def gauge_bound(g,d,extra_error=0.):
    u,s,vh=np.linalg.svd(g,full_matrices=False)
    if np.linalg.norm(s)==0:
        return dict(primal=0.,upper=2*extra_error,direction=np.zeros_like(g),success=True,message='zero input')
    def g4(t):return np.dot(t,t)**2+np.sum(t**4)/d
    def jac(t):return 4*(np.dot(t,t)*t+t**3/d)
    t0=s/g4(s)**.25
    fit=minimize(lambda t:-np.dot(s,t),t0,jac=lambda t:-s,method='SLSQP',bounds=[(0,None)]*len(s),
        constraints={'type':'ineq','fun':lambda t:1-g4(t),'jac':lambda t:-jac(t)},options={'ftol':1e-13,'maxiter':500})
    t=np.maximum(fit.x,0)
    if not np.all(np.isfinite(t)) or np.linalg.norm(t)==0:t=t0
    c=(u*t)@vh
    w=omega(c,d); c/=w
    w=omega(c,d)
    primal=float(np.vdot(g,c).real/w)
    gradient=(np.vdot(c,c).real*c+c@(c.conj().T@c)/d)/w**3
    residual=np.linalg.norm(g-primal*gradient,'fro')
    err=envelope(max(g.shape),np.linalg.norm(g,'fro'))+extra_error
    support=max(0.,primal)+residual+err
    fallback=np.linalg.norm(g,'fro')/(1+1/(d*min(g.shape)))**.25+err
    upper=2*np.nextafter(min(support,fallback),np.inf)
    # A radial contraction ensures the saved candidate lies inside the gauge ball.
    c/=omega(c,d)*(1+envelope(max(g.shape)))
    return dict(primal=max(0.,2*primal-2*err),upper=float(upper),direction=c,
        success=bool(fit.success),message=str(fit.message))


def radius_lower(m_lower,alpha_upper,mh_upper):
    if m_lower<=0:return 0.
    b=5*mh_upper
    root=2*m_lower/(alpha_upper+np.sqrt(alpha_upper**2+4*b*m_lower)) if b else m_lower/alpha_upper
    return float(np.nextafter(min(.25,root**2)*(1-1e-10),0.))


def obs_distance(psi,x,u,k,q):
    """Distance under the displayed temporal basis plus a polar-defect bound."""
    n=len(psi); d=8-q
    z=psi/np.linalg.norm(psi); base=x/np.linalg.norm(x)
    zz=z.reshape(k,8)@u.conj(); xx=base.reshape(k,8)@u.conj()
    a=zz[:,:q].reshape(-1); b=xx[:,:q].reshape(-1)
    block=np.outer(a,a.conj())-np.outer(b,b.conj())
    y=zz[:,q:]; y0=xx[:,q:]
    residual=y@y.conj().T-y0@y0.conj().T
    dist=float(np.sqrt(np.linalg.norm(block,'fro')**2+np.linalg.norm(residual,'fro')**2/d))
    defect=np.linalg.norm(u.conj().T@u-np.eye(8),'fro')
    if defect>=.01:raise AssertionError('basis not safely close to unitary')
    err=envelope(n)+8*defect/(1+np.sqrt(1-defect))
    return dist,float(np.nextafter(dist+err,np.inf))
