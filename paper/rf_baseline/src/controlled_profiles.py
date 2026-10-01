"""Global one-dimensional concave E2 profiles; outward mpmath intervals.

The reduction and concavity proof are in PROOF_PACKAGE.md. Decimal outputs
are endpoint strings from 90-digit interval arithmetic; float columns are
only plotting conveniences.
"""
import mpmath as mp
import numpy as np
import json
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING

mp.mp.dps=100
mp.iv.dps=90
FAMILIES=[('mixed',s) for s in ('preserving','cross','tangent','residual')]+[
          ('quadratic',s) for s in ('preserving','cross','tangent','residual','mixed')]


def directed_serialization(value, upper):
    """Serialize an exact mpf binary rational with verified outward rounding."""
    sign,man,exponent,bitcount=value._mpf_
    numerator=(-1 if sign else 1)*int(man)
    denominator=1
    if exponent>=0:numerator <<= exponent
    else:denominator <<= -exponent
    with localcontext() as ctx:
        ctx.prec=95
        ctx.rounding=ROUND_CEILING if upper else ROUND_FLOOR
        dec=Decimal(numerator)/Decimal(denominator)
    f=float(value)
    if (upper and mp.mpf(f)<value) or (not upper and mp.mpf(f)>value):
        f=float(np.nextafter(f,np.inf if upper else -np.inf))
    assert (mp.mpf(f)>=value and mp.mpf(str(dec))>=value) if upper else (mp.mpf(f)<=value and mp.mpf(str(dec))<=value)
    return f,str(dec),json.dumps([int(sign),str(man),int(exponent),int(bitcount)],separators=(',',':'))


def grid_radius(j,ctx=mp.mp):
    return ctx.mpf(10)**(-6)*ctx.sqrt(10)**j if j<11 else ctx.mpf(1)/4


def grid_epsilon(j,ctx=mp.mp):
    return ctx.mpf(0) if j==0 else ctx.mpf(10)**(j-9)


def profile(u,delta,epsilon,order,kind,ctx=mp.mp):
    s=delta/ctx.sqrt(2)*u
    t=delta**2/2*(1-u*u)
    gamma2=1-s-t
    if order=='mixed':
        f=2*ctx.sqrt(s*t)
        if kind=='preserving': return (1+epsilon)*f
    else:
        f=t
        if kind=='preserving': return (1+epsilon)*f
    if epsilon==0: return f
    if kind=='cross': return f+2*epsilon*ctx.sqrt(s*gamma2)
    if kind=='tangent': return f+2*epsilon*ctx.sqrt(t*gamma2)
    if kind=='residual': return f+epsilon*s
    if kind=='mixed': return f+2*epsilon*ctx.sqrt(s*t)
    raise ValueError(kind)


def derivative(u,delta,epsilon,order,kind):
    s=delta/mp.sqrt(2)*u; sp=delta/mp.sqrt(2)
    t=delta**2/2*(1-u*u); tp=-delta**2*u
    z=1-s-t; zp=-sp-tp
    def dsqrt(a,ap): return ap/(2*mp.sqrt(a))
    f=2*dsqrt(s*t,sp*t+s*tp) if order=='mixed' else tp
    if kind=='preserving': return (1+epsilon)*f
    if not epsilon: return f
    if kind=='cross': return f+2*epsilon*dsqrt(s*z,sp*z+s*zp)
    if kind=='tangent': return f+2*epsilon*dsqrt(t*z,tp*z+t*zp)
    if kind=='residual': return f+epsilon*sp
    if kind=='mixed': return f+2*epsilon*dsqrt(s*t,sp*t+s*tp)
    raise ValueError(kind)


def perturbation_envelope(delta,epsilon,order,kind):
    if kind=='preserving': return epsilon*(mp.sqrt(2)*delta**mp.mpf('1.5') if order=='mixed' else delta**2/2)
    if kind=='cross': return epsilon*2**mp.mpf('.75')*mp.sqrt(delta)
    if kind=='tangent': return epsilon*mp.sqrt(2)*delta
    if kind=='residual': return epsilon*delta
    if kind=='mixed': return epsilon*mp.sqrt(2)*delta**mp.mpf('1.5')
    raise ValueError(kind)


def solve(j,e,order,kind):
    delta=grid_radius(j); epsilon=grid_epsilon(e)
    # The endpoint cases have exact maximizers. All other profiles are
    # concave, so a derivative sign change brackets the global maximum.
    if order=='quadratic' and (kind in ('preserving','tangent') or not epsilon):
        left=right=mp.mpf(0)
    elif order=='quadratic' and kind=='residual':
        u=epsilon/(mp.sqrt(2)*delta)
        if u>=1: left=right=mp.mpf(1)
        else: left=max(mp.mpf(0),u-mp.mpf('1e-75')); right=min(mp.mpf(1),u+mp.mpf('1e-75'))
    else:
        tiny=mp.mpf('1e-70')
        left=tiny; right=1-tiny
        dr=derivative(right,delta,epsilon,order,kind)
        if dr>=0:
            # A finite derivative at u=1 is needed only for quadratic cross.
            assert order=='quadratic' and kind=='cross'
            left=right=mp.mpf(1)
        else:
            assert derivative(left,delta,epsilon,order,kind)>0
            for _ in range(180):
                mid=(left+right)/2
                if derivative(mid,delta,epsilon,order,kind)>0: left=mid
                else: right=mid
    u=(left+right)/2
    f=profile(u,delta,epsilon,order,kind)
    # Verify the derivative bracket with outward intervals by differentiating
    # explicit polynomial profiles via dual (value, derivative) arithmetic.
    di=grid_radius(j,mp.iv); ei=grid_epsilon(e,mp.iv)
    def ivnum(x): return mp.iv.mpf(mp.nstr(x,100))
    def ivder(v):
        s=di/mp.iv.sqrt(2)*v; sp=di/mp.iv.sqrt(2)
        t=di**2/2*(1-v*v); tp=-di**2*v; z=1-s-t; zp=-sp-tp
        out=(sp*t+s*tp)/mp.iv.sqrt(s*t) if order=='mixed' else tp
        if kind=='preserving': return (1+ei)*out
        if e==0:return out
        if kind=='cross':return out+ei*(sp*z+s*zp)/mp.iv.sqrt(s*z)
        if kind=='tangent':return out+ei*(tp*z+t*zp)/mp.iv.sqrt(t*z)
        if kind=='residual':return out+ei*sp
        return out+ei*(sp*t+s*tp)/mp.iv.sqrt(s*t)
    if left!=right:
        assert ivder(ivnum(left)).a>=0 and ivder(ivnum(right)).b<=0
    lo=profile(ivnum(u),di,ei,order,kind,mp.iv).a
    hi=profile(mp.iv.mpf([mp.nstr(left,100),mp.nstr(right,100)]),di,ei,order,kind,mp.iv).b
    # Endpoint values are evaluated too, as a redundant global check.
    for end in [0,1]:
        hi=max(hi,profile(mp.iv.mpf(end),di,ei,order,kind,mp.iv).b)
    lower=mp.mpf(lo._mpi_[0]); upper=mp.mpf(hi._mpi_[1])
    assert lower<=f<=upper
    rel=(upper-lower)/max(abs(lower),mp.mpf('1e-100'))
    base=(mp.mpf(2)/3)**mp.mpf('.75')*delta**mp.mpf('1.5') if order=='mixed' else delta**2/2
    be=perturbation_envelope(delta,epsilon,order,kind)
    lf,ld,lb=directed_serialization(lower,False)
    uf,ud,ub=directed_serialization(upper,True)
    return dict(order=order,perturbation=kind,radius_index=j,epsilon_index=e,delta=float(delta),epsilon=float(epsilon),
        u=float(u),loss_lower=lf,loss_upper=uf,lower_decimal=ld,
        upper_decimal=ud,lower_binary_mpf=lb,upper_binary_mpf=ub,relative_bracket_width=float(rel),unperturbed_loss=float(base),
        perturbation_bound=float(be),window_25=bool(be<=base/4),status='CERTIFIED_GLOBAL' if rel<=mp.mpf('1e-5') else 'UNRESOLVED')


def canonical_task(order,kind,epsilon):
    h=np.zeros((3,3),complex)
    if order=='mixed':h[1,2]=h[2,1]=1
    else:h[1,1]=-1
    e=np.zeros_like(h)
    if kind=='preserving':e=h.copy()
    elif kind=='cross':e[0,2]=e[2,0]=1
    elif kind=='tangent':e[0,1]=e[1,0]=1
    elif kind=='residual':e[2,2]=-1
    else:e[1,2]=e[2,1]=1
    return h+epsilon*e


def witness(row):
    d=row['delta']; s=d/np.sqrt(2)*row['u']; t=d*d/2*(1-row['u']**2)
    # Phases minimize the independently tested task; all entries real.
    wsign=1.; ysign=-1.
    if row['perturbation']=='tangent':wsign=-1.; ysign=1.
    return np.array([np.sqrt(1-s-t),wsign*np.sqrt(t),ysign*np.sqrt(s)],complex)
