# v16 Proof Package

Status: PROVABLE AS STATED (with the explicit domains below).
This package records new local consequences; v15's main classification is the canonical inherited theorem.

## Assumptions and notation
Fix finite K>=1, 1<=q<T, d=T-q, n=Kq and a unit resolved vector x. States are normalized rank-one projectors; H and E are Hermitian, with scalar intercepts. For 0<=delta<=1/4 the existing exact coordinates are u=gamma*x+delta*W, Y=sqrt(delta)*C, W perpendicular to x, c=vec_r(C), gamma>=0, with
$$\Omega_d(C)^4+2\|W\|^2\le1,\qquad \gamma^2=1-\delta\|C\|_F^2-\delta^2\|W\|^2.$$
All norms of matrices below are operator norms unless F is shown. Task blocks g,b,Kr,J,Kt are the linear functions of H in v15; f_H is the base score. L and U are extrema over the full compatible set; R=(U-L)/2.

## Claim P: perturbation of centered endpoints
For H+E versus H, each centered endpoint and R changes by at most
$$B_E=\alpha_E\sqrt\delta+(\sqrt2\|b_E\|+\|K_E^r\|)\delta+\sqrt2\|J_E\|\delta^{3/2}+\tfrac12\|K_E^t\|\delta^2,$$
where alpha_E=2 Omega_d^vee(unvec_r(g_E)).

### Strategy and dependency map
Exact budget -> exact task expansion -> uniform pointwise difference -> extrema inequalities. No differentiability of an optimizer is required.

### Proof
1. The budget gives Omega(C)<=1, ||C||_F<=1 and ||W||<=1/sqrt(2). Also 0<=gamma<=1.
2. Apply the exact expansion to E. Its cross term is bounded in absolute value by alpha_E sqrt(delta), by the definition of the dual gauge. Its next two terms are bounded by delta*(sqrt(2)||b_E||+||Kr_E||), its mixed term by sqrt(2)delta^(3/2)||J_E||, and its last term by delta^2||Kt_E||/2.
3. The resulting pointwise bound is for (F_{H+E}-f_{H+E})-(F_H-f_H); subtracting base scores prevents an intercept or base-shift ambiguity. If two continuous functions differ by at most B on one compact domain, their respective minima and maxima differ by at most B: apply both pointwise inequalities and take the relevant extremum. Halfwidth then changes by at most (B+B)/2=B. At delta=0 both centered functions are zero. This proves the claim.
4. If an ideal centered endpoint/downward loss has |D_0-c delta^p|<=e_0(delta), then |D_E-c delta^p|<=e_0+B_E. Consequently e_0+B_E<=c delta^p/4 is a sufficient 25% finite-window criterion. It is not necessary. Structural perturbations preserve an order only when the required zero blocks stay zero AND its leading block remains nonzero. A fixed nonzero perturbed g invokes the existing square-root branch, irrespective of a finite higher-order window.

## Claim S: one-sided Schur complement
Assume g=b=0 and Kr strictly positive definite. Put A=(Kr)^(-1)J^dagger and S=Kt-J(Kr)^(-1)J^dagger on x-perp. If the tangent space is zero, the base is a global minimum. Otherwise, if lambda=min eigenvalue(S)<0,
$$\Delta_H^-(\delta)=-\tfrac12\lambda\delta^2+o(\delta^2).$$
If S is PSD the lower endpoint is flat for every nonnegative tolerance.

### Strategy and dependencies
Unscaled exact budget -> completing the square -> universal lower bound and matching normalized pure-state witness. Strict positive definiteness is used only to define A and the nonnegative square.

### Proof
1. In the orthogonal decomposition span{x} plus x-perp plus residual, g=b=0 gives
$$H-hI=0\oplus\begin{pmatrix}Kt&J\\J^\dagger&Kr\end{pmatrix}.$$
Thus for any normalized pure state with tangent w and residual y the centered score is exactly
$$ (y+Aw)^\dagger Kr(y+Aw)+w^\dagger S w.$$
2. If S is PSD, this expression is nonnegative for every normalized state, not only nearby ones. The base belongs to every compatible set, so every lower endpoint equals f_H. A zero-dimensional tangent has only the Kr square and satisfies the same conclusion.
3. If lambda<0, the budget ||E_q(rho)-z_x||_F^2=Omega_d(Y)^4+2||w||^2<=delta^2 gives w^dagger S w>=lambda||w||^2>=lambda delta^2/2. Hence Delta^-<=-lambda delta^2/2.
4. Choose a unit minimum-eigenvalue vector v of S. Set w=t*v, y=-t*A*v and kappa=Omega_d(unvec_r(A*v))^4. Let
$$t^2=\frac{\delta^2}{1+\sqrt{1+\kappa\delta^2}}.$$
Then 2t^2+kappa*t^4=delta^2, including kappa=0. For all sufficiently small delta, t^2(1+||Av||^2)<=1, so u=sqrt(1-t^2(1+||Av||^2))*x+w yields a normalized pure state with the required observation budget. Its centered score is lambda*t^2, since the square vanishes.
5. Since t^2=delta^2/2+O(delta^4), the two bounds give Delta^-=-lambda delta^2/2+O(delta^4), which implies the claimed little-oh remainder. Constants depend on this fixed H,x, including ||(Kr)^(-1)||; no uniform claim is made as Kr becomes singular. This proves both cases.

## Claim G: distance to failure and Gaussian converse
Let a strictly win at the resolved-only base rho_x. Let
$$\mathcal B_a=\{\rho:\exists b\ne a,\ F_b(\rho)\ge F_a(\rho)\},\qquad r_* = \inf_{\rho\in\mathcal B_a}\|E_q(\rho)-z_x\|_F.$$
The infimum of the empty set is +infinity. Nonempty bad sets give 0<r_*<infinity and r_* is the first tolerance where the whole compatible set fails strict class-a invariance.
In the existing real D-coordinate independent Gaussian model with known sigma>0, suppose a measurable possibly randomized N-sample rule has joint false acceptance <=eta under EVERY pure state and returns a at rho_x with probability >=1-eta, for 0<eta<1/2. Then
$$N\ge 4\sigma^2[\Phi^{-1}(1-\eta)]^2/r_*^2.$$

### Strategy and dependencies
Compactness and the singleton base fiber -> nearest bad state -> two simple Gaussian hypotheses -> total variation. Combine with the existing anchored upper-bound rule for the matched small-margin conclusion.

### Proof
1. The pure-state set is compact in finite dimension. The bad set is a finite union of closed score inequalities, hence compact. If nonempty, the continuous observation distance attains a finite minimum. A zero minimum would give a bad state in the base fiber, which contains only rho_x; strict base winning excludes it. Thus r_*>0.
2. For delta<r_* no bad state is compatible. At delta=r_* a minimizing bad state is compatible, so strict invariance fails. Compactness of each nonempty compatible set and finiteness of the head imply positive pairwise lower endpoints exactly before this first failure. If the bad set is empty the class wins globally and no finite failure/lower-bound assertion is needed.
3. Let rho_* attain r_*, and A be the event that the rule returns a. P_x(A)>=1-eta. Because a is not a strict winner at rho_*, the uniform joint-error condition implies P_*(A)<=eta, even when rho_* is a tie. Therefore TV(P_x^N,P_*^N)>=1-2eta. Independent rule randomization cannot enlarge total variation: integrate its acceptance probability, a function in [0,1], against the two laws.
4. In Frobenius-orthonormal real coordinates the two N-sample laws have covariance sigma^2 I and mean separation sqrt(N)*r_*. Rotate the observation vector so that its first coordinate is the normalized mean difference. The remaining coordinates have identical laws; the likelihood-ratio event is the halfspace with first coordinate beyond the midpoint. Integrating its one-dimensional normal distributions gives
$$\operatorname{TV}=2\Phi(\sqrt N r_*/(2\sigma))-1.$$
Combining with Step 3 and the monotonicity of Phi yields the stated bound.
5. In a fixed binary task/intercept-only margin family with Delta^-(s)~c s^p, c,p>0, the existing inversion gives r_*(m)~(m/c)^(1/p). Use S=gamma*r_*(m), 0<gamma<1, in the existing anchored rule; for sufficiently small m it lies below 1/4 and the rule achieves the two requirements with N<=ceil(4sigma^2 b_{D,eta}^2/(gamma*r_*)^2), with at least one sample. The lower bound applies to every eligible rule, so at fixed D,sigma,eta the least sample order is Theta(m^(-2/p)). This is pointwise power at rho_x with uniform joint error, not uniform power, growing-dimensional optimality, an adaptive stopping theorem, or a physical RF sample law.

## Claim C: decision differences and common-mode example
All strict winner inequalities depend on H_a-H_b and intercept differences. Adding one Hermitian A to every head cancels identically. At K=1,q=1,d=2,x=e1, let A=E13+E31, B=E12+E21, H1=A+B,H2=A-B. Raw residual gradients are (1,1) and (-1,1), whose stacked rank is two. The difference gradient is (2,0), whose rank is one. The existing design rank theorem therefore requires two extra directions for both raw score activations to vanish and one for the decision difference. No finite-radius optimum follows from this rank statement.

## Claim E2: exact scalar reductions for the controlled experiments
Use an exact orthonormal triple (x,w,v), with w resolved tangent and v residual, d=1; all task blocks outside the triple are zero. Let s=|v amplitude|^2, t=|w amplitude|^2, A_delta=delta^2/2, gamma=sqrt(1-s-t). The budget is 2s^2+2t<=delta^2. Extra orthogonal components only consume this budget/normalization and cannot increase the following nonnegative downward objectives, so removing them and putting unused norm into x preserves or improves a maximum.
For H_3/2=E23+E32, the preserving/cross/tangential-linear/residual-linear losses are respectively
$$2(1+\varepsilon)\sqrt{st},\quad 2\sqrt{s}(\sqrt t+\varepsilon\gamma),\quad 2\sqrt{t}(\sqrt s+\varepsilon\gamma),\quad 2\sqrt{st}+\varepsilon s.$$
For H_2=-E22 the preserving/cross/tangential-linear/residual-linear/mixed losses are respectively
$$ (1+\varepsilon)t,\quad t+2\varepsilon\sqrt{s}\gamma,\quad t+2\varepsilon\sqrt t\gamma,\quad t+\varepsilon s,\quad t+2\varepsilon\sqrt{st}.$$
The complex phases attaining these losses can be chosen simultaneously within each (separately tested) perturbation family: flip the residual for cross, flip the tangent for tangential-linear, and choose opposite tangent/residual phases for mixed coupling. Intercepts are zero for centered losses.
For 0<=epsilon<=0.1 and 0<delta<=1/4, t<=1/32, s<=1/(4sqrt(2)), 1-s-t>0.79 and 1-s-2t>0.76. Each displayed objective is nondecreasing in t. The only derivative differences to check are sqrt(s)*(1/sqrt(t)-epsilon/gamma)>0 for the mixed cross family and 1-epsilon*sqrt(s)/gamma>0 for quadratic cross. The tangential terms have derivative proportional to 1-s-2t>0. Continuity covers t=0. Therefore t=A_delta-s^2 at a maximum and s lies in [0,delta/sqrt(2)].
The resulting profiles are concave: their nontrivial square roots involve f1=A_delta*s-s^3, f2=(1-A_delta)*s-s^2+s^3, or f3=A_delta*(1-A_delta)-A_delta*s+(2A_delta-1)*s^2+s^3-s^4. On this interval f1''<=0, f2''=-2+6s<0, and f3''=4A_delta-2+6s-12s^2<0. They are nonnegative, so their square roots are concave; A_delta-s^2 is concave. Thus derivative bracketing or supporting tangents gives certified full-interval maxima, not an unsupported local optimum. Endpoint cases are checked separately. The two unperturbed losses are exactly (2/3)^(3/4)delta^(3/2) and delta^2/2.

## Corrections or missing assumptions
None beyond the explicit domains above. The numerical implementations must still be verified against these mathematical objects.

## Open risks
The Schur remainder is not uniform near a singular Kr. Empirical RF applicability/design gains and raw-IQ reconstruction are not established by this proof package. Numerical validation is not external mathematical review.
