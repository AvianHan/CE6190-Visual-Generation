# Why refresh placement and constraints matter

This analysis supports the speed–fidelity comparison. The proofs are conditional or elementary constructions, not a proof that one named method dominates another.

## 1. A budgeted decision problem

At sampling step $k$, choose $a_k\in\{0,1\}$: compute the block-stack residual when $a_k=1$, otherwise reuse the cached residual. The output head and sampler still run. With a latency budget $B$, the empirical question is

$$\min_\pi\;\mathbb E[d(x_\pi,x_{\rm full})]
\quad\text{subject to}\quad \mathbb E[T_\pi]\le B.$$

Here $\pi$ is a compute/reuse policy, and $d$ can be LPIPS to the paired full-computation output. This objective describes fidelity; prompt adherence and VBench dimensions must be evaluated separately. Our finite threshold grids estimate trade-offs, not the global optimum.

## 2. Equal computation counts do not imply equal error

**Proposition.** Two schedules with identical compute counts, protected endpoints, and maximum reuse length can have different final errors. Their ranking can reverse with the residual trajectory.

**Proof by construction.** Use four scalar Euler updates of size $1/4$, a zero stem and an identity head, so the block residual is the velocity. Reuse means holding the last computed velocity. Compare:

| Step | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Schedule A | compute | compute | reuse | compute |
| Schedule B | compute | reuse | compute | compute |

Both compute three times and reuse once. If the true velocities are $(0,1,1,1)$, the full output is $3/4$: A is exact, while B gives $1/2$. If velocities are $(0,0,1,1)$, the full output is $1/2$: B is exact, while A gives $1/4$. Thus compute count alone cannot determine error. Equal operation costs also give equal runtime in this toy model. This full-velocity-reuse example is a special case of residual caching, not a fitted model of Wan or FLUX.

**Connection to results.** Near 2.45× on FLUX, Tea/Mag/Sen all compute eleven of 28 stages, yet mean LPIPS differs. The counterexample explains why such a difference is possible; it does not establish that placement alone causes the measured difference or that early refreshes always help.

## 3. Local cache error can propagate through later updates

Let $\widetilde z_k$ and $z_k$ be cached and full-computation trajectories from identical initial noise. At the *same cached state*, define

$$\delta_k=\widetilde v(\widetilde z_k,t_k)-v(\widetilde z_k,t_k).$$

If the output head is $L^H_k$-Lipschitz and cached residual error is $e^r_k$, then $\|\delta_k\|\le L^H_k\|e^r_k\|$. For Wan's two-branch CFG, replace this bound by

$$d_k=|1-s|L^H_{k,u}\|e^r_{k,u}\|+|s|L^H_{k,c}\|e^r_{k,c}\|.$$

This follows by the triangle inequality. It is a worst-case bound, not measured CFG amplification; FLUX's embedded guidance does not use this two-forward expression.

For an Euler sampler with increment $h_k$ and an $L_k$-Lipschitz full velocity field, subtracting the cached and full updates gives

$$E_{k+1}\le(1+|h_k|L_k)E_k+|h_k|d_k,\qquad E_0=0.$$

Repeated substitution proves

$$\boxed{E_N\le\sum_{k=0}^{N-1}|h_k|\!\prod_{j=k+1}^{N-1}(1+|h_j|L_j)\,d_k.}$$

A refresh removes the current local cache approximation; it does not undo an already changed latent. The bound motivates accounting for both approximation size and its position. It does not prove that early steps are always more important: step sizes, local errors, contraction and directional cancellation can change the result.

**Actual sampler boundary.** FLUX uses FlowMatch Euler; Wan uses UniPC. For a multistep solver, include stored predictions/history in a state $y_k$. If its full update is $A_k$-Lipschitz in that state and its response to prediction perturbation is bounded by $B_k\|\delta_k\|$, then

$$\|\widetilde y_{k+1}-y_{k+1}\|\le A_k\|\widetilde y_k-y_k\|+B_k\|\delta_k\|.$$

The same induction applies with $A_j,B_k$. We have not estimated these constants or certified UniPC error in our implementation. A latent-error bound is not a guarantee on LPIPS, semantics or VBench.

## 4. A conditional Sen-FLUX plateau certificate

**Counting lemma.** If the first step computes and at most $K$ consecutive reuses are allowed, $M$ compute steps cover at most $M(K+1)$ stages. Hence

$$M\ge\lceil N/(K+1)\rceil.$$

If the first $g$ steps additionally compute, write $N-g=q(K+1)+r$, $0\le r\le K$. At least $q$ further computes are needed; the pattern $C^g(R^KC)^qR^r$ attains

$$\boxed{M_{\min}=g+\lfloor(N-g)/(K+1)\rfloor.}$$

All 200 delivered Sen-FLUX threshold-2 traces have $N=28$, $K=3$, observed $g=6$, and $M=11$, attaining this conditional bound. The first six steps use a strict threshold; **Sen does not hard-code them as computes**. The K-only bound is seven, not eleven. Likewise, Wan's $N=50,K=3$ only implies thirteen, not its observed seventeen.

**Threshold interval.** Fix a recorded deterministic trajectory and all settings except the main threshold. Let $L$ be the largest logged error at a post-prefix reuse, and $U$ the smallest error at a post-prefix refresh not forced by the cap (or infinity if none). Any $\epsilon\in(L,U]$ preserves all decisions: identical prior decisions give identical states/errors; reuse inequalities remain true, unforced refresh inequalities remain false, and cap-triggered refreshes remain forced. Induction proves the complete path stays unchanged.

Our read-only audit finds 3,400 post-prefix reuses and 1,000 cap-triggered refreshes, with no unforced refresh. Thus $L=1.606399434983263$, $U=\infty$. Using a margin such as $\epsilon\ge1.607$ avoids quoting a rounded floating-point boundary. This is a conditional inference for the delivered 200 inputs, **not a new threshold experiment** or a guarantee on unseen prompts. K/start threshold/profile linkage is reconstructed from delivered provenance; the original per-image JSON did not bind all these settings at runtime.

Evidence: [audit output](../analysis/sen_plateau_audit.json), [full proof and assumptions](sen_plateau_proof.md). Reproduce without models:

```bash
python scripts/audit_sen_plateau.py results/drawbench_flux/sen_plateau_records.zip --output analysis/sen_plateau_audit.json
```

## 5. Denoising speedup is not complete-generation speedup

Let $A>0$ be unchanged non-denoising time, $D$ baseline denoising time, and $r$ the denoising speedup including cache-decision overhead. Then

$$S(r)=\frac{A+D}{A+D/r},\qquad \lim_{r\to\infty}S(r)=\frac{A+D}{A}.$$

This follows directly from adding runtime components (the standard Amdahl argument). In the A100 main experiment, $T_0\approx271.33$ s and $D\approx172.74$ s imply $A\approx98.59$ s and a **conditional** ceiling of about 2.75×. Sea's 2.06× denoising speedup becomes 1.49× complete-generation speedup. Changing the decoder, hardware, timing boundary or other retained work changes this ceiling. The remainder is not assumed to be entirely VAE time.

## 6. Improvements motivated by this analysis

- **Detect saturation before widening thresholds.** Once every refresh is cap-triggered, changing the main threshold cannot improve this fixed batch. A K or early-threshold change is a separate, untested quality trade-off.
- **Compare adaptive decisions with schedule replay.** Replaying the certified schedule preserves this batch under the proof's assumptions and could avoid gate calculations. Runtime benefit is unmeasured; reuse on unseen inputs needs new validation.
- **Weight error proxies by sampling-stage influence.** The propagation bound motivates a stage-dependent threshold. Practical proxies are not certified local-error bounds, so this remains a proposed extension. SenCache already discusses dynamic thresholds; we do not claim this general idea as a new method.

## Sources

The proofs above are self-contained. The primary papers supply method context:

- [SenCache, §4.3–4.5 and §6](https://arxiv.org/html/2602.24208v1): sensitivity estimates, reuse cap and threshold extensions.
- [SeaCache, §4.1–4.2](https://arxiv.org/html/2602.18993v2): spectral decision proxy.
- [UniPC, §3 and Appendix B](https://arxiv.org/html/2302.04867v4): multistep predictor–corrector context.
- [Amdahl (1967)](https://doi.org/10.1145/1465482.1465560): fixed-fraction speedup.

Measured values come from the archived main results and the delivered FLUX records, not paper result tables.
