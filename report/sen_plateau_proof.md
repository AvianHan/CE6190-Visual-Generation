# A conditional plateau certificate for the delivered Sen-FLUX runs

This is an analysis of existing records, not a new generation experiment.

## Implementation and units

The delivered `src/flux_cache.py`, `SenGate.decide`, reuses a cached residual
exactly when a cache exists, the estimated error is strictly below the active
threshold, and the number of preceding consecutive reuses is less than K.
The log key `sen_errors` stores the estimate divided by the positive latent
scale. Thus it can be compared directly with the unscaled thresholds.
For 28 sampling steps the switch is `round(28*0.2)=6`; the first six steps
use threshold_start=0.045 and subsequent steps use the main threshold.
This is NOT a hard rule that the first six steps must compute. In contrast,
the project's `fixed_schedule` explicitly forces its first six steps.

K=3 permits THREE consecutive reuses; a maximally skipped cycle has FOUR
steps: C R R R. K is not the cycle length. FLUX has one network call per
sampling step. Wan's paired conditional/unconditional calls must instead be
counted as one scheduling decision per sampling step, with two block-stack
computations on a refresh.

## Conditional count bound

Let N be the number of sampling steps and C the number of refresh steps.
If the first step computes and no reuse run exceeds K,

    C >= ceil(N/(K+1)).

Every compute can support at most K following reuses, so
N <= (K+1)C. Repeating C R^K and truncating achieves the structural bound.
Threshold comparisons can force extra computes, so a particular model need
not attain it.

If the first g steps additionally compute, the tighter conditional bound is

    C >= g + floor((N-g)/(K+1)).

After the prefix, write N-g=q(K+1)+r with 0<=r<=K. The sequence
C^g (R^K C)^q R^r attains the bound. For necessity, with c extra computes
the remaining interval has at most c+1 reuse runs: N-g <= c+(c+1)K.

For the audited FLUX traces, N=28, K=3 and the observed prefix has g=6,
so the conditional minimum is 6+floor(22/4)=11. All 200 runs attain it:

    1111110001000100010001000100  (1=compute, 0=reuse)

Without the observed prefix, the unconditional K-only bound is 7, not 11.
For Wan at N=50 and K=3 it is 13, not the observed 17; the Wan early phase
uses a different length (10 sampling steps under the paired-call convention).

## Schedule-preserving threshold interval

Fix one recorded reference execution, including inputs, initial random noise,
weights, calibrated sensitivities, early threshold, K and implementation.
Assume forward/sampler computations are deterministic and the main threshold
affects only the gate comparison.

On the reference path after the switch, let R contain reuse events and U
contain refresh events where the skip cap/other hard guards did not force
the refresh. Let e_i be the normalized logged error. Define

    L = max(e_i for i in R),
    U_bound = min(e_i for i in U), or +infinity if U is empty.

Any new main threshold epsilon in (L, U_bound] preserves the reference path.
Proof by induction: identical prior decisions give identical state, cached
features and sensitivities, and hence identical e_i. Reference reuse events
still satisfy e_i<epsilon. Unguarded refresh events still satisfy
e_i>=epsilon. Cap-forced refreshes remain forced. Thus the next decision and
state agree. The first six steps agree because their threshold is unchanged.

The audit verifies every logged decision using the declared K=3 and early
threshold0.045. Across all 200 threshold-2 records:

- Every first-six-step decision is compute; the noninitial early normalized
  errors are at least 0.35405504423704093, above0.045.
- Every post-prefix refresh occurs after three consecutive reuses.
- There are no unguarded post-prefix refreshes, so U_bound=+infinity.
- The maximum error on all post-prefix reuse events is
  L=1.606399434983263 (`db_070`, step9).

Therefore epsilon>1.606399434983263 is a conditional, record-derived plateau
interval for this batch. The lower endpoint is OPEN because the source uses
a strict less-than comparison. This conclusion does not require all errors
to be below2: errors at already cap-forced refreshes may exceed2.

The interval is stated in the normalized arithmetic of the mathematical
gate. The implementation multiplies epsilon by the latent scale and logs
error divided by that scale; floating-point rounding matters arbitrarily
close to the displayed endpoint. A practical certified choice should retain
a positive numerical margin, for example epsilon>=1.607, rather than reuse
the rounded endpoint itself. The original epsilon=2 has margin0.3936005650.

## Limits of the claim

The certificate is for these 200 prompts and seed0, not unseen inputs or
different profiles, resolutions, samplers, K values, or early thresholds.
Original per-image JSONs omit K, the starting threshold, and the runtime
profile hash. Those settings are supplied by the delivered source/provenance;
their historical per-run binding remains retrospective. No altered-threshold
generation was run. Identical scheduling does not establish good quality,
optimal scheduling, general threshold monotonicity, or universal early-stage
importance. Byte-identical outputs require the stated deterministic execution
assumption, rather than following from matching decision strings alone.

To reproduce the read-only audit, run `scripts/audit_sen_plateau.py results/drawbench_flux/sen_plateau_records.zip
--output REPORT_JSON`. It uses Python's standard library and reads the
original ZIP without extraction or modification.
