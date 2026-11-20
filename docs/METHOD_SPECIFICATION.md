# Historical FLEX point specification

## Explicit grouping modes

The public `grouping` argument distinguishes two official FLEX specifications.
`grouping="state"` is the backward-compatible default described below.
`grouping="cohort"` replaces the state nuisance and treatment groups with first-
treatment cohorts (zero for never treated), matching `group(cohort)`. Thus states
adopting together share group fixed effects, group-covariate slopes, and post-
treatment cohort-time coefficients. Calendar-time terms remain common. Cohort
targets use observation-weighted covariate means when observation weights are
supplied. Cohort mode requires never-treated observations and rejects always-
treated schedules through the general assignment checks.

The cohort fast path aggregates cached state-period cross-products to cohort-
period blocks for each assignment. This preserves state-cluster deletions and
assignment reshuffles without creating observation-level dummy matrices. Native
clustered standard errors for cohort grouping are outside the current certified
scope and fail explicitly.

Derived directly from the archived `fast_flex.py` and its import dependencies. This is a source specification, not a claim of new paper or native-inference certification.

## Sample and timing

Use complete observations for outcome, state, time and supplied covariates; the historical optional-weight path additionally retains positive weights. Each state has an explicit first-treatment year g(s), with zero denoting never treated. D(s,t)=1 when g(s)>0 and t>=g(s). The historical production call used no observation weights and included age, educ, ofcolor and marstat in every DGP.

## Pooled regression

This is a single pooled least-squares regression. Its nuisance design F contains an intercept, additive state and time indicators, ordinary X, state-by-X interactions, and time-by-X interactions. Omitted baseline groups avoid redundant nuisance columns. For each observed treated-post state-time cell c, append Z_c=1{(s,t)=c} and every Z_c X column. Only post-treatment cells are included: no treatment leads.

With A=[F,Z,ZX], estimate b=argmin sum_i w_i (Y_i-A_i b)^2, where w_i=1 in historical production. The historical reference uses `numpy.linalg.lstsq` with its documented source settings. Finite rank-deficient solutions are flagged and retained by the old runner; this is not a blanket guarantee of estimand identification.

The coefficient on Z_c plus the unweighted observed within-cell mean of X times the Z_c X coefficients defines ATT_c. Aggregate to cohort-time, cohort, and finally overall ATT with target weights N_c. With production `weighting="both"`, these nested weighted means equal sum_c N_c ATT_c / sum_c N_c. Covariate effects are allowed to vary by state, time, and treated-post cell.

## Comparison observations and estimand

Never-treated and not-yet-treated observations constrain the nuisance surface. Saturated post-cell terms identify deviations from that surface. The target is an observation-mass weighted mean over treated-post cells, with no forced alignment to another estimator's chosen aggregation. The covariate nuisance specification differs from an additive common-slope BJS first stage; the post-cell covariate terms matter as well.

For the optional weighted archival path, regression and target masses use weights, but within-cell X means are unweighted. Its separately returned `native_att` uses observation-weighted effect averages, which can differ from the saved canonical `att`. Production uses the latter. Weighted acceptance must preserve this distinction rather than silently changing it.

The archival `base_target_N` is looked up in counts built only from treated-post rows, so it is zero. Consequently its `contrast_weight` equals target mass for these inputs. This behavior is recorded, not repaired without understanding its consequences.

## Missing cells and failure

Only observed treated-post cells enter the target. No missing cell is synthesized. The old routine reports no-treated-post and rank-deficiency diagnostics. New inference must require all state deletions and explicitly reject undefined deletion estimands; no such inference is implemented in this archival stage.

## Intended inference scope

Native clustered inference now reproduces official flexdid unconditional state-clustered linearization after user authorization. Jackknife uses Student t(G-1), RI preserves state timing counts, Replace holds Y fixed, and Subtract removes a fixed full-sample scalar effect before placebo assignment. Arbitrary heterogeneous-effect validity of Subtract is not established. Add is N/A.

