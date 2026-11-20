# Historical FLEX provenance



Inspected 2026-09-08T00:26:13.677802+00:00.



## Identified production point implementation



Authoritative readable source: `<historical-local-path>/fast_flex.py`.

`run_expanded_didint_pilot.py` imports `estimators.fast_flex.estimate_flex`; `run_bjs_or_flex` calls it with covariates `age`, `educ`, `ofcolor`, `marstat`, `weighting="both"`, and `return_components=True`. The omitted specification argument defaults to `lags_only`. No observation-weight column is supplied.

The saved scalar is `result.att`, the canonical ATT(s,t)-first aggregation with target mass N(s,t), rather than an independently selected FLEX variant. Four covariates are included even for the no-X outcome DGP.

`run_metadata.json`, `report.md`, the saved output labels, and readable imports agree on this path. The metadata has no Git hash, so the archived current source is a reconstruction target whose numerical identity must be tested against saved replications. The source header still says provisional, whereas the later run report describes point certification. Neither statement establishes clustered-inference certification.

All source hashes, modification times, alternatives, and existing repository states are in the adjacent manifests. Original research files remain untouched. Bytecode is provenance evidence only.



## Historical design



Prepared inputs contain 4,959 observations, years 2000 through 2013, early states 15 and 22 treated in 2004, late states 23 and 54 treated in 2009, and never-treated state 21. Saved disturbances permit direct replay without generating new Monte Carlo replications. The folder is named 10000rep_production, while its metadata purpose calls it diagnostic/preproduction; this inconsistency is retained rather than hidden.



## Native clustered inference: resolved extension



The specified Python production runner does not save a clustered SE, statistic, or p-value. Its result object contains an aggregation dictionary with a dispersion-based SE helper, but this is not a state-clustered regression sandwich: the final calculation uses dispersion of cohort aggregates, M/(M-1), and a normal reference. The runner does not export these helper inference values.

The separate Stata point-certification wrapper uses `flexdid ..., specification(lagsonly) vce(robust)` and saves only the point estimate. Observation-robust inference cannot establish a state-clustered native convention.

Older `Unpool-RA/DID-INT simulations with CPS Data/Code/flexmaker.ado` and the August 2026 copy call `areg ..., vce(cluster state)` (through the cluster argument), followed by `margins`, but include both lead and lag cell terms. They return only `flexcoeff`. This is a different specification from the requested lags-only Python production path; its inference must not silently be grafted onto that path as historical fact.

At the provenance stage the user was asked to clarify the clustered-inference target. The user then authorized original official clustering. The implementation, certification, seven-estimator integration, and release tagging have now been completed for the documented uniform-weight scope.



## Preservation



Historical FLEX and BJS source copies are retained in the separate research archive. The released package is self-contained and does not import that archive.



## Scope



The planned FLEX inference API is native_cluster, jackknife, RI, Replace, and Subtract. Add is N/A. No production Monte Carlo has been launched.

## Saved replay evidence

Six archived fits reproduced stored scalar estimates to 3.56e-12 and 180 components to 1.45e-11. See HISTORICAL_REPLAY.md. This resolves point-source identity, but cannot establish clustered inference absent from the saved outputs.

## Authorized clustered extension

The user subsequently instructed: "Yes, add their original clustering". This authorizes official flexdid state clustering with unconditional margins for the identified lags-only point estimator. This is explicitly a new inference feature, not a claim that the old Python production files stored clustered results. Version and hashes are in manifests/official_comparator.json.

