# Official FLEX clustered inference

User-authorized extension: official `flexdid` 2.5 (06jun2026), lags-only, `vce(cluster state)`, overall observation-level ATT, followed by `margins ... vce(unconditional)`. Original Python production stored point estimates only. Comparator SHA-256 hashes are in manifests/official_comparator.json.

Let A be the pooled nuisance and treated-cell-interacted design, b its OLS coefficient, u=Y-Ab, and l the sample-average treatment contrast over treated-post observations. Let delta_i denote the fitted individual treatment effect, D_i treatment status, M=sum D_i, N sample size, K=rank(A), and G the number of states. For state g define

    coefficient_score_g = sum_{i in g} u_i A_i (A'A)^+ l
    composition_score_g = sum_{i in g} D_i (delta_i - theta) / M
    psi_g = coefficient_score_g + composition_score_g
    V(theta) = [G/(G-1)] [(N-1)/(N-K)] sum_g psi_g^2.

The second term propagates the sampled covariate and treatment-target composition used by unconditional margins. Omitting it gives a different variance. State and time nuisance parameters and post-cell regressors all contribute to K, using actual rank for redundant post-cell terms.

Native statistic is (theta-null)/sqrt(V); the official reported reference is Student t(G-1). This happens to agree with the project's jackknife df but the variances remain separate. The reference uses the full pooled design and stable pseudoinverse of its weighted design; the fast path evaluates equivalent scores from untreated state-period sufficient statistics. All reference fits use an invertible centered/scaled covariate basis for conditioning.

## Certification

Six official fixtures cover no covariates, ordinary covariates, balanced and unequal cells, and missing cells (official `usercohort()` supplied). Point, native SE, statistic and p-value are compared at full precision. See OFFICIAL_CERTIFICATION.md and official_comparison.csv.

## Weights

Point and resampling paths retain the historical positive-WLS-weight convention, including its unweighted within-cell covariate means. Native clustering is currently certified for uniform weights only. Nonuniform weights raise `InferenceError(native_nonuniform_weights_unsupported)`: official aweight unconditional margins differs from the straightforward weighted score extension, and with covariates its weighted aggregation target also differs from the historical Python target. These differences are documented, not forced into agreement. Production CPS uses no observation weights and is within the certified scope.

## Sources

- [Official flexdid distribution](https://ideas.repec.org/c/boc/bocode/s459517.html), installed ado sources, version and hashes pinned.
- [Stata margins manual](https://www.stata.com/manuals/rmargins.pdf): unconditional margins accounts for sampling variation in covariates.
- [Deb, Norton, Wooldridge and Zabel, FLEX paper](https://www.nber.org/papers/w33026), associated method reference. The numerical certification target is the pinned official software, not an inferred paper-only convention.
