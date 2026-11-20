# FLEX stress tests

Release-candidate rerun, 19 November 2026: 30 deterministic designs, comprising
27 full comparison passes, one identified-margin check with a redundant slope,
and two expected failures (deleting the last treated school; no treated schools).
Zero unexplained failures. A disappearing singleton cohort is not itself a
failure when another estimable treatment effect remains. Designs include
balanced/unequal samples, missing cells, one and two cohorts, 2/2/46,
heterogeneous and zero-average effects, covariates, weighted point/resampling,
shuffled rows, strings and nonconsecutive labels.

Each passing design compares full/reference points, components, weights, all deletions, two fixed RI assignments, and both placebo inner p-value arrays. Heterogeneous designs are numerical diagnostics, not proof of Subtract validity. Native nonuniform weights are explicitly N/A. See stress_test_results.csv and certification_summary.json.

Eight separate processes reproduced serial point/native/JK/RI/Replace/Subtract results exactly. Fixed assignment generation separately tested all small support and 999 unique 2/2/46 draws. Five unittest methods cover these parameterized cases.

New test runs write machine-specific numerical diagnostics to ignored
`output/tests/` (or `FASTFLEX_TEST_OUTPUT`), preserving the tracked release record.
