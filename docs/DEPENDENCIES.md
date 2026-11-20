# Runtime dependencies

Python >=3.10 and NumPy >=1.24. An isolated Python import test blocked pandas, SciPy, statsmodels, linearmodels and imports from the other estimator repositories; point, native cluster and jackknife calls passed. pandas and official Stata flexdid are optional validation dependencies only. Runtime code performs no file I/O. Normal Python bytecode caching may be disabled with PYTHONDONTWRITEBYTECODE=1, as the Monte Carlo runner does.
