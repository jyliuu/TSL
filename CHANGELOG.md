# Changelog

All notable changes to TSL are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com) and the project adheres to
[Semantic Versioning](https://semver.org).

## [0.6.0] - 2026-10-14

### Breaking changes

- Bagged fits (`n_trees > 1`, `consensus=0`) fit each grid to its own bootstrap
  resample of the rows by default, in Rust and Python, so fitted models differ from
  0.5.0. Pass `bootstrap=False` (Rust: `.bootstrap(false)`) to fit every grid to all
  rows (#52)
- `TSLRegressor` and `TSL.fit` default to `max_bins=4095` quantile bins per feature
  instead of every distinct value, so default Python fits differ from 0.5.0. Pass
  `max_bins="exact"` for every distinct value; `TSL.fit(max_bins=None)` still selects
  it. The Rust builders keep every distinct value by default (#54)

### Features

- **stage:** Bagged grids fit bootstrap resamples again (`bootstrap`), restoring the
  row resampling of bagging (#52)
- **forest:** Fits take per-row sample weights: `fit_boosted_weighted` in Rust and
  `sample_weight=` on `TSLRegressor.fit` and `TSL.fit` (#52)
- **tsl-py:** `TSLRegressor` accepts `max_bins`: an integer of at least 2 for quantile
  bins, or `"exact"` (#54)

### Performance

- Exact-candidate fits are about 1.4× faster (33 s → 24 s on the California housing
  test configuration) with bit-identical models (#53)
- At the 4095-bin default, Python fits are 1.2–5× faster than with exact candidates
  on eight OpenML datasets, with similar accuracy (#54)

### Documentation

- Sample weights and bootstrap resampling, with a bootstrap section in the bagging
  math page (#52)
- The hyperparameters guide explains how `max_bins` picks split candidates (#54)

## [0.5.0] - 2026-10-12

### Breaking changes

- `colsample_bytree` is renamed `colsample_bysplit` in `TSLRegressor`, `TSL.fit`,
  `GridTensor.fit` and R `tsl()`, and defaults to 1.0. Each random split proposal draws
  that fraction of the features that still have split positions; in earlier releases
  values below 1 had almost no effect, so 1.0 keeps default fits as they were (#48, #50)
- `prior_sample_size` and `bagged` are removed from `TSLRegressor`, `TSL.fit` and R
  `tsl()`; neither affected a fit (#48, #50)
- Fitted models can differ from 0.4.0: histogram bin edges snap to distinct-value
  boundaries and rank-deficient two-tensor systems take bounded descent steps (#46)
- Rust API: `FittingState` drops `interval_id` and stores the curvature statistics as one
  struct per row, and `FittingAction` has only `ApplySplit`, `ApplyResplit` and
  `ApplyMerge` (#46, #47)

### Features

- **stage:** Bagged grids can fit toward a lockstep branch consensus (`consensus`,
  `consensus_interval`, `consensus_anchors`) with nonnegative stage calibration
  (`nonnegative_scaling`); `TSL.fit` accepts `max_bins` (#46)
- **r:** The R package builds on the current core (#50)

### Performance

- Consensus fits are 2.5× faster across six OpenML datasets, with bit-identical models (#46)
- Partial dependence and ICE are about 7× and 4× faster (#47)

### Bug Fixes

- `colsample_bysplit` subsamples features on every split proposal (#48)
- Model floats round-trip exactly through JSON (#46)

### Refactor

- The fitting code is split into modules of one responsibility each, and unit tests sit
  inline as general behavior checks (#47)

## [0.4.0] - 2026-08-03

### Features

- **grid:** Add cost-complexity structural selection (#38)

## [0.3.0] - 2026-06-03

### Documentation

- Align README and docs with the code, link the docs site (#24)

### Refactor

- **tsl-py:** Rename import package `tsl_py` to `tensorsl` — the import name now
  matches the `tensorsl` PyPI distribution; `import tsl_py` no longer works

## [0.2.0] - 2026-06-03

- Add R wrapper `tensorsl` with S3 interface and ggplot2 interpretability layer

## [0.1.3] - 2026-06-02

### Features

- **plot:** Complete plotting overhaul (#22)
## [0.1.1] - 2026-06-02

### Refactor

- Solve boosting OLS with a pure-Rust SVD, dropping OpenBLAS (#19)
## [0.1.0] - 2026-06-01

### Bug Fixes

- Handle non-contiguous arrays in fit and predict (#6)
- **tsl-py:** Use a package-local README for the sdist (#17)
- **tsl-py:** Keep package metadata free of direct-URL dependencies (#18)

### Documentation

- Polish readme
- Documentation overhaul (README, CLAUDE.md, CONTRIBUTING) (#4)
- Enlarge readme logo and example figures (#5)
- MkDocs documentation site + GitHub Pages deployment (#7)
