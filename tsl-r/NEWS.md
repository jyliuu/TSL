# tslr 0.1.0

* First release: R bindings for the `tsl` (Tensor Separation Learning) Rust
  crate via extendr.
* `tsl(x, y, ...)` fits a boosted TSL model; `predict()` and `print()` S3
  methods are provided. Hyperparameters mirror the Python `TSLRegressor`.
* `tsl_components()` extracts the fitted glass-box structure: per stage, the
  OLS scalings and the aggregated and per-tree grid tensors in two-tensor form
  (per-feature backbone, tilt, splits, and the branch scalars).
* Installs from GitHub via `pak::pak("jyliuu/TSL/tsl-r")` /
  `remotes::install_github("jyliuu/TSL", subdir = "tsl-r")`. The core is pure
  Rust, so no system numerical libraries are required.

## Known follow-ups

* **CRAN.** The package is structurally CRAN-ready (vendoring placeholders in
  `src/Makevars.in`), but submission still needs the crates vendored offline
  into `src/rust/vendor.tar.xz` (including the `tsl_rust` git dependency) and
  build-time tuning.
* Windows and macOS CI are not yet exercised (CI covers Linux).
* Higher-level interpretability (partial dependence, ICE, feature importance)
  and model serialisation are not yet exposed, though `tsl_components()` already
  gives access to the raw fitted components.
