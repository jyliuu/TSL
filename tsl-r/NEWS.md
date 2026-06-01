# tslr 0.1.0

* First release: R bindings for the `tsl` (Tensor Separation Learning) Rust
  crate via extendr.
* `tsl(x, y, ...)` fits a boosted TSL model; `predict()` and `print()` S3
  methods are provided. Hyperparameters mirror the Python `TSLRegressor`.
* Installs from GitHub via `pak::pak("jyliuu/TSL/tsl-r")` /
  `remotes::install_github("jyliuu/TSL", subdir = "tsl-r")`. The core is pure
  Rust, so no system numerical libraries are required.

## Known follow-ups

* **CRAN.** The package is structurally CRAN-ready (vendoring placeholders in
  `src/Makevars.in`), but submission still needs the crates vendored offline
  into `src/rust/vendor.tar.xz` (including the `tsl_rust` git dependency) and
  build-time tuning.
* Windows and macOS CI are not yet exercised (CI covers Linux).
* Interpretability (partial dependence, ICE, feature importance) and model
  serialisation are not yet exposed.
