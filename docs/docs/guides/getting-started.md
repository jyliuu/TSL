# Getting started

## Install (Python)

TSL's core is a Rust crate; `tsl-py` builds it as a native extension via maturin during
`pip install`, so **a working Rust toolchain is required first** (Python ≥ 3.10).

```sh
# 1. install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. install TSL from GitHub (builds the Rust extension at install time)
pip install "git+https://github.com/jyliuu/TSL.git#subdirectory=tsl-py"

# optional extras:
pip install "tsl-py[plots] @ git+https://github.com/jyliuu/TSL.git#subdirectory=tsl-py"     # matplotlib + tsl_py.plot
pip install "tsl-py[examples] @ git+https://github.com/jyliuu/TSL.git#subdirectory=tsl-py"  # EBM, XGBoost, SepALS for comparisons
```

The build compiles the Rust extension at install time, so it needs a Rust toolchain
(install via [rustup](https://rustup.rs)). The core is pure Rust — no system math
libraries are required.

## First fit

`TSLRegressor` is a drop-in scikit-learn regressor:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from tsl_py.sklearn import TSLRegressor

rng = np.random.default_rng(0)
X = rng.uniform(0.0, 1.0, size=(1000, 2))
y = 2.0 * X[:, 1] + X[:, 0] - 0.5 * X[:, 0] * X[:, 1] + 3.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = TSLRegressor(epochs=3, n_trees=4, n_iter=25, seed=1)
model.fit(X_train, y_train)

print(f"Test R^2: {model.score(X_test, y_test):.4f}")
```

The lower-level `TSL.fit(...)` classmethod returns `(model, fit_result)` and takes the same
flat hyperparameters; it expects **float64** arrays. See the
[Python API](../code/python-api.md) and the
[Hyperparameters](hyperparameters.md) reference.

## Install and fit (R)

The R package **`tensorsl`** ([R API](../code/r-api.md)) compiles the same Rust core as a static
library at build time, so it too needs only a Rust toolchain (`rustc >= 1.80`, from
[rustup](https://rustup.rs)) — no system math libraries. It lives in the `tsl-r/`
subdirectory, so installers need the subdirectory:

```r
# pak (owner/repo/subdir):
pak::pak("jyliuu/TSL/tsl-r")

# remotes / devtools:
remotes::install_github("jyliuu/TSL", subdir = "tsl-r")
```

`tsl()` fits a boosted model with the same hyperparameters as the Python `TSLRegressor`, so a
fit with the same data and `seed` reproduces the Python results:

```r
library(tensorsl)

set.seed(1)
n <- 500
x <- matrix(runif(n * 3, -2, 2), ncol = 3, dimnames = list(NULL, c("a", "b", "c")))
y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(n, sd = 0.1)

fit <- tsl(x, y, epochs = 20L, seed = 42L, verbosity = 0L)
preds <- predict(fit, x)
```

`tsl_components()` extracts the fitted glass-box structure, and the `plot_*()` /
`autoplot()` helpers render the interpretability diagnostics in ggplot2. See the
[R API](../code/r-api.md) reference.

## Building from a clone (development)

This is a Cargo workspace (root crate `tsl_rust` + member `tsl-py`).

**Rust core** — always pass `--release` (the tests run numerical workloads; debug
is far too slow):

```sh
cargo build --release
cargo test -p tsl_rust --release                 # full core test suite
cargo test -p tsl_rust --release test_name        # a single test by name
```

**Python wrapper** — maturin cannot auto-detect a Python 3.14, so build against a 3.13 venv
with `VIRTUAL_ENV` set:

```sh
# from tsl-py/
VIRTUAL_ENV=/path/to/.venv /path/to/.venv/bin/maturin develop
/path/to/.venv/bin/python -m pytest python/tests/
```

`tsl-py` is an `extension-module` cdylib, so its Rust test harness can't link libpython on
Linux — exercise that crate through the Python tests, not `cargo test`.

## Next steps

- [The model](../math/model.md) — what TSL actually fits.
- [Hyperparameters](hyperparameters.md) — every knob, what it does, and where it maps.
- [Python API](../code/python-api.md) / [R API](../code/r-api.md) — the per-language reference.
- [Examples](../index.md#examples) — reproduce the paper figures.
- [Visualization dashboard](visualizing.md) — replay how a model was built with `tslviz`.
