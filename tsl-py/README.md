# tensorsl

Tensor Separation Learning (TSL) — a glass-box regression model. A fitted model is a sum of
*stages*, where each stage is the ordered difference of two non-negative rank-1 products of
univariate functions. This keeps the model inspectable while still capturing interactions
between features.

`tensorsl` is the Python package (imported as `tsl_py`): a `fit`/`predict` estimator that
slots into typical ML pipelines, backed by a Rust core.

## Install

```bash
pip install tensorsl
```

The package builds from source on install for now, so you need a Rust toolchain and a BLAS
library (e.g. OpenBLAS) available. Prebuilt wheels are planned.

## Usage

```python
import numpy as np
from tsl_py import TSLRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 5))
y = X[:, 0] * X[:, 1]

model = TSLRegressor().fit(X, y)
preds = model.predict(X)
```

The fitted model also exposes per-feature partial dependence and feature importances for
interpretation.

## Links

- Documentation: https://jyliuu.github.io/TSL/
- Source: https://github.com/jyliuu/TSL

## License

MIT
