from tsl_py._tsl_py import TSL, GridTensor, StagePredictor, FitResult
from tsl_py.sklearn import TSLRegressor

__all__ = ["TSL", "GridTensor", "StagePredictor", "FitResult", "TSLRegressor"]


def __getattr__(name):
    """Lazy import of the optional plot subpackage so importing tsl_py
    does not require matplotlib."""
    if name == "plot":
        import importlib
        mod = importlib.import_module("tsl_py.plot")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'tsl_py' has no attribute {name!r}")
