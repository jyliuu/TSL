"""Benchmark cost-complexity pruning on a fixed train/test split.

Run from the repository root after installing the local extension with maturin.
The script emits one JSON record per model seed followed by an aggregate record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from tensorsl import TSL, __version__ as tensorsl_version


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y_true
    squared_error = error * error
    return {
        "rmse": float(np.mean(squared_error) ** 0.5),
        "mae": float(np.mean(np.abs(error))),
        "r2": 1.0
        - float(np.sum(squared_error) / np.sum((y_true - np.mean(y_true)) ** 2)),
    }


def model_split_counts(model: TSL) -> tuple[int, int]:
    primary = sum(
        len(axis)
        for stage in model.stage_predictors
        for axis in stage.combined_grid_tensor.splits
    )
    bags = sum(
        len(axis)
        for stage in model.stage_predictors
        for grid in stage.grid_tensors
        for axis in grid.splits
    )
    return primary, bags


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_bytes(repo_root: Path, *args: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def benchmark_provenance(repo_root: Path) -> dict[str, object]:
    status = git_output(repo_root, "status", "--porcelain")
    tracked_patch = git_bytes(repo_root, "diff", "--binary", "HEAD", "--")
    tracked_files = git_output(repo_root, "diff", "--name-only", "HEAD", "--")
    return {
        "git": {
            "revision": git_output(repo_root, "rev-parse", "HEAD"),
            "branch": git_output(repo_root, "branch", "--show-current"),
            "dirty": None if status is None else bool(status),
            "tracked_patch_sha256": (
                None if tracked_patch is None else hashlib.sha256(tracked_patch).hexdigest()
            ),
            "tracked_modified_files": (
                None if tracked_files is None else tracked_files.splitlines()
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tensorsl": tensorsl_version,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "threading": {
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "logical_cpus": os.cpu_count(),
        },
        "benchmark_script_sha256": file_sha256(Path(__file__)),
    }


def summarize(
    records: list[dict[str, object]],
    config: dict[str, object],
    provenance: dict[str, object],
) -> dict[str, object]:
    numeric_paths = {
        "seconds": lambda row: row["seconds"],
        "primary_splits": lambda row: row["primary_splits"],
        "bag_splits": lambda row: row["bag_splits"],
        "train_rmse": lambda row: row["train"]["rmse"],
        "test_rmse": lambda row: row["test"]["rmse"],
        "test_mae": lambda row: row["test"]["mae"],
        "test_r2": lambda row: row["test"]["r2"],
    }
    metrics = {}
    for name, getter in numeric_paths.items():
        values = np.asarray([getter(row) for row in records], dtype=np.float64)
        metrics[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return {
        "kind": "summary",
        "dataset": records[0]["dataset"],
        "complexity_penalty": records[0]["complexity_penalty"],
        "seeds": [row["seed"] for row in records],
        "config": config,
        "provenance": provenance,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Headerless CSV; target is the last column")
    parser.add_argument("--name", default=None, help="Dataset label in JSON output")
    parser.add_argument("--complexity-penalty", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n-trees", type=int, default=16)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--split-try", type=int, default=16)
    parser.add_argument("--min-interval-samples", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    data = np.loadtxt(args.data, delimiter=",")
    x = np.ascontiguousarray(data[:, :-1])
    y = np.ascontiguousarray(data[:, -1])
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.split_seed,
    )
    dataset_name = args.name or args.data.stem
    model_params = {
        "epochs": args.epochs,
        "decay": 1.0,
        "n_trees": args.n_trees,
        "n_iter": args.n_iter,
        "split_try": args.split_try,
        "colsample_bytree": 1.0,
        "alpha": 0.0,
        "complexity_penalty": args.complexity_penalty,
        "min_split_loss": 0.0,
        "min_interval_samples": args.min_interval_samples,
        "refinement_strategy": "l2",
        "prior_sample_size": 0.0,
        "update_clamp": float("inf"),
        "tilt_tau": 0.01,
        "tilt_rho": 0.0,
        "split_strategy": "random",
        "top_k": 10,
        "must_fill_all_k": True,
        "similarity_threshold": 0.0,
        "bagged": False,
        "verbosity": 0,
        "visualdb": None,
    }
    model_config = {**model_params, "update_clamp": "infinity"}
    config = {
        "dataset": {
            "name": dataset_name,
            "path": str(args.data),
            "sha256": file_sha256(args.data),
            "size_bytes": args.data.stat().st_size,
            "rows": len(y),
            "features": x.shape[1],
        },
        "split": {"seed": args.split_seed, "test_size": args.test_size},
        "model": model_config,
        "seeds": args.seeds,
    }
    provenance = benchmark_provenance(repo_root)

    records = []
    for seed in args.seeds:
        started = time.perf_counter()
        model, fit_result = TSL.fit(
            x_train,
            y_train,
            seed=seed,
            **model_params,
        )
        elapsed = time.perf_counter() - started
        primary_splits, bag_splits = model_split_counts(model)
        record = {
            "kind": "run",
            "dataset": dataset_name,
            "rows": len(y),
            "features": x.shape[1],
            "seed": seed,
            "complexity_penalty": args.complexity_penalty,
            "seconds": elapsed,
            "fit_mse": float(fit_result.err),
            "train": regression_metrics(y_train, np.asarray(model.predict(x_train))),
            "test": regression_metrics(y_test, np.asarray(model.predict(x_test))),
            "primary_splits": primary_splits,
            "bag_splits": bag_splits,
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    print(json.dumps(summarize(records, config, provenance), sort_keys=True))


if __name__ == "__main__":
    main()
