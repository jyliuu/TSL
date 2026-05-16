#!/usr/bin/env python3
"""
Script to fetch OpenML datasets and save them as X, y format.

This script fetches datasets from OpenML and saves them as combined CSVs
(features + target, last column is the target) into the script's directory
by default.

You can fetch:
- All datasets (or filtered by task type)
- Datasets from a specific task collection (study) ID
- Specific datasets by their IDs
"""

import argparse
import os
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    # Fallback: create a no-op tqdm
    class tqdm:
        def __init__(self, iterable=None, desc=None, **kwargs):
            self.iterable = iterable
            self.desc = desc

        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @staticmethod
        def write(s):
            print(s)


def preprocess_openml_data(X, y):
    """
    Preprocess OpenML data to handle categorical variables.
    Uses TargetEncoder for categorical variables.

    Args:
        X: Feature data (DataFrame or array-like)
        y: Target data (Series or array-like)

    Returns:
        Tuple of (X_processed, y_processed) as numpy arrays
    """
    # Convert to DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Detect feature types
    categorical_features = list(X.select_dtypes(include=["category", "object"]).columns)
    numerical_features = list(X.select_dtypes(include=["number"]).columns)
    boolean_features = list(X.select_dtypes(include=["bool"]).columns)

    # If no categorical features, return as-is (but include boolean columns)
    if len(categorical_features) == 0:
        # Convert boolean to int and combine with numerical
        X_combined = X.copy()
        for col in boolean_features:
            X_combined[col] = X_combined[col].astype(int)
        return X_combined.values, y.astype(float).values

    # Use TargetEncoder for categorical variables, passthrough for numerical and boolean
    transformers = []
    if len(numerical_features) > 0:
        transformers.append(("num", "passthrough", numerical_features))
    if len(boolean_features) > 0:
        transformers.append(("bool", "passthrough", boolean_features))
    if len(categorical_features) > 0:
        transformers.append(("cat", TargetEncoder(), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    X_processed = preprocessor.fit_transform(X, y)

    # Convert to dense array if sparse
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Check for NaN values
    if np.isnan(X_processed).any():
        raise ValueError(
            "Input data contains NaN values after preprocessing. Please clean the data before proceeding."
        )

    return X_processed, y.astype(float).values


def fetch_and_save_dataset(data_id, output_dir, min_samples=None, max_samples=None):
    """
    Fetch a single OpenML dataset and save it as X, y format.

    Parameters
    ----------
    data_id : int
        OpenML dataset ID
    output_dir : str
        Directory to save the dataset
    min_samples : int, optional
        Minimum number of samples to include dataset
    max_samples : int, optional
        Maximum number of samples to include dataset

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Fetch dataset using openml.datasets.get_dataset (same as run_cluster_experiment.py)
        dataset = openml.datasets.get_dataset(data_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

        # Check sample size constraints
        n_samples = len(X)
        if min_samples is not None and n_samples < min_samples:
            return False, f"Skipped: {n_samples} samples < {min_samples} minimum"
        if max_samples is not None and n_samples > max_samples:
            return False, f"Skipped: {n_samples} samples > {max_samples} maximum"

        # Preprocess the data (handles categorical variables, converts to numpy arrays)
        try:
            X_array, y_array = preprocess_openml_data(X, y)
        except Exception as e:
            return False, f"Preprocessing failed: {str(e)}"

        # Create filename base
        dataset_name = dataset.name
        # Sanitize filename
        safe_name = "".join(
            c for c in dataset_name if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        safe_name = safe_name.replace(" ", "_")
        filename_base = f"{data_id}_{safe_name}"

        # Save as combined CSV (X + y, last column is y)
        combined = np.column_stack([X_array, y_array])
        output_path = os.path.join(output_dir, f"{filename_base}.csv")
        np.savetxt(output_path, combined, delimiter=",", fmt="%.8f")

        return True, f"Saved: {n_samples} samples, {X_array.shape[1]} features"

    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Fetch all OpenML datasets and save as X, y format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default: the script's own directory)",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["regression", "classification", "both"],
        default="both",
        help="Filter by task type (default: both)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        help="Minimum number of samples to include",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to include",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of datasets to fetch (for testing)",
    )
    parser.add_argument(
        "--data-ids",
        type=str,
        help="Comma-separated list of specific dataset IDs to fetch",
    )
    parser.add_argument(
        "--task-collection-id",
        type=int,
        help="OpenML task collection (study) ID to fetch all datasets from (e.g., 353)",
    )

    args = parser.parse_args()

    # Create output directory (relative to script location if relative path)
    if os.path.isabs(args.output_dir):
        output_dir = Path(args.output_dir)
    else:
        # Relative to script directory
        script_dir = Path(__file__).parent
        output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")

    # Get list of datasets
    if args.task_collection_id:
        # Fetch datasets from a task collection (study)
        print(f"Fetching task collection (study) ID: {args.task_collection_id}...")
        try:
            # Try using list_tasks with study_id filter first (faster and more reliable)
            print("Fetching tasks from OpenML (this may take a moment)...")
            import time

            start_time = time.time()

            # Method 1: Try list_tasks with study_id filter
            try:
                print("Attempting to list tasks by study ID...")
                tasks_df = openml.tasks.list_tasks(
                    output_format="dataframe", study_id=args.task_collection_id
                )

                if tasks_df is not None and len(tasks_df) > 0:
                    task_ids = tasks_df["tid"].tolist()
                    elapsed = time.time() - start_time
                    print(
                        f"Found {len(task_ids)} tasks in collection (took {elapsed:.1f}s)"
                    )
                else:
                    raise ValueError("No tasks found via list_tasks")
            except Exception as e1:
                # Method 2: Fallback to get_suite
                print(f"list_tasks failed ({e1}), trying get_suite...")
                start_time = time.time()
                try:
                    suite = openml.study.get_suite(args.task_collection_id)
                    elapsed = time.time() - start_time
                    print(f"Suite fetched! (took {elapsed:.1f}s)")
                    task_ids = suite.tasks
                    print(f"Found {len(task_ids)} tasks in collection")
                except Exception as e2:
                    # Method 3: Try get_study then get_suite
                    print(
                        f"get_suite failed ({e2}), trying get_study then get_suite..."
                    )
                    start_time = time.time()
                    study = openml.study.get_study(args.task_collection_id)
                    elapsed = time.time() - start_time
                    print(f"Study fetched! (took {elapsed:.1f}s)")
                    suite = study.get_suite()
                    task_ids = suite.tasks
                    print(f"Found {len(task_ids)} tasks in collection")

            # Extract unique dataset IDs from tasks
            data_ids = []
            processed_datasets = set()
            for task_id in task_ids:
                try:
                    task = openml.tasks.get_task(task_id)
                    dataset_id = task.dataset_id
                    if dataset_id not in processed_datasets:
                        data_ids.append(dataset_id)
                        processed_datasets.add(dataset_id)
                except Exception as e:
                    print(f"Warning: Could not get dataset from task {task_id}: {e}")
                    continue

            print(f"Found {len(data_ids)} unique datasets in collection")
        except Exception as e:
            print(f"Error fetching task collection: {e}")
            import traceback

            traceback.print_exc()
            return
    elif args.data_ids:
        # Fetch specific datasets
        data_ids = [int(id.strip()) for id in args.data_ids.split(",")]
        print(f"Fetching {len(data_ids)} specified datasets...")
    else:
        # List all datasets
        print("Fetching list of all OpenML datasets...")
        datasets_df = openml.datasets.list_datasets(output_format="dataframe")

        # Filter by task type if specified
        if args.task_type != "both":
            if args.task_type == "regression":
                datasets_df = datasets_df[datasets_df["NumberOfClasses"] == 0]
            elif args.task_type == "classification":
                datasets_df = datasets_df[datasets_df["NumberOfClasses"] > 0]

        # Get dataset IDs
        data_ids = datasets_df["did"].tolist()

        if args.limit:
            data_ids = data_ids[: args.limit]
            print(f"Limited to first {args.limit} datasets")

        print(f"Found {len(data_ids)} datasets to fetch")

    # Fetch and save datasets
    successful = 0
    failed = 0
    skipped = 0

    iterator = tqdm(data_ids, desc="Fetching datasets") if HAS_TQDM else data_ids
    for data_id in iterator:
        success, message = fetch_and_save_dataset(
            data_id,
            str(output_dir),
            min_samples=args.min_samples,
            max_samples=args.max_samples,
        )

        if success:
            successful += 1
            tqdm.write(f"[{data_id}] ✓ {message}")
        else:
            if "Skipped" in message:
                skipped += 1
            else:
                failed += 1
            tqdm.write(f"[{data_id}] ✗ {message}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(data_ids)}")
    print(f"  Output directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
