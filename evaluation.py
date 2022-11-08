from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

import pickle

import os

from eval_utils import Dataset, Results, arguments, do_evaluations, DEFAULT_SEED, HERE, METHODS, METRICS, eval_method, set_seed


if __name__ == "__main__":
    args = arguments()

    if not args.validation_datasets:
        args.validation_datasets = "cc_valid"

    if not args.test_datasets:
        args.test_datasets = "cc_test"

    # We need to create some directories for this to work
    out_dir = os.path.join(args.result_path, "results", "tabular", "multiclass")
    os.mkdir(out_dir,
        parents=True, exist_ok=True
    )

    # We ignore the flags datasets
    filter_f = lambda d: d.name != "flags"  # noqa: ignore

    valid_datasets = Dataset.fetch(args.validation_datasets, only=filter_f)
    test_datasets = Dataset.fetch(args.test_datasets, only=filter_f)

    all_datasets = valid_datasets + test_datasets
    all_datasets = all_datasets

    if not args.load_predefined_results:
        result = do_evaluations(args, all_datasets)
    else:

        def read(_path: Path) -> dict:
            with _path.open("rb") as f:
                return pickle.load(f)

        d = {
            path.stem: read(path)
            for path in args.predefined_results_path.iterdir()
            if path.is_file()
        }
        result = Results.from_dict(
            d,
            datasets=all_datasets,
            recorded_metrics=args.recorded_metrics,
        )

    result.df.to_csv(os.path.join(out_dir, "results.csv"), index=True)