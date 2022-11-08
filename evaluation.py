from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

import pickle

import os

import time

from eval_utils import Dataset, Results, arguments, do_evaluations_slurm, DEFAULT_SEED, HERE, METHODS, METRICS, eval_method, set_seed
from tabpfn.scripts.tabular_metrics import (calculate_score, time_metric)

if __name__ == "__main__":
    args = arguments()

    if not args.validation_datasets:
        args.validation_datasets = "cc_valid"

    if not args.test_datasets:
        args.test_datasets = "cc_test"

    # We need to create some directories for this to work
    out_dir = os.path.join(args.result_path, "results", "tabular", "multiclass", f"{time.time()}")
    os.makedirs(out_dir, exist_ok=True
    )

    # We ignore the flags datasets
    filter_f = lambda d: d.name != "flags"  # noqa: ignore

    valid_datasets = Dataset.fetch(args.validation_datasets, only=filter_f)
    test_datasets = Dataset.fetch(args.test_datasets, only=filter_f)

    all_datasets = valid_datasets + test_datasets
    all_datasets = all_datasets
    # base_path = os.path.join('/work/dlclarge1/rkohli-results_tabpfn_180/results_1667931216')

    # print(args.result_path)
    if not args.load_predefined_results:
        result, jobs = do_evaluations_slurm(args, all_datasets, slurm=True)
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

    for key in jobs:
        result[key] = jobs[key].result()

    datasets_as_lists = [d.as_list() for d in all_datasets]

    # This will update the results in place
    for metric in args.recorded_metrics:
        metric_f = METRICS[metric]
        calculate_score(
            metric=metric_f,
            name=metric,
            global_results=result,
            ds=datasets_as_lists,
            eval_positions=args.eval_positions,
        )

    # We also get the times
    calculate_score(
        metric=time_metric,
        name="time",
        global_results=result,
        ds=datasets_as_lists,
        eval_positions=args.eval_positions,
    )
    final_results = Results.from_dict(
            result,
            datasets=all_datasets,
            recorded_metrics=args.recorded_metrics + ["time"],
        )
    final_results.df.to_csv(os.path.join(out_dir, "results.csv"), index=True)
    # result.df.to_csv(os.path.join(args.result_path, "results.csv"), index=True)