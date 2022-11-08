from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np


import tabpfn.scripts.tabular_baselines as tb
from tabpfn.scripts.tabular_metrics import (calculate_score, count_metric,
                                            time_metric)

from eval_utils import Dataset, DEFAULT_SEED, HERE, METHODS, METRICS, N_SPLITS, eval_method, set_seed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path",
        type=Path,
        help="Where the results path is",
        default=HERE,
    )
    parser.add_argument("--gpu", action="store_true", help="GPU's available?")
    parser.add_argument(
        "--times",
        nargs="+",
        type=int,
        default=[30],
        help="Times to evaluate (seconds)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[DEFAULT_SEED],
        help="Seeds to evaluate on",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="The number of splits to evaluate",
    )

    parser.add_argument(
        "--validation_datasets",
        nargs="+",
        type=int,
        help="The validation datasets",
    )
    parser.add_argument(
        "--test_datasets",
        nargs="+",
        type=int,
        help="The test datasets",
    )

    parser.add_argument(
        "--optimization_metrics",
        type=str,
        choices=METRICS,
        help="Metrics to optimize for (if possible)",
        default=["roc"],
    )
    parser.add_argument(
        "--result_metrics",
        type=str,
        nargs="+",
        choices=METRICS,
        help="Metrics to calculate for results",
        default=["roc", "cross_entropy", "acc", "brier_score", "ece"],
    )

    parser.add_argument(
        "--methods",
        choices=METHODS.keys(),
        nargs="+",
        type=str,
        help="The methods to evaluate",
        default=["svm_default"],
    )
    parser.add_argument(
        "--fetch_only",
        action="store_true",
        help="Whether to only fetch results and not run anything",
    )

    # Transformer args
    parser.add_argument(
        "--bptt",
        type=int,
        help="Transformer sequence length",
        default=2000,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite results if they already exist",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plots", type=Path, help="Where to output plots to")  

    args = parser.parse_args()

    if not args.validation_datasets:
        args.validation_datasets = "cc_valid"

    if not args.test_datasets:
        args.test_datasets = "cc_test"

    # We need to create some directories for this to work
    (args.result_path / "results" / "tabular" / "multiclass").mkdir(
        parents=True, exist_ok=True
    )

    # We ignore the flags datasets
    filter_f = lambda d: d.name != "flags"  # noqa: ignore

    valid_datasets = Dataset.fetch(args.validation_datasets, only=filter_f)
    test_datasets = Dataset.fetch(args.test_datasets, only=filter_f)
    all_datasets = valid_datasets + test_datasets
    all_datasets = all_datasets[:2]

    results = {}
    for seed, method, metric, time in product(
        args.seeds,
        args.methods,
        args.optimization_metrics,
        args.times,
    ):
        metric_f = METRICS[metric]
        metric_name = tb.get_scoring_string(metric_f, usage="")

        key = f"{method}_time_{time}_{metric_name}"

        set_seed(seed=seed)

        for split_id in range(args.splits):
            key += f"_split_{split_id}"
            results[key] = eval_method(
                datasets=all_datasets,
                label=method,
                result_path=args.result_path,
                classifier_evaluator=METHODS[method],
                eval_positions=[1_000],  # It's a constant basically
                fetch_only=args.fetch_only,
                verbose=args.verbose,
                max_time=time,
                metric_used=metric_f,
                split_id=split_id,
                overwrite=args.overwrite,
                seed=seed,
                bptt=args.bptt,
            )

    all_datasets_as_lists = [d.as_list() for d in all_datasets]
    # This will update the results in place
    for metric in args.result_metrics:
        metric_f = METRICS[metric]
        calculate_score(
            metric=metric_f,
            name=metric,
            global_results=results,
            ds=all_datasets_as_lists,
            eval_positions=[1_000] + [-1],
        )

    # We also do some other little bits
    for agg in ["sum", "mean"]:
        calculate_score(
            metric=time_metric,
            name="time",
            global_results=results,
            ds=all_datasets_as_lists,
            eval_positions=[1_000] + [-1],
            aggregator=agg,
        )

    calculate_score(
        metric=count_metric,
        name="count",
        global_results=results,
        ds=all_datasets_as_lists,
        eval_positions=[1_000] + [-1],
        aggregator="sum",
    )

    # print(results)