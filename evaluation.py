from __future__ import annotations

from argparse import Namespace
from itertools import product
from pathlib import Path
from typing import Any, Dict, List
import warnings 

import numpy as np

import pickle

import os

import torch

<<<<<<< HEAD
from eval_utils import Dataset, Results, METRICS, arguments, do_evaluations_parallel, get_executer, do_evaluations
=======
from eval_utils import Dataset, Results, arguments, do_evaluations_ensemble, DEFAULT_SEED, HERE, METHODS, METRICS, eval_method, set_seed
>>>>>>> 9856fcf (code pasted in colab)
from tabpfn.scripts.tabular_metrics import (calculate_score, time_metric)


def calculate_metrics(
    datasets: List[Dataset],
    recorded_metrics: List[str],
    eval_positions: List[int],
    results: Dict[str, Any]
):
    datasets_as_lists = [d.as_list() for d in datasets]

    # This will update the results in place
    for metric in recorded_metrics:
        metric_f = METRICS[metric]
        calculate_score(
            metric=metric_f,
            name=metric,
            global_results=results,
            ds=datasets_as_lists,
            eval_positions=eval_positions,
        )

    # We also get the times
    calculate_score(
        metric=time_metric,
        name="time",
        global_results=results,
        ds=datasets_as_lists,
        eval_positions=eval_positions,
    )

    return Results.from_dict(
        results,
        datasets=datasets,
        recorded_metrics=recorded_metrics + ["time"],
    )


def post_process_chunks_result(
    chunk_size: int,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merges chunked results. 

    Args:
        chunk_size (int): size of chunk, i.e, number of datasets in a chunk
        result (Dict[str, Any]): chunked results

    Returns:
        Dict[str, Any]: _description_
    """
    final_results = {}
    for key in result:
        final_results[key] = []
        new_item = {}
        sum_aggregate_metric = torch.tensor(0.0)
        for i, item in enumerate(result[key]):
            individual_result = item.result()
            sum_aggregate_metric += individual_result['sum_aggregate_metric']
            new_item = {**new_item, **individual_result}
        new_item.pop('sum_aggregate_metric', None)
        new_item['mean_metric'] = sum_aggregate_metric / ((i+1)*chunk_size)
        final_results[key] = new_item

    return final_results

if __name__ == "__main__":
    args = arguments()

    print(args)

    if not args.validation_datasets:
        args.validation_datasets = "cc_valid"
    elif len(args.validation_datasets) == 1 and args.validation_datasets[-1] < 0:
        args.validation_datasets = None

    if not args.test_datasets:
        args.test_datasets = "cc_test"
    elif len(args.test_datasets) == 1 and args.test_datasets[-1] < 0:
        args.test_datasets = None

    # We need to create some directories for this to work
    out_dir = os.path.join(args.result_path, "results", "tabular", "multiclass") # , f"{time.time()}")
    os.makedirs(out_dir, exist_ok=True
    )

    # We ignore the flags datasets
    filter_f = lambda d: d.name != "flags"  # noqa: ignore

    valid_datasets = []
    test_datasets = []
    if args.validation_datasets is not None:
        valid_datasets = Dataset.fetch(args.validation_datasets, only=filter_f)
    if args.test_datasets is not None:
        test_datasets = Dataset.fetch(args.test_datasets, only=filter_f)

    all_datasets = valid_datasets + test_datasets
    all_datasets = all_datasets
<<<<<<< HEAD
    log_folder = os.path.join(args.result_path, "log_test/")
    
    if not args.load_predefined_results:
        if args.slurm:
            if args.parallel:
                result = do_evaluations_parallel(args, all_datasets, log_folder=log_folder)
                result = post_process_chunks_result(args.chunk_size, result=result)
            else:
                assert not args.overwrite, f"Passed 'overwrite' flag. For running method on new splits please use parallel execution"
                if not args.fetch_only:
                    warnings.warn(f"Not passed 'fetch_only' flag. For running method on new splits please use parallel execution")
                slurm_executer = get_executer(
                    partition=args.partition,
                    log_folder=log_folder,
                    total_job_time_secs=3600,  # 1 hr to collect the results
                    gpu=args.gpu
                    )
                job = slurm_executer.submit(do_evaluations, args, all_datasets)
                result = job.result()
        else:
            result = do_evaluations(args, all_datasets)

        # Calculate metrics for results

        result = calculate_metrics(
            datasets=all_datasets,
            recorded_metrics=args.recorded_metrics,
            eval_positions=args.eval_positions,
            results=result)
    else:
=======
    # base_path = os.path.join('/work/dlclarge1/rkohli-results_tabpfn_180/results_1667931216')

    if args.ensemble:
        result = do_evaluations_ensemble(args, all_datasets)
    # print(args.result_path)
    # if not args.load_predefined_results:
    #     result = do_evaluations_slurm(args, all_datasets, slurm=args.slurm, chunk_size=args.chunk_size)
    # else:
>>>>>>> 9856fcf (code pasted in colab)

    #     def read(_path: Path) -> dict:
    #         with _path.open("rb") as f:
    #             return pickle.load(f)

    #     d = {
    #         path.stem: read(path)
    #         for path in args.predefined_results_path.iterdir()
    #         if path.is_file()
    #     }
    #     result = Results.from_dict(
    #         d,
    #         datasets=all_datasets,
    #         recorded_metrics=args.recorded_metrics,
    #     )

<<<<<<< HEAD
    result.df.to_csv(os.path.join(args.result_path, "results.csv"), index=True)
=======
    # # Post processing as the results are currently Dict[key, List[Dict]] make them Dict[key, Dict]
    # final_results = post_process_chunks_result(args, result)

    # datasets_as_lists = [d.as_list() for d in all_datasets]

    # # This will update the results in place
    # for metric in args.recorded_metrics:
    #     metric_f = METRICS[metric]
    #     calculate_score(
    #         metric=metric_f,
    #         name=metric,
    #         global_results=final_results,
    #         ds=datasets_as_lists,
    #         eval_positions=args.eval_positions,
    #     )

    # # We also get the times
    # calculate_score(
    #     metric=time_metric,
    #     name="time",
    #     global_results=final_results,
    #     ds=datasets_as_lists,
    #     eval_positions=args.eval_positions,
    # )
    # final_results = Results.from_dict(
    #         final_results,
    #         datasets=all_datasets,
    #         recorded_metrics=args.recorded_metrics + ["time"],
    #     )
    # final_results.df.to_csv(os.path.join(out_dir, "results.csv"), index=True)

    result.df.to_csv(os.path.join(args.result_path, "predefined_results.csv"), index=True)
>>>>>>> 9856fcf (code pasted in colab)
