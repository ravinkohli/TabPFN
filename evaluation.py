from __future__ import annotations

from argparse import Namespace
from itertools import product
from pathlib import Path
from typing import Any, Dict

import numpy as np

import pickle

import os

import time
import torch

from eval_utils import Dataset, Results, METRICS, arguments, do_evaluations_parallel, get_executer, get_executer_params, do_evaluations
from tabpfn.scripts.tabular_metrics import (calculate_score, time_metric)



def post_process_chunks_result(
    chunk_size: int,
    result: Dict[str, Any],
    slurm: bool = False
) -> Dict[str, Any]:
    """
    Merges chunked results. 

    Args:
        chunk_size (int): size of chunk, i.e, number of datasets in a chunk
        result (Dict[str, Any]): chunked results
        slurm (bool, optional): Whether results are to be fetched from slurm or not. Defaults to False.

    Returns:
        Dict[str, Any]: _description_
    """
    final_results = {}
    for key in result:
        final_results[key] = []
        new_item = {}
        sum_aggregate_metric = torch.tensor(0.0)
        for i, item in enumerate(result[key]):
            individual_result = item.result() if slurm else item
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
    log_folder = os.path.join(args.result_path, "log_test/")
    
    if not args.load_predefined_results:
        if args.slurm:
            if args.parallel:
                result = do_evaluations_parallel(args, all_datasets, log_folder=log_folder, chunk_size=args.chunk_size)
            else:
                # slurm expects time in minutes. 
                total_job_time = 60
                slurm_executer = get_executer(args.partition)(folder=log_folder)
                slurm_executer.update_parameters(**get_executer_params(total_job_time, args.partition, args.gpu)
                                    )
                job = slurm_executer.submit(do_evaluations, args, all_datasets)
                result = job.result()
        else:
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

    result.df.to_csv(os.path.join(args.result_path, "results.csv"), index=True)