from __future__ import annotations

from typing import Any, Dict, List
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)
import os
from copy import deepcopy

import torch

from eval_utils import Dataset, Results, METRICS, arguments, do_evaluations_parallel, get_executer, do_evaluations
from eval_utils import Dataset, Results, arguments, do_evaluations_ensemble
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
            try:
                individual_result = item.result()
                sum_aggregate_metric += individual_result['sum_aggregate_metric']
                new_item = {**new_item, **individual_result}
            except Exception as e:
                print(f"Failed for {key} with {repr(e)}")
   
        new_item.pop('sum_aggregate_metric', None)
        new_item['mean_metric'] = sum_aggregate_metric / ((i+1)*chunk_size)
        final_results[key] = new_item

    return final_results

if __name__ == "__main__":
    args = arguments()

    print(args)

    # We need to create some directories for this to work
    out_dir = os.path.join(args.result_path, "results", "tabular", "multiclass") # , f"{time.time()}")
    os.makedirs(out_dir, exist_ok=True)

    # We ignore the flags datasets
    filter_f = lambda d: d.name != "flags"  # noqa: ignore

    if args.datasets is None:
        valid_datasets = Dataset.fetch("cc_valid", only=filter_f, subsample_flag=args.subsample)
        test_datasets = Dataset.fetch("cc_test", only=filter_f, subsample_flag=args.subsample)
        all_datasets = valid_datasets + test_datasets
    else:
        all_datasets = Dataset.fetch(args.datasets, only=filter_f, subsample_flag=args.subsample)

    data_file_suffix = None
    if args.datasets is None:
        data_file_suffix = 'test_valid'
    elif isinstance(args.datasets, str):
        data_file_suffix = args.datasets
    elif isinstance(args.datasets, (int, list)):
        data_file_suffix = "custom_dids"+str(args.datasets)

    if args.subsample:
        data_file_suffix += "_subsample"
    log_folder = os.path.join(args.result_path, "log_test/")

    all_methods = deepcopy(args.methods)
    for method in all_methods:
        args.methods = [method]
        if args.ensemble:
            # assumes that args.methods passed provides baseline methods to combine with 1 tabpfn classifier
            # can be done locally.
            if args.slurm:
                print("Running ensemble on slurm")
                slurm_executer = get_executer(
                    partition=args.partition,
                    log_folder=log_folder,
                    total_job_time_secs=args.slurm_job_time,
                    gpu=args.gpu
                    )
                job = slurm_executer.submit(do_evaluations_ensemble, args, all_datasets)
                print(f"Started job with job_id: {job.job_id}")
                result = job.result()
            else:
                result = do_evaluations_ensemble(args, all_datasets)

        else:
            if args.slurm:
                if args.parallel:
                    # runs each split, method on "args.chunk_size" datasets as paralle jobs. 
                    jobs = do_evaluations_parallel(args, all_datasets, log_folder=log_folder)
                    for experiment_key, chunked_job in jobs.items():
                        for i, single_job in enumerate(chunked_job):
                            print(f"Running chunk: {i} for {experiment_key} with job_id: {single_job.job_id}")
                    result = post_process_chunks_result(args.chunk_size, result=jobs)
                else:
                    # submits only 1 job with all methods and datasets and splits
                    # should only be used for small experiments or collecting results. 
                    # Max time hardcoded to 1 hr.
                    assert not args.overwrite, f"Passed 'overwrite' flag. For running method on new splits please use parallel execution"
                    if not args.fetch_only:
                        warnings.warn(f"Not passed 'fetch_only' flag. For running method on new splits please use parallel execution")
                    slurm_executer = get_executer(
                        partition=args.partition,
                        log_folder=log_folder,
                        total_job_time_secs=args.slurm_job_time,  # 1 hr to collect the results
                        gpu=args.gpu
                        )
                    job = slurm_executer.submit(do_evaluations, args, all_datasets)
                    print(f"Started job with job_id: {job.job_id}")
                    result = job.result()

            else:
                # running locally
                result = do_evaluations(args, all_datasets)

        # Calculate metrics for results
        result = calculate_metrics(
            datasets=all_datasets,
            recorded_metrics=args.recorded_metrics,
            eval_positions=args.eval_positions,
            results=result)

        result.df.to_csv(os.path.join(args.result_path, f"results_{method}_{data_file_suffix}.csv"), index=True)
