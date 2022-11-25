from __future__ import annotations
import argparse
import time
from typing import Any, List, Dict, Callable
import os
from functools import partial

import numpy as np
import sklearn.model_selection
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from pandas.api.types import is_sparse

from eval_utils import METRICS, get_executer
import baselines
import pandas as pd
import openml


TOTAL_CHUNKS = 13

all_dataset_ids_numerical = [
    # 41986, 41988, 41989, 1053, 40996, 40997, 4134, 41000, 554, 41002, 44, 1590, 44089, 44090, 44091, 60, 1596, 41039, 1110, 44120, 1113, 44121, 44122, 44123, 44124, 44125, 1119, 44126, 44127, 44128, 44129, 44130, 44131, 150, 152, 153, 1181, 159, 160, 1183, 1185, 180, 1205, 182, 1209, 41146, 41147, 1212, 1214, 1216, 1218, 1219, 1222, 41671, 41162, 1226, 41163, 41164, 41166, 720, 41168, 41169, 725, 1240, 1241, 1242, 734, 735, 42206, 737, 40685, 246, 42742, 761, 250, 251, 252, 42746, 254, 42750, 256, 257, 258, 261, 266, 267, 269, 271, 279, 803, 816, 819, 823, 833, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 846, 847, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1368, 351, 357, 871, 42343, 1393, 1394, 1395, 42395, 4541, 42435, 1476, 1477, 1478, 1486, 976, 979, 40923, 23517, 1503, 43489, 1507, 42468, 42477, 41972, 1526, 41982
    # 44, 152, 153, 251, 256, 267, 269, 351, 357, 720, 725, 734, 735, 737, 761, 803, 816, 819, 823, 833, 846, 847, 871, 976, 979, 1053, 1119, 1240, 1241, 1242, 1486, 1507, 1590, 4134, 23517, 41146, 41147, 41162, 42206, 42343, 42395, 42435, 42477, 42742, 60, 150, 159, 160, 180, 182, 250, 252, 254, 261, 266, 271, 279, 554, 1110, 1113, 1183, 1185, 1209, 1214, 1222, 1226, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1368, 1393, 1394, 1395, 1476, 1477, 1478, 1503, 1526, 1596, 4541, 40685, 40923, 40996, 40997, 41000, 41002, 41039, 41163, 41164, 41166, 41168, 41169, 41671, 41972, 41982, 41986, 41988, 41989, 42468, 42746, 44089, 44090, 44091, 44120, 44121, 44122, 44123, 44124, 44125, 44126, 44127, 44128, 44129, 44130, 44131
    152, 153, 1352, 1353, 1355, 1356, 1359, 1361, 1362, 41000, 41671, 1240, 44089, 44090, 44091, 44120, 44121, 44122, 44123, 44124, 44125, 44126, 44127, 44128, 44129, 44130, 44131
]
all_dataset_ids_categorical = [
    24, 26, 154, 179, 274, 350, 720, 881, 923, 959, 981, 993, 1110, 1112, 1113, 1119, 1169, 1240, 1461, 1486, 1503, 1568, 1590, 4534, 4541, 40517, 40672, 40997, 40998, 41000, 41002, 41003, 41006, 41147, 41162, 41440, 41672, 42132, 42192, 42193, 42206, 42343, 42344, 42345, 42477, 42493, 42732, 42734, 42742, 42746, 42750, 43044, 43439, 43489, 43607, 43890, 43892, 43898, 43903, 43904, 43920, 43922, 43923, 43938, 44156, 44157, 44159, 44160, 44161, 44162, 44186
]

METHODS = {
    # hist_gradient_boosting_default
    "hist_gradient_boosting": partial(baselines.hist_gradient_boosting_metric, no_tune={}),
    # mlp
    "mlp": partial(baselines.sklearn_mlp_metric, no_tune={}),
    # decision_tree
    "decision_tree": partial(baselines.decision_tree_metric, no_tune={}),
    # random forest
    "rf": partial(baselines.random_forest_metric, no_tune={}),
    # logistic classification
    "logistic": partial(baselines.logistic_metric, no_tune={}),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datasets",
    nargs="+",
    type=int,
    help="List of test datasets",
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
    "--seeds",
    nargs="+",
    type=int,
    help="The seeds to evaluate",
    default=[3],
)
parser.add_argument(
    "--metric",
    type=str,
    choices=list(METRICS.keys())
)
parser.add_argument(
    "--result_dir",
    type=str,
    default="."
)
parser.add_argument(
    "--slurm_job_time",
    type=float,
    default=3600,
    help="Time to allot for slurm jobs. Not used for parallel execution",
)
parser.add_argument(
    "--recorded_metrics",
    type=str,
    nargs="+",
    choices=METRICS,
    help="Metrics to calculate for results",
    default=["roc", "cross_entropy", "acc", "brier_score", "ece"],
)
parser.add_argument(
    "--partition", type=str, default="bosch_cpu-cascadelake"
    )
parser.add_argument(
    "--chunk_id", type=int, default=1
    )
parser.add_argument("--slurm", action="store_true", help="Run on slurm?")
parser.add_argument("--overwrite", action="store_true", help="Overwrite previous results?")

args = parser.parse_args()


@dataclass
class Dataset:
    """Small helper class just to name entries in the loaded pickled datasets."""

    name: str
    X: np.ndarray
    y: np.ndarray
    categorical_columns: list[int]
    numerical_feats: list[int]
    # Seems to be some things about how the dataset was constructed
    info: dict
    # Only 'multiclass' is known?
    task_type: str

    @classmethod
    def fetch(
        self,
        identifier: str | int | list[int],
        seed: np.random.RandomState | None = None,
        only: Callable | None = None,
    ) -> list[Dataset]:
        if isinstance(identifier, int):
            identifier = [identifier]
            datasets = Dataset.from_openml(identifier, seed)
        elif isinstance(identifier, list):
            datasets = Dataset.from_openml(identifier, seed)
        else:
            raise ValueError(identifier)

        if only:
            return list(filter(only, datasets))
        else:
            return datasets

    @classmethod
    def from_openml(
        self,
        dataset_id: int | list[int],
        seed: np.random.RandomState | None = None,
        multiclass: bool = True,
    ) -> list[Dataset]:
        # TODO: should be parametrized, defaults taken from ipy notebook
        if not isinstance(dataset_id, list):
            dataset_id = [dataset_id]

        datasets, _ = load_openml_list(
            dataset_id,
            seed=seed
        )
        return [
            Dataset(  # type: ignore
                *entry,
                task_type="multiclass" if multiclass else "binary",
            )
            for entry in datasets
        ]

    def as_list(self) -> list:
        """How the internals expect a dataset to look like."""
        return [
            self.name,
            self.X,
            self.y,
            self.categorical_columns,
            self.numerical_feats,
            self.info,
        ]


def get_openml_classification(did):
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array",
        target=dataset.default_target_attribute
    )
    return X, y, list(np.where(categorical_indicator)[0]), list(np.where(~np.array(categorical_indicator))[0])

def load_openml_list(dids, seed):
    datasets = []
    openml_list = openml.datasets.list_datasets(dids)
    print(f'Number of datasets: {len(openml_list)}')

    datalist = pd.DataFrame.from_dict(openml_list, orient="index")

    for ds in datalist.index:
        entry = datalist.loc[ds]

        print('Loading', entry['name'], entry.did, '..')

        if entry['NumberOfClasses'] == 0.0:
            raise Exception("Regression not supported")
            #X, y, categorical_feats, attribute_names = get_openml_regression(int(entry.did), max_samples)
        else:
            X, y, categorical_feats, numerical_feats = import_open_ml_data(int(entry.did), rng=seed) # get_openml_classification(int(entry.did))


        datasets += [[entry['name'], X, y, categorical_feats, numerical_feats, None]]

    return datasets, datalist

def import_open_ml_data(openml_task_id=None, max_num_samples=10_000, rng=None) -> pd.DataFrame:
    """
    WARNING Depreciated, use import_real_data
    :param int openml_task_id:
    :param path_to_file:
    :return:
    """
    if openml_task_id is None:
        raise ValueError('Not implemented yet')

    # task = openml.tasks.get_task(openml_task_id)  # download the OpenML task
    dataset = openml.datasets.get_dataset(dataset_id=openml_task_id, download_data=False)
    # retrieve categorical data for encoding
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    categorical_indicator = np.array(categorical_indicator)
    print("{} categorical columns".format(sum(categorical_indicator)))
    print("{} columns".format(X.shape[1]))
    y_encoder = LabelEncoder()
    # remove missing values
    missing_rows_mask = X.isnull().any(axis=1)
    if sum(missing_rows_mask) > X.shape[0] / 5:
        print("Removed {} rows with missing values on {} rows".format(
            sum(missing_rows_mask), X.shape[0]))
    X = X[~missing_rows_mask]
    y = y[~missing_rows_mask]

    n_rows_non_missing = X.shape[0]
    if n_rows_non_missing == 0:
        print("Removed all rows")
        return None

    print("removing {} categorical features among {} features".format(sum(categorical_indicator), X.shape[1]))
    X = X.to_numpy()[:, ~categorical_indicator]  # remove all categorical columns
    if X.shape[1] == 0:
        print("removed all features, skipping this task")
        return None

    y = y_encoder.fit_transform(y)

    if not (max_num_samples is None):
        # max_num_samples = int(max_num_samples)
        if max_num_samples < X.shape[0]:
            indices = rng.choice(range(X.shape[0]), max_num_samples, replace=False)
            X = X[indices]
            y = y[indices]
    return X, y, [], [] #, list(np.where(categorical_indicator)[0]), list(np.where(~np.array(categorical_indicator))[0])

def evaluate_method(
    args,
    metric_function,
    method,
    dataset,
    seed,
    recording_metrics=None,
    overwrite=False,
    preprocess=True
):

    ds_name, X, y, categorical_feats, numerical_feats, _ = dataset.as_list()

    if is_sparse(X):
        print(f"Skipping {ds_name} as it is sparse")
        return None, None
    train_indices, test_indices = sklearn.model_selection.train_test_split(
            list(range(X.shape[0])),
            random_state=seed
        )
    method_func = METHODS[method]
    start_time = time.time()
    try:
        path = os.path.join(args.result_dir, f"results_{method}_{ds_name}_{seed}.npy")

        if os.path.exists(path) and not overwrite:
            result = np.load(open(path, "rb"), allow_pickle=True)
            test_score, pred, _ = result
        else:

            test_score, pred, _ = method_func(
                X[train_indices],
                y[train_indices],
                X[test_indices],
                y[test_indices],
                categorical_feats,
                numerical_feats,
                metric_used=metric_function,
                seed=seed,
                max_time=0,
                preprocess=preprocess)
            test_score = test_score.item()
            total_time = time.time()-start_time
            np.save(open(path, "wb"), (test_score, pred, total_time))
    except Exception as e:
        # raise e
        print(repr(e))
        return None, None
    
    metric_dict = {}
    if recording_metrics is not None:
        for metric in recording_metrics:
            print(f"Calculating metric: {metric}")
            metric_func = METRICS[metric]
            score = metric_func(y[test_indices], pred)
            metric_dict[metric] = score.item() if hasattr(score, 'item') else score
    else:
        metric_dict["test_score"] =  test_score
    metric_dict["time"] = time.time()-start_time
    print(f"Result: score: {metric_dict} and time: {time.time()-start_time}")
    return {"scores": metric_dict, "pred": pred}


if __name__ == '__main__':
    if args.datasets is None:
        dataset_ids =  all_dataset_ids_numerical # all_dataset_ids_categorical +
    else:
        dataset_ids = args.datasets 

    metric_function = METRICS[args.metric]

    os.makedirs(args.result_dir, exist_ok=True)


    # total_datasets = len(dataset_ids)
    # chunk_size = int(total_datasets / TOTAL_CHUNKS)
    # start_index = args.chunk_id * chunk_size
    # end_index = (args.chunk_id + 1) * chunk_size
    # if end_index > len(dataset_ids):
    #     datasets = Dataset.fetch(dataset_ids[start_index:])
    # else:
    #     datasets = Dataset.fetch(dataset_ids[start_index:end_index])

    dataset_id_to_name = {}
    results = {}
    for method in args.methods:
        results[method] = {}
        for dataset_id in dataset_ids:
            # dataset_name = dataset.name
            results[method][dataset_id] = {}
            for seed in args.seeds:
                rng = np.random.RandomState(seed)
                dataset = Dataset.fetch(dataset_id, rng)[0]
                if dataset_id not in dataset_id_to_name:
                    dataset_id_to_name[dataset_id] = dataset.name
                print(f"Running: {method}, for seed: {seed}, on {dataset.name}")
                if args.slurm:
                    print("Running ensemble on slurm")
                    log_folder = os.path.join(args.result_dir, "log_test")
                    slurm_executer = get_executer(
                        partition=args.partition,
                        log_folder=log_folder,
                        total_job_time_secs=args.slurm_job_time,
                        gpu=False
                        )
                    try:
                        result = slurm_executer.submit(
                            evaluate_method,
                            args,
                            metric_function,
                            method,
                            dataset,
                            seed,
                            recording_metrics=args.recorded_metrics,
                            overwrite=args.overwrite,
                            preprocess=False
                        )
                        print(f"Started job with job_id: {result.job_id}")
                    except Exception as e:
                        print(repr(e))
                        result = None
                else:
                    result = evaluate_method(
                        args=args,
                        metric_function=metric_function,
                        method=method,
                        dataset=dataset,
                        seed=seed,
                        recording_metrics=args.recorded_metrics,
                        overwrite=args.overwrite,
                        preprocess=False)
                results[method][dataset_id][seed] = result
                del dataset

   
    # result.df.to_csv(os.path.join(args.result_path, "results.csv"), index=True)
    dataset_names = set([dataset for dataset in dataset_id_to_name.values()])
    index = pd.MultiIndex.from_product(
        [args.methods, args.seeds],
        names=[
            "method",
            "seed",
        ],
    )

    metrics = args.recorded_metrics + ["time"]

    columns = pd.MultiIndex.from_product(
        [metrics, dataset_names],
        names=["metric", "dataset"],
    )

    df = pd.DataFrame(columns=columns, index=index)
    df.sort_index(inplace=True)
    print_string = "all jobs successfully submitted" if args.slurm else "All results ran successfully"
    print(print_string)
    for method in results:
        for dataset_id in results[method]:
            for seed in results[method][dataset_id]:
                row = (method, seed)
                result = results[method][dataset_id][seed]
                dataset_name = dataset_id_to_name[dataset_id]
                if args.slurm:
                    print(f"Waiting for the result odf {method}, for seed: {seed}, on {dataset_name}")
                    result = result.result()
                else:
                    result = result
                for metric in result["scores"]:
                    df.loc[row, (metric, dataset_name)] = result["scores"][metric]

    df.to_csv(os.path.join(args.result_dir, f"result.csv"))

# seeds = 545, 385, 287, 721, 834
# python run_sklearn.py --result_dir '/work/dlclarge2/rkohli-run_gael_benchmark/selected' --metric acc --methods decision_tree --seeds 545 --slurm