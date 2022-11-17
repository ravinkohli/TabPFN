from __future__ import annotations
import argparse
import time
from itertools import product
from typing import Callable
import os
import numpy as np
import sklearn.model_selection
from dataclasses import dataclass
from pandas.api.types import is_sparse

from eval_utils import METRICS, get_executer
import baselines
import pandas as pd
import openml
from functools import partial


TOTAL_CHUNKS = 13

all_dataset_ids_numerical = [
    # 41986, 41988, 41989, 1053, 40996, 40997, 4134, 41000, 554, 41002, 44, 1590, 44089, 44090, 44091, 60, 1596, 41039, 1110, 44120, 1113, 44121, 44122, 44123, 44124, 44125, 1119, 44126, 44127, 44128, 44129, 44130, 44131, 150, 152, 153, 1181, 159, 160, 1183, 1185, 180, 1205, 182, 1209, 41146, 41147, 1212, 1214, 1216, 1218, 1219, 1222, 41671, 41162, 1226, 41163, 41164, 41166, 720, 41168, 41169, 725, 1240, 1241, 1242, 734, 735, 42206, 737, 40685, 246, 42742, 761, 250, 251, 252, 42746, 254, 42750, 256, 257, 258, 261, 266, 267, 269, 271, 279, 803, 816, 819, 823, 833, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 846, 847, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1368, 351, 357, 871, 42343, 1393, 1394, 1395, 42395, 4541, 42435, 1476, 1477, 1478, 1486, 976, 979, 40923, 23517, 1503, 43489, 1507, 42468, 42477, 41972, 1526, 41982
    # 44, 152, 153, 251, 256, 267, 269, 351, 357, 720, 725, 734, 735, 737, 761, 803, 816, 819, 823, 833, 846, 847, 871, 976, 979, 1053, 1119, 1240, 1241, 1242, 1486, 1507, 1590, 4134, 23517, 41146, 41147, 41162, 42206, 42343, 42395, 42435, 42477, 42742, 60, 150, 159, 160, 180, 182, 250, 252, 254, 261, 266, 271, 279, 554, 1110, 1113, 1183, 1185, 1209, 1214, 1222, 1226, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1368, 1393, 1394, 1395, 1476, 1477, 1478, 1503, 1526, 1596, 4541, 40685, 40923, 40996, 40997, 41000, 41002, 41039, 41163, 41164, 41166, 41168, 41169, 41671, 41972, 41982, 41986, 41988, 41989, 42468, 42746, 44089, 44090, 44091, 44120, 44121, 44122, 44123, 44124, 44125, 44126, 44127, 44128, 44129, 44130, 44131
    152, 153, 1240, 1352, 1353, 1355, 1356, 1359, 1361, 1362, 41000, 41671
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
    "--partition", type=str, default="bosch_cpu-cascadelake"
    )
parser.add_argument(
    "--chunk_id", type=int, default=1
    )
parser.add_argument("--slurm", action="store_true", help="Run on slurm?")

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
        only: Callable | None = None,
    ) -> list[Dataset]:
        if isinstance(identifier, int):
            identifier = [identifier]
            datasets = Dataset.from_openml(identifier)
        elif isinstance(identifier, list):
            datasets = Dataset.from_openml(identifier)
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
        multiclass: bool = True,
    ) -> list[Dataset]:
        # TODO: should be parametrized, defaults taken from ipy notebook
        if not isinstance(dataset_id, list):
            dataset_id = [dataset_id]

        datasets, _ = load_openml_list(
            dataset_id,
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

def load_openml_list(dids):
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
            X, y, categorical_feats, numerical_feats = get_openml_classification(int(entry.did))


        datasets += [[entry['name'], X, y, categorical_feats, numerical_feats, None]]

    return datasets, datalist


def evaluate_method(
    args,
    metric_function,
    method,
    dataset,
    seed
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

        test_score, pred, _ = method_func(
            X[train_indices],
            y[train_indices],
            X[test_indices],
            y[test_indices],
            categorical_feats,
            numerical_feats,
            metric_used=metric_function,
            seed=seed,
            max_time=0)

        np.save(open(path, "wb"), np.array(pred))
    except Exception as e:
        # raise e
        print(repr(e))
        return None, None
    total_time = time.time()-start_time
    print(f"Result: score: {test_score} and time: {total_time}")
    return test_score.item(), total_time

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

    datasets = Dataset.fetch(dataset_ids)
    dataset_names = set([dataset.name for dataset in datasets])
    index = pd.MultiIndex.from_product(
        [args.methods, args.seeds],
        names=[
            "method",
            "seed",
        ],
    )

    metrics = [args.metric, "time"]

    columns = pd.MultiIndex.from_product(
        [metrics, dataset_names],
        names=["metric", "dataset"],
    )

    df = pd.DataFrame(columns=columns, index=index)
    df.sort_index(inplace=True)

    results = {}
    for method in args.methods:
        results[method] = {}
        for dataset in datasets:
            dataset_name = dataset.name
            results[method][dataset_name] = {}
            for seed in args.seeds:
                
                print(f"Running: {method}, for seed: {seed}, on {dataset_name}")
                if args.slurm:
                    print("Running ensemble on slurm")
                    log_folder = os.path.join(args.result_dir, "log_test")
                    slurm_executer = get_executer(
                        partition=args.partition,
                        log_folder=log_folder,
                        total_job_time_secs=args.slurm_job_time,
                        gpu=False
                        )
                    result = slurm_executer.submit(
                        evaluate_method,
                        args,
                        metric_function,
                        method,
                        dataset,
                        seed
                    )
                    print(f"Started job with job_id: {result.job_id}")
                else:
                    result = evaluate_method(
                        args=args,
                        metric_function=metric_function,
                        method=method,
                        dataset=dataset,
                        seed=seed)
                results[method][dataset_name][seed] = result

    print_string = "all jobs successfully submitted" if args.slurm else "All results ran successfully"
    print(print_string)
    for method in results:
        for dataset_name in results[method]:
            for seed in results[method][dataset_name]:
                row = (method, seed)
                result = results[method][dataset_name][seed]
                if args.slurm:
                    print(f"Waiting for the result odf {method}, for seed: {seed}, on {dataset_name}")
                    result = result.result()
                else:
                    result = result
                df.loc[row, (args.metric, dataset_name)] = result[0]
                df.loc[row, ("time", dataset_name)] = result[1]

    df.to_csv(os.path.join(args.result_dir, f"result_{method}_{args.chunk_id}.csv"))

# seeds = 545, 385, 287, 721, 834
