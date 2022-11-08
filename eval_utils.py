from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

import random

import torch

import tabpfn.scripts.tabular_baselines as tb
from tabpfn.datasets import load_openml_list, open_cc_dids, open_cc_valid_dids
from tabpfn.scripts.tabular_baselines import clf_dict
from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts.tabular_metrics import (accuracy_metric, auc_metric,
                                            brier_score_metric,
                                            cross_entropy, ece_metric)

DEFAULT_SEED = 42

HERE = Path(__file__).parent.resolve().absolute()

METRICS = {
    "roc": auc_metric,
    "cross_entropy": cross_entropy,
    "acc": accuracy_metric,
    "brier_score": brier_score_metric,
    "ece": ece_metric,
}

PREDFINED_DATASET_PATHS = HERE / "tabpfn" / "datasets"
PREDEFINED_DATASET_COLLECTIONS = {
    "cc_valid": {
        "ids": open_cc_valid_dids,
        "path": PREDFINED_DATASET_PATHS / "cc_valid_datasets_multiclass.pickle",
    },
    "cc_test": {
        "ids": open_cc_dids,
        "path": PREDFINED_DATASET_PATHS / "cc_test_datasets_multiclass.pickle",
    },
}

def set_seed(seed):
    # Setting up reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@dataclass
class Dataset:
    """Small helper class just to name entries in the loaded pickled datasets."""

    name: str
    X: torch.Tensor
    y: torch.Tensor
    categorical_columns: list[int]
    attribute_names: list[str]
    # Seems to be some things about how the dataset was constructed
    info: dict
    # Only 'multiclass' is known?
    task_type: str

    @classmethod
    def fetch(
        self, identifier: str | int | list[int], only: Callable | None = None
    ) -> list[Dataset]:
        if isinstance(identifier, str) and identifier in PREDEFINED_DATASET_COLLECTIONS:
            datasets = Dataset.from_predefined(identifier)
        elif isinstance(identifier, int):
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
    def from_pickle(self, path: Path, task_types: str) -> list[Dataset]:
        with path.open("rb") as f:
            raw = pickle.load(f)

        return [Dataset(*entry, task_type=task_types) for entry in raw]  # type: ignore

    @classmethod
    def from_predefined(self, name: str) -> list[Dataset]:
        assert name in PREDEFINED_DATASET_COLLECTIONS
        path = PREDEFINED_DATASET_COLLECTIONS[name]["path"]

        return Dataset.from_pickle(path, task_types="multiclass")

    @classmethod
    def from_openml(
        self,
        dataset_id: int | list[int],
        filter_for_nan: bool = False,
        min_samples: int = 100,
        max_samples: int = 2_000,
        num_feats: int = 100,
        return_capped: bool = False,
        shuffled: bool = True,
        multiclass: bool = True,
    ) -> list[Dataset]:
        # TODO: should be parametrized, defaults taken from ipy notebook
        if not isinstance(dataset_id, list):
            dataset_id = [dataset_id]

        datasets, _ = load_openml_list(
            dataset_id,
            filter_for_nan=filter_for_nan,
            num_feats=num_feats,
            min_samples=min_samples,
            max_samples=max_samples,
            return_capped=return_capped,
            shuffled=shuffled,
            multiclass=multiclass,
        )
        return [
            Dataset(  # type: ignore
                *entry,
                task_type="multiclass" if multiclass else "binary",
            )
            for entry in datasets
        ]

    def as_list(self) -> list:
        """How the internals expect a dataset to look like"""
        return [
            self.name,
            self.X,
            self.y,
            self.categorical_columns,
            self.attribute_names,
            self.info,
        ]


# Predefined methods with `no_tune={}` inidicating they are not tuned
METHODS = {
    # svm
    "svm": tb.svm_metric,
    "svm_default": partial(tb.svm_metric, no_tune={}),
    # gradient boosting
    "gradient_boosting": tb.gradient_boosting_metric,
    "gradient_boosting_default": partial(tb.gradient_boosting_metric, no_tune={}),
    # gp
    "gp": clf_dict["gp"],
    "gp_default": partial(
        clf_dict["gp"],
        no_tune={"params_y_scale": 0.1, "params_length_scale": 0.1},
    ),
    # lightgbm
    "lightgbm": clf_dict["lightgbm"],
    "lightgbm_default": partial(clf_dict["lightgbm"], no_tune={}),
    # xgb
    "xgb": clf_dict["xgb"],
    "xgb_default": partial(clf_dict["xgb"], no_tune={}),
    # random forest
    "random_forest": clf_dict["random_forest"],
    "rf_default": partial(clf_dict["random_forest"], no_tune={}),
    "rf_default_n_estimators_10": partial(
        clf_dict["random_forest"], no_tune={"n_estimators": 10}
    ),
    "rf_default_n_estimators_32": partial(
        clf_dict["random_forest"], no_tune={"n_estimators": 32}
    ),
    # knn
    "knn": clf_dict["knn"],
    # logistic classification
    "logistic": clf_dict["logistic"],
    # Transformers
    "transformer_cpu_N_1": partial(
        clf_dict["transformer"], device="cpu", N_ensemble_configurations=1
    ),
    "transformer_cpu_N_8": partial(
        clf_dict["transformer"], device="cpu", N_ensemble_configurations=8
    ),
    "transformer_cpu_N_32": partial(
        clf_dict["transformer"], device="cpu", N_ensemble_configurations=32
    ),
    "transformer_gpu_N_1": partial(
        clf_dict["transformer"], device="cuda", N_ensemble_configurations=1
    ),
    "transformer_gpu_N_8": partial(
        clf_dict["transformer"], device="cuda", N_ensemble_configurations=8
    ),
    "transformer_gpu_N_32": partial(
        clf_dict["transformer"], device="cuda", N_ensemble_configurations=32
    ),
}


def eval_method(
    datasets: list[Dataset],
    label: str,
    classifier_evaluator: Callable,
    max_time: int | None,
    metric_used: Callable,
    split_id: int,
    result_path: Path,
    seed: int = DEFAULT_SEED,
    append_metric: bool = True,
    fetch_only: bool = False,
    verbose: bool = False,
    eval_positions: list[int] = [1000],
    bptt: int = 2000,
    overwrite: bool = False,
):
    """Evaluate a given method."""
    if max_time is not None:
        label += f"_time_{max_time}"

    if append_metric:
        label += f"_{tb.get_scoring_string(metric_used, usage='')}"

    if isinstance(classifier_evaluator, partial):
        device = classifier_evaluator.keywords.get("device", "cpu")
    else:
        device = "cpu"

    task_type = "multiclass"
    if any(d.task_type != task_type for d in datasets):
        raise RuntimeError("Not sure how to handle this yet")

    return evaluate(
        datasets=[d.as_list() for d in datasets],
        model=classifier_evaluator,
        method=label,
        bptt=bptt,
        base_path=result_path,
        eval_positions=eval_positions,
        device=device,
        overwrite=overwrite,
        save=True,
        metric_used=metric_used,
        path_interfix=task_type,
        fetch_only=fetch_only,
        split_number=split_id,
        verbose=verbose,
        max_time=max_time,
        seed=seed,
    )
