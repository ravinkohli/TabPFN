from __future__ import annotations

import argparse
import pickle
import re
from tqdm import tqdm
from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import chain, product
from pathlib import Path
import random
from typing import Any, Callable, Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D

import tabpfn.scripts.tabular_baselines as tb
from tabpfn.datasets import load_openml_list, open_cc_dids, open_cc_valid_dids
from tabpfn.scripts.tabular_baselines import clf_dict
from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts.tabular_metrics import (accuracy_metric, auc_metric,
                                            brier_score_metric,
                                            calculate_score, cross_entropy,
                                            ece_metric, time_metric)
from submitit import SlurmExecutor, AutoExecutor


HERE = Path(__file__).parent.resolve().absolute()

METRICS = {
    "roc": auc_metric,
    "cross_entropy": cross_entropy,
    "acc": accuracy_metric,
    "brier_score": brier_score_metric,
    "ece": ece_metric,
}

PREDEFINED_RESULTS_PATH = HERE / "TabPFNResults" / "all_results"
PREDFINED_DATASET_PATHS = HERE / "tabpfn" / "datasets" # / "TabPFN" 
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


LABEL_NAMES = {
    "transformer": "TabPFN",
    "transformer_gpu_N_1": "TabPFN GPU (N_ens =  1)",
    "transformer_gpu_N_8": "TabPFN GPU (N_ens =  8)",
    "transformer_gpu_N_32": "TabPFN GPU (N_ens = 32)",
    "transformer_cpu_N_1": "TabPFN CPU (N_ens =  1)",
    "transformer_cpu_N_8": "TabPFN CPU (N_ens =  8)",
    "transformer_cpu_N_32": "TabPFN CPU (N_ens = 32)",
    "autogluon": "Autogluon",
    "autosklearn2": "Autosklearn2",
    "gp_default": "default GP (RBF)",
    "gradient_boosting": "tuned Grad. Boost.",
    "gradient_boosting_default": "default Grad. Boost.",
    "lightgbm": "tuned LGBM",
    "lightgbm_default": "default LGBM",
    "gp": "tuned GP (RBF)",
    "logistic": "tuned Log. Regr.",
    "knn": "tuned KNN",
    "catboost": "tuned Catboost",
    "xgb": "tuned XGB",
    "xgb_default": "default XGB",
    "svm": "tuned SVM",
    "svm_default": "default SVM",
    "random_forest": "tuned Random Forest",
    "rf_default_n_estimators_10": "Rand. Forest (N_est =  10)",
    "rf_default_n_estimators_32": "Rand. Forest (N_est =  32)",
    "rf_default": "Rand. Forest (N_est = 100)",
}
FAMILY_NAMES = {
    "gp": "GP",
    "gradient_boosting": "Grad. Boost",
    "knn": "KNN",
    "lightgbm": "LGBM",
    "logistic": "Log. Regr.",
    "rf": "RF",
    "svm": "SVM",
    "transformer_cpu": "TabPFN CPU",
    "transformer_gpu": "TabPFN GPU",
    "xgb": "XGB",
    "catboost": "CatBoost",
}


class BoschSlurmExecutor(SlurmExecutor):
    def _make_submission_command(self, submission_file_path):
        return ["sbatch", str(submission_file_path), '--bosch']


PARTITION_TO_EXECUTER = {
    'bosch': BoschSlurmExecutor,
    'other': AutoExecutor

}

def get_executer_class(partition: str) -> SlurmExecutor:
    if 'bosch' in partition:
        key = 'bosch'
    else:
        key = 'other'
    return PARTITION_TO_EXECUTER[key]


def get_executer_params(timeout: float, partition: str, gpu: bool = False) -> Dict[str, Any]:
    if gpu:
        return {'timeout_min': int(timeout), 'slurm_partition': partition, 'slurm_tasks_per_node': 1, 'slurm_gres': "gpu:1"}
    else:
        return {'time': int(timeout), 'partition': partition, 'mem_per_cpu': 6000, 'nodes': 1, 'cpus_per_task': 1, 'ntasks_per_node': 1}


def get_executer(partition: str, log_folder: str, gpu: bool=False, total_job_time_secs: float = 3600):
    slurm_executer = get_executer_class(partition)(folder=log_folder)
    slurm_executer.update_parameters(**get_executer_params(np.ceil(total_job_time_secs/60), partition, gpu))
    return slurm_executer


def set_seed(seed: int):
    # Setting up reproducibility
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
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

    @property
    def categorical(self) -> bool:
        return len(self.categorical_columns) == len(self.attribute_names)

    @property
    def numerical(self) -> bool:
        return len(self.categorical_columns) == 0

    @property
    def mixed(self) -> bool:
        return not self.numerical and not self.categorical

    @classmethod
    def fetch(
        self,
        identifier: str | int | list[int],
        only: Callable | None = None,
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
        """How the internals expect a dataset to look like."""
        return [
            self.name,
            self.X,
            self.y,
            self.categorical_columns,
            self.attribute_names,
            self.info,
        ]


@dataclass
class Row:
    time: int
    pos: int
    method: str
    split: int
    metric: str
    metric_value: dict[str, float]


@dataclass
class Results:
    # Big ass predefined dictionary
    df: pd.DataFrame

    @classmethod
    def from_dict(
        self,
        d: dict,
        datasets: list[Dataset],
        recorded_metrics: list[str],
        *,
        dropna: bool = True,
    ) -> Results:
        # TODO: we could extract dataset_names for the dict but it's not ordered well
        #   for that. Likewise for the recorded_metrics
        #
        # We do a lot of parsing here to massage things into a nice table
        # Extract all the times listed in the keys
        pattern = re.compile(
            r"(?P<method>\w+)"
            r"_time_(?P<time>\d+(\.\d+)?)"
            r"(_)?(?P<metric>\w+)"
            r"_split_(?P<split>\d+)"
        )

        groups = []
        for key in d:
            match = re.match(pattern, key)
            if not match:
                raise ValueError(key)

            groups.append(match.groupdict())

        matches = pd.DataFrame(groups)

        # The unique, methods, times, metrics and splits present
        methods = list(matches["method"].unique())
        times = list(matches["time"].astype(float).unique())
        metrics = list(matches["metric"].unique())
        splits = list(matches["split"].astype(int).unique())

        # Next we extract all the eval_positions
        _eval_positions = set()
        for v in d.values():
            _eval_positions.update(v["eval_positions"])
        eval_positions = sorted(_eval_positions)

        # Dataset names...
        dataset_names = sorted([d.name for d in datasets])

        # We flatten out the fit_time and inference_time of best_config
        for (k, v), pos, dataset in product(d.items(), eval_positions, datasets):
            old_best_configs_key = f"{dataset.name}_best_configs_at_{pos}"

            best_config_key = f"{dataset.name}_best_config"
            inference_time_key = f"{dataset.name}_inference_time_at_{pos}"
            fit_time_key = f"{dataset.name}_fit_time_at_{pos}"

            # If there is a best config
            if any(v.get(old_best_configs_key, [])):
                assert len(v[old_best_configs_key]) == 1

                best_config = v[old_best_configs_key][0]

                v[inference_time_key] = best_config.get("inference_time", np.nan)
                v[fit_time_key] = best_config.get("fit_time", np.nan)
                v[best_config_key] = best_config.copy()
                del v[old_best_configs_key]
            else:
                v[inference_time_key] = np.nan
                v[fit_time_key] = np.nan
                v[best_config_key] = np.nan

        index = pd.MultiIndex.from_product(
            [methods, metrics, times, eval_positions, splits],
            names=[
                "method",
                "optimization_metric",
                "optimization_time",
                "eval_position",
                "split",
            ],
        )

        metrics = recorded_metrics + ["time", "inference_time", "fit_time"]
        columns = pd.MultiIndex.from_product(
            [metrics, dataset_names],
            names=["metric", "dataset"],
        )

        df = pd.DataFrame(columns=columns, index=index)
        df.sort_index(inplace=True)

        for k, v in d.items():
            match = re.match(pattern, k)
            if match is None:
                raise ValueError(k)

            method = match.group("method")
            time = float(match.group("time"))
            opt_metric = match.group("metric")
            split = int(match.group("split"))

            for dataset, metric, pos in product(dataset_names, metrics, eval_positions):
                row = (method, opt_metric, time, int(pos), split)
                col = (metric, dataset)

                value = v.get(f"{dataset}_{metric}_at_{pos}", np.nan)

                df.loc[row, col] = value

        # Drop full NaN rows
        if dropna:
            df = df[df.any(axis=1)]

        return Results(df)

    def at(
        self,
        *,
        method: str | list[str] | None = None,
        optimization_metric: str | list[str] | None = None,
        optimization_time: float | list[float] | None = None,
        split: int | list[int] | None = None,
        eval_position: int | list[int] | None = None,
        dataset: str | list[str] | None = None,
        metric: str | list[str] | None = None,
    ) -> Results:
        """Use this for slicing in to the dataframe to get what you need"""
        df = self.df
        items = {
            "method": method,
            "optimization_time": optimization_time,
            "optimization_metric": optimization_metric,
            "split": split,
            "eval_position": eval_position,
        }
        for name, item in items.items():
            if item is None:
                continue
            idx: list = item if isinstance(item, list) else [item]
            df = df[df.index.get_level_values(name).isin(idx)]
            if not isinstance(item, list):
                df = df.droplevel(name, axis="index")

        if dataset:
            _dataset = dataset if isinstance(dataset, list) else [dataset]
            df = df.T.loc[df.T.index.get_level_values("dataset").isin(_dataset)].T
            if not isinstance(dataset, list):
                df = df.droplevel("dataset", axis="columns")

        if metric:
            _metric = metric if isinstance(metric, list) else [metric]
            df = df.T.loc[df.T.index.get_level_values("metric").isin(_metric)].T
            if not isinstance(metric, list):
                df = df.droplevel("metric", axis="columns")

        return Results(df)

    @property
    def methods(self) -> list[str]:
        return list(self.df.index.get_level_values("method").unique())

    @property
    def optimization_metrics(self) -> list[str]:
        return list(self.df.index.get_level_values("optimization_metric").unique())

    @property
    def optimization_times(self) -> list[float]:
        return list(self.df.index.get_level_values("optimization_time").unique())

    @property
    def eval_positions(self) -> list[int]:
        return list(self.df.index.get_level_values("eval_position").unique())

    @property
    def datasets(self) -> list[str]:
        return list(self.df.columns.get_level_values("dataset").unique())

    @property
    def metrics(self) -> list[str]:
        return list(self.df.columns.get_level_values("metric").unique())


@dataclass
class Plotter:
    result: Results

    def overall_plot(
        self,
        *,
        eval_position: int = 1_000,
        optimization_time: float = 30.0,
        optimization_metric: str = "roc_auc",
        metric: str = "acc",
        legend: str = "box",  # box, text
        highlighted_families: Sequence[str] = FAMILY_NAMES.keys(),
        # (
        #    "transformer_cpu",
        #    "transformer_gpu",
        #    "xgb",
        #    "rf",
        # ),
        ax: plt.Axes,
    ) -> plt.Axes:
        assert all(f in FAMILY_NAMES for f in highlighted_families)
        quantile_pairs = [(0.05, 0.95), (0.25, 0.75)]
        quantile_mark = [(0.05, 0.95), (0.25, 0.75)]
        quantiles = sorted(set(chain.from_iterable(quantile_pairs)))

        s_point = 50
        s_median = 100
        alpha_point = 0.1
        alpha_family_join = 0.1
        q_alpha = {0: 0.2, 0.05: 0.3, 0.25: 0.5}
        q_linewidth = {0: 1, 0.05: 2, 0.25: 3}

        r = self.result.at(
            optimization_metric=optimization_metric,
            optimization_time=optimization_time,
            eval_position=eval_position,
            metric=[metric, "time"],
        )

        # metric        acc       time
        # method split
        # gp     0      0.786164  39.354000
        #        1      0.786164  38.317375
        # ...           ...        ...
        # xgb    19     0.794751   0.148113
        #        20     0.794751   0.148113
        df = r.df.groupby(["method", "split"]).mean().T.groupby("metric").mean().T

        # For dataset cross dataset aggregation
        # df = r.df.unstack(level="method").mean().unstack("metric").reset_index()

        #          | time                               metric
        # quantile | 0, 0.05, 0.25, 0.75, 0.95, 1, 0, 0.05, 0.25, 0.75, 0.95, 1
        # ---------------------------------------------------------------------
        # gp       |
        # ...      |
        # xgb      |
        qs = df.groupby("method").quantile(quantiles).unstack()
        qs.columns.names = [qs.columns.names[0], "quantiles"]

        #        | acc time
        # method |
        # gp     |
        # ...    |
        # xgb    |
        medians = df.groupby("method").agg({metric: "median", "time": "median"})

        families = set(map(Plotter.family, r.methods))

        palette = {
            h: c for h, c in zip(families, sns.color_palette(n_colors=len(families)))
        }

        # Tiny feint blobs for all points
        # methods = df.index.get_level_values("method")
        # df["family"] = [self.family(m) for m in method_list]
        # df["style"] = self.styles(method_list)
        # sns.scatterplot(
        # data=df,
        # x="time",
        # y=metric,
        # hue=hue,
        # style=style,
        # alpha=alpha_point,
        # ax=ax,
        # legend=False,
        # palette=palette,
        # s=s_point,
        # )

        # Quantiles
        # For each (method, quantile) we draw a H on both the time and metric axis
        # time
        times = qs["time"]
        metric_values = qs[metric]
        for method, (q_low, q_high) in product(qs.index, quantile_pairs):

            x = medians.loc[method]["time"]
            time_low = times[q_low].loc[method]
            time_high = times[q_high].loc[method]

            y = medians.loc[method][metric]
            metric_low = metric_values[q_low].loc[method]
            metric_high = metric_values[q_high].loc[method]

            family = Plotter.family(method)

            # Time
            time_marker = "|" if (q_low, q_high) in quantile_mark else None
            ax.plot(
                [time_low, time_high],
                [y, y],
                c=palette[family],
                alpha=q_alpha[q_low],
                linewidth=q_linewidth[q_low],
                marker=time_marker,
            )

            # Metric
            metric_marker = "_" if (q_low, q_high) in quantile_mark else None
            ax.plot(
                [x, x],
                [metric_low, metric_high],
                c=palette[family],
                alpha=q_alpha[q_low],
                linewidth=q_linewidth[q_low],
                marker=metric_marker,
            )

        # Big blob for medians
        medians["family"] = [Plotter.family(i) for i in medians.index]
        markers = self.markers(sorted(medians.index, key=lambda x: LABEL_NAMES[x]))

        for key, group in medians.groupby("method"):
            sns.scatterplot(
                data=group,
                x="time",
                y=metric,
                hue="family",
                ax=ax,
                palette=palette,
                s=s_median,
                marker=markers[key],
            )

        # https://matplotlib.org/stable/gallery/misc/transoffset.html#sphx-glr-gallery-misc-transoffset-py
        text_offset = mtransforms.offset_copy(ax.transData, x=10, y=20, units="dots")
        for family, group in medians.groupby("family"):
            if family not in highlighted_families:
                continue
            # Sort by the time axis
            xs, ys = zip(*sorted(zip(group["time"], group[metric])))
            ax.plot(xs, ys, c=palette[family], linestyle="--", alpha=alpha_family_join)

            l_xs = len(xs)
            mid_x, mid_y = xs[l_xs // 2], ys[l_xs // 2]
            ax.text(
                mid_x,
                mid_y,
                FAMILY_NAMES[family],
                transform=text_offset,
                c=palette[family],
                fontweight="bold",
            )

        ax.set_xscale("log")
        ticks = {0.5: "0.5s", 1: "1s", 5: "5s", 15: "15s", 30: "30s", 60: "1min"}
        ax.set_xticks(list(ticks.keys()))
        ax.set_xticklabels(list(ticks.values()))

        # We unfortunatly have to create a manual legend just due to seaborn not being
        # very flexible in that respect
        family_methods = sorted([(self.family(m), m) for m in set(medians.index)])

        items = [
            (
                family,
                LABEL_NAMES[method],
                Line2D(
                    [],
                    [],
                    color=palette[family],
                    marker=markers[method],
                    linestyle="",
                ),
            )
            for family, method in family_methods
        ]
        # Sort just by family and label
        _, labels, handles = zip(*sorted(items, key=lambda x: x[:2]))

        # create a legend only using the items
        ax.legend(
            handles,
            labels,
            title="Method",
            fontsize=10,
        )

        ax.set_xlabel("Time taken (s)")
        ax.set_ylabel(metric)

        return ax

    @classmethod
    def family(cls, method: str) -> str:
        for f in FAMILY_NAMES:
            if method.startswith(f):
                return f

        # Exceptions
        if "random_forest" in method:
            return "rf"

        return method

    @classmethod
    def markers(cls, methods: Iterable[str]) -> dict[str, str]:
        markers = ["o", "v", "s", "D", "8", "X", "*"]
        styles: dict[str, str] = {}

        counter: Counter[str] = Counter()
        for method in methods:
            family = cls.family(method)
            idx = counter[family]
            styles[method] = markers[idx]
            counter[family] += 1

        return styles


# Predefined methods with `no_tune={}` inidicating they are not tuned
METHODS = {
    # svm
    "svm": tb.svm_metric,
    "svm_default": partial(tb.svm_metric, no_tune={}),
    # autogluon
    "autogluon": tb.autogluon_metric,
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
    # catboost
    "catboost": clf_dict["catboost"],
    "catboost_default": partial(clf_dict["catboost"], no_tune={}),
    "catboost_gpu": partial(clf_dict["catboost"], gpu_id=0),
    "catboost_default_gpu": partial(clf_dict["catboost"], no_tune={}, gpu_id=0),
    # xgb
    "xgb": clf_dict["xgb"],
    "xgb_default": partial(clf_dict["xgb"], no_tune={}),
    "xgb_default_gpu": partial(clf_dict['xgb'], no_tune={}, gpu_id=0),
    "xgb_gpu": partial(clf_dict['xgb'], gpu_id=0),
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
    # naiveautoml
    "naiveautoml": clf_dict["naiveautoml"],
    # Transformers
    "transformer_cpu_N_1": partial(
        clf_dict["transformer"], device="cpu", N_ensemble_configurations=1
    ),
    "transformer_cpu_N_4": partial(
        clf_dict["transformer"], device="cpu", N_ensemble_configurations=4
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
    "transformer_gpu_N_4": partial(
        clf_dict["transformer"], device="cuda", N_ensemble_configurations=4
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
    max_time: float | None,
    metric_used: Callable,
    split: int,
    eval_positions: list[int],
    result_path: Path,
    append_metric: bool = True,
    fetch_only: bool = False,
    verbose: bool = False,
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
        max_splits=1,
        overwrite=overwrite,
        save=True,
        metric_used=metric_used,
        path_interfix=task_type,
        fetch_only=fetch_only,
        split_id=split,
        verbose=verbose,
        max_time=max_time,
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path",
        type=Path,
        help="Where the results path is",
        default=HERE,
    )
    parser.add_argument("--gpu", action="store_true", help="GPU's available?")
    parser.add_argument("--slurm", action="store_true", help="Run on slurm?")
    parser.add_argument(
        "--times",
        nargs="+",
        type=float,
        default=[30],
        help="Times to evaluate (seconds)",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="The splits to evaluate",
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
        "--chunk_size",
        type=int,
        default=10,
        help="Size of chunks to process the datasets",
    )
    parser.add_argument(
        "--optimization_metrics",
        type=str,
        choices=METRICS,
        help="Metrics to optimize for (if possible)",
        default=["roc"],
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
    parser.add_argument("--eval_positions", nargs="+", type=int, default=[1_000])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite results if they already exist",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plots", type=Path, help="Where to output plots to")
    parser.add_argument("--load_predefined_results", action="store_true")
    parser.add_argument(
        "--predefined_results_path", type=Path, default=PREDFINED_DATASET_PATHS
    )
    parser.add_argument(
        "--partition", type=str, default="bosch_cpu-cascadelake"
    )
    parser.add_argument("--parallel", action="store_true",
                        help="Paralleise execution of each split"
    )
    return parser.parse_args()


def do_evaluations(args: argparse.Namespace, datasets: list[Dataset]) -> Results:
    results = {}
    for method, metric, time, split in product(
        args.methods,
        args.optimization_metrics,
        args.times,
        range(1, args.splits+1),
    ):
        metric_f = METRICS[metric]
        metric_name = tb.get_scoring_string(metric_f, usage="")
        key = f"{method}_time_{time}{metric_name}_split_{split}"

        results[key] = eval_method(
            datasets=datasets,
            label=method,
            result_path=args.result_path,
            classifier_evaluator=METHODS[method],
            eval_positions=args.eval_positions,  # It's a constant basically
            fetch_only=args.fetch_only,
            verbose=args.verbose,
            max_time=time,
            metric_used=metric_f,
            split=split,
            overwrite=args.overwrite,
        )

    return results


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def do_evaluations_parallel(args: argparse.Namespace, datasets, log_folder: str) -> Results:
    jobs = {}
    for method, metric, time, split in product(
        args.methods,
        args.optimization_metrics,
        args.times,
        range(1, args.splits+1),
    ):

        metric_f = METRICS[metric]
        metric_name = tb.get_scoring_string(metric_f, usage="")
        key = f"{method}_time_{time}{metric_name}_split_{split}"
        for sub_datasets in tqdm(chunks(list(datasets), args.chunk_size)):
            set_seed(seed=split)

            if key not in jobs:
                jobs[key] = []

            # give atleast 1 min per split and 1.5 times the opt_time
            total_job_time = max(time * 1.5, 60) * args.chunk_size
            slurm_executer = get_executer(
                partition=args.partition,
                log_folder=log_folder,
                total_job_time_secs=total_job_time,
                gpu=args.gpu
                )
            jobs[key].append(slurm_executer.submit(eval_method,
            datasets=sub_datasets,
            label=method,
            result_path=args.result_path,
            classifier_evaluator=METHODS[method],
            eval_positions=args.eval_positions,  # It's a constant basically
            fetch_only=args.fetch_only,
            verbose=args.verbose,
            max_time=time,
            metric_used=metric_f,
            split=split,
            overwrite=args.overwrite)
            )

    return jobs
