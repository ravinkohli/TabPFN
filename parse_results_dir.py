import os
import glob

import re

methods = [
    'svm',
    'gradient_boosting',
    'catboost',
    'svm_default',
    'gradient_boosting_default',
    'gp',
    'gp_default',
    'autosklearn2',
    'lightgbm',
    'lightgbm_default',
    'catboost_default',
    'xgb',
    'xgb_onehot',
    'xgb_default',
    'xgb_default_onehot',
    'random_forest',
    'rf_default',
    'rf_default_n_estimators_10',
    'rf_default_n_estimators_32',
    'knn',
    'logistic',
    'transformer_cpu_N_1',
    'transformer_cpu_N_4',
    'transformer_cpu_N_8',
    'transformer_cpu_N_32']

pattern = re.compile(
            r"results_(?P<method>\w+)_time_(?P<time>\d+(\.\d+)?)(_)?(?P<metric>\w+)_(?P<dataset>\w+)_\d+_\d+_(?P<split>\d+).npy"
        )

groups = []

for file in os.listdir("/work/dlclarge1/rkohli-results_tabpfn_400/results/tabular/multiclass/"):
    match = re.match(pattern, os.path.basename(file))
    if not match:
        continue
    groups.append(match.groupdict())


method_result_files = {}
for group in groups:
    if group["method"] not in method_result_files:
        method_result_files[group["method"]] = {}
    if group["dataset"] not in method_result_files[group["method"]]:
        method_result_files[group["method"]][group["dataset"]] = 0
    else:
        method_result_files[group["method"]][group["dataset"]] += 1

for method in method_result_files:
    print(f"{method} with datasets: {len(method_result_files[method])}")

    
