import sys
tabpfn_path = '../../'
sys.path.insert(0, tabpfn_path)


from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

from sklearn.linear_model import LogisticRegression
import time

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

CV = 5
MULTITHREAD = -1 # Number of threads baselines are able to use at most
MAX_EVALS = 10_000
param_grid, param_grid_hyperopt = {}, {}

def eval_complete_f(x, y, test_x, test_y, key, clf_, metric_used, seed, max_time, no_tune):
    start_time = time.time()
    clf = clf_(**no_tune)
    clf.fit(x, y)

    if hasattr(clf, 'predict_proba'):
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    inference_time = time.time() - start_time

    predictions = np.argmax(pred, axis=1) if len(pred.shape) > 1 else pred
    metric = metric_used(test_y, predictions)


    return metric, pred, inference_time

def preprocess_impute(x, y, test_x, test_y, impute, one_hot, standardize, cat_features=[], numerical_features=[]):
    import warnings
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn

    transformers=[
        ("imputer", SimpleImputer(missing_values=np.nan, strategy='mean'), list(range(x.shape[1])))
    ]
 
    if one_hot:
        transformers.append(("one_hot",  OneHotEncoder(handle_unknown='ignore', sparse=False), cat_features))

    ct = ColumnTransformer(
        transformers=transformers, remainder="drop"
    )
    x_transformed = ct.fit_transform(x)
    test_x_transformed = ct.transform(test_x)

    if standardize:
        scaler = MinMaxScaler()
        scaler.fit(x_transformed)
        x_transformed, test_x_transformed = scaler.transform(x_transformed), scaler.transform(test_x_transformed)

    le = LabelEncoder()
    y_transformed = le.fit_transform(y)
    test_y_transformed = le.transform(test_y)

    return x_transformed, y_transformed, test_x_transformed, test_y_transformed

def logistic_metric(x, y, test_x, test_y, cat_features, numerical_features, metric_used, seed, max_time=300, no_tune=None):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=True, impute=True, standardize=True
                                             , cat_features=cat_features, numerical_features=numerical_features)

    def clf_(**params):
        return LogisticRegression(n_jobs=MULTITHREAD, **params)

    return eval_complete_f(x, y, test_x, test_y, 'logistic', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)


## Random Forest
# Search space from
# https://www.kaggle.com/code/emanueleamcappella/random-forest-hyperparameters-tuning/notebook
def random_forest_metric(x, y, test_x, test_y, cat_features, numerical_features, metric_used, seed, max_time=300, no_tune={}):
    from sklearn.ensemble import RandomForestClassifier

    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=True, impute=True, standardize=False,
                                             cat_features=cat_features, numerical_features=numerical_features)
    def clf_(**params):
        return RandomForestClassifier(n_jobs=MULTITHREAD, random_state=seed, **params)

    return eval_complete_f(x, y, test_x, test_y, 'random_forest', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)

## Gradient Boosting
param_grid_hyperopt['decision_tree'] = {}
def decision_tree_metric(x, y, test_x, test_y, cat_features, numerical_features, metric_used, seed, max_time=300, no_tune=None):
    from sklearn import tree
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=True, impute=True, standardize=False,
                                             cat_features=cat_features, numerical_features=numerical_features)

    def clf_(**params):
        return tree.DecisionTreeClassifier(**params)

    return eval_complete_f(x, y, test_x, test_y, 'decision_tree', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)

## Gradient Boosting
param_grid_hyperopt['hist_gradient_boosting'] = {}
def hist_gradient_boosting_metric(x, y, test_x, test_y, cat_features, numerical_features, metric_used, seed, max_time=300, no_tune=None):
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn import ensemble
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=True, impute=False, standardize=False,
                                             cat_features=cat_features, numerical_features=numerical_features)

    def clf_(**params):
        return ensemble.HistGradientBoostingClassifier(**params)

    return eval_complete_f(x, y, test_x, test_y, 'hist_gradient_boosting', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)


## mlp
param_grid_hyperopt['mlp'] = {}
def sklearn_mlp_metric(x, y, test_x, test_y, cat_features, numerical_features, metric_used, seed, max_time=300, no_tune=None):
    from sklearn import neural_network
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=True, impute=True, standardize=True,
                                             cat_features=cat_features,
                                             numerical_features=numerical_features)
    def clf_(**params):
        return neural_network.MLPClassifier(**params)

    return eval_complete_f(x, y, test_x, test_y, 'mlp', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)


## Tabnet
# https://github.com/dreamquark-ai/tabnet
#param_grid['tabnet'] = {'n_d': [2, 4], 'n_steps': [2,4,6], 'gamma': [1.3], 'optimizer_params': [{'lr': 2e-2}, {'lr': 2e-1}]}

clf_dict = {
    'random_forest': random_forest_metric,
    'logistic': logistic_metric,
    'mlp': sklearn_mlp_metric,
    'hist_gradient_boosting': hist_gradient_boosting_metric,
    'decision_tree': decision_tree_metric
    }
