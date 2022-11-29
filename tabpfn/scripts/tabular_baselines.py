import sys
tabpfn_path = '../../'
sys.path.insert(0, tabpfn_path)

import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

import math
import sklearn
import os
#from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from torch import nn
from sklearn.impute import SimpleImputer


from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

import torch
import itertools
from tabpfn.scripts import tabular_metrics
import pandas as pd
from tqdm import tqdm
from tabpfn.utils import remove_outliers
from tabpfn.scripts.autopytorch_baselines import well_tuned_simple_nets_metric
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
import time

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials , space_eval, rand
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

CV = 5
MULTITHREAD = -1 # Number of threads baselines are able to use at most
MAX_EVALS = 10_000
param_grid, param_grid_hyperopt = {}, {}

def get_scoring_direction(metric_used):
    # Not needed
    if metric_used.__name__ == tabular_metrics.auc_metric.__name__:
        return -1
    elif metric_used.__name__ == tabular_metrics.cross_entropy.__name__:
        return 1
    else:
        raise Exception('No scoring string found for metric')

def is_classification(metric_used):
    if metric_used.__name__ == tabular_metrics.auc_metric.__name__ or metric_used.__name__ == tabular_metrics.cross_entropy.__name__:
        return 'classification'
    elif metric_used.__name__ == tabular_metrics.auc_metric.__name__:
        return -1

# Loss

def get_scoring_string(metric_used, multiclass=True, usage="sklearn_cv"):
    if metric_used.__name__ == tabular_metrics.auc_metric.__name__:
        if usage == 'sklearn_cv':
            return 'roc_auc_ovo'
        elif usage == 'autogluon':
            return 'log_loss' # Autogluon crashes when using 'roc_auc' with some datasets usning logloss gives better scores;
                              # We might be able to fix this, but doesn't work out of box.
                              # File bug report? Error happens with dataset robert and fabert
            # if multiclass:
            #     return 'roc_auc_ovo_macro'
            # else:
            #     return 'roc_auc'
        elif usage == 'tabnet':
            return 'logloss' if multiclass else 'auc'
        elif usage == 'autosklearn':
            import autosklearn.classification
            if multiclass:
                return autosklearn.metrics.log_loss # roc_auc only works for binary, use logloss instead
            else:
                return autosklearn.metrics.roc_auc
        elif usage == 'catboost':
            return 'MultiClass' # Effectively LogLoss, ROC not available
        elif usage == 'xgb':
            return 'logloss'
        elif usage == 'lightgbm':
            if multiclass:
                return 'auc'
            else:
                return 'binary'
        return 'roc_auc'
    elif metric_used.__name__ == tabular_metrics.cross_entropy.__name__:
        if usage == 'sklearn_cv':
            return 'neg_log_loss'
        elif usage == 'autogluon':
            return 'log_loss'
        elif usage == 'tabnet':
            return 'logloss'
        elif usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.log_loss
        elif usage == 'catboost':
            return 'MultiClass' # Effectively LogLoss
        return 'logloss'
    elif metric_used.__name__ == tabular_metrics.r2_metric.__name__:
        if usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.r2
        elif usage == 'sklearn_cv':
            return 'r2' # tabular_metrics.neg_r2
        elif usage == 'autogluon':
            return 'r2'
        elif usage == 'xgb': # XGB cannot directly optimize r2
            return 'rmse'
        elif usage == 'catboost': # Catboost cannot directly optimize r2 ("Can't be used for optimization." - docu)
            return 'RMSE'
        else:
            return 'r2'
    elif metric_used.__name__ == tabular_metrics.root_mean_squared_error_metric.__name__:
        if usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.root_mean_squared_error
        elif usage == 'sklearn_cv':
            return 'neg_root_mean_squared_error' # tabular_metrics.neg_r2
        elif usage == 'autogluon':
            return 'rmse'
        elif usage == 'xgb':
            return 'rmse'
        elif usage == 'catboost':
            return 'RMSE'
        else:
            return 'neg_root_mean_squared_error'
    elif metric_used.__name__ == tabular_metrics.mean_absolute_error_metric.__name__:
        if usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.mean_absolute_error
        elif usage == 'sklearn_cv':
            return 'neg_mean_absolute_error' # tabular_metrics.neg_r2
        elif usage == 'autogluon':
            return 'mae'
        elif usage == 'xgb':
            return 'mae'
        elif usage == 'catboost':
            return 'MAE'
        else:
            return 'neg_mean_absolute_error'
    else:
        raise Exception('No scoring string found for metric')

def eval_f(params, clf_, x, y, metric_used):
    scores = cross_val_score(clf_(**params), x, y, cv=CV, scoring=get_scoring_string(metric_used, usage='sklearn_cv'))
    if get_scoring_string(metric_used, usage='sklearn_cv') == 'r2' or get_scoring_string(metric_used, usage='sklearn_cv') == 'neg_log_loss':
        return np.nanmean(scores)
    
    return -np.nanmean(scores)

def eval_complete_f(x, y, test_x, test_y, key, clf_, metric_used, seed, max_time, no_tune):
    start_time = time.time()
    def stop(trial):
        return time.time() - start_time > max_time, []

    trials = None
    if no_tune is None:
      default = eval_f({}, clf_, x, y, metric_used)
      trials = Trials()
      best = fmin(
          fn=lambda params: eval_f(params, clf_, x, y, metric_used),
          space=param_grid_hyperopt[key],
          algo=rand.suggest,
          rstate=np.random.RandomState(seed),
          early_stop_fn=stop,
          trials=trials,
          catch_eval_exceptions=True,
          verbose=True,
          # The seed is deterministic but varies for each dataset and each split of it
          max_evals=MAX_EVALS)
      best_score = np.min([t['result']['loss'] for t in trials.trials])
      if best_score < default:
        best = space_eval(param_grid_hyperopt[key], best)
      else:
        best = {}
    else:
      best = no_tune.copy()

    start = time.time()
    clf = clf_(**best)
    clf.fit(x, y)
    fit_time = time.time() - start
    start = time.time()
    if is_classification(metric_used):
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    inference_time = time.time() - start
    metric = metric_used(test_y, pred)
    
    best = {'best': best}
    best['fit_time'] = fit_time
    best['inference_time'] = inference_time

    best['trials'] = trials

    return metric, pred, best#, times

def preprocess_impute(x, y, test_x, test_y, impute, one_hot, standardize, cat_features=[]):
    import warnings
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn

    x, y, test_x, test_y = x.cpu().numpy(), y.cpu().long().numpy(), test_x.cpu().numpy(), test_y.cpu().long().numpy()

    if impute:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(x)
        x, test_x = imp_mean.transform(x), imp_mean.transform(test_x)

    if one_hot:
        def make_pd_from_np(x):
            data = pd.DataFrame(x)
            for c in cat_features:
                data.iloc[:, c] = data.iloc[:, c].astype('int')
            return data
        x, test_x = make_pd_from_np(x),  make_pd_from_np(test_x)
        transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_features)], remainder="passthrough")
        transformer.fit(x)
        x, test_x = transformer.transform(x), transformer.transform(test_x)

    if standardize:
        scaler = MinMaxScaler()
        scaler.fit(x)
        x, test_x = scaler.transform(x), scaler.transform(test_x)

    return x, y, test_x, test_y

import torch
import random
from tqdm import tqdm
def transformer_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300, device='cpu', N_ensemble_configurations=3, classifier=None):
    from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

    if classifier is None:
      classifier = TabPFNClassifier(device=device, N_ensemble_configurations=N_ensemble_configurations, seed=seed)
    classifier.fit(x, y)
    print('Train data shape', x.shape, ' Test data shape', test_x.shape)
    pred = classifier.predict_proba(test_x)

    metric = metric_used(test_y, pred)

    return metric, pred, None


def naiveatuoml_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300, max_hpo_iterations=None):
    from naiveautoml import NaiveAutoML

    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False
                                             , cat_features=cat_features
                                             , impute=False
                                             , standardize=False)
    # MAX_HPO_ITERATIONS = 10
    multiclass = False
    if is_classification(metric_used):
        multiclass = len(np.unique(y)) > 2
    
    max_time = max_time if max_hpo_iterations is None else None
    max_iters = max_hpo_iterations if max_hpo_iterations is not None else MAX_EVALS
    classifier = NaiveAutoML(
        max_hpo_iterations=max_iters,
        scoring=get_scoring_string(metric_used, multiclass=multiclass, usage="sklearn_cv"),
        timeout=max_time)

    classifier.fit(x, y)
    print('Train data shape', x.shape, ' Test data shape', test_x.shape)
    predict_function = 'predict_proba' if is_classification(metric_used) else 'predict'
    pred = getattr(classifier, predict_function)(test_x)

    metric = metric_used(test_y, pred)

    return metric, pred, None


## Auto Gluon
# WARNING: Crashes for some predictors for regression
def autogluon_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300):
    from autogluon.tabular import TabularPredictor
    # Preprocess basically doesn't do anything here
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False
                                             , cat_features=cat_features
                                             , impute=False
                                             , standardize=False)
    train_data = pd.DataFrame(np.concatenate([x, y[:, np.newaxis]], axis=1))
    test_data = pd.DataFrame(np.concatenate([test_x, test_y[:, np.newaxis]], axis=1))
    if is_classification(metric_used):
        problem_type = 'multiclass' if len(np.unique(y)) > 2 else 'binary'
    else:
        problem_type = 'regression'
    # AutoGluon automatically infers datatypes, we don't specify the categorical labels
    predictor = TabularPredictor(
        label=train_data.columns[-1],
        eval_metric=get_scoring_string(metric_used, usage='autogluon', multiclass=(len(np.unique(y)) > 2)),
        problem_type=problem_type
        ## seed=int(y[:].sum()) doesn't accept seed
    ).fit(
        train_data=train_data,
        time_limit=max_time,
        presets=['best_quality']
        # The seed is deterministic but varies for each dataset and each split of it
    )

    if is_classification(metric_used):
        pred = predictor.predict_proba(test_data, as_multiclass=True).values
    else:
        pred = predictor.predict(test_data).values

    metric = metric_used(test_y, pred)

    return metric, pred, predictor.fit_summary()

## AUTO Sklearn
def autosklearn_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300):
    import autosklearn.classification
    return autosklearn2_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=max_time, version=1)

def autosklearn2_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300, version=2):
    # Basically doesn't do anything here
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=False,
                                             cat_features=cat_features,
                                             impute=False,
                                             standardize=False)

    def make_pd_from_np(x):
        data = pd.DataFrame(x)
        for c in cat_features:
            data.iloc[:, c] = data.iloc[:, c].astype('category')
        return data

    x = make_pd_from_np(x)
    test_x = make_pd_from_np(test_x)

    if is_classification(metric_used):
        if version == 2:
            from autosklearn.experimental.askl2 import AutoSklearn2Classifier
            clf_ = AutoSklearn2Classifier
        else:
            from autosklearn.classification import AutoSklearnClassifier
            clf_ = AutoSklearnClassifier
    else:
        if version == 2:
            raise Exception("AutoSklearn 2 doesn't do regression.")
        from autosklearn.regression import AutoSklearnRegressor
        clf_ = AutoSklearnRegressor

    clf = clf_(
        time_left_for_this_task=int(max_time),
        memory_limit=4_000,
        n_jobs=MULTITHREAD,
        seed=seed,
        metric=get_scoring_string(
            metric_used,
            usage='autosklearn',
            multiclass=len(np.unique(y)) > 2
        )
    )

    # fit model to data
    clf.fit(x, y)

    if is_classification(metric_used):
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, clf.leaderboard()

param_grid_hyperopt['ridge'] = {
    'max_iter': hp.randint('max_iter', 50, 500)
    , 'fit_intercept': hp.choice('fit_intercept', [True, False])
    , 'alpha': hp.loguniform('alpha', -5, math.log(5.0))}  # 'normalize': [False],

def ridge_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300):
    if is_classification(metric_used):
        raise Exception("Ridge is only applicable to pointwise Regression.")

    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=True, impute=True, standardize=True,
                                             cat_features=cat_features)
    def clf_(**params):
        return Ridge(tol=1e-4, random_state=seed, **params)

    start_time = time.time()

    def stop(trial):
        return time.time() - start_time > max_time, []
    
    best = fmin(
        fn=lambda params: eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['ridge'],
        algo=rand.suggest,
        rstate=np.random.RandomState(seed),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=10000)
    best = space_eval(param_grid_hyperopt['ridge'], best)

    clf = clf_(**best)
    clf.fit(x, y)

    pred = clf.predict(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best

def lightautoml_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300):
    from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
    from lightautoml.tasks import Task
    
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False, impute=False, standardize=False
                                             , cat_features=cat_features)

    roles = {'target': str(x.shape[-1])}
    task = Task('multiclass', metric = lambda x, y : metric_used(x, y, numpy=True))
    automl = TabularUtilizedAutoML(task=task,
                           timeout=max_time,
                           cpu_limit=4,  # Optimal for Kaggle kernels
                           general_params={'use_algos': [['linear_l2',
                                                          'lgb', 'lgb_tuned']]})

    tr_data = np.concatenate([x, np.expand_dims(y, -1)], -1)
    tr_data = pd.DataFrame(tr_data, columns=[str(k) for k in range(0, x.shape[-1] + 1)])
    oof_pred = automl.fit_predict(tr_data, roles=roles)
    te_data = pd.DataFrame(test_x, columns=[str(k) for k in range(0, x.shape[-1])])

    probabilities = automl.predict(te_data).data
    probabilities_mapped = probabilities.copy()

    class_map = automl.outer_pipes[0].ml_algos[0].models[0][0].reader.class_mapping
    if class_map:
        column_to_class = {col: class_ for class_, col in class_map.items()}
        for i in range(0, len(column_to_class)):
            probabilities_mapped[:, int(column_to_class[int(i)])] = probabilities[:, int(i)]

    metric = metric_used(test_y, probabilities_mapped)

    return metric, probabilities_mapped, None

param_grid_hyperopt['lightgbm'] = {
    'num_leaves': hp.randint('num_leaves', 5, 50)
    , 'max_depth': hp.randint('max_depth', 3, 20)
    , 'learning_rate': hp.loguniform('learning_rate', -3, math.log(1.0))
    , 'n_estimators': hp.randint('n_estimators', 50, 2000)
    #, 'feature_fraction': 0.8,
    #, 'subsample': 0.2
    , 'min_child_weight': hp.choice('min_child_weight', [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4])
    , 'subsample': hp.uniform('subsample', 0.2, 0.8)
    , 'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 0.8)
    , 'reg_alpha': hp.choice('reg_alpha', [0, 1e-1, 1, 2, 5, 7, 10, 50, 100])
    , 'reg_lambda': hp.choice('reg_lambda', [0, 1e-1, 1, 5, 10, 20, 50, 100])
}  # 'normalize': [False],


def lightgbm_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300, no_tune=None):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False, impute=False, standardize=False
                                             , cat_features=cat_features)
    from lightgbm import LGBMClassifier

    def clf_(**params):
        return LGBMClassifier(categorical_feature=cat_features, use_missing=True
                              , objective=get_scoring_string(metric_used, usage='lightgbm', multiclass=len(np.unique(y)) > 2), **params)

    return eval_complete_f(x, y, test_x, test_y, 'lightgbm', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)

param_grid_hyperopt['logistic'] = {
    'penalty': hp.choice('penalty', ['l1', 'l2', 'none'])
    , 'max_iter': hp.randint('max_iter', 50, 500)
    , 'fit_intercept': hp.choice('fit_intercept', [True, False])
    , 'C': hp.loguniform('C', -5, math.log(5.0))}  # 'normalize': [False],


def logistic_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300, no_tune=None):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=True, impute=True, standardize=True
                                             , cat_features=cat_features)

    def clf_(**params):
        return LogisticRegression(solver='saga', tol=1e-4, n_jobs=MULTITHREAD, **params)

    return eval_complete_f(x, y, test_x, test_y, 'logistic', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)


## Random Forest
# Search space from
# https://www.kaggle.com/code/emanueleamcappella/random-forest-hyperparameters-tuning/notebook
param_grid_hyperopt['random_forest'] = {'n_estimators': hp.randint('n_estimators', 20, 200),
               'max_features': hp.choice('max_features', ['auto', 'sqrt']),
               'max_depth': hp.randint('max_depth', 1, 45),
               'min_samples_split': hp.choice('min_samples_split', [5, 10])}
def random_forest_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300, no_tune=None):
    from sklearn.ensemble import RandomForestClassifier

    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=False, impute=True, standardize=False,
                                             cat_features=cat_features)

    def clf_(**params):
        if is_classification(metric_used):
            return RandomForestClassifier(n_jobs=MULTITHREAD, **params)
        return RandomForestClassifier(n_jobs=MULTITHREAD, **params)

    return eval_complete_f(x, y, test_x, test_y, 'random_forest', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)

## Gradient Boosting
param_grid_hyperopt['gradient_boosting'] = {}
def gradient_boosting_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300, no_tune=None):
    from sklearn import ensemble
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=True, impute=True, standardize=True,
                                             cat_features=cat_features)

    def clf_(**params):
        if is_classification(metric_used):
            return ensemble.GradientBoostingClassifier(**params)
        return ensemble.GradientBoosting(**params)

    return eval_complete_f(x, y, test_x, test_y, 'gradient_boosting', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)

## SVM
param_grid_hyperopt['svm'] = {'C': hp.choice('C', [0.1,1, 10, 100]), 'gamma': hp.choice('gamma', ['auto', 'scale']),'kernel': hp.choice('kernel', ['rbf', 'poly', 'sigmoid'])}
def svm_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300, no_tune=None):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=True, impute=True, standardize=True,
                                             cat_features=cat_features)

    def clf_(**params):
        if is_classification(metric_used):
            return sklearn.svm.SVC(probability=True,**params)
        return sklearn.svm.SVR(**params)

    return eval_complete_f(x, y, test_x, test_y, 'svm', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)

## KNN
param_grid_hyperopt['knn'] = {'n_neighbors': hp.randint('n_neighbors', 1,16)
                              }
def knn_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300, no_tune=None):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=True, impute=True, standardize=True,
                                             cat_features=cat_features)

    def clf_(**params):
        if is_classification(metric_used):
            return neighbors.KNeighborsClassifier(n_jobs=MULTITHREAD, **params)
        return neighbors.KNeighborsRegressor(n_jobs=MULTITHREAD, **params)

    return eval_complete_f(x, y, test_x, test_y, 'knn', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)

## GP
param_grid_hyperopt['gp'] = {
    'params_y_scale': hp.loguniform('params_y_scale', math.log(0.05), math.log(5.0)),
    'params_length_scale': hp.loguniform('params_length_scale', math.log(0.1), math.log(1.0))
}
def gp_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300, no_tune=None):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=True, impute=True, standardize=True,
                                             cat_features=cat_features)

    def clf_(params_y_scale=None,params_length_scale=None, **params):
        kernel = params_y_scale * RBF(params_length_scale) if params_length_scale is not None else None
        if is_classification(metric_used):
            return GaussianProcessClassifier(kernel=kernel, random_state=seed, **params)
        else:
            return GaussianProcessRegressor(kernel=kernel, ranom_state=seed, **params)

    return eval_complete_f(x, y, test_x, test_y, 'gp', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)

## Tabnet
# https://github.com/dreamquark-ai/tabnet
#param_grid['tabnet'] = {'n_d': [2, 4], 'n_steps': [2,4,6], 'gamma': [1.3], 'optimizer_params': [{'lr': 2e-2}, {'lr': 2e-1}]}

# Hyperparameter space from dreamquarks implementation recommendations
param_grid_hyperopt['tabnet'] = {
    'n_d': hp.randint('n_d', 8, 64),
    'n_steps': hp.randint('n_steps', 3, 10),
    'max_epochs': hp.randint('max_epochs', 50, 200),
    'gamma': hp.uniform('relax', 1.0, 2.0),
    'momentum': hp.uniform('momentum', 0.01, 0.4),
}

def tabnet_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300):
    from pytorch_tabnet.tab_model import TabNetClassifier
    # TabNet inputs raw tabular data without any preprocessing and is trained using gradient descent-based optimisation.
    # However Tabnet cannot handle nans so we impute with mean

    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y, impute=True, one_hot=False, standardize=False)

    def clf_(**params):
        return TabNetClassifier(cat_idxs=cat_features, verbose=True, n_a=params['n_d'], seed=seed, **params)

    def tabnet_eval_f(params, clf_, x, y, metric_used, start_time, max_time):
        if time.time() - start_time > max_time:
            return np.nan

        kf = KFold(n_splits=min(CV, x.shape[0] // 2), random_state=seed, shuffle=True)
        metrics = []

        params = {**params}
        max_epochs = params['max_epochs']
        del params['max_epochs']

        for train_index, test_index in kf.split(x):
            X_train, X_valid, y_train, y_valid = x[train_index], x[test_index], y[train_index], y[test_index]

            clf = clf_(**params)

            clf.fit(
                X_train, y_train,
                # eval_metric=[get_scoring_string(metric_used, multiclass=len(np.unique(y_train)) > 2, usage='tabnet')],
                # eval_set=[(X_valid, y_valid)],
                # patience=15,
                max_epochs=max_epochs
            )
            metrics += [metric_used(y_valid, clf.predict_proba(X_valid))]

        return -np.nanmean(np.array(metrics))

    start_time = time.time()
    def stop(trial):
        return time.time() - start_time > max_time, []

    best = fmin(
        fn=lambda params: tabnet_eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['tabnet'],
        algo=rand.suggest,
        rstate=np.random.RandomState(seed),
        early_stop_fn=stop,
        max_evals=1000)
    best = space_eval(param_grid_hyperopt['tabnet'], best)
    max_epochs = best['max_epochs']
    del best['max_epochs']

    clf = clf_(**best)
    clf.fit(x, y, max_epochs=max_epochs) # , max_epochs=mean_best_epochs[best_idx]

    pred = clf.predict_proba(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best


# Catboost
# Hyperparameter space: https://arxiv.org/pdf/2106.03253.pdf

param_grid_hyperopt['catboost'] = {
    'learning_rate': hp.loguniform('learning_rate', math.log(math.pow(math.e, -5)), math.log(1)),
    'random_strength': hp.randint('random_strength', 1, 20),
    'l2_leaf_reg': hp.loguniform('l2_leaf_reg', math.log(1), math.log(10)),
    'bagging_temperature': hp.uniform('bagging_temperature', 0., 1),
    'leaf_estimation_iterations': hp.randint('leaf_estimation_iterations', 1, 20),
    'iterations': hp.randint('iterations', 100, 4000), # This is smaller than in paper, 4000 leads to ram overusage
}

def catboost_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300, no_tune=None, gpu_id=None):
    from catboost import CatBoostClassifier
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False
                                             , cat_features=cat_features
                                             , impute=False
                                             , standardize=False)

    # Nans in categorical features must be encoded as separate class
    x[:, cat_features], test_x[:, cat_features] = np.nan_to_num(x[:, cat_features], -1), np.nan_to_num(
        test_x[:, cat_features], -1)
    
    if gpu_id is not None:
         gpu_params = {'task_type':"GPU", 'devices':gpu_id}
    else:
        gpu_params = {}

    def make_pd_from_np(x):
        data = pd.DataFrame(x)
        for c in cat_features:
            data.iloc[:, c] = data.iloc[:, c].astype('int')
        return data

    x = make_pd_from_np(x)
    test_x = make_pd_from_np(test_x)

    def clf_(**params):
        if is_classification(metric_used):
            return CatBoostClassifier(
                                    loss_function=get_scoring_string(metric_used, usage='catboost'),
                                    thread_count = MULTITHREAD,
                                    used_ram_limit='4gb',
                                    random_seed=seed,
                                    logging_level='Silent',
                                    cat_features=cat_features,
                                    **gpu_params,
                                    **params)
        else:
            return CatBoostRegressor(
                loss_function=get_scoring_string(metric_used, usage='catboost'),
                thread_count=MULTITHREAD,
                used_ram_limit='4gb',
                random_seed=seed,
                logging_level='Silent',
                cat_features=cat_features,
                **gpu_params,
                **params)

    return eval_complete_f(x, y, test_x, test_y, 'catboost', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)


# XGBoost
# Hyperparameter space: https://arxiv.org/pdf/2106.03253.pdf
param_grid_hyperopt['xgb'] = {
    'learning_rate': hp.loguniform('learning_rate', -7, math.log(1)),
    'max_depth': hp.randint('max_depth', 1, 10),
    'subsample': hp.uniform('subsample', 0.2, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1),
    'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
    'alpha': hp.loguniform('alpha', -16, 2),
    'lambda': hp.loguniform('lambda', -16, 2),
    'gamma': hp.loguniform('gamma', -16, 2),
    'n_estimators': hp.randint('n_estimators', 100, 4000), # This is smaller than in paper
}

def xgb_metric(x, y, test_x, test_y, cat_features, metric_used, seed, max_time=300, no_tune=None, gpu_id=None, preprocess='standard'):
    import xgboost as xgb
    # XGB Documentation:
    # XGB handles categorical data appropriately without using One Hot Encoding, categorical features are experimetal
    # XGB handles missing values appropriately without imputation
    
    if gpu_id is not None:
        print("Running on gpu")
        gpu_params = {'tree_method':'gpu_hist', 'gpu_id':gpu_id}
    else:
        gpu_params = {}
    one_hot_encode = 'one_hot' in preprocess
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=one_hot_encode,
                                             cat_features=cat_features,
                                             impute=False,
                                             standardize=False)

    def clf_(**params):
        if is_classification(metric_used):
            return xgb.XGBClassifier(use_label_encoder=False,
                                     nthread=MULTITHREAD,
                                     random_state=seed,
                                     **params,
                                     **gpu_params,
                                     eval_metric=get_scoring_string(metric_used, usage='xgb') # AUC not implemented
            )
        else:
            return xgb.XGBRegressor(use_label_encoder=False,
                                    nthread=MULTITHREAD,
                                    random_state=seed,
                                    **params,
                                    **gpu_params,
                                    eval_metric=get_scoring_string(metric_used, usage='xgb')  # AUC not implemented
                                    )
                                    
    return eval_complete_f(x, y, test_x, test_y, 'xgb', clf_,
                           metric_used=metric_used,
                           seed=seed,
                           max_time=max_time,
                           no_tune=no_tune)

"""
LEGACY UNUSED
"""

## Ridge
from sklearn.linear_model import RidgeClassifier
param_grid['ridge'] = {'alpha': [0, 0.1, .5, 1.0, 2.0], 'fit_intercept': [True, False]} # 'normalize': [False],
def ridge_metric(x, y, test_x, test_y, cat_features, metric_used):
    import warnings
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn

    x, y, test_x, test_y = x.cpu(), y.cpu(), test_x.cpu(), test_y.cpu()
    x, test_x = torch.nan_to_num(x), torch.nan_to_num(test_x)

    clf = RidgeClassifier(n_jobs=MULTITHREAD)

    # create a dictionary of all values we want to test for n_neighbors
    # use gridsearch to test all values for n_neighbors
    clf = GridSearchCV(clf, param_grid['ridge'], cv=min(CV, x.shape[0]//2)
                       , scoring=get_scoring_string(metric_used)
                       , n_jobs=MULTITHREAD)
    # fit model to data
    clf.fit(x, y.long())

    pred = clf.decision_function(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred

def mlp_acc(x, y, test_x, test_y, hyperparameters):
    num_layers, hidden_dim, activation_module, fixed_dropout_prob, is_binary_classification, epochs, lr, weight_decay = hyperparameters
    num_features = x.shape[1]

    x, y = x.to(device), y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    def get_model():
        model = nn.Sequential(*[
            module for layer_idx in range(num_layers) for module in [
                nn.Linear(hidden_dim if layer_idx > 0 else num_features,
                          2 if layer_idx == num_layers - 1 else hidden_dim),
                torch.nn.Identity() if layer_idx == num_layers - 1 else activation_module(),
                torch.nn.Identity() if layer_idx == num_layers - 1 else torch.nn.Dropout(p=fixed_dropout_prob,
                                                                                         inplace=False)]
        ])
        if is_binary_classification:
            model.add_module(str(len(model)), torch.nn.Softmax(dim=1))  # TODO might also just do an round!?
        return model

    model = get_model().to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x)[:, 1]
        # Compute Loss

        loss = criterion(y_pred.squeeze(), y.float())

        # print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()

    model.eval()
    pred_y = model(test_x)[:, 1] > 0.5
    acc = (pred_y == test_y).float().mean()
    return acc

clf_dict = {'gp': gp_metric
, 'transformer': transformer_metric
, 'random_forest': random_forest_metric
                , 'knn': knn_metric
                , 'catboost': catboost_metric
                , 'tabnet': tabnet_metric
                , 'xgb': xgb_metric
                , 'lightgbm': lightgbm_metric
            , 'ridge': ridge_metric
                , 'logistic': logistic_metric
           , 'autosklearn': autosklearn_metric
             , 'autosklearn2': autosklearn2_metric
            , 'autogluon': autogluon_metric,
            'cocktail': well_tuned_simple_nets_metric,
            "naiveautoml": naiveatuoml_metric}
