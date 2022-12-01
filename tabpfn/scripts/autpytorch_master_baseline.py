import numpy as np
import os
import time
import pandas as pd
import random
import tempfile

import json
import numpy as np
import openml
import pandas as pd
from typing import List

from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.constants import MIN_CATEGORIES_FOR_EMBEDDING_MAX


def get_updates_for_autopytorch_tabular(categorical_indicator: List[bool]):
    
    """
    These updates mimic the autopytorch tabular paper.
    Returns:
    ________
    search_space_updates - HyperparameterSearchSpaceUpdates
        The search space updates like setting different hps to different values or ranges.
    """

    search_space_updates = HyperparameterSearchSpaceUpdates()

    # updates for autopytorch tabular paper
    has_cat_features = any(categorical_indicator)
    has_numerical_features = not all(categorical_indicator)

    search_space_updates = HyperparameterSearchSpaceUpdates()

    # architecture head
    search_space_updates.append(
        node_name='network_head',
        hyperparameter='__choice__',
        value_range=['no_head'],
        default_value='no_head',
    )
    search_space_updates.append(
        node_name='network_head',
        hyperparameter='no_head:activation',
        value_range=['relu'],
        default_value='relu',
    )

    # weights initialisation
    search_space_updates.append(
        node_name='network_init',
        hyperparameter='__choice__',
        value_range=['NoInit'],
        default_value='NoInit',
    )
    search_space_updates.append(
        node_name='network_init',
        hyperparameter='NoInit:bias_strategy',
        value_range=['Zero'],
        default_value='Zero',
    )

    # backbone architecture choices
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='__choice__',
        value_range=['ShapedResNetBackbone'],
        default_value='ShapedResNetBackbone',
    )

    # resnet backbone
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:resnet_shape',
        value_range=['funnel'],
        default_value='funnel',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:num_groups',
        value_range=[1, 4],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:blocks_per_group',
        value_range=[1, 3],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:output_dim',
        value_range=[32, 512],
        default_value=64,
        log=True
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:max_units',
        value_range=[32, 512],
        default_value=64,
        log=True
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:activation',
        value_range=['relu'],
        default_value='relu',
    )

    # training updates
    # lr scheduler
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='__choice__',
        value_range=['CosineAnnealingWarmRestarts'],
        default_value='CosineAnnealingWarmRestarts',
    )

    # optimizer
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='__choice__',
        value_range=['AdamWOptimizer'],
        default_value='AdamWOptimizer',
    )
    # adamw
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamWOptimizer:lr',
        value_range=[1e-4, 1e-1],
        default_value=1e-3,
        log=True
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamWOptimizer:beta1',
        value_range=[0.9],
        default_value=0.9,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamWOptimizer:beta2',
        value_range=[0.999],
        default_value=0.999,
    )
    search_space_updates.append(
        node_name='data_loader',
        hyperparameter='batch_size',
        value_range=[16, 512],
        default_value=128,
        log=True
    )

    # preprocessing
    if has_numerical_features:
        search_space_updates.append(
            node_name='feature_preprocessor',
            hyperparameter='__choice__',
            value_range=['NoFeaturePreprocessor', 'TruncatedSVD'],
            default_value='NoFeaturePreprocessor',
        )
        search_space_updates.append(
            node_name='feature_preprocessor',
            hyperparameter='TruncatedSVD:target_dim',
            value_range=[0.1, 0.9],
            default_value=0.4,
        )
        search_space_updates.append(
            node_name='imputer',
            hyperparameter='numerical_strategy',
            value_range=['mean'],
            default_value='mean',
        )
        search_space_updates.append(
            node_name='scaler',
            hyperparameter='__choice__',
            value_range=['StandardScaler'],
            default_value='StandardScaler',
        )

    if has_cat_features:
        search_space_updates.append(
            node_name='column_splitter',
            hyperparameter='min_categories_for_embedding',
            value_range=(1, MIN_CATEGORIES_FOR_EMBEDDING_MAX),
            default_value=3
        )

    return search_space_updates


def get_smac_object(
    scenario_dict,
    seed: int,
    ta,
    ta_kwargs,
    n_jobs: int,
    initial_budget: int,
    max_budget: int,
    dask_client,
):
    """
    This function returns an SMAC object that is gonna be used as
    optimizer of pipelines.
    Args:
        scenario_dict (typing.Dict[str, typing.Any]): constrain on how to run
            the jobs.
        seed (int): to make the job deterministic.
        ta (typing.Callable): the function to be intensified by smac.
        ta_kwargs (typing.Dict[str, typing.Any]): Arguments to the above ta.
        n_jobs (int): Amount of cores to use for this task.
        initial_budget (int):
            The initial budget for a configuration.
        max_budget (int):
            The maximal budget for a configuration.
        dask_client (dask.distributed.Client): User provided scheduler.
    Returns:
        (SMAC4AC): sequential model algorithm configuration object
    """
    from smac.intensification.simple_intensifier import SimpleIntensifier
    from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
    from smac.scenario.scenario import Scenario
    from smac.facade.smac_ac_facade import SMAC4AC
    # multi-fidelity is disabled, that is why initial_budget and max_budget
    # are not used.
    rh2EPM = RunHistory2EPM4LogCost

    return SMAC4AC(
        scenario=Scenario(scenario_dict),
        rng=seed,
        runhistory2epm=rh2EPM,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        initial_configurations=None,
        run_id=seed,
        intensifier=SimpleIntensifier,
        dask_client=dask_client,
        n_jobs=n_jobs,
    )


def get_incumbent_results(
    run_history_file: str,
    search_space
):
    """
    Get the incumbent configuration and performance from the previous run HPO
    search with AutoPytorch.
    Args:
        run_history_file (str):
            The path where the AutoPyTorch search data is located.
        search_space (ConfigSpace.ConfigurationSpace):
            The ConfigurationSpace that was previously used for the HPO
            search space.
    Returns:
        config, incumbent_run_value (Tuple[ConfigSpace.Configuration, float]):
            The incumbent configuration found from HPO search and the validation
            performance it achieved.
    """
    from smac.runhistory.runhistory import RunHistory
    run_history = RunHistory()
    run_history.load_json(
        run_history_file,
        search_space,
    )

    run_history_data = run_history.data
    sorted_runvalue_by_cost = sorted(run_history_data.items(), key=lambda item: item[1].cost)
    incumbent_run_key, incumbent_run_value = sorted_runvalue_by_cost[0]
    config = run_history.ids_config[incumbent_run_key.config_id]
    return config, incumbent_run_value

def well_tuned_simple_nets_metric(X_train, y_train, X_test, y_test, categorical_indicator, metric_used, seed, max_time=300, nr_workers=1):
    """Install:
    git clone https://github.com/automl/Auto-PyTorch.git
    cd Auto-PyTorch
    git checkout regularization_cocktails
    From the page, not needed for me at least: conda install gxx_linux-64 gcc_linux-64 swig
    conda create --clone CONDANAME --name CLONENAME
    conda activate CLONENAME
    pip install -r requirements.txt (I checked looks like nothing should break functionality of our project not sure about baselines, thus a copied env is likely good :))
    pip install -e .
    """
    #os.environ.get('SLURM_JOBID', '')
    categorical_indicator = np.array([i in categorical_indicator for i in range(X_train.shape[1])])
    with tempfile.TemporaryDirectory(prefix=f"{len(X_train)}_{len(X_test)}_{max_time}") as temp_dir:
        from autoPyTorch.api.tabular_classification import TabularClassificationTask
        from autoPyTorch.datasets.resampling_strategy import HoldoutValTypes, NoResamplingStrategyTypes
        from autoPyTorch.data.tabular_validator import TabularInputValidator
        from autoPyTorch.datasets.tabular_dataset import TabularDataset
        from autoPyTorch import metrics
        # append random folder to temp_dir to avoid collisions
        rand_int = str(random.randint(1,1000))
        temp_dir = os.path.join(temp_dir, 'temp_'+rand_int)
        out_dir = os.path.join(temp_dir, 'out_'+rand_int)

        start_time = time.time()

        X_train, y_train, X_test, y_test = X_train.cpu().numpy(), y_train.cpu().long().numpy(), X_test.cpu().numpy(), y_test.cpu().long().numpy()

        def safe_int(x):
            assert np.all(x.astype('int64') == x) or np.any(x != x), np.unique(x) # second condition for ignoring nans
            return pd.Series(x, dtype='category')

        X_train = pd.DataFrame({i: safe_int(X_train[:,i]) if c else X_train[:,i] for i, c in enumerate(categorical_indicator)})
        X_test = pd.DataFrame({i: safe_int(X_test[:,i]) if c else X_test[:,i] for i, c in enumerate(categorical_indicator)})


        if isinstance(y_train[1], bool):
            y_train = y_train.astype('bool')
        if isinstance(y_test[1], bool):
            y_test = y_test.astype('bool')

        # number_of_configurations_limit = 840 # hard coded in the paper
        epochs = 105
        func_eval_time = min(1000, max_time/2)

        resampling_strategy_args = {
            'val_share': len(y_test)/(len(y_test)+len(y_train)),
        }

        search_space_updates = get_updates_for_autopytorch_tabular(
            categorical_indicator,
        )


        ############################################################################
        # Build and fit a classifier
        # ==========================
        # if we use HPO, we can use multiple workers in parallel
        nr_workers = 1

        api = TabularClassificationTask(
            temporary_directory=temp_dir,
            output_directory=out_dir,
            delete_tmp_folder_after_terminate=False,
            delete_output_folder_after_terminate=False,
            resampling_strategy=HoldoutValTypes.stratified_holdout_validation,
            resampling_strategy_args=resampling_strategy_args,
            ensemble_size=50,
            ensemble_nbest=50,
            max_models_on_disc=50,
            search_space_updates=search_space_updates,
            seed=seed,
            n_jobs=nr_workers,
            n_threads=1,
        )

        # No early stopping, train on cpu
        pipeline_update = {
            'early_stopping': -1,
            "device": 'cpu',
        }

        # Early stopping disabled for the experiments on LPER, training on cpu
        api.set_pipeline_config(**pipeline_update)

        ############################################################################
        # Search for the best hp configuration
        # ====================================
        # We search for the best hp configuration only in the case of a cocktail ingredient
        # that has hyperparameters.
        print('temp_dir',temp_dir)
        # print(max_time, min(func_eval_time, max_time, number_of_configurations_limit))

        api.search(
            X_train=X_train.copy(),
            y_train=y_train.copy(),
            X_test=X_test.copy(),
            y_test=y_test.copy(),
            optimize_metric='balanced_accuracy',
            total_walltime_limit=max_time,
            memory_limit=12000,
            func_eval_time_limit_secs=min(func_eval_time, max_time),
            get_smac_object_callback=get_smac_object,
        )

        print(api.sprint_statistics())
        
        duration = time.time() - start_time
        print(f'Time taken: {duration}')

        # predict_function = __getattribute__(get_predict_function(metric_object))
        # train_predictions = predict_function(X_train)
        test_predictions = api.predict_proba(X_test)
        metric = metric_used(y_test, test_predictions.squeeze())
        # print(f'Time taken: {duration} for {metric} metric')
        return metric, test_predictions, None

