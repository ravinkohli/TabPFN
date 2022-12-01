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


def get_updates_for_autopytorch_tabular(categorical_indicator: List[bool]):
    from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
    # from autoPyTorch.constants import MIN_CATEGORIES_FOR_EMBEDDING_MAX


    """
    These updates mimic the autopytorch tabular paper.
    Returns:
    ________
    search_space_updates - HyperparameterSearchSpaceUpdates
        The search space updates like setting different hps to different values or ranges.
    """


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
        value_range=['ShapedResNetBackbone', 'ShapedMLPBackbone'],
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
        hyperparameter='ShapedResNetBackbone:dropout_shape',
        value_range=['funnel'],
        default_value='funnel',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:max_dropout',
        value_range=[0, 1],
        default_value=0.5,
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
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:use_skip_connection',
        value_range=[True],
        default_value=True,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:use_batch_norm',
        value_range=[True],
        default_value=True,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:shake_shake_update_func',
        value_range=['shake-shake'],
        default_value='shake-shake',
    )
    # mlp backbone
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedMLPBackbone:mlp_shape',
        value_range=['funnel'],
        default_value='funnel',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedMLPBackbone:num_groups',
        value_range=[1, 5],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedMLPBackbone:output_dim',
        value_range=[64, 1024],
        default_value=64,
        log=True
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedMLPBackbone:max_units',
        value_range=[64, 1024],
        default_value=64,
        log=True
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedMLPBackbone:activation',
        value_range=['relu'],
        default_value='relu',
    )

    # training updates
    # lr scheduler
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='__choice__',
        value_range=['CosineAnnealingLR'],
        default_value='CosineAnnealingLR',
    )
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='CosineAnnealingLR:T_max',
        value_range=[50],
        default_value=50,
    )
    # optimizer
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='__choice__',
        value_range=['AdamOptimizer', 'SGDOptimizer'],
        default_value='AdamOptimizer',
    )
    # adam
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamOptimizer:lr',
        value_range=[1e-4, 1e-1],
        default_value=1e-3,
        log=True
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamOptimizer:use_weight_decay',
        value_range=[True],
        default_value=True,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamOptimizer:weight_decay',
        value_range=[1e-5, 1e-1],
        default_value=1e-3,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamOptimizer:beta1',
        value_range=[0.9],
        default_value=0.9,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamOptimizer:beta2',
        value_range=[0.999],
        default_value=0.999,
    )
    # sgd
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='SGDOptimizer:lr',
        value_range=[1e-4, 1e-1],
        default_value=1e-3,
        log=True
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='SGDOptimizer:use_weight_decay',
        value_range=[True],
        default_value=True,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='SGDOptimizer:weight_decay',
        value_range=[1e-5, 1e-1],
        default_value=1e-3,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='SGDOptimizer:momentum',
        value_range=[0.1, 0.999],
        default_value=0.1,
        log=True
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
            node_name='encoder',
            hyperparameter='__choice__',
            value_range=['OneHotEncoder'],
            default_value='OneHotEncoder',
        )
    # trainer
    trainer_choices = ['StandardTrainer', 'MixUpTrainer']
    search_space_updates.append(
        node_name='trainer',
        hyperparameter='__choice__',
        value_range=trainer_choices,
        default_value=trainer_choices[0],
    )
    for trainer_choice in trainer_choices:
        search_space_updates.append(
            node_name='trainer',
            hyperparameter=f'{trainer_choice}:use_lookahead_optimizer',
            value_range=[False],
            default_value=False,
        )
        search_space_updates.append(
            node_name='trainer',
            hyperparameter=f'{trainer_choice}:use_snapshot_ensemble',
            value_range=[False],
            default_value=False,
        )
        search_space_updates.append(
            node_name='trainer',
            hyperparameter=f'{trainer_choice}:use_stochastic_weight_averaging',
            value_range=[False],
            default_value=False,
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

def autopytorch_master_default_metric(X_train, y_train, X_test, y_test, categorical_indicator, metric_used, seed, max_time=300, nr_workers=1):
    """Install:
    git clone https://github.com/automl/Auto-PyTorch.git
    cd Auto-PyTorch
    git checkout reg_cocktails
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

        if hasattr(X_train, 'cpu'):
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
        # No early stopping, train on cpu
        pipeline_update = {
            'early_stopping': 20,
            "device": 'cpu',
        }

        # Early stopping disabled for the experiments on LPER, training on cpu

        api = TabularClassificationTask(
            temporary_directory=temp_dir,
            output_directory=out_dir,
            delete_tmp_folder_after_terminate=False,
            delete_output_folder_after_terminate=False,
            resampling_strategy=HoldoutValTypes.stratified_holdout_validation,
            resampling_strategy_args=resampling_strategy_args,
            ensemble_size=1,
            ensemble_nbest=1,
            max_models_on_disc=5,
            search_space_updates=search_space_updates,
            seed=seed,
            n_jobs=nr_workers,
            n_threads=1,
        )

        api.set_pipeline_config(**pipeline_update)
        ############################################################################
        # Search for the best hp configuration
        # ====================================
        # We search for the best hp configuration only in the case of a cocktail ingredient
        # that has hyperparameters.
        print('temp_dir',temp_dir)
        # print(max_time, min(func_eval_time, max_time, number_of_configurations_limit))      

        ############################################################################
        # Refit on the best hp configuration
        # ==================================
        print(f"Fitting pipeline with {epochs} epochs")

        dataset = api.get_dataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            resampling_strategy=NoResamplingStrategyTypes.no_resampling,
        )
        dataset.is_small_preprocess = False

        search_space = api.get_search_space(dataset)
        configuration = search_space.get_default_configuration()
        print(f"Default configuration: {configuration}")

        fitted_pipeline, run_info, run_value, dataset = api.fit_pipeline(
            configuration=configuration,
            dataset=dataset,
            budget_type='epochs',
            budget=epochs,
            run_time_limit_secs=max_time,
            eval_metric='balanced_accuracy',
            memory_limit=12000,
        )

        X_train = dataset.train_tensors[0]
        y_train = dataset.train_tensors[1]
        X_test = dataset.test_tensors[0]
        y_test = dataset.test_tensors[1]

        if fitted_pipeline is not None:
            test_predictions = fitted_pipeline.predict(X_test)
        else:
            raise ValueError(f"Fitting autopytorch failed due to: {run_value.additional_info}")

        metric = metric_used(y_test, test_predictions.squeeze())
        duration = time.time() - start_time

        print(f'Time taken: {duration} for {metric} metric')
        return metric, test_predictions.squeeze(), None

