import numpy as np
import os
import time
import pandas as pd
import random
import tempfile


def get_updates_for_regularization_cocktails(
    categorical_indicator: np.ndarray):
    """
    These updates replicate the regularization cocktail paper search space.
    Args:
        categorical_indicator (np.ndarray)
            An array that indicates whether a feature is categorical or not.
    Returns:
    ________
        pipeline_update, search_space_updates, include_updates (Tuple[dict, HyperparameterSearchSpaceUpdates, dict]):
            The pipeline updates like number of epochs, budget, seed etc.
            The search space updates like setting different hps to different values or ranges.
            Lastly include updates, which can be used to include different features.
    """
    from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

    include_updates = dict()
    include_updates['network_embedding'] = ['NoEmbedding']
    include_updates['network_init'] = ['NoInit']

    has_cat_features = any(categorical_indicator)
    has_numerical_features = not all(categorical_indicator)

    def str2bool(v):
        if isinstance(v, bool):
            return [v, ]
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return [True, ]
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return [False, ]
        elif v.lower() == 'conditional':
            return [True, False]
        else:
            raise ValueError('No valid value given.')
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

    # backbone architecture
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='__choice__',
        value_range=['ShapedResNetBackbone'],
        default_value='ShapedResNetBackbone',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:resnet_shape',
        value_range=['brick'],
        default_value='brick',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:num_groups',
        value_range=[2],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:blocks_per_group',
        value_range=[2],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:output_dim',
        value_range=[512],
        default_value=512,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:max_units',
        value_range=[512],
        default_value=512,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:activation',
        value_range=['relu'],
        default_value='relu',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:shake_shake_update_func',
        value_range=['even-even'],
        default_value='even-even',
    )

    # training updates
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='__choice__',
        value_range=['CosineAnnealingWarmRestarts'],
        default_value='CosineAnnealingWarmRestarts',
    )
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='CosineAnnealingWarmRestarts:n_restarts',
        value_range=[3],
        default_value=3,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='__choice__',
        value_range=['AdamWOptimizer'],
        default_value='AdamWOptimizer',
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamWOptimizer:lr',
        value_range=[1e-3],
        default_value=1e-3,
    )
    search_space_updates.append(
        node_name='data_loader',
        hyperparameter='batch_size',
        value_range=[128],
        default_value=128,
    )

    # preprocessing
    search_space_updates.append(
        node_name='feature_preprocessor',
        hyperparameter='__choice__',
        value_range=['NoFeaturePreprocessor'],
        default_value='NoFeaturePreprocessor',
    )

    if has_numerical_features:
        print('has numerical features')
        search_space_updates.append(
            node_name='imputer',
            hyperparameter='numerical_strategy',
            value_range=['median'],
            default_value='median',
        )
        search_space_updates.append(
            node_name='scaler',
            hyperparameter='__choice__',
            value_range=['StandardScaler'],
            default_value='StandardScaler',
        )

    if has_cat_features:
        print('has cat features')
        search_space_updates.append(
            node_name='imputer',
            hyperparameter='categorical_strategy',
            value_range=['constant_!missing!'],
            default_value='constant_!missing!',
        )
        search_space_updates.append(
            node_name='encoder',
            hyperparameter='__choice__',
            value_range=['OneHotEncoder'],
            default_value='OneHotEncoder',
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

    # if the cash formulation of the cocktail is not activated,
    # otherwise the methods activation will be chosen by the SMBO optimizer.


    # No early stopping and train on gpu
    pipeline_update = {
        'early_stopping': -1,
        'min_epochs': 105,
        'epochs': 105,
        "device": 'cpu',
    }

    return pipeline_update, search_space_updates, include_updates

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


def well_tuned_simple_nets_metric(X_train, y_train, X_test, y_test, categorical_indicator, metric_used, seed, max_time=300, nr_workers=1, **kwargs):
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
        # temp_dir = "/work/dlclarge1/rkohli-run-autopytorch/roc_bug_fix_debug/"
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

        # number_of_configurations_limit = 400 # for run on gaels datasets
        epochs = 105
        func_eval_time = min(1000, max_time/2)

        resampling_strategy_args = {
            'val_share': len(y_test)/(len(y_test)+len(y_train)),
        }

        pipeline_update, search_space_updates, include_updates = get_updates_for_regularization_cocktails(
            categorical_indicator,
        )


        ############################################################################
        # Build and fit a classifier
        # ==========================
        # if we use HPO, we can use multiple workers in parallel
        # if number_of_configurations_limit == 0:
        #     nr_workers = 1

        print(f"Running with: {nr_workers}")
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
            include_components=include_updates,
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

        api.search(
            X_train=X_train.copy(),
            y_train=y_train.copy(),
            X_test=X_test.copy(),
            y_test=y_test.copy(),
            optimize_metric='roc_auc',
            # optimize_metric='balanced_accuracy',
            total_walltime_limit=max_time,
            all_supported_metrics=False,
            memory_limit=12000,
            func_eval_time_limit_secs=min(func_eval_time, max_time),
            enable_traditional_pipeline=False,
            get_smac_object_callback=get_smac_object,
            # smac_scenario_args={
            #     'runcount_limit': number_of_configurations_limit,
            # },
        )

        test_predictions = api.predict_proba(X_test=X_test)
        print(test_predictions)
        ############################################################################
        # Refit on the best hp configuration
        # ==================================
        input_validator = TabularInputValidator(
            is_classification=True,
        )
        input_validator.fit(
            X_train=X_train.copy(),
            y_train=y_train.copy(),
            X_test=X_test.copy(),
            y_test=y_test.copy(),
        )

        dataset = TabularDataset(
            X=X_train,
            Y=y_train,
            X_test=X_test,
            Y_test=y_test,
            seed=seed,
            validator=input_validator,
            resampling_strategy=NoResamplingStrategyTypes.no_resampling,
        )
        dataset.is_small_preprocess = False
        print(f"Fitting pipeline with {epochs} epochs")

        search_space = api.get_search_space(dataset)
        # only when we perform hpo will there be an incumbent configuration
        # otherwise take a default configuration.
        configuration, incumbent_run_value = get_incumbent_results(
            os.path.join(
                temp_dir,
                'smac3-output',
                'run_{}'.format(seed),
                'runhistory.json'),
            search_space,
        )
        print(f"Incumbent configuration: {configuration}")
        print(f"Incumbent trajectory: {api.trajectory}")

        fitted_pipeline, run_info, run_value, dataset = api.fit_pipeline(
            configuration=configuration,
            budget_type='epochs',
            budget=epochs,
            dataset=dataset,
            run_time_limit_secs=func_eval_time,
            eval_metric='roc_auc',
            # eval_metric='balanced_accuracy',
            memory_limit=12000,
        )

        X_train = dataset.train_tensors[0]
        y_train = dataset.train_tensors[1]
        X_test = dataset.test_tensors[0]
        y_test = dataset.test_tensors[1]

        if fitted_pipeline is not None:
            test_predictions = fitted_pipeline.predict_proba(X_test)

        metric = metric_used(y_test, test_predictions.squeeze())
        duration = time.time() - start_time

        print(f'Time taken: {duration} for {metric} metric')
        return metric, test_predictions, None
