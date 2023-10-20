import pandas as pd
import os
import sys

import numpy as np
import mlflow

from experiment.batch.auto_profile_semi_supervised_experiment import AutoProfileSemiSupervisedPdMExperiment
from experiment.batch.unsupervised_experiment import UnsupervisedPdMExperiment
from experiment.batch.incremental_semi_supervised_experiment import IncrementalSemiSupervisedPdMExperiment
from experiment.batch.semi_supervised_experiment import SemiSupervisedPdMExperiment
from pipeline.pipeline import PdMPipeline
from method.profile_based import ProfileBased
from method.ocsvm import OneClassSVM
from method.isolation_forest import IsolationForest
from method.usad import usad
from method.TranADPdM import TranADPdM
from method.NPsemi import NeighborProfileSemi
from method.dist_k_Semi import Distance_Based_Semi
from pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple
from preprocessing.record_level.default import DefaultPreProcessor
from postprocessing.default import DefaultPostProcessor
from thresholding.constant import ConstantThresholder
from preprocessing.record_level.min_max_scaler import MinMaxScaler
from constraint_functions.constraint import self_tuning_constraint_function, incremental_constraint_function, \
    combine_constraint_functions, auto_profile_max_wait_time_constraint, incremental_max_wait_time_constraint
from method.ltsf_linear.ltsf_linear import LTSFLinear
from method.lof_semi import LocalOutlierFactor
from method.HBOS import HBOS
from method.PCA import PCA_semi
from method.cnn import Cnn
from method.forecast import ForecastingAnomalyPredictionMethod
from postprocessing.self_tuning import SelfTuningPostProcessor
from utils import loadDataset
from utils import automatic_parameter_generation
from utils.utils import calculate_mango_parameters
from postprocessing.min_max_scaler import MinMaxPostProcessor


def execute(method_names_to_run,dataset_name="smd", MAX_RUNS=100, MAX_JOBS=16, INITIAL_RANDOM=4, setup=40):
    print(f"script: {dataset_name}/run_online.py")

    tracking_uri = mlflow.get_tracking_uri()
    print("MLflow Tracking URI:", tracking_uri)

    dataset = loadDataset.get_dataset(f"{dataset_name}", setup_1_period=setup, reset_after_fail=False)
    experiments = [
        AutoProfileSemiSupervisedPdMExperiment,
    ]

    experiment_names = [
        f'TSB Auto profile {dataset_name}',
    ]

    methods = [
        IsolationForest,
        OneClassSVM,
        Distance_Based_Semi,
        ProfileBased,
        NeighborProfileSemi,
        LocalOutlierFactor,
        HBOS,
        PCA_semi,
        Cnn,
        ForecastingAnomalyPredictionMethod
    ]

    method_names = [
        'IF',
        'OCSVM',
        'KNN',
        'PB',
        'NP',
        'LOF',
        'HBOS',
        'PCA',
        'CNN',
        'ForecastingAnomalyPrediction'
    ]

    param_space_dict_per_method = [automatic_parameter_generation.online_technique(method_name, dataset["max_wait_time"],multivariate=False) for method_name in method_names]

    for current_method, current_method_param_space, current_method_name in zip(methods, param_space_dict_per_method, method_names):
        print(f"Current method: {current_method}")
        if current_method_name not in method_names_to_run:
            continue

        postprocessor = DefaultPostProcessor
        if len(sys.argv) > 1:
            if sys.argv[1] == 'minmax':
                postprocessor = MinMaxPostProcessor

        my_pipeline = PdMPipeline(
            steps={
                'preprocessor': DefaultPreProcessor,
                'method': current_method,
                'postprocessor': postprocessor,
                'thresholder': ConstantThresholder,
            },
            dataset=dataset,
            auc_resolution=100
        )

        for experiment, experiment_name in zip(experiments, experiment_names):
            current_param_space_dict = {
                'thresholder_threshold_value': [0.5],
            }

            current_param_space_dict['profile_size'] = automatic_parameter_generation.profile_values(dataset["max_wait_time"])

            for key, value in current_method_param_space.items():
                current_param_space_dict[f'method_{key}'] = value

            num, jobs, initial_random = calculate_mango_parameters(current_param_space_dict, MAX_JOBS,
                                                                   INITIAL_RANDOM,
                                                                   MAX_RUNS)

            my_experiment = experiment(
                experiment_name=experiment_name + ' ' + current_method_name,
                target_data=dataset['target_data'],
                target_sources=dataset['target_sources'],
                pipeline=my_pipeline,
                param_space=current_param_space_dict,
                num_iteration=num,
                n_jobs=jobs,
                initial_random=initial_random,
                artifacts='./artifacts/' + experiment_name + ' artifacts',
                constraint_function=auto_profile_max_wait_time_constraint(my_pipeline),
                debug=True,
                log_best_scores = True,
                optimization_param= 'AD1_AUC'
            )

            best_params = my_experiment.execute()
            print(experiment_name)
            print(best_params)
