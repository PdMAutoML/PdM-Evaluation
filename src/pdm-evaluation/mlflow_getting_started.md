# Getting started with MlFlow

When executing an experiment with our framework a folder 'mlruns' will be created that contains information and details about the experiments you executed. 

You can still use the framework without bothering about what the 'mlruns' directory contains and just take the best hyperparameters and result as shown in the example_run_me.ipynb notebook. But in case you want to perform experimental comparison you may find useful information below.

Because MlFlow and our framework have both the concept of Experiments we start with providing a clarification below.

An MlFlow experiment corresponds to one experiment instance of our framework (for example experiment.batch.auto_profile_semi_supervised_experiment.AutoProfileSemiSupervisedPdMExperiment). Each experiment of our framework will execute several configurations of the chosen method and flavor during the hyperparameter tuning process and for each one of these methods and flavors with specific hyperparameter values will test different thresholds and store metric results only for the best threshold in MlFlow in order to enable querying and comparison capebilities across different configurations.

For inspecting MlFlow results you can follow the steps below:

Activate the environment

```
conda activate PdM-Evaluation
```

Run the MlFlow server (in the same directory level as the 'mlruns' directory)

```
mlflow server --host 127.0.0.1 --port 8080
```

Go to https://127.0.0.1:8080/ and you can navigate through the executed experiments.