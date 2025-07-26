# from experimental_runs_configuration.edp.run_unsupervised import execute as execute_edp_unsupervised

from experimental_runs_configuration.bhd.run_online import execute as execute_bhd_online
# from experimental_runs_configuration.xjtu.run_incremental import execute as execute_xjtu_incremental
# from experimental_runs_configuration.xjtu.run_unsupervised import execute as execute_xjtu_unsupervised

# from experimental_runs_configuration.azure.run_online import execute as execute_azure_online
# from experimental_runs_configuration.azure.run_incremental import execute as execute_azure_incremental
# from experimental_runs_configuration.azure.run_unsupervised import execute as execute_azure_unsupervised

# from experimental_runs_configuration.bhd.run_online import execute as execute_bhd_online
# from experimental_runs_configuration.bhd.run_incremental import execute as execute_bhd_incremental
# from experimental_runs_configuration.bhd.run_unsupervised import execute as execute_bhd_unsupervised

# from experimental_runs_configuration.ai4i.run_online import execute as execute_ai4i_online
# from experimental_runs_configuration.ai4i.run_incremental import execute as execute_ai4i_incremental
# from experimental_runs_configuration.navarchos.run_unsupervised import execute as execute_navarchos_unsupervised

from utils import loadDataset

methods_to_run = [
        # 'IF',
        # 'OCSVM',
        # 'PB',
        # 'KNN',
        # 'NP',
        # 'LOF',
        # 'TRANAD',
        # 'LTSF',
        # 'USAD'
    # 'CNN'
       'CHRONOS'
]

methods_to_run_unsupervised = [
        # 'NP',
        #'KNN',
        #'IF',
        #'LOF',
        # 'SAND',
        # 'CHRONOS'
]

MAX_JOBS = 1
MAX_RUNS = 6
INITIAL_RANDOM = 1

# dataset = loadDataset.get_dataset("bhd")


# execute_edp_unsupervised(method_names_to_run=methods_to_run_unsupervised, MAX_RUNS=MAX_RUNS, MAX_JOBS=MAX_JOBS, INITIAL_RANDOM=INITIAL_RANDOM)
execute_bhd_online(method_names_to_run=methods_to_run, MAX_RUNS=MAX_RUNS, MAX_JOBS=MAX_JOBS, INITIAL_RANDOM=INITIAL_RANDOM)
# execute_azure_incremental(method_names_to_run=methods_to_run, MAX_RUNS=MAX_RUNS, MAX_JOBS=MAX_JOBS, INITIAL_RANDOM=INITIAL_RANDOM)
# execute_ai4i_incremental(method_names_to_run=methods_to_run, MAX_RUNS=MAX_RUNS, MAX_JOBS=MAX_JOBS, INITIAL_RANDOM=INITIAL_RANDOM)

# execute_bhd_unsupervised(method_names_to_run=methods_to_run_unsupervised, MAX_RUNS=MAX_RUNS, MAX_JOBS=MAX_JOBS, INITIAL_RANDOM=INITIAL_RANDOM)

# execute_xjtu_incremental(method_names_to_run=methods_to_run, MAX_RUNS=MAX_RUNS, MAX_JOBS=MAX_JOBS, INITIAL_RANDOM=INITIAL_RANDOM)
# (method_names_to_run=methods_to_run_unsupervised, MAX_RUNS=MAX_RUNS, MAX_JOBS=MAX_JOBS, INITIAL_RANDOM=INITIAL_RANDOM)
#execute_ai4i_online(method_names_to_run=methods_to_run, MAX_RUNS=MAX_RUNS, MAX_JOBS=MAX_JOBS, INITIAL_RANDOM=INITIAL_RANDOM)
