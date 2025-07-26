from time import sleep

import streamlit as st
st.set_page_config(initial_sidebar_state="collapsed",layout="wide")

import subprocess
import os
import socket


def is_port_in_use(host, port):
    """Check if a given port is in use on the specified host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def run_mlflow_server():
    """Function to run MLflow server if it's not already running."""
    host = "127.0.0.1"
    port = 8080

    if is_port_in_use(host, port):
        st.write(f"MLflow server is already running at http://{host}:{port}.")
    else:
        st.write("Starting MLflow server...")
        subprocess.Popen(["mlflow", "ui", "--host", host, "--port", str(port)])
        st.write(f"MLflow server started at http://{host}:{port}.")


def run_experiment(params):
    """Function to run your ML experiment."""
    # You would replace this with the actual code to run your experiment
    st.write("Running experiment with parameters:", params)
    # Example command to run your tool
    # subprocess.run(["python", "your_ml_tool.py", "--param1", params["param1"], "--param2", params["param2"]])

def change_flavors():
    options_dict = {
        "Semi-Supervised": ["Online", "Sliding", "Historical"],
        "Unsupervised": ["Unsupervised"],
    }
    st.session_state.flavor_options = options_dict[st.session_state.methodlogy]

    st.session_state.method_options = method_options_dict[st.session_state.methodlogy]

def create_buttons():


    dataset, flavorsc, prepro,methodse,postpro = st.columns([1, 2,2,2,2])
    with dataset:
        #dictionaries/{st.session_state.dataset_name}.pickle


        display_dataset=alldataset.copy()
        display_dataset.extend(dictionary_names)
        dataset_name = st.selectbox(
            "Select Dataset",
            display_dataset,
            key='dataset_name'
        )
    with flavorsc:
        methodlogy = st.selectbox(
            "Select Methodology",
            ["Semi-Supervised", "Unsupervised"], on_change=change_flavors, key='methodlogy'
        )
        # Define the options for each category

        # Second parameter: Multi-select based on the first selection
        multi_flavor_select = st.multiselect(
            "Select flavors",
            st.session_state.flavor_options, key='multi_flavor_select'
        )
    with prepro:
        Pre_processing_select = st.selectbox(
            "Pre-processing",
            ["Default","Keep Features","MinMax Scaler (semi)","Windowing (one column)", "Mean Aggregator"], key='Pre_processing_select'
        )

    with methodse:
        multi_method_select = st.multiselect(
            "Methods",
            st.session_state.method_options, key='multi_method_select'
        )
    with postpro:
        Post_processing_select = st.selectbox(
            "Post-processing",
            ["Default","Dynamic Threshold","Moving2T","SelfTuning"], key='Post_processing_select'
        )
if st.button('Home'):
    st.switch_page("pages/mainUI.py")
alldataset=[
                "cmapss", "navarchos", "femto",
                "ims", "edp-wt", "metropt-3", "xjtu", "bhd", "azure", "ai4i"
            ]

folder_path = "./DataFolder/dictionaries"
filenames = os.listdir(folder_path)
filenames = [f for f in filenames if os.path.isfile(os.path.join(folder_path, f))]
dictionary_names = [os.path.basename(filename).split(".")[0] for filename in filenames]

# UI for experiment configuration
st.title("Experiment Configuration")
method_options_dict = {
        "Semi-Supervised":['LOF','KNN','PB','NP','OCSVM','IF','LTSF','TRANAD','USAD','HBOS','PCA','CNN','DummyIncrease','CHRONOS'],
        "Unsupervised":['LOF','KNN','NP','IF','SAND','DAMP','HBOS','PCA']
    }
if "flavor_options" not in st.session_state:
    st.session_state.flavor_options = ["Online", "Sliding", "Historical"]
    st.session_state.method_options = method_options_dict["Semi-Supervised"]
    st.session_state.selected_method_options = []
    st.session_state.multi_flavor_select=[]
    st.session_state.multi_method_select=[]
    st.session_state.Pre_processing_select="Default"
    st.session_state.Post_processing_select="Default"
create_buttons()



if st.button('Parameters'):
    # Capture the experiment configuration
    current_name=st.session_state.dataset_name
    # print(alldataset)
    # print(dictionary_names)
    if current_name in dictionary_names:
        current_name=f"dictionaries/{current_name}.pickle"
    # print(current_name)
    params = {
        "dataset_name": current_name,
        "methodology": st.session_state.methodlogy,
        "select_flavors": st.session_state.multi_flavor_select,
        "pre-processing": st.session_state.Pre_processing_select,
        "methods": st.session_state.multi_method_select,
        "post-processing": st.session_state.Post_processing_select,
    }
    if "parameters" not in st.session_state:
        st.session_state.parameters=params
    else:
        st.session_state.parameters=params
    sleep(0.1)
    st.switch_page("pages/pages0.py")

    # Run the experiment
    # run_experiment(params)

    # # Start MLflow server if not already running
    # run_mlflow_server()
    #
    # # Display MLflow UI
    # st.write("Experiment started. Check the results [here](http://127.0.0.1:8080).")
    #st.markdown('<iframe src="http://127.0.0.1:8080" width="100%" height="600"></iframe>', unsafe_allow_html=True)