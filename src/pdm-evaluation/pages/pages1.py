import tempfile
import threading
import time

import streamlit as st
st.set_page_config(initial_sidebar_state="collapsed",layout="wide")
from utils.automatic_parameter_generation import online_technique,profile_values,unsupervised_technique,incremental_windows
from utils.automatic_parameter_generation import pre_proccessing_params, post_proccessing_params
import subprocess
import os
import socket
import streamlit as st
import sys
import io
from multiprocessing import Process
import pickle



# def get_parameters(method_name):
#
#     if params["methodology"]=="Semi-Supervised":
#         online_technique(method_name,params["dataset"]["max_wait_time"])
#     else:
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






def get_flavor_parameters(flavor_name,dataset):
    experiment_names=["Online", "Sliding", "Historical","Unsupervised"]
    #parameters_ = [["profile_size"], ['initial_incremental_window_length','incremental_window_length','incremental_slide'], [],[]]

    if flavor_name=="Online":
        p_dict={
            "profile_size":profile_values(dataset["max_wait_time"])
        }
        return p_dict
    elif flavor_name=="Sliding":
        p_dict={}
        incremental_slide, initial_incremental_window_length, incremental_window_length = incremental_windows(
            dataset["max_wait_time"])
        p_dict['initial_incremental_window_length'] = initial_incremental_window_length
        p_dict['incremental_window_length'] = incremental_window_length
        p_dict['incremental_slide'] = incremental_slide
        return p_dict
    else:
        return {}

def get_method_parameters(method_name,dataset):

    if params["methodology"]=="Semi-Supervised":
        return online_technique(method_name,dataset["max_wait_time"])
    else:
        return unsupervised_technique(method_name, dataset["max_wait_time"])

def get_preproces_parameters(method_name,dataset):
    return pre_proccessing_params(method_name, dataset["max_wait_time"])

def get_postproces_parameters(method_name,dataset):
    return post_proccessing_params(method_name, dataset["max_wait_time"])



if "parameters" in st.session_state:
    params=st.session_state.parameters
    #for key in params.keys():
    #    print(f"{key}: {params[key]}")
else:
    assert False,"No params"

dataset = params["dataset"]

if st.button('Home'):
    st.switch_page("pages/mainUI.py")


st.markdown("*Dataset Metrics:*")
name, evaluation_param,datalengths = st.columns([1,1,1])
with name:
    st.markdown(f"Dataset: {params['dataset_name']}")
with evaluation_param:
    st.markdown(f"predictive horizon (PH) : {dataset['predictive_horizon']} ")
    st.markdown(f"lead : {dataset['lead']}")
    st.markdown(f"VUS slide : {dataset['slide']}")
    st.markdown(f"Optimization Metric : {params['optimization_metric']}")
with datalengths:
    st.markdown(f"*min_target_scenario_len*: {dataset['min_target_scenario_len']}")
    st.markdown(f"*max_wait_time*: {dataset['max_wait_time']}")

st.markdown("*Set Parameters:*")


mango, flavor_param,pre_param,method_param,post_param = st.columns([1, 2,2,2,2])

with mango:
    temp = st.text_input("Mango MAX RUNS:", value=1, key="MAX_RUNS")
    temp2 = st.text_input("Mango MAX JOBS:", value=1, key="MAX_JOBS")
    temp3 = st.text_input("Mango INITIAL_RANDOM:", value=1, key="INITIAL_RANDOM")

with flavor_param:
    for flavorname in params["select_flavors"]:
        with st.popover(f"{flavorname} params"):
            temp_dict=get_flavor_parameters(flavorname,dataset)
            for parameter_name in temp_dict:
                values_to_display=""
                for v in temp_dict[parameter_name]:
                    if isinstance(v, str):
                        values_to_display=f"{values_to_display} \"{v}\","
                    else:
                        values_to_display=f"{values_to_display} {v},"
                if len(values_to_display)>0:
                    values_to_display=values_to_display[:-1]
                temp = st.text_input(parameter_name+":",value=values_to_display,key=flavorname+"_"+parameter_name)

with pre_param:
    methodname=params["pre-processing"]
    with st.popover(f"{methodname} params"):
        temp_dict=get_preproces_parameters(methodname,dataset)
        for parameter_name in temp_dict:
            values_to_display=""
            for v in temp_dict[parameter_name]:
                if isinstance(v, str):
                    values_to_display=f"{values_to_display} \"{v}\","
                else:
                    values_to_display=f"{values_to_display} {v},"
            if len(values_to_display)>0:
                values_to_display=values_to_display[:-1]
            temp = st.text_input(parameter_name+":",value=values_to_display,key="pre_"+params["pre-processing"]+"_"+parameter_name)



with method_param:
    for methodname in params["methods"]:
        with st.popover(f"{methodname} params"):
            agree = st.checkbox("Use automatic parameter (ignore given)",key=f"automatic_{methodname}")
            temp_dict=get_method_parameters(methodname,dataset)
            for parameter_name in temp_dict:
                values_to_display=""
                for v in temp_dict[parameter_name]:
                    if isinstance(v, str):
                        values_to_display=f"{values_to_display} \"{v}\","
                    else:
                        values_to_display=f"{values_to_display} {v},"
                if len(values_to_display)>0:
                    values_to_display=values_to_display[:-1]
                temp = st.text_input(parameter_name+":",value=values_to_display,key="method_"+methodname+"_"+parameter_name)


with post_param:
    methodname = params["post-processing"]
    with st.popover(f"{methodname} params"):
        temp_dict=get_preproces_parameters(methodname,dataset)
        for parameter_name in temp_dict:
            values_to_display=""
            for v in temp_dict[parameter_name]:
                if isinstance(v, str):
                    values_to_display=f"{values_to_display} \"{v}\","
                else:
                    values_to_display=f"{values_to_display} {v},"
            if len(values_to_display)>0:
                values_to_display=values_to_display[:-1]
            temp = st.text_input(parameter_name+":",value=values_to_display,key="post_"+methodname+"_"+parameter_name)

Is_Running=False
BEST_configuration=None
if st.button('Run') and Is_Running==False:

    params["MAX_RUNS"]=int(st.session_state["MAX_RUNS"])
    params["MAX_JOBS"]=int(st.session_state["MAX_JOBS"])
    params["INITIAL_RANDOM"]=int(st.session_state["INITIAL_RANDOM"])

    # experimental parameters:
    for flavorname in params["select_flavors"]:
        temp_dict = get_flavor_parameters(flavorname, dataset)
        to_submit={}
        for parameter_name in temp_dict:
            input_string=st.session_state[flavorname+"_"+parameter_name]
            input_string=input_string.split(",")
            values=[]
            for input in input_string:
                if "\"" in input or "\'" in input:
                    fvalue=input.replace(" ","").replace("\"","").replace("\'","")
                elif "." in input:
                    fvalue=float(str(input).replace(" ", ""))
                else:
                    fvalue = int(str(input).replace(" ", ""))
                values.append(fvalue)
            to_submit[parameter_name]=values
        params[flavorname + "_parameters"]=to_submit


    methodname = params["pre-processing"]
    temp_dict = get_preproces_parameters(methodname, dataset)
    to_submit={}
    for parameter_name in temp_dict:
        input_string=st.session_state["pre_"+methodname+"_"+parameter_name]
        input_string=input_string.split(",")
        values=[]
        for inputsp in input_string:
            input = inputsp.replace(" ", "")
            if input == "None":
                fvalue = None
            elif input == "True":
                fvalue = True
            elif input == "False":
                fvalue = False
            elif "\"" in input or "\'" in input:
                fvalue = input.replace(" ", "").replace("\"", "").replace("\'", "")
            elif "." in input:
                fvalue = float(input)
            else:
                fvalue = int(input.replace(" ", ""))
            values.append(fvalue)
        to_submit[parameter_name]=values
    params["pre-processing_parameters"]=to_submit

    for methodname in params["methods"]:
        temp_dict = get_method_parameters(methodname, dataset)
        to_submit={}
        params[f"automatic_{methodname}"]=st.session_state[f"automatic_{methodname}"]
        for parameter_name in temp_dict:
            input_string=st.session_state["method_"+methodname+"_"+parameter_name]
            input_string=input_string.split(",")
            values=[]
            for inputsp in input_string:
                input=inputsp.replace(" ", "")
                if input=="None":
                    fvalue=None
                elif input=="True":
                    fvalue = True
                elif input=="False":
                    fvalue = False
                elif "\"" in input or "\'" in input:
                    fvalue=input.replace(" ","").replace("\"","").replace("\'","")
                elif "." in input:
                    fvalue=float(input)
                else:
                    fvalue = int(input.replace(" ", ""))
                values.append(fvalue)
            to_submit[parameter_name]=values
        params["method_"+methodname + "_parameters"]=to_submit

    methodname = params["post-processing"]
    temp_dict = get_postproces_parameters(methodname, dataset)
    to_submit={}
    for parameter_name in temp_dict:
        input_string=st.session_state["post_"+methodname+"_"+parameter_name]
        input_string=input_string.split(",")
        values=[]
        for inputsp in input_string:
            input = inputsp.replace(" ", "")
            if input == "None":
                fvalue = None
            elif input == "True":
                fvalue = True
            elif input == "False":
                fvalue = False
            elif "\"" in input or "\'" in input:
                fvalue = input.replace(" ", "").replace("\"", "").replace("\'", "")
            elif "." in input:
                fvalue = float(input)
            else:
                fvalue = int(input.replace(" ", ""))
            values.append(fvalue)
        to_submit[parameter_name]=values
    params["post-processing_parameters"]=to_submit



    run_mlflow_server()
    Is_Running=True



if Is_Running:
    serialized_dict = pickle.dumps(params)


    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Write the serialized dictionary to the temporary file
        temp_file.write(serialized_dict)
        temp_file_path = temp_file.name

    ENVBIN = sys.exec_prefix
    try:
        BIN2 = os.path.join(ENVBIN,'python')

        process  = subprocess.Popen([BIN2,"-u", "./experimental_runs_configuration/allexperimentsUI.py", temp_file_path],  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    except Exception as e:
        BIN2 = os.path.join(ENVBIN, 'bin/python')

        process = subprocess.Popen(
            [BIN2, "-u", "./experimental_runs_configuration/allexperimentsUI.py", temp_file_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    #st.title("Terminal Output:")
    with st.popover(f"Terminal Output:"):
        while process.poll() is None:
            line = process.stdout.readline()
            if not line:
                continue
            st.write(line.strip())
        Is_Running=False
    os.remove(temp_file_path)
