from time import sleep

import streamlit as st
st.set_page_config(initial_sidebar_state="collapsed",layout="wide")


from utils import loadDataset



if "parameters" in st.session_state:
    params=st.session_state.parameters
    #for key in params.keys():
    #    # print(f"{key}: {params[key]}")
else:
    assert False,"No params"

with st.spinner('Loading Dataset...'):
    dataset = loadDataset.get_dataset(params["dataset_name"])


if st.button('Home'):
    st.switch_page("pages/mainUI.py")

st.markdown("*Dataset Metrics:*")
name, evaluation_param,datalengths = st.columns([1,1,1])
with name:
    st.markdown(f"Dataset: {params['dataset_name']}")
with evaluation_param:
    ph = st.text_input("predictive horizon (PH):", value=dataset["predictive_horizon"], key="predictive_horizon")
    lead = st.text_input("lead:", value=dataset["lead"], key="lead_time")
    vus_slide = st.text_input("VUS slide:", value=dataset["slide"], key="vus_slide")
    optimization_metric=st.selectbox("Optimization Metric:",["AD1_AUC","AD1_f1","VUS_VUS_PR"],key="optimization_metric")
with datalengths:
    st.markdown(f"*min_target_scenario_len*: {dataset['min_target_scenario_len']}")
    max_wait_time = st.text_input("max_wait_time", value=dataset["max_wait_time"], key="max_wait_time")


if st.button('Next'):

    dataset["predictive_horizon"] = st.session_state.predictive_horizon.replace("\"","").replace("\'","")
    dataset["lead"] = st.session_state.lead_time.replace("\"","").replace("\'","")
    dataset["slide"]=int(st.session_state.vus_slide)
    dataset["max_wait_time"]=int(st.session_state.max_wait_time)

    params["optimization_metric"] = st.session_state.optimization_metric
    params["dataset"] = dataset
    st.session_state.parameters=params
    sleep(0.1)
    st.switch_page("pages/pages1.py")

