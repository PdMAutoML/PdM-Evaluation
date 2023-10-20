import streamlit as st
import os
import pandas as pd
st.set_page_config(initial_sidebar_state="collapsed", layout="wide")

def load_sources():
    uploaded_files = st.file_uploader("Choose Target Data CSV files", accept_multiple_files=True, type=["csv"])
    if uploaded_files:
        csv_names = [os.path.basename(file.name) for file in uploaded_files]
        st.session_state.csv_names = csv_names
        st.session_state.uploaded_files = uploaded_files


def load_historic_sources():
    uploaded_files = st.file_uploader("Choose Normal Historic Data CSV files", accept_multiple_files=True, type=["csv"])
    if uploaded_files:
        csv_names = [os.path.basename(file.name) for file in uploaded_files]
        st.session_state.csv_names_historic = csv_names
        st.session_state.uploaded_files_historic = uploaded_files


def load_labels():

    uploaded_files = st.file_uploader("Choose labels (optional)", accept_multiple_files=True, type=[".out"])
    st.markdown("Labels files have to named as data source files, with different ending (.out). E.g.: "
                + "labels for source_0.csv should have source_0.out name. To pass lead, use an additional column to each file.")
    if uploaded_files:
        st.session_state.uploaded_labels = uploaded_files
        csv_names = [os.path.basename(file.name) for file in uploaded_files]
        st.session_state.label_names = csv_names
def load_events():
    uploaded_files = st.file_uploader("Choose events (optional)", accept_multiple_files=True, type=["csv"])
    if uploaded_files:
        st.session_state.uploaded_events = uploaded_files
        st.session_state.event_names=[os.path.basename(file.name) for file in uploaded_files]
def create_buttons():
    datasources,historic, labels_lead, events,run_to_failure = st.columns([1,1, 1, 1,1])
    st.text_input("Dataset name:", value="Name",key="dataset_name_is")
    with historic:
        load_historic_sources()
    with datasources:
        load_sources()

    with labels_lead:
        load_labels()

    with events:
        load_events()
    with run_to_failure:
        isRTF=st.checkbox("Run to Failure (Ignore labels)", key=f"run_to_failure")

    if st.button("Next"):
        read_dataframes()

        if len(st.session_state.dataframes_historic)!=0:
            if len(st.session_state.dataframes_historic)!=len(st.session_state.dataframes):
                assert False, "Historic sources need to be tha same as Target ones."
            for name in st.session_state.f_names:
                if name in st.session_state.names_historic:
                    continue
                else:
                    assert False, "Source {name} not appeared in Historic sources."


        if len(st.session_state.dataframes_labels)==0 and st.session_state.run_to_failure==False and  len(st.session_state.dataframe_events)==0:
            st.markdown(f":red[You have to pass at least on of labels, events or set Run to Failure True]")
            assert False,"You have to pass at least on of labels, events or set Run to Failure True"
        if len(st.session_state.dataframes_labels)>0:
            # print("Considering labels (Ignore run to failure)")
            if len(st.session_state.dataframes_labels)!=len(st.session_state.dataframes):
                st.markdown(f":red[Inconsistent Label files {st.session_state.uploaded_labels} with Data files {st.session_state.uploaded_files}]")
                assert False, f"Inconsistent Label files {st.session_state.uploaded_labels} with Data files {st.session_state.uploaded_files}"
            else:
                final_dataframes=[]
                for lb,lbname in zip( st.session_state.dataframes_labels,st.session_state.label_names):
                    pos = st.session_state.csv_names.index(f"{lbname.split('.')[0]}.csv")
                    if pos ==-1:
                        st.markdown(f":red[Inconsistent file names, {lbname.split('.')[0]}.csv not found in data names]")
                        assert False, f"Inconsistent file names, {lbname.split('.')[0]}.csv not found in data names"
                    datadf=st.session_state.dataframes[pos]
                    final_dataframes.append(datadf)
                    if len(lb.index)!=len(datadf):
                        st.markdown(
                            f":red[Inconsistent length of labels and dataframes for files {lbname}, {st.session_state.csv_names[pos]}: {len(lb.index)}!={len(datadf)}]")
                        assert False, f"Inconsistent length of labels and dataframes for files {lbname}, {st.session_state.csv_names[pos]}: {len(lb.index)}!={len(datadf)}"
                # if all ok then reordering dataframes
                st.session_state.dataframes=final_dataframes
                st.session_state.f_names=[nam.split(".")[0] for nam in st.session_state.label_names]
            st.session_state.run_to_failure=False
        if len(st.session_state.dataframes_historic)!=0:
            f_historic=[]
            for name in st.session_state.f_names:
                f_historic.append(st.session_state.dataframes_historic[st.session_state.names_historic.index(name)])
            st.session_state.names_historic=st.session_state.f_names
            st.session_state.dataframes_historic=f_historic


        if "parameters" not in st.session_state:

            params={
                "csv_names": st.session_state.csv_names,
                "label_names": st.session_state.label_names,

                "uploaded_files":st.session_state.uploaded_files,
                "uploaded_labels":st.session_state.uploaded_labels,
                "uploaded_events":st.session_state.uploaded_events,
                "dataframes":st.session_state.dataframes,
                "dataframes_labels":st.session_state.dataframes_labels,
                "dataframe_events":st.session_state.dataframe_events,
            }

            st.session_state.parameters = params
        if "run_to_failure_is" not in st.session_state:
            st.session_state.run_to_failure_is=st.session_state.run_to_failure
        if "dataset_name" not in st.session_state:
            st.session_state.dataset_name=st.session_state.dataset_name_is
        st.switch_page("pages/databuilder2.py")



def read_dataframes():
    if len(st.session_state.uploaded_files) ==0:
        assert False, "No provided data"
    else:
        dataframes = []
        names = []
        for file,nam in zip(st.session_state.uploaded_files,st.session_state.csv_names):
            df = pd.read_csv(file)
            dataframes.append(df)
            names.append(nam.split(".")[0])
        st.session_state.dataframes = dataframes
        st.session_state.f_names=names

    dataframes_historic = []
    names_historic=[]

    for file, nam in zip(st.session_state.uploaded_files_historic, st.session_state.csv_names_historic):
        df = pd.read_csv(file)
        dataframes.append(df)
        names.append(nam.split(".")[0])
    st.session_state.dataframes_historic = dataframes_historic
    st.session_state.names_historic = names_historic

    # read label dataframes
    if st.session_state.run_to_failure==False:
        if len(st.session_state.uploaded_files) !=0:
            dataframes_labels = []
            for file in st.session_state.uploaded_labels:
                df = pd.read_csv(file)
                dataframes_labels.append(df)
            st.session_state.dataframes_labels = dataframes_labels

    # read event dataframe
    if len(st.session_state.uploaded_events) != 0:
        dataframes_events = []
        for file in st.session_state.uploaded_events:
            df = pd.read_csv(file)
            dataframes_events.append(df)
        st.session_state.dataframe_events = dataframes_events

if "csv_names" not in st.session_state:
    st.session_state.csv_names = []
    st.session_state.csv_names_historic = []
    st.session_state.label_names=[]
    st.session_state.names_historic = []
    st.session_state.f_names = []
    st.session_state.event_names = []


    st.session_state.uploaded_files = []
    st.session_state.uploaded_labels = []
    st.session_state.uploaded_events = []

    st.session_state.uploaded_files_historic = []

    st.session_state.dataframes=[]

    st.session_state.dataframes_historic = []
    st.session_state.dataframes_labels=[]
    st.session_state.dataframe_events=[]
create_buttons()
