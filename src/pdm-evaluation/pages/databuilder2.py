import streamlit as st
import os
import pandas as pd
st.set_page_config(initial_sidebar_state="collapsed", layout="wide")


# st.session_state.csv_names = []
# st.session_state.label_names=[]
#
# st.session_state.uploaded_files = []
# st.session_state.uploaded_labels = []
# st.session_state.uploaded_events = []
# st.session_state.dataframes=[]
# st.session_state.dataframes_labels=[]
# st.session_state.dataframe_events=[]
# st.session_state.run_to_failure

if "parameters" in st.session_state:
    params=st.session_state.parameters
    #for key in params.keys():
    #    # print(f"{key}: {params[key]}")
else:
    assert False,"No params"

# print(st.session_state.run_to_failure_is)

def event_columns_match(namefile,common_columns):
    st.markdown("Match Columns with the ")
    #["date", "type", "source", "description"]:
    st.selectbox("Date Column (*)",common_columns,key=f'{namefile}_date')
    st.selectbox("Source Column (*)",common_columns,key=f'{namefile}_source')
    choises=["Empty"]
    for col in common_columns:
        choises.append(col)
    st.selectbox("type (optinal)", choises, key=f'{namefile}_type')
    st.selectbox("Description (optinal)",choises,key=f'{namefile}_description')







def handle_events():
    for df,namefile in zip(st.session_state.dataframe_events,st.session_state.event_names):
        with st.popover(f"Formulate {namefile}"):
            columnsdf=df.columns
            event_columns_match(namefile.split(".")[0], columnsdf)

    if st.button("Combine"):
        columns = ["date", "type", "source", "description"]

        # Create an empty DataFrame with the specified columns
        combined_df = pd.DataFrame(columns=columns)
        for df, namefile_w_ in zip(st.session_state.dataframe_events, st.session_state.event_names):
            namefile=namefile_w_.split(".")[0]
            dfcor=df.copy()
            for origin_col in ["date", "type", "source", "description"]:
                tempcolname=st.session_state[f"{namefile}_{origin_col}"]
                if tempcolname=="Empty":
                    dfcor[origin_col]=["undefined" for ind in dfcor.index]
                else:
                    dfcor[origin_col]=dfcor[tempcolname]
            dfcor=dfcor[["date", "type", "source", "description"]]
            combined_df = pd.concat([combined_df, dfcor], ignore_index=True)
        st.session_state.event_data=combined_df
        st.markdown(f"Event data head:")
        st.dataframe(combined_df.head())
        st.markdown(f"Event data tail:")
        st.dataframe(combined_df.tail())
def handle_timestamp():
    with st.popover(f"Data sources Timestamp"):
        st.checkbox("Generate Timestamps (Ignore Below)", key=f"generate_timestamp")
        common_columns= set(st.session_state.dataframes[0].columns)
        for df in st.session_state.dataframes[1:]:
            common_columns.intersection_update(df.columns)
        for df in st.session_state.dataframes_historic:
            common_columns.intersection_update(df.columns)
        column_timestamp = st.selectbox(
            "Select a column appeared in all files as Timestamp",
            common_columns,
            key='column_timestamp'
        )
    if st.button("Set Timestamps"):
        st.session_state.dataframes_ready = []
        st.session_state.historic_ready = []
        if st.session_state.generate_timestamp:
            for df in st.session_state.dataframes:
                timestamps = pd.date_range(start='2000-01-01 00:00:00', periods=len(df.index), freq='T')
                dfn = df.copy()
                dfn['dates']=timestamps
                st.session_state.dataframes_ready.append(dfn)
            for df in st.session_state.dataframes_historic:
                timestamps = pd.date_range(start='2000-01-01 00:00:00', periods=len(df.index), freq='T')
                dfn = df.copy()
                dfn['dates']=timestamps
                st.session_state.historic_ready.append(dfn)
        else:
            for df in st.session_state.dataframes:
                dfn=df.copy()
                dfn['dates']=pd.to_datetime(df[st.session_state.column_timestamp])
                dfn=dfn.drop([st.session_state.column_timestamp],axis=1)
                st.session_state.dataframes_ready.append(dfn)
            for df in st.session_state.dataframes_historic:
                dfn = df.copy()
                dfn['dates'] = pd.to_datetime(df[st.session_state.column_timestamp])
                dfn = dfn.drop([st.session_state.column_timestamp], axis=1)
                st.session_state.historic_ready.append(dfn)
        st.markdown(f":green[Data sources timestamp setting completed]")
        st.dataframe(st.session_state.dataframes_ready[0].head())


def mainpage():
    datasources, labels_lead, events = st.columns([1, 1, 1])

    with datasources:
        handle_timestamp()
    with labels_lead:
        if len(st.session_state.dataframes_labels)==0:
            minsenario=min([len(df.index) for df in st.session_state.dataframes])
            st.markdown(f"Predictive Horizon and lead should be provided since no labels are detected.")
            ph = st.text_input("Predictive Horizon (PH) eg: 100 or \"100 days\":", value=minsenario//10,key="predictive_horizon")
            lead = st.text_input("Lead:", value=max(2,minsenario//100), key="lead_time")
        vus_slide = st.text_input("VUS slide:", value=1, key="vus_slide")

    if len(st.session_state.dataframe_events)>0:
        with events:
            handle_events()
    if st.button("Next"):
        dataset=Next()
        # print(dataset)
        st.session_state.dataset = dataset
        st.switch_page("pages/databuilder3.py")
def Next():
    dataset={}

    if len(st.session_state.dataframe_events)>0 and st.session_state.event_data is None:
        assert False, "Configure Event Data before continue to the next step!"
    if len(st.session_state.dataframe_events)==0:
        st.session_state.event_data = pd.DataFrame(columns=["date", "type", "source", "description"])
    if len(st.session_state.dataframes_ready)==0:
        assert False,"No timestamps for Data sources!"
    else:
        dataset["target_data"]=st.session_state.dataframes_ready
        dataset["target_sources"] = st.session_state.f_names
        dataset["dates"] = 'dates'
        dataset["historic_data"] = st.session_state.historic_ready
        dataset["historic_sources"] = st.session_state.names_historic
        dataset["event_data"] = st.session_state.event_data
        dataset["run_to_failure"] = st.session_state.run_to_failure_is

        if len(st.session_state.dataframes_labels)==0:
            if "\"" in st.session_state.predictive_horizon or "'" in st.session_state.predictive_horizon:
                dataset["predictive_horizon"] = st.session_state.predictive_horizon.replace("\"","").replace("'","")
            else:
                dataset["predictive_horizon"] = int(st.session_state.predictive_horizon)

            if "\"" in st.session_state.lead_time or "'" in st.session_state.lead_time:
                dataset["lead"] = st.session_state.lead_time.replace("\"","").replace("'","")
            else:
                dataset["lead"] = int(st.session_state.lead_time)
        else:
            # store ranges
            anomaly_ranges=[]
            lead_ranges=[]
            for lb in st.session_state.dataframes_labels:
                labels=lb[lb.columns[0]].values
                if len(lb.columns)>1:
                    lead=lb[lb.columns[1]].values
                else:
                    lead = [0 for i in range(len(labels))]
                anomaly_ranges.append(labels)
                lead_ranges.append(lead)
            dataset["lead"] =lead_ranges
            dataset["predictive_horizon"] = anomaly_ranges
            dataset["anomaly_ranges"] = True
        dataset["slide"]=int(st.session_state.vus_slide)

        return dataset


if "dataframes_ready" not in st.session_state:
    st.session_state.dataframes_ready = []
    st.session_state.historic_ready = []
    st.session_state.event_data = None
mainpage()