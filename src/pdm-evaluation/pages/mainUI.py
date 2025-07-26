
import streamlit as st
st.set_page_config(initial_sidebar_state="collapsed", layout="wide")
# Delete all the items in Session state
for key in st.session_state.keys():
    del st.session_state[key]



Databuild,runbuild = st.columns([1, 1])

with Databuild:
    if st.button("Dataset Builder"):
        st.switch_page("pages/datasetBuilder.py")

with runbuild:
    if st.button("Run Experiment"):
        st.switch_page("pages/run.py")

