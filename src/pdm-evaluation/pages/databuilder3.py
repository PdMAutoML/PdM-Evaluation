import pickle

import streamlit as st
import os
import pandas as pd
st.set_page_config(initial_sidebar_state="collapsed", layout="wide")
from pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple


#st.session_state.dataset

DESCRIPTION_OPTIONS=["*"]
DESCRIPTION_OPTIONS.extend(st.session_state.dataset["event_data"]["description"].unique())
TYPE_OPTIONS=["*"]
TYPE_OPTIONS.extend(st.session_state.dataset["event_data"]["type"].unique())
SOURCE_OPTIONS=["*"]
SOURCE_OPTIONS.extend(st.session_state.f_names)
TARGET_SOURCE_OPTIONS=["*","="]
TARGET_SOURCE_OPTIONS.extend(st.session_state.f_names)

def add_preference(key):
    #st.session_state.event_preferences[key]
    #pos=len(st.session_state.event_preferences[key].keys())
    st.multiselect("Select Description",DESCRIPTION_OPTIONS
        , key=f'pef_{key}_desc'
    )
    st.multiselect("Select Type", TYPE_OPTIONS
                   , key=f'pef_{key}_type'
                   )
    st.multiselect("Select Source", SOURCE_OPTIONS
                   , key=f'pef_{key}_source'
                   )
    st.multiselect("Select Target Source", TARGET_SOURCE_OPTIONS
                   , key=f'pef_{key}_target_source'
                   )

    if st.button(f"Add preference to {key}"):
        pos = len(st.session_state.event_preferences[key].keys())
        toadd=[st.session_state[f'pef_{key}_desc'],
               st.session_state[f'pef_{key}_type'],
               st.session_state[f'pef_{key}_source'],
               st.session_state[f'pef_{key}_target_source']
               ]


        st.session_state.event_preferences[key][pos]=toadd

def clear_star(choices):
    if "*" in choices:
        return ["*"]
    if "=" in choices:
        return ["="]
    return choices
def combinations(desc,type,source,target):
    combination_prefs=[]
    for desk_ch in clear_star(desc):
        for type_ch in clear_star(type):
            for source_ch in clear_star(source):
                for target_ch in clear_star(target):
                    combination_prefs.append([desk_ch,type_ch,source_ch,target_ch])
    return combination_prefs
def add_new_category():
    if st.session_state.new_pref_category!="failure" and st.session_state.new_pref_category!="reset":
        if st.session_state.new_pref_category not in  st.session_state.event_preferences:
            st.session_state.event_preferences[st.session_state.new_pref_category]={}


def event_preferences():
    # event_preferences: EventPreferences = {
    #     'failure': [
    #         EventPreferencesTuple(description='*', type='fail', source='*', target_sources='=')
    #     ],
    #     'reset': [
    #         EventPreferencesTuple(description='*', type='reset', source='*', target_sources='='),
    #         EventPreferencesTuple(description='*', type='fail', source='*', target_sources='=')
    #     ]
    # }
    set,display= st.columns([1, 1])

    with set:
        if "event_preferences" not in st.session_state:
            st.session_state.event_preferences= {}
            #if st.session_state.dataset["run_to_failure"]==False:
            st.session_state.event_preferences["failure" ] = {}
            st.session_state.event_preferences["reset" ] = {}

        st.markdown("These are use to annotate reset, failures or other user defined events."
                    + "\n When no labels are passed and Run to failure is not checked, failures events are required."
                    + "In case of Run to Failure or passed labels, the \"failure\" cannot be used in event preferences"
                    + "\n '*' means 'any', '=' means 'same as source'")
        required =""
        if len(st.session_state.dataframes_labels) ==0 and st.session_state.dataset["run_to_failure"] == False:
            required = "*"
        edit, deletec = st.columns([3, 1])
        uppernames=[]
        with edit:
            for key in st.session_state.event_preferences.keys():
                uppernames.append(key)
                if key=="failure" and st.session_state.dataset["run_to_failure"]:
                    continue
                with st.popover(f"Add preference to: {key}"):
                    add_preference(key)
        with deletec:
            for name in uppernames:
                if name not in ['failure','reset']:
                    if st.button(f"delete",key=f"delete_{name}"):
                        del st.session_state.event_preferences[name]
        st.text_input("Add category: ",key="new_pref_category",on_change=add_new_category)

    with (display):
        st.markdown(f"Event Preferences in form \n [(Description),(Type),(Source),(Target Source)]")
        for key in st.session_state.event_preferences:
            #with st.popover(f"{key}"):
            st.markdown(f"{key}:")
            show,deletec = st.columns([4, 1])
            allkeys=st.session_state.event_preferences[key].keys()
            names=[]
            for innerkey in allkeys:
                with show:
                    todisp=[]
                    for array in st.session_state.event_preferences[key][innerkey]:
                        todispin="("
                        for membe in clear_star(array):
                            md=membe.replace("*", "any")
                            todispin+=f'{md}, '
                        todispin+=")"
                        todisp.append(todispin)
                    st.markdown(f"---- {todisp}")
                names.append(innerkey)
            with deletec:
                for name in names:
                    if st.button(f"delete",key=f"{key}_{name}"):
                        del st.session_state.event_preferences[key][name]

    if st.button("Save Dataset"):
        final_event_prefs={}
        for key in st.session_state.event_preferences:
            final_event_prefs[key]=[]
            for innerkey in st.session_state.event_preferences[key]:
                choices=st.session_state.event_preferences[key][innerkey]
                prefs=combinations(choices[0],choices[1],choices[2],choices[3])
                print(prefs) # TODO: CHECK IF THE STAR REMAIN
                for pref in prefs:
                    new_prefs=[]
                    for prefin in pref:
                        if prefin != "":
                            new_prefs.append(prefin)
                        else:
                            new_prefs.append("*")
                    final_event_prefs[key].append(EventPreferencesTuple(description=new_prefs[0], type=new_prefs[1], source=new_prefs[2], target_sources=new_prefs[3]))

        st.session_state.dataset["event_preferences"]=final_event_prefs
        # print(st.session_state.dataset.keys())
        with open(f'./DataFolder/dictionaries/{st.session_state.dataset_name}.pickle', 'wb') as handle:
            pickle.dump(st.session_state.dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        st.switch_page("pages/mainUI.py")

event_preferences()
