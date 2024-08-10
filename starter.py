import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Homepage",
    layout="centered",
    initial_sidebar_state="auto"
    )

df=pd.DataFrame({"col1":[1,2,3],
                 "col2":[4,5,6]})

if "df" not in st.session_state:
    st.session_state["df"]=df