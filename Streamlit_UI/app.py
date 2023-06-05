import streamlit as st
import dashboard
import classifyPage

st.set_page_config(
    page_title="Classify Arabic Digits",
    page_icon="🤖",
    layout="wide")

PAGES = {
    "Classify Arabic Digits": dashboard,
    # "Classify Arabic Digits": classifyPage
}

st.sidebar.title("Arabic Digit Classifier")

st.sidebar.write("Arabic Digit Classifier is a tool that utilizes the power of Deep Learning to classify spoken arabic digits.")

# st.sidebar.subheader('Navigation:')
# selection = st.sidebar.radio("", list(PAGES.keys()))

# page = PAGES[dashboard]

dashboard.app()