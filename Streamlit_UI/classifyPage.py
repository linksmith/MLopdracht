import streamlit as st
import util

def app():
    st.image('./Streamlit_UI/Header.gif', use_column_width=True)
    st.write("Upload a wav file of spoken Arabic to find out which digit is being spoken.")
    st.markdown('*Need a some wav files to test? Visit this [link]("https://github.com/kanakmi/Deforgify/tree/main/Model%20Training/dataset")*')
    file_uploaded = st.file_uploader("Choose the Wav File", type=["wav"])
    if file_uploaded is not None:
        res = util.classify_image(file_uploaded)
        c1, buff, c2 = st.columns([2, 0.5, 2])
        c1.image(file_uploaded, use_column_width=True)
        c2.subheader("Classification Result")
        c2.write("This wav file is classified as **{}**.".format(res['label'].title()))
