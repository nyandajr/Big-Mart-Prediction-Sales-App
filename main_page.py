import streamlit as st
from predict_page import show_predict_page
from visualization_page import show_visualization_page
import sklearn


page = st.sidebar.selectbox("Select App or Visualize side bar", ("Visualize", "predict"))

if page == "predict":
    show_predict_page()
else:
    show_visualization_page()
