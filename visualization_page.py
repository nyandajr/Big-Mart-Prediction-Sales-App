import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Loading data
big_mart_data = pd.read_csv('Train.csv')
def show_visualization_page():
    st.title("Visualizations")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Item_Weight distribution
    st.subheader("Item_Weight distribution")
    plt.figure(figsize=(6,6))
    sns.histplot(big_mart_data['Item_Weight'])
    st.pyplot()
    
    # Item Visibility distribution
    st.subheader("Item Visibility distribution")
    plt.figure(figsize=(6,6))
    sns.histplot(big_mart_data['Item_Visibility'])
    st.pyplot()
    
    # Item MRP distribution
    st.subheader("Item MRP distribution")
    plt.figure(figsize=(6,6))
    sns.histplot(big_mart_data['Item_MRP'])
    st.pyplot()
    
    # Item Outlet sales
    st.subheader("Item Outlet sales")
    plt.figure(figsize=(6,6))
    sns.histplot(big_mart_data['Item_Outlet_Sales'])
    st.pyplot()
    
    # Outlet_Establishment_Year
    st.subheader("Outlet establishment")
    plt.figure(figsize=(6,6))
    sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
    st.pyplot()
    
    # Item_Fat_Content
    st.subheader("Item_Fat_Content column")
    plt.figure(figsize=(6,6))
    sns.countplot(x='Item_Fat_Content', data=big_mart_data)
    st.pyplot()
    
    # Item_Type
    st.subheader("Item_Type column")
    plt.figure(figsize=(26,6))
    sns.countplot(x='Item_Type', data=big_mart_data)
    st.pyplot()
    
    # Outlet_Size
    st.subheader("Outlet_Size column")
    plt.figure(figsize=(6,6))
    sns.countplot(x='Outlet_Size', data=big_mart_data)
    st.pyplot()
    
    
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Coyright @Nyanda Jr</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
