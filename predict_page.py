import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

###### Data collection and preprocessor
# Loading data
big_mart_data = pd.read_csv('Train.csv')

big_mart_data["Item_Weight"].mean()
# fillling the missing values in item weight column with mean value
big_mart_data["Item_Weight"].fillna(big_mart_data["Item_Weight"].mean(),inplace = True)


#Replacing the missing values in the outlet size column with mode
mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
miss_values = big_mart_data['Outlet_Size'].isnull()  

big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])




big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
big_mart_data['Item_Fat_Content'].value_counts()
###### Label Encoding
encoder = LabelEncoder()
big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])

big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])

big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])

big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])

big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])

big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])

big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

###### Splitting features and target
X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']

###### Splitting the data into Training data & Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

###### Machine Learning Model Training
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)
###### Evaluation
# prediction on training data
training_data_prediction = regressor.predict(X_train)
# R squared Value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared value = ', r2_train)
# prediction on test data
test_data_prediction = regressor.predict(X_test)
# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value = ', r2_test)


def  show_predict_page():
    st.title("Big Mart Sales Prediction")

    Item_Identifier = st.number_input("Item Identifier")
    Item_Weight = st.number_input("Item Weight")
    Item_Fat_Content = st.number_input("Item Fat Content (0-Low Fat, 1-Regular)")
    Item_Visibility = st.number_input("Item Visibility")
    Item_Type = st.number_input("Item Type")
    Item_MRP = st.number_input("Item MRP")
    Outlet_Identifier = st.number_input("Outlet Identifier")
    Outlet_Establishment_Year = st.number_input("Outlet Establishment Year")
    Outlet_Size = st.number_input("Outlet Size (0-Small, 1-Medium, 2-High)")
    Outlet_Location = st.number_input("Outlet Location (0-Tier 1, 1-Tier 2, 2-Tier 3)")
    Outlet_Type = st.number_input("Outlet Type (0-Grocery Store, 1-Supermarket Type1, 2-Supermarket Type2, 3-Supermarket Type3)")



    def predict_sales(Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location, Outlet_Type):
        # Define an array to hold the input data
        input_data = np.array([[Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location, Outlet_Type]])

        # Perform necessary data pre-processing and transformations on the input data
        # ...

        # Load the trained model
        # ...

        # Use the model to make a prediction on the input data
        prediction = regressor.predict(input_data)

        # Return the prediction
        return prediction

    if st.button("Predict"):
        result = predict_sales(Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location, Outlet_Type)
        st.success("The predicted sales is {result} $")



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

