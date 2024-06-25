import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import streamlit as st
import numpy as np

data_set = pd.read_csv('House_KTM.csv')

X = data_set[['Aana', 'Road']]
Y = data_set[['Price']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
Norm_X_train = scaler.fit_transform(X_train)

sdgr = SGDRegressor(max_iter=100)
sdgr.fit(Norm_X_train, Y_train)

st.title("KTM HOUSE PRICE PREDICTION")
x0 = st.number_input('Enter the Size of House (in Ana)?', step=0.1, value=0.0)
x1 = st.number_input('Enter the Road Size?', step=0.1, value=0.0)

predicted_value = None
if x0 > 0 and x1 > 0:
    user_data = np.array([[x0, x1]])
    user_data_norm = scaler.transform(user_data.reshape(1, -1))
    predicted_value = sdgr.predict(user_data_norm)

if predicted_value is not None:
    st.success(f"The predicted price of house is Rs.{predicted_value[0]:.2f}cores")
else:
    st.warning("Please Enter all the value")
