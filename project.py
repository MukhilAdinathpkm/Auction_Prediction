import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import requests
import os

# Define the URL of your model on GitHub
model_url = 'https://github.com/MukhilAdinathpkm/Auction_Prediction/raw/main/my_model1.h5'

# Download the model
model_path = 'my_model1.h5'
if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)

# Load the model
model = load_model(model_path)

# Define mean and standard deviation values for top selected features
mean = np.array([71.29230769230769, 0.45314685314685316, 3.248951048951049])
std = np.array([8.097369977286826, 1.0648532357505802, 1.8215284814679016])

def standardize_data(data, mean, std):
    return (data - mean) / std

def predict(features):
    try:
        features_array = np.array(features).reshape(1, -1)
        x_standardized = standardize_data(features_array, mean, std)
        prediction = model.predict(x_standardized)
        result = (prediction[0][0] > 0.5)
        return "TRUE" if result else "FALSE"
    except ValueError:
        return "Invalid input. Please enter numeric values."

# Streamlit app interface
st.title('AUCTION VERIFICATION APP')

process_b1_capacity = st.number_input('Enter process.b1.capacity', format="%.2f")
process_b2_capacity = st.number_input('Enter process.b2.capacity', format="%.2f")
process_b3_capacity = st.number_input('Enter process.b3.capacity', format="%.2f")
process_b4_capacity = st.number_input('Enter process.b4.capacity', format="%.2f")
property_price = st.number_input('Enter property.price', format="%.2f")
property_product = st.number_input('Enter property.product', format="%.2f")
property_winner = st.number_input('Enter property.winner', format="%.2f")

features = [property_price, property_winner, property_product]

if st.button('Predict'):
    result = predict(features)
    st.write(f'Prediction: {result}')
