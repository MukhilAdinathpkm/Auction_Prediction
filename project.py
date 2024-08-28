import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model(r"C:\Users\ADMIN\Downloads\my_model1.h5")  # Ensure the correct path

# Define mean and standard deviation values for top selected features
mean = np.array([71.29230769230769, 0.45314685314685316, 3.248951048951049])
std = np.array([8.097369977286826, 1.0648532357505802, 1.8215284814679016])

def standardize_data(data, mean, std):
    return (data - mean) / std

def predict(features):
    try:
        # Process input
        features_array = np.array(features).reshape(1, -1)
        
        # Standardize input data
        x_standardized = standardize_data(features_array, mean, std)
        
        # Predict
        prediction = model.predict(x_standardized)
        
        # Interpret result
        result = (prediction[0][0] > 0.5)  # Assuming binary classification
        return "TRUE" if result else "FALSE"
    except ValueError:
        return "Invalid input. Please enter numeric values."

# Streamlit app interface
# st.markdown(
#     """
#     <style>
#     body {
#         background: linear-gradient(to right, #ff7e5f, #feb47b);
#         color: #333333;
#     }
#     .main {
#         background-color: #ffffff;
#         border-radius: 15px;
#         padding: 20px;
#         box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
#     }
#     .stButton > button {
#         background-color: #ff7e5f;
#         color: white;
#         border-radius: 10px;
#         padding: 10px 20px;
#     }
#     .stTextInput > div > label {
#         font-size: 18px;
#         font-weight: bold;
#         color: #4B4B4B;
#     }
#     .stTextInput > div > input {
#         margin-bottom: 10px;
#     }
#     h1 {
#         color: #4B4B4B;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
st.title('AUCTION VERIFICATION APP')

# User inputs for all features
process_b1_capacity = st.number_input('Enter process.b1.capacity', format="%.2f")
process_b2_capacity = st.number_input('Enter process.b2.capacity', format="%.2f")
process_b3_capacity = st.number_input('Enter process.b3.capacity', format="%.2f")
process_b4_capacity = st.number_input('Enter process.b4.capacity', format="%.2f")
property_price = st.number_input('Enter property.price', format="%.2f")
property_product = st.number_input('Enter property.product', format="%.2f")
property_winner = st.number_input('Enter property.winner', format="%.2f")

# Only include top selected features for prediction
features = [
    property_price,
    property_winner,
    property_product
]

if st.button('Predict'):
    result = predict(features)
    st.write(f'Prediction: {result}')
