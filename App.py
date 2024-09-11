import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'predict.h5'
model = load_model(model_path)

# Function to read CSV and extract the 'Rate' column
def load_data(file):
    df = pd.read_csv(file)
    series = df['Rate'].values  # Assuming the column is named 'Rate'
    return series

# Function to normalize data
def normalize_data(series):
    scaler = MinMaxScaler(feature_range=(0, 1))
    series = series.reshape(-1, 1)
    scaled_series = scaler.fit_transform(series)
    return scaled_series, scaler

# Function to predict future values
def predict_future_values(model, series, scaler, time_step=100, n_predictions=10):
    predictions = []
    input_data = series[-time_step:].reshape(1, time_step, 1)
    
    for _ in range(n_predictions):
        pred = model.predict(input_data)
        predictions.append(pred[0, 0])
        
        # Update input_data with the new prediction
        pred = np.reshape(pred, (1, 1, 1))  # Reshape pred to match input_data's shape
        input_data = np.append(input_data[:, 1:, :], pred, axis=1)
    
    # Inverse transform predictions to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Streamlit app
st.title('Time Series Prediction')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    series = load_data(uploaded_file)
    scaled_series, scaler = normalize_data(series)
    
    predictions = predict_future_values(model, scaled_series, scaler)
    
    st.write("Next 10 days predictions:")
    st.write(predictions)
