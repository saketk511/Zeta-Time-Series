import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.models import load_model 

# Step 1: Read the CSV file and extract the 'Rate' column
def load_data(file_path):
    df = pd.read_csv(file_path)
    series = df['Rate'].values  # Assuming the column is named 'Rate'
    return series

# Step 2: Normalize the column values
def normalize_data(series):
    scaler = MinMaxScaler(feature_range=(0, 1))
    series = series.reshape(-1, 1)
    scaled_series = scaler.fit_transform(series)
    return scaled_series, scaler

# Step 3: Create the training data
def create_dataset(series, time_step=100):
    X, y = [], []
    for i in range(len(series) - time_step):
        X.append(series[i:(i + time_step), 0])
        y.append(series[i + time_step, 0])
    return np.array(X), np.array(y)

# Step 4: Build and train the LSTM model
def build_and_train_model(X_train, y_train, epochs=50, batch_size=100):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, activation='relu', return_sequences=False))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    # Save the model
    model.save("predict.h5") 
    return model

# Step 5: Predict the next 10 values
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

# Putting it all together
file_path = 'Rate.csv'
series = load_data(file_path)
scaled_series, scaler = normalize_data(series)
X_train, y_train = create_dataset(scaled_series)

# Reshape X_train to be [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

model = build_and_train_model(X_train, y_train)

predictions = predict_future_values(model, scaled_series, scaler)
print(predictions)