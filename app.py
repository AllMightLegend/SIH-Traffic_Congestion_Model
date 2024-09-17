from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load the trained LSTM model and the scaler
lstm_model = load_model('lstm_traffic_model.h5')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

# Function to preprocess incoming data for the LSTM model
def preprocess_data(data, time_steps=5):
    # Normalize data using the same scaler as during training
    scaled_data = scaler.transform(data)
    # Reshape data for LSTM model
    reshaped_data = np.reshape(scaled_data, (1, time_steps, 1))
    return reshaped_data

@app.route('/predict', methods=['POST'])
def predict_traffic():
    # Get the JSON data from the request
    input_data = request.json
    
    # Convert JSON data to a NumPy array
    vehicle_data = np.array(input_data['vehicles']).reshape(-1, 1)
    
    # Ensure we have enough data points
    if len(vehicle_data) < 5:
        return jsonify({"error": "Insufficient data for prediction. Need at least 5 data points."}), 400
    
    # Preprocess the data for the LSTM model
    processed_data = preprocess_data(vehicle_data[-5:])
    
    # Make prediction
    predicted_vehicles = lstm_model.predict(processed_data)
    predicted_vehicles = float(predicted_vehicles[0][0])  # Convert to standard float
    
    # Inverse transform the prediction
    # Since the prediction is a single value, we need to reshape it to match scaler's expected input
    predicted_vehicles_unscaled = scaler.inverse_transform(np.array([[predicted_vehicles]]))
    
    # Construct the response dictionary
    response = {
        'predicted_vehicles': float(predicted_vehicles_unscaled[0][0]),
    }
    
    # Decide traffic light duration based on predicted traffic volume
    if response['predicted_vehicles'] > 40:
        green_light_duration = 120  # High traffic
    elif response['predicted_vehicles'] > 20:
        green_light_duration = 90   # Medium traffic
    else:
        green_light_duration = 60   # Low traffic

    # Return the prediction and suggested green light duration
    response['green_light_duration'] = green_light_duration
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
