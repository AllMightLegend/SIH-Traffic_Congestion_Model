# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
path="traffic.csv"
data = pd.read_csv(path)  # Replace with actual dataset



# Parse the DateTime column and set it as the index
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)

# Extract the 'Vehicles' column as the feature
vehicles = data[['Vehicles']]

# Normalize the data (scale values between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_vehicles = scaler.fit_transform(vehicles)
import joblib
joblib.dump(scaler, 'scaler.pkl')

# Function to create time-series data for LSTM
def create_dataset(data, time_steps=5):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Define the number of time steps (window size) for LSTM input
time_steps = 5

# Create sequences for LSTM model
X, y = create_dataset(scaled_vehicles, time_steps)

# Reshape X for LSTM input (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into training and testing sets (80% training, 20% testing)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential()

# Add LSTM layers with dropout to prevent overfitting
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))

# Add a dense output layer for the predicted vehicle count
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

# Evaluate the model's performance
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Inverse transform the predictions to get the actual vehicle count
y_pred_inverse = scaler.inverse_transform(y_pred)

# Compare predictions to actual values
for i in range(10):
    print(f'Predicted Vehicles: {y_pred_inverse[i][0]}, Actual: {scaler.inverse_transform([[y_test[i]]])[0][0]}')


# %%
model.save("lstm_traffic_model.h5")

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import time

# Load the pre-trained LSTM model
model = load_model('lstm_traffic_model.h5')

# Load the fitted scaler from the saved file
scaler = joblib.load('scaler.pkl')

# Function to preprocess real-time data
def preprocess_real_time_data(data, scaler, time_steps=5):
    # Normalize data using the same scaler as during training
    scaled_data = scaler.transform(data)
    # Reshape data for LSTM model
    reshaped_data = np.reshape(scaled_data, (1, time_steps, 1))
    return reshaped_data

# Function to simulate real-time traffic data collection
def get_real_time_data():
    # Simulate vehicle count between 5 and 50
    vehicles = np.random.randint(5, 50)
    return np.array([[vehicles]])

# Initialize a list to store real-time data (sliding window)
real_time_data = []

# Simulate real-time traffic signal control system
while True:
    # Fetch real-time traffic data
    current_traffic_data = get_real_time_data()
    real_time_data.append(current_traffic_data[0][0])  # Append the new data point

    # Ensure we have at least 'time_steps' data points for LSTM (i.e., 5 data points)
    if len(real_time_data) >= 5:
        # Use the last 5 data points for prediction (sliding window)
        data_for_prediction = np.array(real_time_data[-5:]).reshape(-1, 1)

        # Preprocess the data
        processed_data = preprocess_real_time_data(data_for_prediction, scaler)

        # Predict the traffic volume for the next period
        predicted_vehicles = model.predict(processed_data)
        predicted_vehicles = scaler.inverse_transform(predicted_vehicles)

        # Adaptive signal management: adjust green light duration based on predicted traffic volume
        if predicted_vehicles[0][0] > 40:  # Heavy traffic threshold
            green_light_duration = 120  # Longer green light duration for high traffic
        elif predicted_vehicles[0][0] > 20:
            green_light_duration = 90  # Medium traffic
        else:
            green_light_duration = 60  # Low traffic

        print(f"Predicted Vehicles: {predicted_vehicles[0][0]}, Green Light Duration: {green_light_duration} seconds")

    # Simulate updating the traffic light system
    time.sleep(10)  # Wait 10 seconds before fetching new data

# %%
import joblib
import cv2
import numpy as np
import requests
import torch
import pygame
from tensorflow.keras.models import load_model

# Load the pre-trained YOLOv5 model from PyTorch Hub
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the pre-trained LSTM model and scaler
lstm_model = load_model('lstm_traffic_model.h5')
scaler = joblib.load('scaler.pkl')

# Function to process each frame and detect vehicles
def detect_vehicles(frame):
    results = yolo_model(frame)
    detected = results.pandas().xyxy[0]
    vehicle_count = 0

    for _, row in detected.iterrows():
        if row['name'] in ['car', 'truck', 'bus', 'motorcycle']:
            vehicle_count += 1

    results.render()
    return vehicle_count

# Function to preprocess real-time data for LSTM
def preprocess_real_time_data(data, scaler, time_steps=5):
    scaled_data = scaler.transform(data)
    reshaped_data = np.reshape(scaled_data, (1, time_steps, 1))
    return reshaped_data

# Function to manage traffic lights based on predictions
def manage_traffic(vehicle_count, real_time_data, lstm_model, scaler):
    real_time_data.append(vehicle_count)

    if len(real_time_data) >= 5:
        data_for_prediction = np.array(real_time_data[-5:]).reshape(-1, 1)
        processed_data = preprocess_real_time_data(data_for_prediction, scaler)
        predicted_vehicles = lstm_model.predict(processed_data)
        predicted_vehicles = scaler.inverse_transform(predicted_vehicles)

        if predicted_vehicles[0][0] > 40:
            green_light_duration = 120
        elif predicted_vehicles[0][0] > 20:
            green_light_duration = 90
        else:
            green_light_duration = 60

        return predicted_vehicles[0][0], green_light_duration
    return None, None

# Function to send vehicle count data to the backend
def send_data_to_backend(vehicle_data):
    backend_url = 'http://localhost:5000/predict'  # Replace with your actual backend URL
    try:
        response = requests.post(backend_url, json={"vehicles": vehicle_data})
        response.raise_for_status()  # Raise an exception for HTTP error responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to backend: {e}")
        return {"error": str(e)}

# Pygame simulation setup
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Traffic Congestion Simulation")

    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    # Colors for traffic lights
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)

    # Traffic light state
    traffic_light_color = GREEN

    # Access the CCTV feed
    video_path = '2252223-uhd_3840_2160_30fps.mp4'  # Replace with your actual video path or camera feed URL
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # Frame interval for capturing one frame per second

    real_time_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if it's time to capture the next frame based on interval
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        # Resize the frame to fit the screen
        frame_resized = cv2.resize(frame, (640, 480))  # Resize to 640x480 pixels; adjust as needed
        cv2.imshow('Traffic CCTV Feed', frame_resized)

        # Detect vehicles and get the count
        vehicle_count = detect_vehicles(frame_resized)

        # Use LSTM model to manage traffic lights based on vehicle count
        predicted_vehicles, green_light_duration = manage_traffic(vehicle_count, real_time_data, lstm_model, scaler)

        # Prepare vehicle count data for backend
        if predicted_vehicles is not None:
            response = send_data_to_backend(real_time_data[-5:])
            print(f"Backend response: {response}")

        # Update the traffic light color based on the prediction
        if green_light_duration == 120:
            traffic_light_color = GREEN
        elif green_light_duration == 90:
            traffic_light_color = YELLOW
        else:
            traffic_light_color = RED

        # Pygame rendering
        screen.fill(WHITE)
        pygame.draw.circle(screen, traffic_light_color, (400, 300), 50)

        # Display detected vehicle count
        vehicle_count_text = font.render(f"Detected Vehicles: {vehicle_count}", True, (0, 0, 0))
        screen.blit(vehicle_count_text, (50, 50))

        # Display predicted vehicle count
        if predicted_vehicles is not None:
            predicted_text = font.render(f"Predicted Vehicles: {int(predicted_vehicles)}", True, (0, 0, 0))
            green_light_text = font.render(f"Green Light Duration: {green_light_duration}s", True, (0, 0, 0))
            screen.blit(predicted_text, (50, 100))
            screen.blit(green_light_text, (50, 150))

        pygame.display.flip()

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                return

        # Control the simulation speed
        clock.tick(30)  # Adjust to control the frame rate of the simulation

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

# Run the simulation
run_simulation()
# %%
