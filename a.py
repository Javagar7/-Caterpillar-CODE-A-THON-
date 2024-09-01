import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'QWJhaWt1bWFyIEk='
CORS(app)

def ml(machine, component):
    # Load dataset
    file_path = 'test.csv'
    df = pd.read_csv(file_path)

    # Merge component and parameter into one column
    df['component_parameter'] = df.apply(lambda row: f"{row['Component']}_{row['Parameter']}", axis=1)
    df['Time'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')

    # Pivot the dataframe to create separate columns for each component-parameter combination
    df_pivot = df.pivot_table(index=['Time', 'Machine'], columns='component_parameter', values='Value', fill_value=0)
    df_pivot.reset_index(inplace=True)

    df_pivot.to_csv('transformed_dataset.csv', index=False)

    df = df_pivot.sort_values(by='Time')
    df.set_index('Time', inplace=True)
    df = df[df['Machine'] == machine]

    # Check column names for debugging
    print("Columns in the dataframe after pivot:")
    print(df.columns)

    thresholds = {
        'Engine_Oil Pressure': {'low': 25, 'high': 65, 'probability': 2},
        'Engine_Speed': {'low': float('-inf'), 'high': 1800, 'probability': 1},
        'Engine_Temperature': {'low': float('-inf'), 'high': 105, 'probability': 2},
        'Drive_Brake Control': {'low': 1, 'high': float('inf'), 'probability': 1},
        'Drive_Transmission Pressure': {'low': 200, 'high': 450, 'probability': 1},
        'Drive_Pedal Sensor': {'low': float('-inf'), 'high': 4.7, 'probability': 0},
        'Fuel_Water in Fuel': {'low': float('-inf'), 'high': 1800, 'probability': 2},
        'Fuel_Level': {'low': 1, 'high': float('inf'), 'probability': 0},
        'Fuel_Pressure': {'low': 35, 'high': 65, 'probability': 0},
        'Fuel_Temperature': {'low': float('-inf'), 'high': 400, 'probability': 2},
        'Misc_System Voltage': {'low': 12.0, 'high': 15.0, 'probability': 2},
        'Misc_Exhaust Gas Temperature': {'low': float('-inf'), 'high': 365, 'probability': 2},
        'Misc_Hydraulic Pump Rate': {'low': float('-inf'), 'high': 125, 'probability': 1},
        'Misc_Air Filter Pressure': {'low': 20, 'high': float('inf'), 'probability': 1}
    }

    component_parameter_mapping = {
        'Engine': ['Engine_Oil Pressure', 'Engine_Speed', 'Engine_Temperature'],
        'Misc': ['Misc_Air Filter Pressure', 'Misc_Exhaust Gas Temperature', 'Misc_Hydraulic Pump Rate', 'Misc_System Voltage'],
        'Drive': ['Drive_Brake Control', 'Drive_Pedal Sensor', 'Drive_Transmission Pressure'],
        'Fuel': ['Fuel_Level', 'Fuel_Pressure', 'Fuel_Temperature', 'Fuel_Water in Fuel']
    }

    probability = {
        0: 'Low',
        1: 'Medium',
        2: 'High'
    }

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data.iloc[i:(i + seq_length)].values
            y = data.iloc[i + seq_length].values
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def predict_future(model, data, seq_length, steps_ahead):
        future_predictions = []
        current_seq = data[-seq_length:].astype(np.float32)  # Last sequence from the data

        for _ in range(steps_ahead):
            prediction = model.predict(current_seq[np.newaxis, :, :])[0]
            future_predictions.append(prediction)
            current_seq = np.append(current_seq[1:], [prediction], axis=0)  # Slide window

        return np.array(future_predictions)

    failure_dates = {}
    for part in component_parameter_mapping[component]:
        if part not in df.columns:
            print(f"Warning: {part} not found in dataframe columns.")
            continue

        seq_length = 10
        data = df[[part]].dropna()

        X, y = create_sequences(data, seq_length)

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(X_train.shape[2])
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=1, epochs=20, verbose=0)

        # Predict future values beyond the dataset
        steps_ahead = 50
        time_step_seconds = 60

        future_predictions = predict_future(model, data.values, seq_length, steps_ahead)

        # Adjust current time to be the current date in 2024
        current_time = df.index[-1]
        if current_time < datetime(2024, 7, 6):
            time_gap = datetime(2024, 7, 6) - current_time
        else:
            time_gap = timedelta(0)
        current_time += time_gap

        for i in range(steps_ahead):
            if future_predictions[i, 0] > thresholds[part]['high'] or future_predictions[i, 0] < thresholds[part]['low']:
                failure_dates[part] = current_time + pd.Timedelta(seconds=(i + 1) * time_step_seconds)
                break

    result_str = "Analysis Results\n"
    if not failure_dates:
        result_str += "No failures predicted within the specified horizon."
    else:
        prob = []
        for param in component_parameter_mapping[component]:
            if param in failure_dates:
                result_str += f"\nThe predicted failure date for {param} is: {failure_dates[param]} and the probability is: {probability[thresholds[param]['probability']]}"
                prob.append(thresholds[param]['probability'])
            else:
                result_str += f"\nNo failure predicted for {param}"

        if prob:
            result_str += f"\nProbability of Total Failure: {probability[max(prob)]}"

    return result_str

@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index.html", msg=None)    

@app.route("/login", methods=["POST", "GET"])
def login():
    machine_type = request.form.get("machine_type")
    component_type = request.form.get("component_type")
    result = ml(machine_type, component_type)
    return jsonify({"msg": result})

if __name__ == "__main__":
    app.run(debug=True)
