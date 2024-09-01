import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from flask import Flask, jsonify, request, render_template, session, redirect, url_for, send_from_directory
from flask_cors import CORS
from datetime import datetime
import os
from PIL import *

app = Flask(__name__)
app.secret_key = 'QWJhaWt1bWFyIEk='
CORS(app)


def ml(machine,component):
        
    
    file_path = 'synthetic_dataset (2).csv'
    
    df = pd.read_csv(file_path)


   
    print(df.head())

    print(df.tail())

    
    print(df.info())

    def merge_component_parameter(row):
        component = row['Component']
        parameter = row['Parameter']
        return f"{component}_{parameter}"

    df['component_parameter'] = df.apply(merge_component_parameter, axis=1)
    df['Time'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')
    df_pivot = df.pivot_table(index=['Time', 'Machine'],
                            columns='component_parameter',
                            values='Value',
                            fill_value=0)

    df_pivot.reset_index(inplace=True)


    df_pivot.to_csv('transformed_dataset.csv', index=False)

    df

    lst=list(df['component_parameter'].unique())
    print(lst)
    df = df.sort_values(by='Time')
    df.set_index('Time',inplace=True)
    df=df[df['Machine']==machine]
    df

    l=float('-inf')
    h=float('inf')
    thresholds = {
        'Engine_Oil Pressure': {'low': 25, 'high': 65, 'probability': 2},
        'Engine_Speed': {'low':l, 'high': 1800, 'probability': 1},
        'Engine_Temparature': {'low':l, 'high': 105, 'probability': 2},
        'Drive_Brake Control': {'low': 1, 'high':h, 'probability': 1},
        'Drive_Transmission Pressure': {'low': 200, 'high': 450, 'probability': 1},
        'Drive_Pedal Sensor': {'low':l, 'high': 4.7, 'probability': 0},
        'Fuel_Water in Fuel': {'low':l, 'high': 1800, 'probability': 2},
        'Fuel_Level': {'low': 1, 'high':h, 'probability': 0},
        'Fuel_Pressure': {'low': 35, 'high': 65, 'probability': 0},
        'Fuel_Temparature': {'low':l, 'high': 400, 'probability': 2},
        'Misc_System Voltage': {'low': 12.0, 'high': 15.0, 'probability': 2},
        'Misc_Exhaust Gas Temparature': {'low':l, 'high': 365, 'probability': 2},
        'Misc_Hydraulic Pump Rate': {'low':l, 'high': 125, 'probability': 1},
        'Misc_Air Filter Pressure': {'low': 20,'high':h, 'probability': 1}
    }

    component_parameter_mapping = {
        'Engine': ['Engine_Oil Pressure', 'Engine_Speed', 'Engine_Temparature'],
        'Misc': ['Misc_Air Filter Pressure', 'Misc_Exhaust Gas Temparature', 'Misc_Hydraulic Pump Rate', 'Misc_System Voltage'],
        'Drive': ['Drive_Brake Control', 'Drive_Pedal Sensor', 'Drive_Transmission Pressure'],
        'Fuel': ['Fuel_Level', 'Fuel_Pressure', 'Fuel_Temparature', 'Fuel_Water in Fuel']
    }

    probability={
        0:'Low',
        1:'Medium',
        2:'High'
    }


    df.shape

    def create_sequences(data, seq_length):
        xs = []
        ys = []
        for i in range(len(data) - seq_length):
            x = data.iloc[i:(i + seq_length)].values
            y = data.iloc[i + seq_length].values
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def predict_future(model, data, seq_length, steps_ahead):
        future_predictions = []
        current_seq = data[-seq_length:].astype(np.float32) 
        for _ in range(steps_ahead):
            prediction = model.predict(current_seq[np.newaxis, :, :])[0]
            future_predictions.append(prediction)
            current_seq = np.append(current_seq[1:], [prediction], axis=0)  
        return np.array(future_predictions)

    failure_dates = {}
    for part in component_parameter_mapping[component]:
        seq_length = 10
        data=df[df['component_parameter']==part]

        data=data.drop(columns=['Parameter','Machine','Id','Component','component_parameter'])

        X, y = create_sequences(data, seq_length)

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, X_train.shape[2])))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(X_train.shape[2]))

        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X_train, y_train, batch_size=1, epochs=20,verbose=0)
        steps_ahead = 50  
        time_step_seconds = 600  

        future_predictions = predict_future(model, data.values, seq_length, steps_ahead)
        print(future_predictions)

        date1 = datetime(2022, 11, 25, 7, 50)
        current_time = data.index[-1]

        for i in range(steps_ahead):
            if future_predictions[i, 0] > thresholds[part]['high'] or future_predictions[i, 0] < thresholds[part]['low']:
                # if((current_time + pd.Timedelta(seconds=(i+1) * time_step_seconds))>date1):
                failure_dates[part] = current_time + pd.Timedelta(seconds=(i+1) * time_step_seconds)
                break
                

    prob=[]
    result=""
    result1="<br><br>Recommendation:"
    if not failure_dates:
        result += "No failures predicted within the specified horizon."
    else:
        result += "<table border='1'>"
        result += "<thead><tr><th class='px-6 py-3 bg-gray-50 text-left text-xs leading-4 font-medium text-gray-500 uppercase tracking-wider'>Parameter</th><th class='px-6 py-3 bg-gray-50 text-left text-xs leading-4 font-medium text-gray-500 uppercase tracking-wider'>Predicted Failure Date</th><th class='px-6 py-3 bg-gray-50 text-left text-xs leading-4 font-medium text-gray-500 uppercase tracking-wider'>Probability</th></tr></thead>"
        result += "<tbody>"

        for param in component_parameter_mapping[component]:
            if param in failure_dates:
                result += f"<tr><td>{param}</td><td>{failure_dates[param]}</td><td>{probability[thresholds[param]['probability']]}</td></tr>"
                prob.append(thresholds[param]['probability'])
                if param=="Engine_Temparature":
                    result1+="<br>Turn off the engine"
                elif param=="Engine_Oil Pressure":
                    result1+="<br>Maintain the oil pressure"
                elif param=="Fuel_Water in Fuel":
                    result1+="<br><br>Stop the vechile"
                elif param=="Engine_Speed":
                    result1+="<br><br>Maintain the speed <br>"
                elif param=="Fuel_Temparature":
                    result1+="<br><br>Maintain the temperature and stop for a while <br>"
                elif param=="Fuel_Pressure":
                    result1+="<br><br>Maintain the fuel pressure <br>"
                elif param=="Fuel_Level":
                    result1+="<br><br>Refill the fuel tank<br>"
                elif param=="Drive_Transmission Pressure":
                    result1+="<br><br>Check the pressure level <br>"
                elif param=="Drive_Pedal Sensor":
                    result1+="<br><br>Check the pedal sensor  <br>"
                elif param=="Drive_Brake Control":
                    result1+="<br><br>Check the brake control <br>"
                elif param=="Misc_Exhaust Gas Temparature":
                    result1+="<br><br>Check the gas temperature <br>"
                elif param=="Misc_Air Filter Pressure":
                    result1+="<br><br>Check the air filter pressure <br>"
                elif param=="Misc_Hydraulic Pump Rate":
                    result1+="<br><br>Check the Hydraulic pump rate <br>"
                elif param=="Misc_System Voltage":
                    result1+="<br><br>Check the system voltage <br>"
            else:
                result += f"<tr><td>{param}</td><td>No Failure</td><td>N/A</td></tr>"

        result += "</tbody></table>"
        print(prob)
        if prob:
            result += f"<br><br>Probability of Total Failure: {probability[max(prob)]}."
        result+=result1
    return result




@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index1.html", msg=None)    

@app.route("/login", methods=["POST", "GET"])
def login():
    machine_type = request.form.get("machine_type")
    component_type = request.form.get("component_type")
    print(f"Machine Type: {machine_type}")
    print(f"Component Type: {component_type}")

    
    result = ml(machine_type, component_type)
    return jsonify({"msg": result})


if __name__ == "__main__":
    app.run(debug=True)