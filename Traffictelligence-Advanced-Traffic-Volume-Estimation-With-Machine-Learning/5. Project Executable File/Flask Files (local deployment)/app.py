from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
model = joblib.load("final_model")
sc = pickle.load(open("scaler.pkl","rb"))
le = pickle.load(open("label_encoder.pkl","rb"))


name = ['temp', 'rain', 'weather', 'time', 'date']
app = Flask(__name__)
@app.route('/')
def loadpage():
    return render_template("index.html")


def addnewcolumn(data):
  
  data['date'] = pd.to_datetime(data['date'], format = '%Y-%m-%d')
  data['time'] = pd.to_datetime(data['time'], format='%H:%M')
  #data['year'] = data['date'].dt.year
  data['hour'] = data['time'].dt.hour
  data['month'] = data['date'].dt.month
  data['day'] = data['date'].dt.day
  #data['weekend'] = np.where(data['date'].dt.day_name().isin(['Saturday','Sunday']),1,0)
  
  data['weekday'] = data['date'].dt.day_name
  

  return data.drop(columns = ['date','time'], axis = 1)

@app.route('/y_predict', methods = ["POST"])
def prediction():
    
    temp = float(request.form["temp"])+273
    rain = request.form["rain"]
    weather = request.form["weather"]
    time = request.form["time"]
    date = request.form["date"]
    
    print(type((request.form["temp"])))
    x_test = [[float(temp),rain,weather,time,date]]
    #print(x_test)
    
    df = pd.DataFrame(x_test,columns = name)
    rd = addnewcolumn(df)
    print(rd.head())
    #rd = rd.to_numpy()
    rd['weather'] = le.fit_transform(rd['weather']).astype('int')
    rd['temp'] = rd['temp'].astype('float')
    rd['rain'] = rd['rain'].astype('float')
    rd['weekday'] = le.fit_transform(rd['weekday']).astype('int')
    print(rd.info())
    p = np.array(sc.transform(rd))
    print(p)
    #p = p.astype(np.float32)
    
    y_pred = model.predict(p)
 
    #prediction = prediction > 0.5
    
    
    return render_template("index.html",prediction_text = int(y_pred[0]) )

    
    
if __name__ == "__main__":
    app.run(debug = False)
