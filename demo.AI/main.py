from flask import Flask,request,jsonify
import joblib
import numpy as np 

app=Flask(__name__)
model=joblib.load('ac_model.pkl')

@app.route("/")

def home():
    return "AC Project"
@app.route("/predict",methods=['POST'])

def predict():
    data=request.json#data will take temp value and humidity value(json java script object notation) it is type of data use store the data store as key value pairs
    temp=data.get('temperature')
    hum=data.get('humidity')
    

    if temp is None or hum is None:
        return jsonify({"error":"Missing temperature or humidity"}),400

    prediction=model.predict(np.array([[temp,hum]]))
    status="ON" if prediction == 1 else "off"
    return jsonify({"ac_status":status})

if __name__== '__main__' :
    app.run(host="0.0.0.0",port=5000,debug=True)