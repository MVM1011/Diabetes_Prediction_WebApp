from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    pedigree = float(request.form['pedigree'])
    age = float(request.form['age'])

    data = np.array([[pregnancies,
                      glucose,
                      blood_pressure,
                      skin_thickness,
                      insulin,
                      bmi,
                      pedigree,
                      age]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Diabetic"
    else:
        result = "Not Diabetic"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)