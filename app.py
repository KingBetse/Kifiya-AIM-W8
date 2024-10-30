from flask import Flask, render_template, request
import numpy as np
import joblib  # Assuming you save your model with joblib

app = Flask(__name__)

# Load your trained model (replace 'model.pkl' with your model file)
model = joblib.load('models/dt_fraud_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input from the form
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)  # Reshape for the model

    # Make a prediction
    prediction = model.predict(features)

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)