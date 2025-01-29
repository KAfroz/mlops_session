from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

#load model

model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    return jsonify({'output': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)