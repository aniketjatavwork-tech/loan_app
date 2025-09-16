from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load model
model = pickle.load(open("loan_model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return "Loan Prediction API is Running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]   # Expecting a list of values
    prediction = model.predict([np.array(data)])
    return jsonify({"prediction": str(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
