from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load model
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

# Home route with frontend form
@app.route("/")
def home():
    return render_template("index.html")

# API endpoint (for developers / JSON requests)
@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["features"]   # Expecting a list of values
    prediction = model.predict([np.array(data)])
    return jsonify({"prediction": str(prediction[0])})

# Frontend form submission
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([np.array(features)])
        result = "✅ Loan Approved" if prediction[0] == 1 else "❌ Loan Rejected"

        return render_template("index.html", prediction_text=result)
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
