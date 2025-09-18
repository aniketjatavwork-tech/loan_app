print("App started running...")

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
import os
import warnings
import pickle

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
linear_model = None
lasso_model = None
is_model_trained = False

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "car_data.csv")
MODEL_PATH_LINEAR = os.path.join(BASE_DIR, "linear_model.pkl")
MODEL_PATH_LASSO = os.path.join(BASE_DIR, "lasso_model.pkl")


def load_and_preprocess_data():
    """Load and preprocess the car dataset"""
    try:
        df = pd.read_csv(DATA_PATH)

        df_encoded = df.copy()

        # Encode categorical variables
        if "Fuel_Type" in df_encoded.columns:
            fuel_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
            df_encoded["Fuel_Type"] = df_encoded["Fuel_Type"].map(fuel_mapping)

        if "Seller_Type" in df_encoded.columns:
            seller_mapping = {"Dealer": 0, "Individual": 1}
            df_encoded["Seller_Type"] = df_encoded["Seller_Type"].map(seller_mapping)

        if "Transmission" in df_encoded.columns:
            transmission_mapping = {"Manual": 0, "Automatic": 1}
            df_encoded["Transmission"] = df_encoded["Transmission"].map(transmission_mapping)

        feature_columns = [
            "Year",
            "Present_Price",
            "Kms_Driven",
            "Fuel_Type",
            "Seller_Type",
            "Transmission",
            "Owner",
        ]
        X = df_encoded[feature_columns]
        y = df_encoded["Selling_Price"]

        X = X.fillna(0)

        return X, y, df_encoded

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None


def train_models():
    """Train Linear Regression and Lasso, then save them"""
    global linear_model, lasso_model, is_model_trained

    X, y, df_encoded = load_and_preprocess_data()
    if X is None or y is None:
        print("❌ Data not loaded correctly.")
        return False

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Train Lasso Regression
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train, y_train)

    # Save models
    pickle.dump(linear_model, open(MODEL_PATH_LINEAR, "wb"))
    pickle.dump(lasso_model, open(MODEL_PATH_LASSO, "wb"))

    is_model_trained = True
    print("✅ Models trained & saved successfully")
    return True


def load_models():
    """Load models from disk if available"""
    global linear_model, lasso_model, is_model_trained
    try:
        if os.path.exists(MODEL_PATH_LINEAR) and os.path.exists(MODEL_PATH_LASSO):
            linear_model = pickle.load(open(MODEL_PATH_LINEAR, "rb"))
            lasso_model = pickle.load(open(MODEL_PATH_LASSO, "rb"))
            is_model_trained = True
            print("✅ Models loaded from disk")
    except Exception as e:
        print(f"❌ Error loading models: {e}")


@app.route("/")
def home():
    return render_template("index.html")  # must be inside templates/


@app.route("/train")
def train():
    if train_models():
        return "✅ Models trained successfully! Go back and try prediction."
    else:
        return "❌ Error training models. Check terminal for details."


@app.route("/predict", methods=["POST"])
def predict():
    global is_model_trained
    if not is_model_trained:
        return "❌ Models are not trained yet. Please train first."

    try:
        year = int(request.form["Year"])
        present_price = float(request.form["Present_Price"])
        kms_driven = int(request.form["Kms_Driven"])
        fuel_type = int(request.form["Fuel_Type"])
        seller_type = int(request.form["Seller_Type"])
        transmission = int(request.form["Transmission"])
        owner = int(request.form["Owner"])

        features = np.array(
            [[year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]]
        )

        prediction = linear_model.predict(features)[0]
        return render_template("result.html", prediction=round(prediction, 2))

    except Exception as e:
        return f"❌ Error: {str(e)}"


if __name__ == "__main__":
    load_models()  # Try loading models at startup
    print("Starting Flask server...")
    app.run(debug=True)
