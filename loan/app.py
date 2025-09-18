from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("loan_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect 11 features in the same order as training
        features = [
            float(request.form['Gender']),
            float(request.form['Married']),
            float(request.form['Dependents']),
            float(request.form['Education']),
            float(request.form['Self_Employed']),
            float(request.form['ApplicantIncome']),
            float(request.form['CoapplicantIncome']),
            float(request.form['LoanAmount']),
            float(request.form['Loan_Amount_Term']),
            float(request.form['Credit_History']),
            float(request.form['Property_Area'])
        ]

        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)

        if prediction[0] == 1:
            message = "Congratulations! 🎉 Your loan is Approved ✅"
        else:
            message = "Sorry! ❌ Your loan application is Rejected."

        return render_template('index.html', popup_message=message)

    except Exception as e:
        return render_template('index.html', popup_message=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

