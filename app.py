from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models and scaler
scaler = joblib.load('scaler_fraud.pkl')
rf_model = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
lstm_model = load_model('lstm_model.h5')
feature_names = joblib.load('feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        input_data = [float(request.form[f'feature_{i}']) for i in range(len(feature_names))]
        new_data = pd.DataFrame([input_data], columns=feature_names)
        new_data_scaled = scaler.transform(new_data)
        new_data_lstm = new_data_scaled.reshape(new_data_scaled.shape[0], new_data_scaled.shape[1], 1)

        # Predictions
        rf_probs = rf_model.predict_proba(new_data_scaled)[:, 1]
        xgb_probs = xgb_model.predict_proba(new_data_scaled)[:, 1]
        lstm_probs = lstm_model.predict(new_data_lstm)[:, 1]
        avg_probs = (rf_probs + xgb_probs + lstm_probs) / 3
        hybrid_pred = (avg_probs >= 0.5).astype(int)

        if hybrid_pred[0] == 1:
            result = " The transaction is predicted to be FRAUD."
        else:
            result = " The transaction is predicted to be NOT FRAUD."

        return render_template('result.html', prediction=result)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
