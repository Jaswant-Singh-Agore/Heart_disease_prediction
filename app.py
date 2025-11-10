from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from src.exception import CustomException
from src.logger import logging
import os

app = Flask(__name__)

# Load model and scaler once at startup
try:
    model = load_model("artifacts/models/best_model.h5")
    scaler = joblib.load("artifacts/scaler.joblib")
    logging.info("Model and Scaler loaded successfully.")
except Exception as e:
    raise CustomException(e)


@app.route('/')
def home():
    # make sure template loads without prediction_text
    return render_template('index.html', prediction_text=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # collect data from form (13 inputs in correct order)
        data = [float(x) for x in request.form.values()]
        arr = np.array([data])
        arr_scaled = scaler.transform(arr)

        # predict
        pred_prob = model.predict(arr_scaled)[0][0]
        prediction = int(pred_prob >= 0.5)

        if prediction == 1:
            result = f"High risk of Heart Disease (Probability: {pred_prob:.2f})"
        else:
            result = f"Low risk of Heart Disease (Probability: {pred_prob:.2f})"

        # return page with prediction
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        logging.error("Prediction failed.")
        raise CustomException(e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
