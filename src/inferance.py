import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from src.logger import logging
from src.exception import CustomException

def predict_sample(sample_dict, feature_order, 
                   model_path="artifacts/models/best_model.h5",
                   scaler_path="artifacts/scaler.joblib"):
    """
    Predict class (0/1) and probability for a single sample.
    """
    logging.info("Inference started")

    try:
        # check model and scaler existence
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        # check feature completeness
        missing = [f for f in feature_order if f not in sample_dict]
        if missing:
            raise CustomException(f"Missing features: {missing}")

        # load model and scaler
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)

        # prepare input array
        x = np.array([[sample_dict[f] for f in feature_order]], dtype=float)
        x_scaled = scaler.transform(x)

        # prediction
        prob = float(model.predict(x_scaled).ravel()[0])
        pred_class = int(prob >= 0.5)

        logging.info(f"Inference completed — class: {pred_class}, prob: {prob:.4f}")
        return {"class": pred_class, "probability": prob}

    except Exception as e:
        raise CustomException(e)
