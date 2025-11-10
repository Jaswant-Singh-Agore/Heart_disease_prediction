# src/run_evaluate.py
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from src.evaluate import evaluate
from src.logger import logging
from src.exception import CustomException

def main():
    logging.info("Evaluation pipeline started")
    try:
        test_path = "data/test_data/heart_test.csv"
        model_path = "artifacts/models/best_model.h5"
        scaler_path = "artifacts/scaler.joblib"

        # Load test data
        df = pd.read_csv(test_path)
        X_test = df.drop(columns=['target']).values.astype(float)
        y_test = df['target'].values

        # Load scaler and scale test data
        scaler = joblib.load(scaler_path)
        X_test_s = scaler.transform(X_test)

        # Load trained model
        model = load_model(model_path)

        # Evaluate
        report, cm, auc = evaluate(
            model, 
            X_test_s, 
            y_test, 
            artifacts_dir="artifacts/figures",
        )

        logging.info(f"Evaluation completed. AUC: {auc:.4f}")

    except Exception as e:
        raise CustomException(e)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Evaluation pipeline failed: {e}")
