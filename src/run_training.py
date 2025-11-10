import os
import joblib
import pandas as pd
from src.data_loader import load_data
from src.data_preprocessing import fit_and_save_scaler
from src.model import build_ann
from src.model_train import train_model
from src.logger import logging
from src.exception import CustomException

def main():
    train_csv = "data/train_data/heart_train.csv"
    test_csv = "data/test_data/heart_test.csv"
    model_path = "artifacts/models/best_model.h5"
    scaler_path = "artifacts/scaler.joblib"

    logging.info("Training pipeline started")

    try:
        # Load training and testing data
        train_df = load_data(train_csv)
        test_df = load_data(test_csv)

        # Split features and target
        X_train = train_df.drop(columns=['target']).values.astype(float)
        y_train = train_df['target'].values
        X_test = test_df.drop(columns=['target']).values.astype(float)
        y_test = test_df['target'].values

        # Fit and save scaler if not already present
        if not os.path.exists(scaler_path):
            fit_and_save_scaler(X_train, scaler_path=scaler_path)
            logging.info("Scaler fitted and saved.")
        else:
            logging.info("Scaler already exists. Using saved scaler.")
        
        # Load the scaler
        scaler = joblib.load(scaler_path)

        # Transform the features
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Build ANN model
        model = build_ann(input_dim=X_train_s.shape[1])

        # Train the model
        train_model(
            model,
            X_train_s, y_train,
            X_val=X_test_s, y_val=y_test,
            epochs=100,
            batch_size=16,
            model_path=model_path
        )

        logging.info(f"Training pipeline completed successfully. Model saved at {model_path}")

    except Exception as e:
        raise CustomException(e)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Training pipeline failed: %s", e)
