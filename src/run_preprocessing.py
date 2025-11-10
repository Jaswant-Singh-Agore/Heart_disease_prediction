from src.data_loader import load_data
from src.data_preprocessing import clean_and_save, split_and_save, fit_and_save_scaler
from src.logger import logging
from src.exception import CustomException
import pandas as pd

def main():
    raw_data = "data/raw_data/heart.csv"
    target_col = "target"
    logging.info("Preprocessing pipeline started")
    
    try:
        # Load raw data
        df = load_data(raw_data)

        # Clean and save processed data
        df_clean = clean_and_save(df, target_col=target_col, save_path="data/processed_data/heart_clean.csv")

        # Split and save train/test data
        train_path, test_path = split_and_save(
            df_clean, 
            target_col=target_col,
            train_path="data/train_data/heart_train.csv",
            test_path="data/test_data/heart_test.csv"
        )

        # Fit and save scaler using training data
        train_df = pd.read_csv(train_path)
        X_train = train_df.drop(columns=[target_col]).values
        fit_and_save_scaler(X_train, scaler_path="artifacts/scaler.joblib")

        logging.info("Preprocessing pipeline completed successfully")

    except Exception as e:
        raise CustomException(e)
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Preprocessing pipeline failed: {e}")
