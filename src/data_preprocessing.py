# src/data_preprocessing.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException

NUMERIC_IMPUTE = "median"
CATEGORICAL_IMPUTE = "most_frequent"


def clean_and_save(df, target_col, save_path="data/processed/heart_clean.csv",
                   numeric_strategy=NUMERIC_IMPUTE, categorical_strategy=CATEGORICAL_IMPUTE):
    """
    Clean dataframe, impute missing values, encode categorical variables, and save processed CSV.
    """
    logging.info("Preprocessing started")
    try:
        df = df.copy()
        
        # Drop duplicates
        before_dropping = df.shape[0]
        df = df.drop_duplicates().reset_index(drop=True)
        logging.info("Dropped %d duplicate rows", before_dropping - df.shape[0])
        
        # Identify numeric and categorical columns
        features = [c for c in df.columns if c != target_col]
        numeric_cols = list(df[features].select_dtypes(include=[np.number]).columns)
        categorical_cols = [c for c in features if c not in numeric_cols]
        
        # Impute numeric columns
        if numeric_cols:
            num_imputer = SimpleImputer(strategy=numeric_strategy)
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
            logging.info("Numeric columns imputed: %s", numeric_cols)
        
        # Impute categorical columns
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy=categorical_strategy)
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            logging.info("Categorical columns imputed: %s", categorical_cols)
        
        # Encode categorical columns
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        else:
            logging.info("No categorical columns to encode")
        
        # Save processed dataframe
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logging.info("Processed data saved at %s", save_path)
        
        return df

    except Exception as e:
        raise CustomException(e)


def split_and_save(df, target_col,
                   train_path="data/train_data/train_heart.csv",
                   test_path="data/test_data/test_heart.csv",
                   test_size=0.2,
                   random_state=42,
                   stratify=True):
    """
    Split dataset into train and test sets and save as CSV.
    """
    logging.info("Splitting data into train/test sets")
    try:
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        stratify_arg = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
        )
        
        train_df = X_train.reset_index(drop=True).join(y_train.reset_index(drop=True))
        test_df = X_test.reset_index(drop=True).join(y_test.reset_index(drop=True))
        
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logging.info("Saved train/test CSVs at %s and %s", train_path, test_path)
        
        return train_path, test_path

    except Exception as e:
        raise CustomException(e)


def fit_and_save_scaler(X_train, scaler_path="artifacts/scaler.joblib"):
    """
    Fit a StandardScaler on training data and save it.
    """
    logging.info("Fitting StandardScaler on training data")
    try:
        scaler = StandardScaler().fit(X_train)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logging.info("Scaler saved at %s", scaler_path)
        return scaler  # returning fitted scaler object for immediate use
    except Exception as e:
        raise CustomException(e)
