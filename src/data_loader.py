import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException

def load_data(path):
    """loading raw data from path"""
    logging.info("loading data from path")
    
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at {path}")
        
        df = pd.read_csv(path)
        
        if df.empty:
            raise ValueError(f"data file is empty at {path}")
        
        logging.info(f"data loaded sucessfully with shape: {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e)
        
        