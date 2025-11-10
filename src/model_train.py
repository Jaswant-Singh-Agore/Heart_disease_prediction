import os
import json
from tensorflow.keras import callbacks
from src.logger import logging
from src.exception import CustomException

def train_model(model, X_train, y_train, X_val=None, y_val=None,
                epochs=100, batch_size=32, model_path="artifacts/models/best_model.h5"):
    """
    Train a compiled model with optional validation data and callbacks.
    """
    logging.info("Model training started")
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # use callbacks only if validation data exists
        cb = []
        if X_val is not None and y_val is not None:
            cb = [
                callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
            ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            verbose=2
        )
        
        # save training history
        hist_path = os.path.splitext(model_path)[0] + "_history.json"
        with open(hist_path, 'w') as f:
            json.dump(history.history, f)
        
        logging.info(f"Model trained successfully and saved to {model_path}")
        return history

    except Exception as e:
        raise CustomException(e)
