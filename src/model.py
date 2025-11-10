import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from src.logger import logging
from src.exception import CustomException

def build_ann(input_dim, lr=1e-3):
    logging.info("model building has started")
    
    try:
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim, )),
            layers.Dropout(0.2),        
            layers.Dense(32, activation='relu'),
            layers.Dense(1,activation='sigmoid')
        ])
        
        model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        logging.info("Model compiled")
        
        return model
        
    except Exception as e:
        raise CustomException(e)