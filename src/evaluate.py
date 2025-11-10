import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from src.logger import logging
from src.exception import CustomException

def evaluate(model, X_test, y_test, artifacts_dir="artifacts/figures"):
    """
    Evaluating the model and checking how it is performing
    """
    logging.info("Evaluating the model")
    
    try:
        os.makedirs(artifacts_dir, exist_ok=True)
        y_pred_prob = model.predict(X_test).ravel()

        # ROC curve calculation
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label="ROC (AUC = {:.3f})".format(roc_auc))
        plt.plot([0, 1], [0, 1], '--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(artifacts_dir, "roc_curve.png")
        plt.savefig(roc_path)
        logging.info("ROC saved to %s (AUC=%.4f)", roc_path, roc_auc)

        # Predictions
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # Report & confusion matrix
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        np.save(os.path.join(artifacts_dir, "confusion_matrix.npy"), cm)

        logging.info("Evaluation completed")
        return report, cm, roc_auc
        
    except Exception as e:
        raise CustomException(e)
