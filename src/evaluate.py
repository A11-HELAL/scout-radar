"""Metrics - in log space and in euros."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true_log, y_pred_log):
    mae = mean_absolute_error(y_true_log, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2 = r2_score(y_true_log, y_pred_log)

    y_true_eur = np.expm1(y_true_log)
    y_pred_eur = np.expm1(y_pred_log)
    medape = np.median(np.abs(y_true_eur - y_pred_eur) / y_true_eur) * 100

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MedAPE_%": round(medape, 1),
    }
