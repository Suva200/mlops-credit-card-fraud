from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
class ModelEvaluator:
    @staticmethod
    def compute_metrics(y_true, y_pred, y_proba):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
            "ROC-AUC": roc_auc_score(y_true, y_proba),
        }

    @staticmethod
    def get_best_model(results, metric="F1-Score"):
        best = max(results, key=lambda k: results[k][metric])
        return best, results[best][metric]
