from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, logger: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute classification metrics from true labels and predictions.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        logger: Optional logging function.

    Returns:
        Dictionary containing evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf = confusion_matrix(y_true, y_pred)

    if logger:
        logger.info(f'[Eval] Accuracy:  {acc:.4f}')
        logger.info(f'[Eval] Precision: {prec:.4f}')
        logger.info(f'[Eval] Recall:    {rec:.4f}')
        logger.info(f'[Eval] F1-Score:  {f1:.4f}')
        logger.info(f'[Eval] Confusion Matrix:\n{conf}')

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': conf.tolist(),
        'classification_report': classification_report(
            y_true, y_pred, output_dict=True
        ),
    }
