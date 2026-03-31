from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute a confusion matrix with an optional mapping to human-readable labels.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        label_names: Optional sequence of label strings where the index matches
            the integer class ID. If not provided, the function will infer the
            set of classes from y_true and sort them.

    Returns:
        cm: Confusion matrix as a 2D NumPy array.
        labels: List of label strings corresponding to the rows/columns of cm.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if label_names is not None:
        labels_int = np.arange(len(label_names))
        cm = confusion_matrix(y_true, y_pred, labels=labels_int)
        labels = list(label_names)
    else:
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        labels = [str(lbl) for lbl in unique_labels]

    return cm, labels

