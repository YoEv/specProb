from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_probe(X_features, y_labels_str, random_state=42, test_size=0.2):
    """
    A generic probing function.
    - Encodes string labels.
    - Splits data.
    - Scales features.
    - Trains a logistic regression model.
    - Returns accuracy, the trained model, and the label encoder.
    """
    le = LabelEncoder()
    y_labels = le.fit_transform(y_labels_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=test_size, random_state=random_state, stratify=y_labels
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    return accuracy, model, le


def run_probe_with_predictions(
    X_features,
    y_labels_str,
    random_state=42,
    test_size=0.2,
):
    """
    Extended probing function for diagnostics.

    - Matches the behaviour of `run_probe` (encoding, split, scaling, training).
    - Additionally returns the test labels and predictions so that callers can
      compute confusion matrices and other detailed metrics.

    Returns:
        accuracy (float): classification accuracy on the test split.
        model: trained sklearn classifier.
        le (LabelEncoder): fitted label encoder for the string labels.
        y_test (ndarray): integer-encoded true labels for the test split.
        y_pred (ndarray): integer-encoded predicted labels for the test split.
        label_names (list[str]): ordered list of label strings corresponding to
            the integer classes.
    """
    le = LabelEncoder()
    y_labels = le.fit_transform(y_labels_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=y_labels,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    label_names = list(le.classes_)

    return accuracy, model, le, y_test, y_pred, label_names
