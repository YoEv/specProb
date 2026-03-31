
import numpy as np
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

EMBEDDINGS_PATH = 'data_artifacts/clap_embeddings_t64.npz'
RANDOM_STATE = 42

# 1. Load data and check for balance
print('--- 1. Data Balance Check ---')
with np.load(EMBEDDINGS_PATH) as data:
    X = data['embeddings']
    y = data['genres']

unique_genres, counts = np.unique(y, return_counts=True)
genre_counts = dict(zip(unique_genres, counts))

print('Genre distribution in the dataset:')
for genre, count in genre_counts.items():
    print(f'  - {genre}: {count} samples')

# Prepare data for modeling
X_reshaped = X.reshape(X.shape[0], -1, X.shape[-1])
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Apply DCT
X_dct = dct(X_reshaped, type=2, axis=-1, norm='ortho')
X_features = X_dct.reshape(X_dct.shape[0], -1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 2. Generate Classification Report (with Macro Average)
print('\n--- 2. Classification Report (including Macro Avg) ---')
report = classification_report(y_test, y_pred, target_names=le.classes_)
print(report)

# 3. Generate Confusion Matrix
print('\n--- 3. Confusion Matrix ---')
print('Labels:', le.classes_)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print('\n--- Analysis Complete ---')
