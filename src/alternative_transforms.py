import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import helper functions from utils.py
from utils import load_and_prepare_data, run_probe

def run_dwt_analysis(X_time, y):
    """
    Analyzes performance using Discrete Wavelet Transform (DWT) features.
    """
    print("\n--- Running Alternative Transform: DWT ---")
    n_samples = X_time.shape[0]
    
    # Apply DWT to each time-series of length 32 for each feature
    # We'll use the 'db1' (Daubechies) wavelet and get approximation (cA) and detail (cD) coeffs
    # This results in two sets of coefficients for each signal.
    coeffs = pywt.dwt(X_time, 'db1', axis=2)
    cA, cD = coeffs
    
    # Concatenate the approximation and detail coefficients along the last axis
    X_dwt = np.concatenate((cA, cD), axis=2)
    print(f"Shape after DWT and concatenation: {X_dwt.shape}")
    
    # Flatten and run probe
    X_dwt_flat = X_dwt.reshape(n_samples, -1)
    accuracy = run_probe(X_dwt_flat, y)
    
    print(f"Accuracy on DWT features: {accuracy:.4f}")
    print("------------------------------------------")
    return accuracy

def run_cnn_analysis(X_time, y):
    """
    Analyzes performance using a 1D CNN as a learnable filter bank.
    """
    print("\n--- Running Alternative Transform: 1D CNN ---")
    n_samples = X_time.shape[0]
    n_features = X_time.shape[1] # 1536
    n_timesteps = X_time.shape[2] # 32
    
    # The CNN expects input shape (n_samples, n_timesteps, n_features)
    # Our data is (n_samples, n_features, n_timesteps), so we need to transpose
    X_cnn_input = X_time.transpose(0, 2, 1)
    print(f"Transposed data for CNN input to shape: {X_cnn_input.shape}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    y_categorical = to_categorical(y_encoded)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_cnn_input, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )

    # Scale data
    # We need to reshape for scaler, fit, and then reshape back
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    # Build the 1D CNN model
    input_layer = Input(shape=(n_timesteps, n_features))
    # Convolutional layers act as learnable filters
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    # Global pooling to aggregate features over time
    x = GlobalAveragePooling1D()(x)
    # Final classification layer
    output_layer = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"Accuracy on 1D CNN (learnable filters): {accuracy:.4f}")
    print("--------------------------------------------------")
    return accuracy

if __name__ == "__main__":
    print("Loading data...")
    X_time, y = load_and_prepare_data()
    
    # Run DWT analysis
    run_dwt_analysis(X_time, y)
    
    # Run CNN analysis
    run_cnn_analysis(X_time, y)
