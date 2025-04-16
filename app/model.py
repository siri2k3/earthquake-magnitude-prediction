"""
This module provides functions for loading and preprocessing earthquake data,
building a Convolutional Neural Network (CNN) model, training the model, and
making predictions for earthquake magnitudes.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING logs

import pandas as pd
import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input  # Import Input layer
from keras.api.models import load_model
from sklearn.model_selection import train_test_split
from typing import Tuple

MODEL_PATH = "saved_model/earthquake_model.keras"  # Update to use the .keras extension
PREPROCESSED_DATA_PATH = "saved_data/preprocessed_data.npz"


def load_and_preprocess_data(csv_path: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the earthquake dataset. If preprocessed data exists, load it
    from a file; otherwise, preprocess the data and save it for future use.

    Args:
        csv_path (str): Path to the CSV file containing earthquake data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Training, validation, and test sets for features and labels.
    """
    # Check if preprocessed data exists
    if os.path.exists(PREPROCESSED_DATA_PATH):
        data = np.load(PREPROCESSED_DATA_PATH)
        return (data['x_train'], data['x_val'], data['x_test'],
                data['y_train'], data['y_val'], data['y_test'])

    # Load the dataset into a pandas DataFrame
    df: pd.DataFrame = pd.read_csv(csv_path)

    # Convert the 'time' column to datetime format
    df['time'] = pd.to_datetime(df['time'])

    # Drop unnecessary columns
    df = df.drop(['dmin', 'magError', 'magNst'], axis=1)

    # Fill missing values with the column mean
    df['nst'] = df['nst'].fillna(df['nst'].mean())
    df['gap'] = df['gap'].fillna(df['gap'].mean())
    df['horizontalError'] = df['horizontalError'].fillna(df['horizontalError'].mean())

    # Extract the year from the 'time' column
    df['year'] = df['time'].dt.year

    # Remove rows with invalid depth or magnitude values
    mask = (df['depth'] < 0) | (df['mag'] < 0)
    df = df[~mask]

    # Select features and labels
    features = ['latitude', 'longitude', 'depth', 'year']
    x: np.ndarray = df[features].values
    y: np.ndarray = df[['mag']].values

    # Ensure the data is in the correct format
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray

    # Split the data into training, validation, and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Reshape the data for the CNN model
    x_train = x_train.reshape(x_train.shape[0], 1, 4, 1)
    x_test = x_test.reshape(x_test.shape[0], 1, 4, 1)
    x_val = x_val.reshape(x_val.shape[0], 1, 4, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    y_val = np.array(y_val).reshape(-1, 1)

    # Save the preprocessed data for future use
    os.makedirs(os.path.dirname(PREPROCESSED_DATA_PATH), exist_ok=True)
    np.savez(PREPROCESSED_DATA_PATH, x_train=x_train, x_val=x_val, x_test=x_test,
             y_train=y_train, y_val=y_val, y_test=y_test)

    return x_train, x_val, x_test, y_train, y_val, y_test


def build_model() -> Sequential:
    """
    Build a Convolutional Neural Network (CNN) model for earthquake magnitude prediction.

    Returns:
        Sequential: Compiled CNN model.
    """
    # Define the CNN architecture
    model: Sequential = Sequential()
    model.add(Input(shape=(1, 4, 1)))  # Use Input layer to define the input shape
    model.add(Conv2D(32, (1, 2), activation='relu', padding='same'))  # No input_shape here
    model.add(MaxPooling2D((1, 2)))
    model.add(Conv2D(64, (1, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D((1, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))

    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def save_model(model: Sequential) -> None:
    """
    Save the trained model to a file in the native Keras format.

    Args:
        model (Sequential): Trained CNN model.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)  # Save in .keras format


def load_trained_model() -> Sequential:
    """
    Load the trained model from a file in the native Keras format.

    Returns:
        Sequential: Loaded CNN model.
    """
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)  # Load from .keras format
    else:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")


def train_model(model: Sequential, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                y_val: np.ndarray) -> Sequential:
    """
    Train the CNN model on the training data.

    Args:
        model (Sequential): Compiled CNN model.
        x_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training label data.
        x_val (np.ndarray): Validation feature data.
        y_val (np.ndarray): Validation label data.

    Returns:
        Sequential: Trained CNN model.
    """
    # Train the model with the training and validation data
    model.fit(x_train, y_train, epochs=50, batch_size=200, validation_data=(x_val, y_val), verbose=0)
    save_model(model)  # Save the trained model
    return model


def predict_magnitude(
        lat: float,
        lon: float,
        dep: float,
        yr: int,
        cnn_model: Sequential
) -> float:
    """
    Predict the earthquake magnitude using the trained CNN model.

    Args:
        lat (float): Latitude of the earthquake location.
        lon (float): Longitude of the earthquake location.
        dep (float): Depth of the earthquake in kilometers.
        yr (int): Year of the earthquake event.
        cnn_model (Sequential): Trained CNN model.

    Returns:
        float: Predicted earthquake magnitude.
    """
    # Prepare the input data for prediction
    input_data: np.ndarray = np.array([[float(lat), float(lon), float(dep), int(yr)]])
    input_data = input_data.reshape(input_data.shape[0], 1, 4, 1)

    # Make the prediction and return the result
    magnitude_prediction: float = float(cnn_model.predict(input_data, verbose=0)[0][0])
    return magnitude_prediction
