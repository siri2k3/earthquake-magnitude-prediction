"""
This module provides utility functions for preprocessing and validating input data
used in earthquake magnitude prediction.
"""

import numpy as np


def preprocess_input(latitude, longitude, depth, year):
    """
    Preprocess the input data for the prediction model.

    Args:
        latitude (float): Latitude of the earthquake location.
        longitude (float): Longitude of the earthquake location.
        depth (float): Depth of the earthquake in kilometers.
        year (int): Year of the earthquake event.

    Returns:
        np.ndarray: Preprocessed input data reshaped for the model.
    """
    input_data = np.array([[float(latitude), float(longitude), float(depth), int(year)]])
    input_data = input_data.reshape(input_data.shape[0], 1, 4, 1)
    return input_data


def validate_input(latitude, longitude, depth, year):
    """
    Validate the input data to ensure it can be converted to the required types.

    Args:
        latitude (str): Latitude of the earthquake location.
        longitude (str): Longitude of the earthquake location.
        depth (str): Depth of the earthquake in kilometers.
        year (str): Year of the earthquake event.

    Returns:
        bool: True if all inputs are valid, False otherwise.
    """
    try:
        float(latitude)
        float(longitude)
        float(depth)
        int(year)
        return True
    except ValueError:
        return False
