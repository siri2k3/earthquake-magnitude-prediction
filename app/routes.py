"""
This module defines the routes for the Flask application, including
the main page and an API endpoint for earthquake magnitude prediction.
"""

from flask import render_template, request, jsonify, Blueprint, current_app
from typing import Optional, Any
from .model import predict_magnitude

main: Blueprint = Blueprint('main', __name__)


@main.route('/', methods=['GET', 'POST'])
def index() -> Any:
    """
    Render the main page of the application. Handles both GET and POST requests.

    For GET requests:
        - Renders the index.html template with no predicted magnitude.

    For POST requests:
        - Extracts input data (latitude, longitude, depth, year) from the form.
        - Uses the model to predict the earthquake magnitude.
        - Renders the index.html template with the predicted magnitude.

    Returns:
        Rendered HTML template for the main page.
    """
    predicted_magnitude: Optional[float] = None
    if request.method == 'POST':
        latitude: float = float(request.form['latitude'])
        longitude: float = float(request.form['longitude'])
        depth: float = float(request.form['depth'])
        year: int = int(request.form['year'])
        model = current_app.config['MODEL']
        predicted_magnitude = predict_magnitude(latitude, longitude, depth, year, model)
        return render_template('index.html', predicted_magnitude=predicted_magnitude)
    return render_template('index.html', predicted_magnitude=predicted_magnitude)


@main.route('/predict', methods=['POST'])
def predict() -> Any:
    """
    API endpoint to predict earthquake magnitude based on input data.

    Expects a JSON payload with the following keys:
        - latitude (float)
        - longitude (float)
        - depth (float)
        - year (int)

    Returns:
        JSON response containing the predicted magnitude.
    """
    if request.content_type != 'application/json':
        return jsonify({'error': 'Unsupported Media Type. Content-Type must be application/json.'}), 415

    data: dict = request.get_json()
    lat: float = float(data.get('latitude'))
    lon: float = float(data.get('longitude'))
    dep: float = float(data.get('depth'))
    yr: int = int(data.get('year'))
    model = current_app.config['MODEL']
    pred: float = predict_magnitude(lat, lon, dep, yr, model)
    return jsonify({'predicted_magnitude': round(pred, 2)})
