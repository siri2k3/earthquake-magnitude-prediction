"""
This module initializes the Flask application, loads the machine learning model,
and registers the application routes.
"""

import tensorflow as tf  # Ensure TensorFlow is imported
from flask import Flask


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    This function initializes the Flask app, loads and trains the machine learning model,
    and registers the application routes.

    Returns:
        Flask: The configured Flask application instance.
    """
    app: Flask = Flask(__name__)

    # Import model utilities
    from .model import load_and_preprocess_data, build_model, train_model, predict_magnitude

    try:
        # Load and train model at startup
        x_train, x_val, x_test, y_train, y_val, y_test = load_and_preprocess_data("Earthquake_of_last_30 days.csv")
        model = build_model()
        model = train_model(model, x_train, y_train, x_val, y_val)
        app.config['MODEL'] = model
        app.config['PREDICT_FUNC'] = predict_magnitude
    except tf.errors.OpError as e:
        # Handle TensorFlow-specific errors
        print(f"TensorFlow error occurred: {e}")
        raise
    except Exception as e:
        # Handle general errors
        print(f"An error occurred during model initialization: {e}")
        raise

    # Register blueprints
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
