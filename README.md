# Earthquake Magnitude Prediction

This project implements a machine learning model to predict earthquake magnitudes using a Convolutional Neural Network (CNN). The model is trained on earthquake data and is accessible through a user-friendly web interface built with Flask.

## Project Structure

```
earthquake-magnitude-prediction
├── app
│   ├── __init__.py
│   ├── routes.py
│   ├── model.py
│   ├── utils.py
│   └── templates
│       └── index.html
├── static
│   └── style.css
├── data
│   └── Earthquake_of_last_30 days.csv
├── requirements.txt
├── run.py
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd earthquake-magnitude-prediction
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```
   python run.py
   ```

5. **Access the application:**
   Open your web browser and go to `http://127.0.0.1:5000`.

## Usage

- Enter the latitude, longitude, depth, and year of the earthquake in the provided input fields.
- Click the "Predict" button to get the predicted magnitude of the earthquake.
- The predicted magnitude will be displayed on the same page.

## Dependencies

- Flask
- Keras
- NumPy
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib
