# -*- coding: utf-8 -*-
"""
Earthquake Magnitude Prediction using CNN

Original file is located at
    https://colab.research.google.com/drive/1p91_sTPGlREo80crzebP8buIhvsvuAdY
"""

# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense
from scipy.stats import ttest_ind

# =========================
# Data Loading
# =========================
uploaded: str = "Earthquake_of_last_30 days.csv"
data_set: str = uploaded

# Read the CSV data into a pandas DataFrame
df: pd.DataFrame = pd.read_csv(uploaded)

# =========================
# Data Preprocessing
# =========================
df['time'] = pd.to_datetime(df['time'])

# Drop columns that don't contribute significantly
df = df.drop(['dmin', 'magError', 'magNst'], axis=1)

# Fill missing values with mean
df['nst'] = df['nst'].fillna(df['nst'].mean())
df['gap'] = df['gap'].fillna(df['gap'].mean())
df['horizontalError'] = df['horizontalError'].fillna(df['horizontalError'].mean())

# =========================
# Data Consistency Checks
# =========================
print(df['latitude'].describe())
print(df['longitude'].describe())
print(df['magType'].value_counts())
print(df[df['mag'] <= 0])
print(df[df['depth'] < 0])

valid_mag_types = ['ml', 'md', 'mb', 'mww', 'mwr', 'mb_lg', 'mw', 'mh', 'mlv', 'mwc']
print(df[~df['magType'].isin(valid_mag_types)])

lat_mask = (df['latitude'] < -90) | (df['latitude'] > 90)
lon_mask = (df['longitude'] < -180) | (df['longitude'] > 180)
print(df[lat_mask | lon_mask])

# =========================
# Data Visualization
# =========================
sns.set_style("darkgrid")

# Histogram of magnitude
sns.histplot(data=df, x='mag', kde=True)
plt.title('Histogram of Magnitude')
plt.show()

# Boxplot of depth
sns.boxplot(data=df, x='depth')
plt.title('Boxplot of Depth')
plt.show()

# Countplot of magType
sns.countplot(data=df, x='magType')
plt.title('Countplot of MagType')
plt.show()

# Scatter plot of depth vs magnitude
plt.scatter(df['depth'], df['mag'])
plt.xlabel('Depth')
plt.ylabel('Magnitude')
plt.title('Scatter plot of Depth vs Magnitude')
plt.show()

# Box plot of earthquake magnitude by type
df.boxplot(column='mag', by='type')
plt.title('Box plot of Earthquake Magnitude by Type')
plt.suptitle('*')
plt.show()

# Correlation heatmap
numerical_features = df.select_dtypes(include=['number']).columns
corr = df[numerical_features].corr()
sns.heatmap(corr, annot=True)
plt.show()

# Additional visualizations
sns.scatterplot(data=df, x="mag", y="depth")
sns.histplot(data=df, x="mag")
sns.boxplot(data=df, x="magType", y="mag")
sns.heatmap(corr, annot=True)
sns.pairplot(df)
plt.show()

# =========================
# Statistical Analysis
# =========================
df['region'] = df['place'].str.extract(',\s(.*$)')
mean_mag_by_region = df.groupby('region')['mag'].mean()
group1 = df[df['mag'] < mean_mag_by_region.mean()]
group2 = df[df['mag'] >= mean_mag_by_region.mean()]
t_stat, p_val = ttest_ind(group1['mag'], group2['mag'], equal_var=False)
print("T-test statistic: ", t_stat)
print("P-value: ", p_val)

# =========================
# Feature Engineering
# =========================
df['year'] = df['time'].dt.year

# Remove rows with invalid depth or magnitude
mask = (df['depth'] < 0) | (df['mag'] < 0)
df = df[~mask]

# =========================
# Data Preparation for Model
# =========================
features: list[str] = ['latitude', 'longitude', 'depth', 'year']
x: np.ndarray = df[features].values
y: np.ndarray = df[['mag']].values

# Split data
x_train: np.ndarray
x_test: np.ndarray
y_train: np.ndarray
y_test: np.ndarray
x_val: np.ndarray
y_val: np.ndarray
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Reshape for CNN
x_train = x_train.reshape(x_train.shape[0], 1, 4, 1)
x_test = x_test.reshape(x_test.shape[0], 1, 4, 1)
x_val = x_val.reshape(x_val.shape[0], 1, 4, 1)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
y_val = np.array(y_val).reshape(-1, 1)

# =========================
# CNN Model Building
# =========================
model: Sequential = Sequential()
model.add(Conv2D(32, (1, 2), activation='relu', input_shape=(1, 4, 1), padding='same'))
model.add(MaxPooling2D((1, 2)))
model.add(Conv2D(64, (1, 2), activation='relu', padding='same'))
model.add(MaxPooling2D((1, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# =========================
# Model Training
# =========================
history = model.fit(x_train, y_train, epochs=50, batch_size=200, validation_data=(x_val, y_val))

# =========================
# Model Evaluation
# =========================
loss, mae = model.evaluate(x_test, y_test, verbose=0)
print('Mean Absolute Error:', mae)

# Metrics calculation
y_pred = model.predict(x_test)
y_test_flat = y_test.reshape(-1)
y_pred_flat = y_pred.reshape(-1)

mae = mean_absolute_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
r2 = r2_score(y_test_flat, y_pred_flat)

print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)
print('R-squared (R2):', r2)


# =========================
# Prediction Function
# =========================
def predict_magnitude(
        lat: float,
        lon: float,
        dep: float,
        yr: int,
        cnn_model: Sequential
) -> float:
    input_data: np.ndarray = np.array([[float(lat), float(lon), float(dep), int(yr)]])
    input_data = input_data.reshape(input_data.shape[0], 1, 4, 1)
    magnitude_prediction: float = float(cnn_model.predict(input_data)[0][0])
    return magnitude_prediction


# =========================
# User Input Prediction
# =========================
latitude: float = float(input("Enter latitude: "))
longitude: float = float(input("Enter longitude: "))
depth: float = float(input("Enter depth: "))
year: int = int(input("Enter year: "))

predicted_magnitude: float = predict_magnitude(latitude, longitude, depth, year, model)
print('Predicted Magnitude:', predicted_magnitude)
print("hi")