# Time Series Forecasting Model for Industrial Equipment using LSTM

This project applies Long Short-Term Memory (LSTM) neural networks to forecast time series data for industrial equipment. The objective is to predict future operational metrics using historical data, enabling better planning and maintenance scheduling.

<img width="185" alt="Image" src="https://github.com/user-attachments/assets/1df2f261-77fa-4272-8791-e1b892fc1021" />

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)

## Overview

Time series forecasting involves predicting future values based on previously observed data points. This project uses an LSTM neural network due to its capability to learn long-term dependencies in time series data. The goal is to generate accurate predictions that can be applied to industrial equipment monitoring and maintenance planning.

## Features

- Data Preprocessing and Normalization
- Time Series Data Visualization
- Model Training using LSTM
- Evaluation of Forecasting Performance
- Plotting Forecasted vs. Actual Values

## Technologies Used

- Python
- TensorFlow/Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

## Dataset

- The dataset contains time-stamped sensor readings from industrial equipment.
- Features include temperature, pressure, vibration levels, and RPM.
- Data is preprocessed to create sequences suitable for LSTM input.
<img width="495" alt="Image" src="https://github.com/user-attachments/assets/19475d71-40b1-4e96-a88d-9257cdf8e355" />

## Model Training

An LSTM model was built and trained using the following configuration:

- **Layers**: 2 LSTM Layers with 50 units each
- **Activation**: ReLU
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam

```python
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val))
```

## Evaluation

The model's performance was evaluated using metrics such as:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}")
```

## Results

- Forecasted values are plotted against actual values for visual comparison.
<img width="185" alt="Image" src="https://github.com/user-attachments/assets/1df2f261-77fa-4272-8791-e1b892fc1021" />
