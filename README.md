# Artificial Neural Networks for Price Forecasting

This project implements artificial neural networks (ANNs) using the NARX model to forecast electricity prices based on the GEFCOM dataset. The aim is to evaluate the performance of different ANN architectures and aggregation techniques.

## Project Overview

The project consists of the following key tasks:

1. **NARX Model Forecasting**:
   - Forecast electricity prices for the period from days 361 to 540 using the NARX model. Evaluate the model's performance in terms of Mean Absolute Error (MAE) by varying the number of neurons in the hidden layer from 1 to 10.

2. **Committee Machine of NARX Networks**:
   - Implement a committee machine approach using multiple NARX networks, each with 5 neurons in the hidden layer. The NARX model will be run 10 times, and the performance of the combined forecasts will be assessed relative to the number of runs.

## Data

The project utilizes the GEFCOM dataset, which includes historical electricity prices and other relevant features that inform the forecasting process.

## Results

The outputs of the project will include forecasts for the specified period, performance metrics for different ANN configurations, and a comparison of the committee machine approach against individual NARX network runs.
