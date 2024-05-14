# Hospital Cost Prediction

This project predicts the treatment cost and doctor consultation fee for different diseases, equipment types, and hospitals using machine learning techniques. It leverages a Random Forest Regression model trained on historical data to make predictions.

## Introduction

The rising cost of healthcare is a significant concern globally. Predicting the cost of treatment and doctor consultation fees can help hospitals and patients better plan for medical expenses. This project provides a simple web interface for users to input a disease and equipment type, and it predicts the top hospitals based on treatment cost and doctor consultation fees.

## Features

- Predict treatment cost and doctor consultation fee for a given disease and equipment type.
- Visualize the top hospitals based on treatment cost and doctor consultation fee.
- Provide user-friendly web interface using Streamlit.
- Utilize Random Forest Regression models for prediction.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- streamlit
- matplotlib

You can install the required Python packages using the following command:

```bash
pip install -r requirements.txt

## Usage
1.Clone the repository:
git clone https://https://github.com/svprferg/Hospital-Cost-Prediction
cd hospital-cost-prediction

2.Install the required Python packages:
pip install -r requirements.txt

3.Run the Streamlit app:
streamlit run app.py

4.Enter the disease and select the equipment type to predict the top hospitals.

5.Optionally, visualize the predictions by clicking the "Visualize" button.
