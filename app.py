import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Users/shwet/Downloads/sss.csv")

# Preprocess the dataset
encoder_disease = LabelEncoder()
encoder_equipment = LabelEncoder()
encoder_hospital = LabelEncoder()
df['Disease'] = encoder_disease.fit_transform(df['Disease'])
df['Equipment Level'] = encoder_equipment.fit_transform(df['Equipment Level'])
df['Hospital Name'] = encoder_hospital.fit_transform(df['Hospital Name'])
X = df[['Disease', 'Equipment Level', 'Hospital Name']]
y = df[['Treatment Cost ($)', 'Doctor Consultation Fee ($)']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regression model for treatment cost
model_cost = RandomForestRegressor(n_estimators=100, random_state=42)
model_cost.fit(X_train, y_train['Treatment Cost ($)'])

# Train a Random Forest Regression model for doctor consultation fee
model_fee = RandomForestRegressor(n_estimators=100, random_state=42)
model_fee.fit(X_train, y_train['Doctor Consultation Fee ($)'])

# Function to predict treatment cost and doctor consultation fee for given disease, equipment type, and hospital
def predict(disease, equipment_type, top_n=5):
    encoded_disease = encoder_disease.transform([disease])[0]
    encoded_equipment = encoder_equipment.transform([equipment_type])[0]
    predictions = []

    for i in range(len(encoder_hospital.classes_)):
        encoded_hospital = i
        prediction_input = [[encoded_disease, encoded_equipment, encoded_hospital]]
        predicted_cost = model_cost.predict(prediction_input)[0]
        predicted_fee = model_fee.predict(prediction_input)[0]
        predictions.append((predicted_cost, predicted_fee, encoded_hospital))

    predictions.sort(reverse=True)  # Sort predictions based on treatment cost from maximum to minimum
    top_hospitals = []

    for i in range(min(top_n, len(predictions))):
        cost, fee, encoded_hospital = predictions[i]
        hospital_name = encoder_hospital.inverse_transform([encoded_hospital])[0]
        top_hospitals.append((hospital_name, cost, fee))

    return top_hospitals

# Streamlit GUI
st.title('Hospital Cost Prediction')

# User input for disease and equipment type
user_disease = st.text_input("Enter the disease:")
user_equipment = st.selectbox("Select the equipment type:", options=['High', 'Medium', 'Low'])

# Button for prediction
if st.button('Predict'):
    # Predict treatment cost and doctor consultation fee
    top_hospitals = predict(user_disease, user_equipment)
    
    # Display top hospitals
    st.write("Top 5 Hospitals (based on maximum to minimum treatment cost):")
    for i, (hospital_name, cost, fee) in enumerate(top_hospitals):
        st.write(f"{i+1}. Hospital Name: {hospital_name}, Treatment Cost: {cost}, Doctor Consultation Fee: {fee}")

# Button for visualization
if st.button('Visualize'):
    # Predict treatment cost and doctor consultation fee
    top_hospitals = predict(user_disease, user_equipment)
    
    # Visualize the predictions
    hospital_names = [hospital_name for hospital_name, _, _ in top_hospitals]
    costs = [cost for _, cost, _ in top_hospitals]
    fees = [fee for _, _, fee in top_hospitals]

    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    ax[0].barh(hospital_names, costs, color='blue')
    ax[0].set_title('Top Hospitals for Treatment (Cost)')
    ax[0].set_xlabel('Treatment Cost ($)')
    ax[0].set_ylabel('Hospital Name')
    ax[1].barh(hospital_names, fees, color='green')
    ax[1].set_title('Top Hospitals for Doctor Consultation Fee')
    ax[1].set_xlabel('Doctor Consultation Fee ($)')
    ax[1].set_ylabel('Hospital Name')
    plt.tight_layout()
    st.pyplot(fig)
