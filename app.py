import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Model and Encoders
model = load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

# Streamlit UI
st.title("CUSTOMER CHURN PREDICTION")

geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
age = st.slider('🎂 Age', 18, 90)
balance = st.number_input('💰 Enter Balance')
credit_score = st.number_input("💳 Enter Credit Score")
estimated_salary = st.number_input('💵 Enter Estimated Salary')
tenure = st.slider('📆 Tenure', 0, 10)
num_of_products = st.slider("📦 Number of Products", 1, 4)
has_crcard = st.selectbox("💳 Has Credit Card", [0, 1])
is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

# Add Predict Button
if st.button("Predict"):
    # Encode Categorical Features
    gender_encoded = label_encoder_gender.transform([gender])[0]
    geography_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geo.get_feature_names_out())

    # Create Input DataFrame
    input_data = pd.DataFrame({
       'CreditScore': [credit_score],
       'Gender': [gender_encoded],
       'Age': [age],
       'Tenure': [tenure],
       'Balance': [balance], 
       'NumOfProducts': [num_of_products],
       'HasCrCard': [has_crcard],
       'IsActiveMember': [is_active_member],
       'EstimatedSalary': [estimated_salary]
    })

    # Combine Encoded Data
    input_data = pd.concat([input_data, encoded_df], axis=1)

    # Ensure Correct Column Order
    input_data = input_data[scaler.feature_names_in_]

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Make Prediction
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]

    # Display Result
    st.subheader("🔮 Prediction Result")
    st.write(f'**Churn Probability:** {prediction_proba:.2F}')
    if prediction_proba > 0.5:
        st.write("🔴 **The customer is likely to churn.**")
    else:
        st.write("🟢 **The customer is likely to stay.**")


