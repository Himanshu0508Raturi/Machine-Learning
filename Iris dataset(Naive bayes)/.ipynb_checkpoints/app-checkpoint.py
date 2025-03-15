import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load model and dataset
model = joblib.load("Naive_bayes_model")
dataset = pd.read_excel("Book1.xlsx")

# Remove 'Id' column if it exists
x = dataset.iloc[:, :-1].drop(columns=["Id"], errors='ignore')
feature_names = x.columns  # Now contains all feature names except 'Id'

# Initialize MinMaxScaler and fit on the dataset (training data)
sc = MinMaxScaler()
sc.fit(x)  # Fit scaler on full dataset (excluding target)

st.title("Iris Flower Prediction")

# Collect user inputs
input_data = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}:", value=0.0, step=0.1)
    input_data.append(value)

# Predict button
if st.button("Predict"):
    # Convert input data into DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Transform using the pre-fitted scaler
    input_scaled = sc.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Display result
    if prediction[0] == 0:
        st.success("Iris-setosa")
    elif prediction[0] == 1:
        st.success("Iris-versicolor")
    else:
        st.success("Iris-virginica")
