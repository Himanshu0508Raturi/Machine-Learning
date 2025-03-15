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

st.set_page_config(page_title="Iris Flower Prediction", layout="centered")
st.title("Iris Flower Prediction")

# ---------------------- FIXED DATASET SAMPLE ----------------------
# Store the dataset sample in session state if not already stored
if "fixed_sample" not in st.session_state:
    species_column = dataset.columns[-1]  # Assuming last column is species
    if dataset[species_column].nunique() >= 3:
        st.session_state.fixed_sample = (
            dataset.groupby(species_column)
            .apply(lambda x: x.sample(min(3, len(x)), random_state=42))  # Fixed random state
            .reset_index(drop=True)
        )
    else:
        st.session_state.fixed_sample = dataset.sample(min(10, len(dataset)), random_state=42)

# Show fixed dataset sample
st.subheader("Sample Data (For Reference)")
st.dataframe(st.session_state.fixed_sample)  # Display stored fixed sample

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
st.markdown("Trained and Deployed by - Himanshu Raturi.")
st.markdown("LinkedIn: https://www.linkedin.com/in/himanshu-raturi-99ab0728b/")
st.markdown("Github: https://github.com/Himanshu0508Raturi/Machine-Learning")
