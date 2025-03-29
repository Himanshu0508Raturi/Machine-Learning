import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model and dataset
model = joblib.load('Heart_disease_prediction/random_forest_model')
dataset = pd.read_csv('Heart_disease_prediction/heart.csv')
scaler = StandardScaler()

# Fit the scaler on the dataset (excluding the target column)
X = dataset.iloc[:, :-1]
scaler.fit(X)

# Extract feature names
feature_names = dataset.columns[:-1]

# Streamlit App
st.title("Heart Disease Prediction App")

# Show a preview of dataset values
st.subheader("Dataset Preview")
st.write("Below are some sample records from the dataset to help you enter valid values.")
st.dataframe(dataset.head(10))  # Show first 10 rows for reference

# Feature Descriptions
st.markdown("""
## 🔹 Feature Descriptions
Each feature in the dataset represents different patient health metrics:

1. **Age** - Age of the patient (in years)
2. **Sex**  
   - `0`: Female  
   - `1`: Male  
3. **Chest Pain Type (cp)**  
   - `0`: Typical Angina  
   - `1`: Atypical Angina  
   - `2`: Non-anginal Pain  
   - `3`: Asymptomatic  
4. **Resting Blood Pressure (trestbps)** - Blood pressure (mm Hg)
5. **Cholesterol (chol)** - Serum cholesterol (mg/dl)
6. **Fasting Blood Sugar (fbs > 120 mg/dl)**  
   - `0`: False  
   - `1`: True  
7. **Resting ECG Results (restecg)**  
   - `0`: Normal  
   - `1`: ST-T Wave Abnormality  
   - `2`: Left Ventricular Hypertrophy  
8. **Max Heart Rate (thalach)** - Maximum heart rate achieved
9. **Exercise-Induced Angina (exang)**  
   - `0`: No  
   - `1`: Yes  
10. **ST Depression (oldpeak)** - Depression induced by exercise
11. **Slope of ST Segment (slope)**  
   - `0`: Upsloping  
   - `1`: Flat  
   - `2`: Downsloping  
12. **Number of Major Vessels (ca)** (0–3)
13. **Thalassemia (thal)**  
   - `0`: Normal  
   - `1`: Fixed Defect  
   - `2`: Reversible Defect  
""")

st.write("### Select values for each feature based on the dataset")

# Input fields using select boxes with predefined values
input_data = []
for feature in feature_names:
    unique_values = sorted(dataset[feature].unique())  # Get unique values from dataset
    value = st.selectbox(f"Select {feature}:", unique_values, index=0)
    input_data.append(value)

# Predict button
if st.button("Predict"):
    # Convert inputs to a DataFrame to match feature names
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Apply StandardScaler to the input data
    input_scaled = scaler.transform(input_df)

    # Make prediction and get probability scores
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)  # Get probability scores

    # Get confidence percentage
    confidence = np.max(prediction_proba) * 100  # Convert to percentage

    # Display the result
    if prediction[0] == 1:
        st.success(f"The model predicts: **Heart Disease**.")
    else:
        st.success(f"The model predicts: **No Heart Disease**.")

    # Show confidence percentage
    st.info(f"🔹 **Confidence Level: {confidence:.2f}%**")

# Footer with user details
st.markdown("""
---
### 🛠️ Developed and Deployed by: **Himanshu Raturi**
🌍 GitHub: https://github.com/Himanshu0508Raturi/Machine-Learning.git
""")
