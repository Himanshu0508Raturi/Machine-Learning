import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

model = joblib.load('naive_bayes_model.joblib')
st.subheader("Heart Failure Prediction")

# Load the training data
df = pd.read_json('training_data.json', orient='records', lines=True)

# Fit the StandardScaler using the training data
sc = StandardScaler()
#sc.fit(df[['age', 'gender', 'chestPain', 'bp', 'Cholesterol', 'bsugar', 'ecg', 'heart_rate', 'angina', 'oldPeak', 'SLOPE']])


# AGE
age = st.number_input("Enter your age", min_value=0, max_value=120)

# GENDER
if(st.selectbox("Select your gender", ["Male", "Female"]) == "Male"):
    gender = 1
else:
    gender = 0

# CHEST PAIN
x = st.selectbox("Select Chest Pain type", ["ASY","NAP","ATA"]) 
if(x == "ASY"):
    chestPain = 0
elif(x == 'NAP'):
    chestPain = 2
else:
    chestPain = 1

# BLOOD PRESSURE
bp = st.number_input("Select Resting BP: ", min_value=100, max_value=200)

#Cholesterol
Cholesterol = st.number_input("Select Cholesterol Level: ", min_value=0, max_value=600)

if(st.selectbox("Blood Sugar: ",['Yes','No']) == 'Yes'):
    bsugar = 1
else:
    bsugar = 0

# ECG
y  = st.selectbox("Enter resting electrocardiogram results: ",["Normal","LVH","ST"])
if(y == "NORMAL"):
    ecg = 1
elif(x == 'ST'):
    ecg = 2
else:
    ecg = 0

# HEART RATE
heart_rate = st.number_input("Enter Maximum heart rate: ",min_value = 60 , max_value = 202)

# ANGINA
if(st.selectbox("exercise induced angina: ", ["Y","N"]) == "Y"):
    angina = 1
else:
    angina = 0

# OLDPEAK
oldPeak = st.number_input("Enter oldPeak: ",min_value = -2.6 , max_value = 6.2)

#SLOPE
z = st.selectbox("ST_Slope: ", ["Up","Flat","Down"])
if(y == "Up"):
    SLOPE = 2
elif(x == 'Flat'):
    SLOPE = 1
else:
    SLOPE = 0

#user_input = [[age, gender, chestPain, bp, Cholesterol, bsugar, ecg, heart_rate, angina, oldPeak, SLOPE]]
#scaled_input = sc.transform(user_input)
#result = model.predict(scaled_input)
result =  model.predict(sc.fit_transform([[age, gender, chestPain, bp, Cholesterol, bsugar, ecg, heart_rate, angina, oldPeak, SLOPE]]))
if result == [0]:
    st.success('Person Not Having Heart Disease')
else:
    st.success("Person Having Heart Disease")


