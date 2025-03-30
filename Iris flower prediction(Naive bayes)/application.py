#1 importing libraries
import uvicorn
from fastapi import FastAPI
from Irispara import Irispara
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#2 create the app object
application = FastAPI()
model = joblib.load("Naive_bayes_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")

#3 Index route , open on port 8000
@application.get('/')
def index():
    return {'message': 'Hello , World' }

#4 Route with a single parameter , return the parameter within a message
# located at: https://127.0.1:8000/AnynameHere
@application.get('/{name}')
def get_name(name: str):
    return {'Welcome This is my first FastAPI deployment ': f'{name}'}

#4 prediction functionality
# make a prediction from the passed JSON data and return the predicted species
@application.post('/predict')
def predict_species(data:Irispara):
    # Convert input to NumPy array
    input_data = np.array([[data.SepalLengthCm, data.SepalWidthCm, data.PetalLengthCm, data.PetalWidthCm]])

     # Apply MinMax Scaling (Using the same scaler from training)
    scaled_data = scaler.transform(input_data)

     # Make prediction
    prediction = model.predict(scaled_data)

    # Convert numerical prediction to species name
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    predicted_species = species_map.get(prediction[0], "Unknown")
    return {"Predicted Species": predicted_species}

#6 Run the API
if __name__ == '__main__':
    uvicorn.run(application,host = '127.0.0.1' , port = 8000)

#uvicorn filename:objectname --reload
# E.g: uvicorn application:application --reload