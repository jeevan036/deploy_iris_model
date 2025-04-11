# app.py
import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_iris

#deploy

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

iris = load_iris()

st.title("🌸 Iris Flower Predictor")
st.write("Enter the measurements below:")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    prediction = model.predict(features)
    species = iris.target_names[prediction[0]]
    st.success(f"🌼 Predicted Species: {species}")
