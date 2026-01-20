import streamlit as st
import pandas as pd
import joblib
import os

# Path handling for deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'wine_cultivar_model.pkl')

# Load model
model = joblib.load(model_path)

st.set_page_config(page_title="Wine Origin Predictor", page_icon="🍷")
st.title("🍷 Wine Cultivar Origin Prediction")

st.write("Enter the chemical properties below to predict the wine origin.")

# Inputs for the 6 features
col1, col2 = st.columns(2)

with col1:
    alcohol = st.number_input("Alcohol", value=13.0)
    malic_acid = st.number_input("Malic Acid", value=2.3)
    ash = st.number_input("Ash", value=2.3)

with col2:
    magnesium = st.number_input("Magnesium", value=100.0)
    flavanoids = st.number_input("Flavanoids", value=2.0)
    color_intensity = st.number_input("Color Intensity", value=5.0)

if st.button("Predict Cultivar"):
    input_data = pd.DataFrame([[alcohol, malic_acid, ash, magnesium, flavanoids, color_intensity]], 
                             columns=['alcohol', 'malic_acid', 'ash', 'magnesium', 'flavanoids', 'color_intensity'])
    
    prediction = model.predict(input_data)[0]
    
    # Map the numeric prediction to a readable name
    cultivars = {0: "Cultivar 1", 1: "Cultivar 2", 2: "Cultivar 3"}
    result = cultivars[prediction]
    
    st.success(f"### The predicted origin is: {result}")