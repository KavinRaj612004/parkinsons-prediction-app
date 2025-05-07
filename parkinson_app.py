import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🧠 Parkinson's Disease Prediction App")

st.markdown("Enter the following voice measurements to predict Parkinson's Disease:")

def user_input():
    features = []
    labels = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
        'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
        'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
        'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]
    for label in labels:
        val = st.number_input(label, format="%.5f")
        features.append(val)
    return np.array(features).reshape(1, -1)

input_data = user_input()

if st.button("Predict"):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    if prediction[0] == 1:
        st.error("⚠️ The person **has Parkinson's Disease**.")
    else:
        st.success("✅ The person **does NOT have Parkinson's Disease**.")
