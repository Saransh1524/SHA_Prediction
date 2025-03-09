import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained models
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("stacked_model.pkl", "rb") as f:
    stacked_model = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Streamlit App
def main():
    st.title("Silent Heart Attack Risk Prediction")
    st.write("Enter the required details below to predict the risk.")
    
    # User input fields
    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(f"{feature}", value=0.0, step=0.1)
    
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    if st.button("Predict"):
        # Get predictions from base models
        rf_pred = rf_model.predict_proba(input_df)[:, 1]
        xgb_pred = xgb_model.predict_proba(input_df)[:, 1]
        svm_pred = svm_model.predict_proba(input_df)[:, 1]
        
        # Stack predictions as input for meta-classifier
        stacked_input = np.column_stack((rf_pred, xgb_pred, svm_pred))
        
        # Get final prediction
        final_prediction = stacked_model.predict(stacked_input)[0]
        
        # Display result
        result = "High Risk" if final_prediction == 1 else "Low Risk"
        st.subheader(f"Prediction: {result}")

if __name__ == "__main__":
    main()