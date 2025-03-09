import os
import pickle
import streamlit as st
import numpy as np

st.title("Silent Heart Attack Prediction")

# Check if files exist before loading
MODEL_PATHS = {
    "rf": "rf_model.pkl",
    "xgb": "xgb_model.pkl",
    "svm": "svm_model.pkl",
    "stacked": "stacked_model.pkl"
}

for model, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        st.error(f"❌ Model file missing: {path}")
    else:
        st.success(f"✅ Found model file: {path}")

# Load models with error handling
def load_model(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Error loading {file_path}: {e}")
        return None

rf_model = load_model(MODEL_PATHS["rf"])
xgb_model = load_model(MODEL_PATHS["xgb"])
svm_model = load_model(MODEL_PATHS["svm"])
stacked_model = load_model(MODEL_PATHS["stacked"])

# Stop execution if any model failed to load
if None in [rf_model, xgb_model, svm_model, stacked_model]:
    st.stop()

st.success("✅ All models loaded successfully!")

# Test prediction button
if st.button("Test Models"):
    try:
        test_input = np.array([[120, 80, 200, 110, 1, 0]])
        rf_pred = rf_model.predict(test_input)
        xgb_pred = xgb_model.predict(test_input)
        svm_pred = svm_model.predict(test_input)
        st.write(f"RF: {rf_pred}, XGB: {xgb_pred}, SVM: {svm_pred}")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
