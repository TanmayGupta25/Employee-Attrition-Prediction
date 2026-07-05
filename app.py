
import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

# Load Metadata
with open('models/metadata.json', 'r') as f:
    metadata = json.load(f)

menu = ["Home", "About Dataset", "About Model", "Prediction", "Developer"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.title("Employee Attrition Prediction")
    st.header("Summer School 2026")
    st.subheader(f"Author: Tanmay Gupta")
    st.write(f"**Problem Type:** {metadata['Problem Type']}")
    st.write(f"**Best Model:** {metadata['Best Model']}")

elif choice == "About Dataset":
    st.title("About Dataset")
    st.write(f"**Dataset Name:** {metadata.get('Dataset Name', 'HR Employee Attrition')}")
    st.write(f"**Target Column:** {metadata['Target Column']}")
    st.write(f"**Problem Type:** {metadata['Problem Type']}")
    with open('models/feature_columns.pkl', 'rb') as f:
        features = joblib.load(f)
    st.write(f"**Number of Features:** {len(features)}")

elif choice == "About Model":
    st.title("About Model")
    st.write(f"**Best Model:** {metadata['Best Model']}")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{metadata['Accuracy']:.4f}")
    col1.metric("Precision", f"{metadata['Precision']:.4f}")
    col2.metric("Recall", f"{metadata['Recall']:.4f}")
    col2.metric("F1 Score", f"{metadata['F1 Score']:.4f}")
    st.write(f"**ROC AUC:** {metadata.get('ROC AUC', metadata.get('ROC-AUC')):.4f}")

elif choice == "Prediction":
    st.title("Prediction Form")
    try:
        model = joblib.load('models/best_model.pkl')
        features = joblib.load('models/feature_columns.pkl')
        
        input_data = {}
        col1, col2 = st.columns(2)
        
        # Simple logic to determine input types for demonstration purposes
        # In a real app, one would use the dtypes from the training dataframe
        for i, col in enumerate(features):
            target_col = col1 if i % 2 == 0 else col2
            # We check for typical categorical names or just provide a placeholder selectbox
            if col in ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']:
                input_data[col] = target_col.selectbox(f"{col}", ["Select Option", "Yes", "No"])
            else:
                input_data[col] = target_col.number_input(f"{col}", value=0)
        
        if st.button("Predict"):
            with st.spinner("Processing..."):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                
                if prediction == 1:
                    st.warning("Prediction: Employee Likely to Leave")
                else:
                    st.success("Prediction: Employee Likely to Stay")
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_df)[0][1]
                    st.write(f"Probability of Attrition: {proba:.2%}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

elif choice == "Developer":
    st.title("Developer Information")
    st.write("**Developer:** Tanmay Gupta")
    st.write("**Programme:** Summer School 2026")
