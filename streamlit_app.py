import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ---- Page Config ----
st.set_page_config(page_title="Walmart Demand Forecast", layout="centered")
st.title("📈 Walmart Demand Forecasting")

# ---- Load Model & Features ----
@st.cache_resource
def load_model():
    try:
        model = joblib.load("Walmart.pkl")
        features = joblib.load("Walmart_features.pkl")
        return model, features
    except FileNotFoundError:
        st.error("❌ 'Walmart.pkl' or 'Walmart_features.pkl' not found in app folder.")
        return None, None

rf_model, feature_columns = load_model()

if rf_model is not None:
    st.success("✅ Model loaded successfully!")

    # ---- Manual Input Section ----
    st.subheader("📝 Enter Feature Values for Forecast")

    # Numeric features
    unit_price = st.number_input("Unit Price", min_value=0.0, value=5.99, step=0.01)
    store_id = st.number_input("Store ID", min_value=1, value=1)
    department_id = st.number_input("Department ID", min_value=1, value=1)

    # Boolean feature
    is_holiday = st.selectbox("Is Holiday", options=[True, False], index=0)

    # Add more features here if your model has them
    # For example: "temperature", "promotion_flag", etc.

    if st.button("Predict Demand"):
        try:
            # Create a DataFrame with user inputs
            df_input = pd.DataFrame({
                "unit_price": [unit_price],
                "store_id": [store_id],
                "department_id": [department_id],
                "IsHoliday": [is_holiday]
                # Add more features here if needed
            })

            # Encode & align features with training model
            df_encoded = pd.get_dummies(df_input)
            df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

            # Predict demand
            prediction = rf_model.predict(df_encoded)[0]

            # Show result
            st.subheader("✅ Forecasted Demand")
            st.write(f"📊 Forecasted Demand: {int(prediction)} units")

            # Optional: Plot bar chart
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(["Forecasted Demand"], [prediction], color="skyblue")
            ax.set_ylabel("Units")
            ax.set_title("Forecasted Demand")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error in prediction: {e}")
