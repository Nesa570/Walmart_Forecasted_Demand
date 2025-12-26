import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ---- Page Config ----
st.set_page_config(page_title="Walmart Demand Forecast", layout="centered")

st.title("📈 Walmart Demand Forecasting App")

# ---- Load Model & Features ----
@st.cache_resource
def load_model():
    try:
        model = joblib.load("Walmart.pkl")
        return model, features
    except FileNotFoundError:
        st.error("❌ Model or feature files not found! Make sure 'Walmart.pkl' and 'Walmart_features.pkl' are in the app folder.")
        return None, None

rf_model, feature_columns = load_model()

if rf_model is not None:
    st.success("Model loaded successfully!")

    # ---- Manual Input Section ----
    st.subheader("📝 Enter Unit Prices for Forecast")

    input_prices = st.text_area(
        "Enter unit prices separated by commas (e.g., 5.99, 6.49, 7.25):",
        value="5.99, 6.49, 7.25"
    )

    if st.button("Predict Demand"):
        try:
            # Convert input to list of floats
            unit_price_list = [float(x.strip()) for x in input_prices.split(",")]

            # Create DataFrame
            df_input = pd.DataFrame({"unit_price": unit_price_list})

            # Encode & align with training features
            df_encoded = pd.get_dummies(df_input)
            df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

            # Predict demand
            predictions = rf_model.predict(df_encoded)
            df_input["Forecasted_Demand"] = predictions.astype(int)

            # Show results table
            st.subheader("✅ Forecast Results")
            st.dataframe(df_input)

            # Plot graph
            st.subheader("📊 Demand vs Unit Price")
            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(df_input["unit_price"], df_input["Forecasted_Demand"], marker="o")
            ax.set_xlabel("Unit Price")
            ax.set_ylabel("Forecasted Demand")
            ax.set_title("Forecasted Demand Based on Unit Price")
            ax.grid(True)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error in prediction: {e}")
