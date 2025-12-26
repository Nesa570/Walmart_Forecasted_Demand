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

# ---- Load Product Lookup ----
@st.cache_data
def load_products():
    try:
        df = pd.read_csv("Walmart_clean.csv")
        return df[["product_id", "product_name"]].drop_duplicates()
    except FileNotFoundError:
        st.warning("⚠️ 'Walmart_clean.csv' not found. Product lookup unavailable.")
        return pd.DataFrame(columns=["product_id", "product_name"])

product_lookup = load_products()

    # ---- Product ID Input ----
    st.subheader("🛒 Enter Product ID")
    product_id = st.number_input("Product ID", min_value=1, value=1, step=1)

    # Auto get product name
    product_name = product_lookup.loc[product_lookup["product_id"] == product_id, "product_name"]
    if not product_name.empty:
        product_name = product_name.values[0]
        st.write(f"**Product Name:** {product_name}")
    else:
        product_name = "Unknown"
        st.warning("❌ Product ID not found in lookup table.")

    # ---- Manual Feature Inputs ----
    st.subheader("📝 Enter Other Feature Values")

    unit_price = st.number_input("Unit Price", min_value=0.0, value=5.99, step=0.01)
    store_id = st.number_input("Store ID", min_value=1, value=1)
    department_id = st.number_input("Department ID", min_value=1, value=1)
    is_holiday = st.selectbox("Is Holiday", options=[True, False], index=0)

    # Add more features if your model requires

    if st.button("Predict Demand"):
        try:
            # Create DataFrame with all inputs
            df_input = pd.DataFrame({
                "unit_price": [unit_price],
                "store_id": [store_id],
                "department_id": [department_id],
                "IsHoliday": [is_holiday]
                # Add more features here
            })

            # Encode & align with training features
            df_encoded = pd.get_dummies(df_input)
            df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

            # Predict demand
            prediction = rf_model.predict(df_encoded)[0]

            # Show result
            st.subheader("✅ Forecasted Demand")
            st.write(f"**Product:** {product_name}")
            st.write(f"**Forecasted Demand:** {int(prediction)} units")

            # Optional: Plot bar chart
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar([product_name], [prediction], color="skyblue")
            ax.set_ylabel("Units")
            ax.set_title("Forecasted Demand")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error in prediction: {e}")
