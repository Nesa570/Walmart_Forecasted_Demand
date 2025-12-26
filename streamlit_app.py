import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor

# ---- Page Config ----
st.set_page_config(page_title="Walmart Demand Forecast", layout="centered")
st.title("📈 Walmart Demand Forecasting on Products")

# ---- Load Model & Features ----
@st.cache_resource
def load_model():
    try:
        model = joblib.load("Walmart.pkl")
        features = joblib.load("Walmart_features.pkl")
        return model, features, True
    except FileNotFoundError:
        st.warning("⚠️ 'Walmart.pkl' or 'Walmart_features.pkl' not found. Using demo model.")
# Create a dummy model
        dummy_model = RandomForestRegressor()
        df_demo = pd.DataFrame({
            "unit_price": [5, 6, 7, 8],
            "store_id": [1, 1, 1, 1],
            "department_id": [1, 1, 1, 1],
            "IsHoliday": [True, False, True, False]
        })
        y_demo = np.array([100, 80, 90, 70])
        dummy_model.fit(df_demo, y_demo)
        return dummy_model, df_demo.columns, False
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        
# Return dummy model so app continues
        dummy_model = RandomForestRegressor()
        df_demo = pd.DataFrame({
            "unit_price": [5],
            "store_id": [1],
            "department_id": [1],
            "IsHoliday": [True]
        })
        dummy_model.fit(df_demo, [50])
        return dummy_model, df_demo.columns, False

# ---- Load model safely ----
rf_model, feature_columns, is_real_model = load_model()

# ---- Load Product Lookup ----
@st.cache_data
def load_products():
    try:
        df = pd.read_csv("Walmart_clean.csv")  # Must contain product_id, product_name, category
        return df[["product_id", "product_name", "category"]].drop_duplicates()
    except Exception:
        st.warning("⚠️ Product lookup CSV not found. Using empty lookup.")
        return pd.DataFrame(columns=["product_id", "product_name", "category"])

product_lookup = load_products()

# ---- Product ID Input ----
st.subheader("🛒 Enter Product ID")
product_id = st.number_input("Product ID", min_value=1, value=1, step=1)

# Auto-fill product name and category
product_info = product_lookup[product_lookup["product_id"] == product_id]
if not product_info.empty:
    product_name = product_info["product_name"].values[0]
    category = product_info["category"].values[0]
    st.write(f"**Product Name:** {product_name}")
    st.write(f"**Category:** {category}")
else:
    product_name = "Unknown"
    category = "Unknown"
    st.warning("❌ Product ID not found in lookup table.")

# ---- Manual Feature Inputs ----
st.subheader("📝 Enter Feature Values")
unit_price = st.number_input("Unit Price", min_value=0.0, value=5.99, step=0.01)
store_id = st.number_input("Store ID", min_value=1, value=1)
department_id = st.number_input("Department ID", min_value=1, value=1)
is_holiday = st.selectbox("Is Holiday", options=[True, False], index=0)

# ---- Predict Button ----
if st.button("Predict Demand"):
    try:
        df_input = pd.DataFrame({
            "unit_price": [unit_price],
            "store_id": [store_id],
            "department_id": [department_id],
            "IsHoliday": [is_holiday]
        })

        df_encoded = pd.get_dummies(df_input)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        prediction = rf_model.predict(df_encoded)[0]

        st.subheader("✅ Forecasted Demand")
        st.write(f"**Product:** {product_name}")
        st.write(f"**Category:** {category}")
        st.write(f"**Forecasted Demand:** {int(prediction)} units")

        # Plot
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar([product_name], [prediction], color="skyblue")
        ax.set_ylabel("Units")
        ax.set_title("Forecasted Demand")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")
