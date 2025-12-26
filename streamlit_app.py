import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1️⃣ Page config
# -----------------------------
st.set_page_config(
    page_title="📈 Walmart Demand Predictor",
    layout="centered"
)

st.title("📊 Walmart Demand Forecast Predictor")
st.caption("Predict demand based on product, unit price, and holiday status")

# -----------------------------
# 2️⃣ Load tiny model & product mapping
# -----------------------------
@st.cache_data
def load_model():
    rf_model = joblib.load("Walmart_tiny.pkl")          # tiny model (<100 KB)
    product_lookup = joblib.load("Walmart_product.pkl") # product_id → product_name mapping
    return rf_model, product_lookup

rf_model, product_lookup = load_model()

# -----------------------------
# 3️⃣ User input
# -----------------------------
st.subheader("Enter Product Details")

product_id = st.selectbox("Select Product ID", list(product_lookup.keys()))
unit_price = st.number_input("Unit Price ($)", min_value=0.0, value=5.0, step=0.1)
is_holiday = st.radio("Is it a holiday?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

# -----------------------------
# 4️⃣ Prepare input
# -----------------------------
X_user = pd.DataFrame({
    "unit_price": [unit_price],
    "IsHoliday": [is_holiday]
})

# -----------------------------
# 5️⃣ Make prediction
# -----------------------------
if st.button("Predict Demand"):
    demand_pred = rf_model.predict(X_user)[0]
    st.success(f"Predicted Demand for **{product_lookup[product_id]}**: {demand_pred:.1f} units")

# -----------------------------
# 6️⃣ Optional: Show product name
# -----------------------------
st.write(f"✅ You selected: **{product_lookup[product_id]}**")
