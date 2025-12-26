import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---- 1. App Configuration ----
st.set_page_config(page_title="📈 Walmart Demand Forecasting", layout="centered")
st.title("📊 Walmart Demand Forecasting")

# ---- 2. Load Model & Features ----
rf_model = joblib.load("Walmart.pkl")

# ---- 3. Load Product Data ----
# Only for mapping product_id to product_name
df_products = pd.read_csv("Walmart_clean.csv")[["product_id", "product_name", "unit_price"]].drop_duplicates()

# ---- 4. User Inputs ----
product_id = st.selectbox("Select Product ID", df_products["product_id"].unique())

# Fetch product name & default unit price
product_info = df_products[df_products["product_id"] == product_id].iloc[0]
product_name = product_info["product_name"]
unit_price = st.number_input("Unit Price", value=float(product_info["unit_price"]))

st.write(f"**Product Name:** {product_name}")

# ---- 5. Prepare Feature Input for Model ----
# Create a single-row dataframe with all features as 0
input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

# Set numerical features
if "unit_price" in input_df.columns:
    input_df["unit_price"] = unit_price

# If product_id is categorical, set corresponding one-hot column to 1
product_col = f"product_id_{product_id}"
if product_col in input_df.columns:
    input_df[product_col] = 1

# ---- 6. Forecast Demand ----
forecasted_demand = rf_model.predict(input_df)[0]
st.success(f"📈 Forecasted Demand: {round(forecasted_demand, 2)} units")

# ---- 7. Optional: Plot Unit Price vs Forecasted Demand ----
import matplotlib.pyplot as plt

# Create a small range of prices around current unit price
price_range = np.linspace(unit_price*0.8, unit_price*1.2, 10)
demand_range = []

for price in price_range:
    input_df["unit_price"] = price
    demand_range.append(rf_model.predict(input_df)[0])

fig, ax = plt.subplots()
ax.plot(price_range, demand_range, marker='o')
ax.set_xlabel("Unit Price")
ax.set_ylabel("Forecasted Demand")
ax.set_title(f"Forecasted Demand vs Unit Price for {product_name}")
st.pyplot(fig)
