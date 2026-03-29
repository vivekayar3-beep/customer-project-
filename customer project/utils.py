import pandas as pd
import streamlit as st
import os

REQUIRED_COLUMNS = [
    'customer_id', 'name', 'gender', 'age', 'email', 'city',
    'membership_tier', 'annual_income', 'purchase_amount', 'quantity',
    'discount_percentage', 'total_spend', 'payment_method',
    'product_category', 'purchase_frequency', 'days_since_last_purchase',
    'customer_lifetime_value', 'purchase_date'
]

@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data.csv")

    if not os.path.exists(file_path):
        st.error(f"❌ Data file not found at: {file_path}")
        st.stop()

    df = pd.read_csv(file_path)

    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Missing columns: {missing}. Some features may not work.")

    # Type enforcement
    int_cols = ['age', 'annual_income', 'purchase_amount', 'quantity',
                'discount_percentage', 'total_spend', 'purchase_frequency',
                'days_since_last_purchase', 'customer_lifetime_value']
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    if 'purchase_date' in df.columns:
        df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')

    # Null check warning
    null_counts = df[[c for c in REQUIRED_COLUMNS if c in df.columns]].isnull().sum()
    critical_nulls = null_counts[null_counts > 0]
    if not critical_nulls.empty:
        st.warning(f"⚠️ Null values detected: {critical_nulls.to_dict()}")

    return df
