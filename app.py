import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data
from model import train_model, load_model, predict_single

st.set_page_config(page_title="Customer Dashboard", layout="wide")
st.title("🚀 Advanced Customer Analytics Dashboard")

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

city_coords = {
    "Mumbai": [19.0760, 72.8777], "Delhi": [28.7041, 77.1025],
    "Bangalore": [12.9716, 77.5946], "Chennai": [13.0827, 80.2707],
    "Ahmedabad": [23.0225, 72.5714], "Hyderabad": [17.3850, 78.4867],
    "Pune": [18.5204, 73.8567], "Kolkata": [22.5726, 88.3639],
    "Jaipur": [26.9124, 75.7873], "Surat": [21.1702, 72.8311],
    "Lucknow": [26.8467, 80.9462], "Nagpur": [21.1458, 79.0882],
    "Indore": [22.7196, 75.8577], "Bhopal": [23.2599, 77.4126],
    "Visakhapatnam": [17.6868, 83.2185], "Patna": [25.5941, 85.1376],
    "Vadodara": [22.3072, 73.1812], "Chandigarh": [30.7333, 76.7794],
    "Rajkot": [22.3039, 70.8022], "Kochi": [9.9312, 76.2673],
}

df['lat'] = df['city'].map(lambda x: city_coords.get(x, [None, None])[0])
df['lon'] = df['city'].map(lambda x: city_coords.get(x, [None, None])[1])

st.sidebar.header("🔍 Filters")
category   = st.sidebar.multiselect("Category",       df['product_category'].unique(), default=df['product_category'].unique())
gender     = st.sidebar.multiselect("Gender",          df['gender'].unique(),           default=df['gender'].unique())
city       = st.sidebar.multiselect("City",            df['city'].unique(),             default=df['city'].unique())
membership = st.sidebar.multiselect("Membership Tier", df['membership_tier'].unique(),  default=df['membership_tier'].unique())
payment    = st.sidebar.multiselect("Payment Method",  df['payment_method'].unique(),   default=df['payment_method'].unique())
age_range  = st.sidebar.slider("Age Range", int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))

date_range = None
if 'purchase_date' in df.columns and df['purchase_date'].notna().any():
    min_d = df['purchase_date'].min().date()
    max_d = df['purchase_date'].max().date()
    date_range = st.sidebar.date_input("Purchase Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

filtered_df = df[
    df['product_category'].isin(category) & df['gender'].isin(gender) &
    df['city'].isin(city) & df['membership_tier'].isin(membership) &
    df['payment_method'].isin(payment) & df['age'].between(age_range[0], age_range[1])
]
if date_range and len(date_range) == 2 and 'purchase_date' in df.columns:
    filtered_df = filtered_df[
        filtered_df['purchase_date'].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    ]

st.write(f"📊 Showing **{len(filtered_df):,}** of {len(df):,} records")

if filtered_df.empty:
    st.warning("🔍 No records match your current filters. Try adjusting the sidebar selections.")
    st.stop()

# KPI CARDS
col1, col2, col3, col4 = st.columns(4)
total_rev  = filtered_df['total_spend'].sum()
full_rev   = df['total_spend'].sum()
avg_spend  = filtered_df['total_spend'].mean()
full_avg   = df['total_spend'].mean()
avg_clv    = filtered_df['customer_lifetime_value'].mean()
full_clv   = df['customer_lifetime_value'].mean()
n_cust     = filtered_df['customer_id'].nunique()
full_cust  = df['customer_id'].nunique()

col1.metric("💰 Revenue",          f"₹{total_rev:,.0f}",  f"{((total_rev-full_rev)/full_rev*100):+.1f}% vs all")
col2.metric("👥 Customers",        f"{n_cust:,}",          f"{n_cust-full_cust:+,} vs all")
col3.metric("📈 Avg Spend",        f"₹{avg_spend:,.0f}",  f"₹{avg_spend-full_avg:+,.0f} vs avg")
col4.metric("🏅 Avg Lifetime Value",f"₹{avg_clv:,.0f}",  f"₹{avg_clv-full_clv:+,.0f} vs avg")

# MAP
st.subheader("🗺️ Customer Map")
map_df = filtered_df.dropna(subset=['lat', 'lon'])
if not map_df.empty:
    city_summary = map_df.groupby(['city', 'lat', 'lon'])['total_spend'].sum().reset_index()
    fig_map = px.scatter_mapbox(city_summary, lat="lat", lon="lon", size="total_spend",
                                color="total_spend", hover_name="city", zoom=4,
                                color_continuous_scale="Viridis", size_max=40)
    fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True, key="map_chart")
else:
    st.info("No location data for current filter.")

# DATA TABLE
st.subheader("📄 Data")
st.dataframe(filtered_df, use_container_width=True)
st.download_button("📥 Download Filtered Data", filtered_df.to_csv(index=False), "filtered_data.csv", mime="text/csv")

# TABS
tabs = st.tabs(["📊 Basic", "📈 Advanced", "📉 Distribution", "🧠 Insights", "📅 Time Series"])

with tabs[0]:
    cat_data = filtered_df.groupby('product_category')['total_spend'].sum().reset_index()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(cat_data, x='product_category', y='total_spend', title="Revenue by Category"), use_container_width=True, key="bar_chart")
        st.plotly_chart(px.area(cat_data, x='product_category', y='total_spend', title="Area Chart"), use_container_width=True, key="area_chart")
    with c2:
        st.plotly_chart(px.pie(cat_data, names='product_category', values='total_spend', hole=0.4, title="Revenue Share"), use_container_width=True, key="donut_chart")
        st.plotly_chart(px.line(cat_data, x='product_category', y='total_spend', title="Line Chart", markers=True), use_container_width=True, key="line_chart")

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.scatter(filtered_df, x='age', y='total_spend', color='gender', opacity=0.6, title="Age vs Spend"), use_container_width=True, key="scatter_chart")
        st.plotly_chart(px.treemap(filtered_df, path=['product_category', 'city'], values='total_spend', title="Revenue Treemap"), use_container_width=True, key="treemap_chart")
    with c2:
        st.plotly_chart(px.scatter(filtered_df, x='age', y='total_spend', size='total_spend', color='product_category', opacity=0.7, title="Bubble Chart"), use_container_width=True, key="bubble_chart")
        radar = filtered_df.groupby('product_category')['total_spend'].mean().reset_index()
        st.plotly_chart(px.line_polar(radar, r='total_spend', theta='product_category', line_close=True, title="Avg Spend Radar"), use_container_width=True, key="radar_chart")

with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(filtered_df, x='total_spend', nbins=40, title="Spend Distribution"), use_container_width=True, key="hist_chart")
        pivot = filtered_df.pivot_table(values='total_spend', index='city', columns='product_category', aggfunc='sum').fillna(0)
        st.plotly_chart(px.imshow(pivot, title="Heatmap: City x Category", color_continuous_scale="Blues"), use_container_width=True, key="heatmap_chart")
    with c2:
        st.plotly_chart(px.box(filtered_df, x='gender', y='total_spend', color='membership_tier', title="Box Plot by Gender & Tier"), use_container_width=True, key="box_chart")
        st.plotly_chart(px.violin(filtered_df, x='product_category', y='total_spend', color='gender', box=True, title="Violin Plot"), use_container_width=True, key="violin_chart")

with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        stacked = filtered_df.groupby(['product_category', 'gender'])['total_spend'].sum().reset_index()
        st.plotly_chart(px.bar(stacked, x='product_category', y='total_spend', color='gender', barmode='stack', title="Stacked Revenue by Category & Gender"), use_container_width=True, key="stacked_chart")
        # Real waterfall
        wf = filtered_df.groupby('product_category')['total_spend'].sum().reset_index().sort_values('total_spend', ascending=False)
        fig_wf = go.Figure(go.Waterfall(orientation="v", x=wf['product_category'], y=wf['total_spend'], connector={"line":{"color":"rgb(63,63,63)"}}))
        fig_wf.update_layout(title="Waterfall — Revenue by Category")
        st.plotly_chart(fig_wf, use_container_width=True, key="waterfall_chart")
    with c2:
        mem = filtered_df.groupby('membership_tier')['total_spend'].mean().reset_index()
        st.plotly_chart(px.bar(mem, x='membership_tier', y='total_spend', color='membership_tier', title="Avg Spend by Membership"), use_container_width=True, key="mem_chart")
        pay = filtered_df.groupby('payment_method')['total_spend'].sum().reset_index()
        st.plotly_chart(px.pie(pay, names='payment_method', values='total_spend', title="Revenue by Payment Method"), use_container_width=True, key="pay_pie")

with tabs[4]:
    if 'purchase_date' in filtered_df.columns and filtered_df['purchase_date'].notna().any():
        ts = filtered_df.copy()
        ts['month'] = ts['purchase_date'].dt.to_period('M').astype(str)
        monthly = ts.groupby('month').agg(revenue=('total_spend','sum'), orders=('customer_id','count')).reset_index()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.line(monthly, x='month', y='revenue', title="Monthly Revenue", markers=True), use_container_width=True, key="monthly_rev")
        with c2:
            st.plotly_chart(px.bar(monthly, x='month', y='orders', title="Monthly Orders"), use_container_width=True, key="monthly_orders")
        cat_monthly = ts.groupby(['month','product_category'])['total_spend'].sum().reset_index()
        st.plotly_chart(px.line(cat_monthly, x='month', y='total_spend', color='product_category', title="Monthly Revenue by Category"), use_container_width=True, key="cat_monthly")
    else:
        st.info("No purchase_date column found.")

# TOP CUSTOMERS
st.subheader("🏆 Top 10 Customers by Spend")
cols_show = ['customer_id','name','city','membership_tier','product_category','total_spend','customer_lifetime_value']
st.dataframe(filtered_df.sort_values('total_spend', ascending=False).head(10)[cols_show], use_container_width=True)

# ML TRAIN
st.subheader("🤖 Machine Learning — Customer Value Classifier")
existing_model, _ = load_model()
if existing_model:
    st.info("💾 Saved model available. Click 'Train Model' to retrain.")

if st.button("🔄 Train Model"):
    with st.spinner("Training Random Forest..."):
        try:
            model, acc, cv_scores, cm, feat_imp = train_model()
            st.session_state.update({'model_trained': True, 'model_acc': acc,
                                     'cv_scores': cv_scores, 'cm': cm, 'feat_imp': feat_imp})
            st.success(f"✅ Accuracy: **{acc:.2%}** | CV mean: **{cv_scores.mean():.2%}** ± {cv_scores.std():.2%}")
        except Exception as e:
            st.error(f"Training failed: {e}")

if st.session_state.get('model_trained'):
    feat_imp = st.session_state['feat_imp']
    cm       = st.session_state['cm']
    c1, c2   = st.columns(2)
    with c1:
        fig_fi = px.bar(x=feat_imp.values, y=feat_imp.index, orientation='h',
                        title="Feature Importance", labels={'x':'Importance','y':'Feature'})
        fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_fi, use_container_width=True, key="feat_imp_chart")
    with c2:
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           x=['Normal','High Value'], y=['Normal','High Value'],
                           labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True, key="conf_matrix")

# PREDICT
st.subheader("🔮 Predict Customer Value")
p1, p2, p3 = st.columns(3)
with p1:
    pred_age   = st.number_input("Age", min_value=18, max_value=70, value=30)
with p2:
    pred_spend = st.number_input("Total Spend", min_value=0, max_value=100000, value=15000, step=500)
with p3:
    st.write(""); st.write("")
    predict_btn = st.button("🔮 Predict")

if predict_btn:
    result = predict_single(pred_age, pred_spend)
    if result is None:
        threshold = df['total_spend'].quantile(0.75)
        if pred_spend >= threshold:
            st.success(f"💰 High Value Customer (spend ≥ ₹{threshold:,.0f})")
        else:
            st.warning(f"👤 Normal Customer (below ₹{threshold:,.0f} — train model for probability)")
    else:
        pred, prob, threshold = result
        if pred == 1:
            st.success(f"💰 **High Value Customer** — Model confidence: {prob:.1%}")
        else:
            st.warning(f"👤 **Normal Customer** — High-value probability: {prob:.1%}")
        st.progress(float(prob), text=f"High-value probability: {prob:.1%}")
