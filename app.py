import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# ---------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# ---------------------------------------------
st.set_page_config(
    page_title="Premium House Price Analytics Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for an attractive, premium feel
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        color: #556B2F;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #3B82F6;
    }
    .result-card {
        background-color: #ECFDF5;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #10B981;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------
# 2. LOAD PRE-TRAINED GRADIENT BOOSTING MODEL
# ---------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model.pkl: {e}")
        return None

model = load_model()

# ---------------------------------------------
# 3. HEADER & HERO SECTION
# ---------------------------------------------
st.markdown('<div class="main-title">🏠 Real Estate Valuation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced predictive dashboard powered by Gradient Boosting Regressor algorithms.</div>', unsafe_allow_html=True)

# ---------------------------------------------
# 4. SIDEBAR INPUT CONTROLS (13 Features Mapping)
# ---------------------------------------------
st.sidebar.image("https://img.icons8.com/clouds/100/000000/home.png", width=100)
st.sidebar.title("Property Configuration")
st.sidebar.markdown("Adjust structural & neighborhood features below:")

# Structural Parameters
st.sidebar.subheader("🏗️ Structural Attributes")
rm = st.sidebar.slider("Average Rooms per Dwelling (RM)", 3.0, 9.0, 6.0, 0.1)
dis = st.sidebar.slider("Weighted Distance to Employment Centers (DIS)", 1.0, 12.0, 4.0, 0.1)
age = st.sidebar.slider("Proportion of Owner-Occupied Units Built Prior to 1940 (AGE)", 0.0, 100.0, 65.0, 1.0)
lstat = st.sidebar.slider("Lower Status Population Percentage (LSTAT)", 1.0, 40.0, 12.0, 0.5)

# Neighborhood & Location Parameters
st.sidebar.subheader("📍 Location & Community Metrics")
crim = st.sidebar.number_input("Per Capita Crime Rate (CRIM)", min_value=0.0, max_value=100.0, value=0.1, step=0.01)
zn = st.sidebar.number_input("Residential Land Zoned for Large Lots % (ZN)", min_value=0.0, max_value=100.0, value=11.0, step=1.0)
indus = st.sidebar.number_input("Non-Retail Business Acres % (INDUS)", min_value=0.0, max_value=30.0, value=11.0, step=0.1)
chas = st.sidebar.selectbox("Bounds Charles River? (CHAS)", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
nox = st.sidebar.slider("Nitric Oxides Concentration (NOX parts/10m)", 0.3, 0.9, 0.5, 0.01)
tax = st.sidebar.number_input("Full-Value Property-Tax Rate per $10,000 (TAX)", min_value=150, max_value=800, value=400, step=10)
ptratio = st.sidebar.slider("Pupil-Teacher Ratio by Town (PTRATIO)", 12.0, 23.0, 18.0, 0.1)
b = st.sidebar.number_input("Proportion of Minority Cohort Metric (B)", min_value=0.0, max_value=400.0, value=350.0, step=10.0)
rad = st.sidebar.slider("Accessibility Index to Radial Highways (RAD)", 1, 24, 4, 1)

# Arrange data for prediction exactly format expected by model.pkl
input_features = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])

# ---------------------------------------------
# 5. MAIN CONTENT - TABS LAYOUT
# ---------------------------------------------
tab1, tab2 = st.tabs(["🔮 Valuation Engine", "📊 Model Parameters & Insights"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Selected Property Snapshot")
        # Build DataFrame representation of properties selected
        df_display = pd.DataFrame(input_features, columns=[
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
        ])
        st.dataframe(df_display.style.highlight_max(axis=0, color='#EBF5FF'), use_container_width=True)
        
        # Trigger Prediction Business Logic
        st.markdown("### Generate Valuation")
        if st.button("✨ Estimate Market Value", type="primary", use_container_width=True):
            if model is not None:
                with st.spinner("Analyzing property metrics and running ensemble inference..."):
                    time.sleep(0.6) # Sleek UX flow delay
                    prediction = model.predict(input_features)[0]
                    
                    # Scaling logic case if output values are in $1000s
                    estimated_value = prediction * 1000 if prediction < 100 else prediction
                    
                st.markdown(f"""
                <div class="result-card">
                    <h3 style='margin:0; color:#065F46;'>Est. Fair Market Valuation Result</h3>
                    <h1 style='margin:10px 0; color:#047857;'>${estimated_value:,.2f}</h1>
                    <p style='margin:0; color:#065F46; font-size:14px;'>Inference successfully computed via GradientBoostingRegressor model pipeline.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Prediction failed. The machine learning model binary could not be found.")

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>💡 Executive Dashboard Advice</h4>
            <p style='font-size: 14px; line-height: 1.6; color:#374151;'>
                The structural variables <b>Average Rooms (RM)</b> and socio-economic feature <b>Lower Status Population (LSTAT)</b> 
                traditionally dominate model feature hierarchies in regression distributions. 
                <br><br>
                Try lowering <b>LSTAT</b> or maximizing <b>RM</b> configurations in the left side menu sidebar to inspect changes in pricing calculations instantly!
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("Model Meta-Information")
    if model is not None:
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.json({
                "Algorithm": "Gradient Boosting Regressor",
                "Loss Metric Specified": "Squared Error",
                "Total Trained Features Input": model.n_features_in_,
                "Base Tree Estimator Strategy": "Friedman MSE Criterion"
            })
        with col_meta2:
            st.info("💡 The backend architecture utilizes Scikit-Learn tree ensemble paths. Each iteration attempts to sequentially resolve residual error gradients left by prior steps.")
