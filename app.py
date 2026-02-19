
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Network Load Estimator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Dynamic Dark Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #00f260, #0575e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1.1rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Custom Input Styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 15px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4);
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.6);
    }

    /* Headers */
    h1, h2, h3 {
        color: white !important;
        font-family: 'Outfit', sans-serif;
    }
    
    h1 {
        text-shadow: 0 0 20px rgba(0, 198, 255, 0.5);
    }

</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
try:
    data = pd.read_csv("network_load.csv")
except FileNotFoundError:
    # Create dummy data if file doesn't exist to prevent crash
    data = pd.DataFrame({
        'Devices': np.random.randint(1, 100, 100),
        'Time': np.random.randint(0, 24, 100),
        'Load': np.random.uniform(0.1, 2.0, 100)
    })
    data.to_csv("network_load.csv", index=False)

X = data[['Devices', 'Time']]
y = data['Load']

lr_model = LinearRegression()
rf_model = RandomForestRegressor()
lr_model.fit(X, y)
rf_model.fit(X, y)

# ---------- SIDEBAR NAVIGATION ----------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208728.png", width=80)
    st.title("Network AI")
    st.markdown("---")
    menu = st.radio(
        "Navigation", 
        ["Home", "Prediction", "Analytics", "History"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    
    # Data Upload
    st.markdown("### 📂 Data Management")
    uploaded_file = st.file_uploader("Upload Network Data (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Data Loaded Successfully!")
            # Re-train models with new data
            X = data[['Devices', 'Time']]
            y = data['Load']
            lr_model.fit(X, y)
            rf_model.fit(X, y)
        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.success("🟢 Online")
    st.markdown(f"**Records:** {len(data)}")

# ---------- HOME PAGE ----------
if menu == "🏠 Home":
    st.title("⚡ AI Crowd-Aware Network Load Estimator")
    st.markdown("### Real-time Crowd Density & Network Traffic Analysis")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Custom Metrics Layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Data Points</div>
            <div class="metric-value">{len(data)}</div>
            <div>Dataset Size</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        avg_load = data['Load'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Load</div>
            <div class="metric-value">{avg_load:.2f} Mbps</div>
            <div>Network Usage</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        max_load = data['Load'].max()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Peak Load</div>
            <div class="metric-value">{max_load:.2f} Mbps</div>
            <div>Max Recorded</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("ℹ️ Navigate to the Prediction tab to estimate network load based on device count and time.")

# ---------- PREDICTION PAGE ----------
elif menu == "📊 Prediction":
    st.title("🚀 Load Prediction Engine")
    st.markdown("Enter the parameters below to predict network congestion.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📡 Input Parameters")
        devices = st.slider("Number of Connected Devices", 1, 100, 25)
        time = st.slider("Time of Day (Hour)", 0, 23, 12)
        model_choice = st.selectbox("Select AI Model", ["Random Forest (Recommended)", "Linear Regression"])

    with col2:
        st.markdown("### 🎯 Result")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Run Prediction", use_container_width=True):
            if model_choice == "Linear Regression":
                pred = lr_model.predict([[devices, time]])[0]
            else:
                pred = rf_model.predict([[devices, time]])[0]

            # Display Result with styling
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2); text-align: center;">
                <h2 style="margin:0; color: #a0a0a0;">Predicted Network Load</h2>
                <h1 style="font-size: 4rem; background: linear-gradient(to right, #4facfe, #00f2fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{pred:.2f} Mbps</h1>
            </div>
            """, unsafe_allow_html=True)

            # Logic for congestion levels
            st.markdown("<br>", unsafe_allow_html=True)
            if pred < 0.8:
                st.success("✅ **Low Congestion**: Network is optimal.")
            elif pred < 1.6:
                st.warning("⚠️ **Medium Congestion**: Moderate traffic detected.")
            else:
                st.error("🚨 **High Congestion**: Network overload imminent!")

            # Save to history
            hist = pd.DataFrame([[devices, time, pred]], columns=["Devices","Time","Prediction"])
            hist.to_csv("history.csv", mode='a', header=False, index=False)

# ---------- ANALYTICS PAGE ----------
elif menu == "📈 Analytics":
    st.title("📊 Network Analytics")
    st.markdown("Deep dive into network performance metrics.")
    
    # 3D visuals using Plotly
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📱 Devices vs Load")
        fig1 = px.scatter(data, x='Devices', y='Load', 
                          color='Load', size='Devices',
                          color_continuous_scale='Viridis',
                          template='plotly_dark')
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.markdown("### ⏰ Time vs Load")
        fig2 = px.line(data.groupby('Time')['Load'].mean().reset_index(), 
                       x='Time', y='Load', 
                       markers=True, line_shape='spline',
                       template='plotly_dark')
        fig2.update_traces(line_color='#00d2ff', line_width=4)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🌐 Load Distribution")
    fig3 = px.histogram(data, x='Load', nbins=20, 
                        color_discrete_sequence=['#ff00cc'],
                        template='plotly_dark')
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧬 Model Performance & Advanced Insights")

    col_metrics1, col_metrics2 = st.columns(2)

    # Calculate metrics
    y_pred_lr = lr_model.predict(X)
    y_pred_rf = rf_model.predict(X)

    with col_metrics1:
        st.markdown("#### Linear Regression Performance")
        st.info(f"MAE: {mean_absolute_error(y, y_pred_lr):.4f}")
        st.info(f"MSE: {mean_squared_error(y, y_pred_lr):.4f}")
        st.info(f"R² Score: {r2_score(y, y_pred_lr):.4f}")

    with col_metrics2:
        st.markdown("#### Random Forest Performance")
        st.success(f"MAE: {mean_absolute_error(y, y_pred_rf):.4f}")
        st.success(f"MSE: {mean_squared_error(y, y_pred_rf):.4f}")
        st.success(f"R² Score: {r2_score(y, y_pred_rf):.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.markdown("### 🔥 Correlation Heatmap")
        corr = data[['Devices', 'Time', 'Load']].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', template='plotly_dark')
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_viz2:
        st.markdown("### 📦 Load Variability (Box Plot)")
        fig_box = px.box(data, x='Time', y='Load', color='Time', template='plotly_dark')
        st.plotly_chart(fig_box, use_container_width=True)

# ---------- HISTORY PAGE ----------
elif menu == "🕘 History":
    st.title("📜 Prediction Log")
    
    try:
        h = pd.read_csv("history.csv", names=["Devices","Time","Prediction"])
        st.markdown("### Recent Predictions")
        st.dataframe(h.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        if st.download_button("Download History as CSV", h.to_csv(index=False), "prediction_history.csv", "text/csv"):
            st.success("Download Started!")

        if st.button("Clear History"):
            with open("history.csv", "w") as f:
                f.write("Devices,Time,Prediction\n")
            st.rerun()
            
    except Exception as e:
        st.warning("No history found or file is empty.")
