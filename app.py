
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Network Load Estimator", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #e3f2fd, #ffffff);
}
.navbar {
    font-size:18px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
data = pd.read_csv("network_load.csv")
X = data[['Devices', 'Time']]
y = data['Load']

lr_model = LinearRegression()
rf_model = RandomForestRegressor()
lr_model.fit(X, y)
rf_model.fit(X, y)

# ---------- NAVIGATION ----------
menu = st.radio("Navigation", ["🏠 Home", "📊 Prediction", "📈 Analytics", "🕘 History"], horizontal=True)

# ---------- HOME ----------
if menu == "🏠 Home":
    st.title("AI Crowd-Aware Network Load Estimator")
    st.subheader("Professional Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(data))
    col2.metric("Avg Load", round(data['Load'].mean(),2))
    col3.metric("Max Load", round(data['Load'].max(),2))

    st.info("Use navigation above to explore features")

# ---------- PREDICTION ----------
elif menu == "📊 Prediction":
    st.title("Prediction Dashboard")

    col1, col2 = st.columns(2)
    devices = col1.slider("Devices", 1, 100)
    time = col2.slider("Time", 0, 23)

    model = st.selectbox("Model", ["Linear Regression", "Random Forest"])

    if st.button("Predict Load"):
        if model == "Linear Regression":
            pred = lr_model.predict([[devices, time]])[0]
        else:
            pred = rf_model.predict([[devices, time]])[0]

        st.success(f"Predicted Load: {pred:.2f}")

        if pred < 0.8:
            st.success("Low Congestion")
        elif pred < 1.6:
            st.warning("Medium Congestion")
        else:
            st.error("High Congestion")

        hist = pd.DataFrame([[devices, time, pred]], columns=["Devices","Time","Prediction"])
        hist.to_csv("history.csv", mode='a', header=False, index=False)

# ---------- ANALYTICS ----------
elif menu == "📈 Analytics":
    st.title("Analytics Dashboard")

    col1, col2 = st.columns(2)

    fig1, ax1 = plt.subplots()
    ax1.scatter(data['Devices'], data['Load'])
    ax1.set_title("Devices vs Load")
    col1.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(data['Time'], data['Load'])
    ax2.set_title("Time vs Load")
    col2.pyplot(fig2)

# ---------- HISTORY ----------
elif menu == "🕘 History":
    st.title("Prediction History")

    try:
        h = pd.read_csv("history.csv", names=["Devices","Time","Prediction"])
        st.dataframe(h, use_container_width=True)
    except:
        st.warning("No history available")
