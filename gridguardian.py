import streamlit as st
import requests
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime

# --------- SETTINGS ---------
LAT = 21.34821000 # Shirppur ,Maharashtra 
LON = 74.88035000
CAPACITY_THRESHOLD = 1000  # Simulated grid max capacity (MW)

# --------- API CALL ---------
@st.cache_data
def get_weather_forecast():
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
        f"&daily=temperature_2m_max,precipitation_sum&timezone=auto"
    )
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame({
        'date': data['daily']['time'],
        'temp_max': data['daily']['temperature_2m_max'],
        'precip': data['daily']['precipitation_sum']
    })
    return df

# --------- TRAINING A SIMPLE ML MODEL ---------
def train_ml_model(df):
    # Using temperature and precipitation to predict demand
    X = df[['temp_max', 'precip']]
    y = df['demand']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict demand for the entire dataset
    df['predicted_demand'] = model.predict(X)
    return df, model

# --------- SIMULATE GRID DEMAND ---------
def simulate_grid_demand(df):
    # Initially simulate demand (this will be replaced by ML model prediction later)
    df['demand'] = 700 + (df['temp_max'] - 25) * 20 + df['precip'] * 10
    return df

# --------- PLOT ---------
def plot_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['predicted_demand'], name='Predicted Demand (MW)', line=dict(color='firebrick')))
    fig.add_trace(go.Scatter(x=df['date'], y=[CAPACITY_THRESHOLD]*len(df), name='Capacity Limit', line=dict(dash='dash')))
    return fig

# --------- UI ---------
st.set_page_config(page_title="GridGuardian", layout="centered")
st.title("⚡ GridGuardian: Blackout Risk Forecast")
st.markdown("This tool predicts blackout risk based on weather-impacted grid demand.")

# Get the weather forecast data
df = get_weather_forecast()
df = simulate_grid_demand(df)

# Train the ML model on historical data
df, model = train_ml_model(df)

# Risk Gauge
df['risk'] = df['predicted_demand'].apply(lambda x: 'Low' if x < 800 else 'Medium' if x < 950 else 'High')
risk_today = df.iloc[0]['risk']
color = {'Low': '🟢', 'Medium': '🟠', 'High': '🔴'}[risk_today]
st.subheader(f"Today's Blackout Risk: {color} **{risk_today}**")

# Show chart
st.plotly_chart(plot_chart(df))

# Show raw data
with st.expander("📊 Show Forecast Data"):
    st.dataframe(df)
