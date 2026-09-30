import json
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Weather Predictor", page_icon="🌤️", layout="wide")

base_cols = ["temp_max", "temp_min", "temp_mean", "rain", "wind_max", "humidity", "pressure"]
lag_cols = ["temp_mean", "temp_max", "temp_min", "rain", "humidity", "pressure", "wind_max"]

CITIES = {
    "Karachi": (24.86, 67.01),
    "Lahore": (31.55, 74.34),
    "Islamabad": (33.68, 73.05),
}

# ---------- loading ----------
@st.cache_resource
def load_models():
    temp_model = joblib.load("temp_model.pkl")
    rain_model = joblib.load("rain_model.pkl")
    columns = json.load(open("feature_columns.json"))
    return temp_model, rain_model, columns

@st.cache_data(ttl=3600)
def fetch_recent(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "past_days": 14, "forecast_days": 1,
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "precipitation_sum", "windspeed_10m_max",
            "relative_humidity_2m_mean", "surface_pressure_mean",
        ],
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["daily"])
    d["time"] = pd.to_datetime(d["time"])
    d = d.set_index("time")
    d.columns = base_cols
    return d

@st.cache_data
def load_history():
    h = pd.read_csv("weather_raw.csv")
    h["time"] = pd.to_datetime(h["time"])
    h = h.set_index("time")
    h.columns = base_cols
    return h

def make_features(raw):
    d = raw[base_cols].copy()
    for lag in [1, 2, 3]:
        for col in lag_cols:
            d[f"{col}_lag{lag}"] = d[col].shift(lag)
    d["temp_roll7"] = d["temp_mean"].rolling(7).mean()
    d["pressure_change"] = d["pressure"].diff()
    doy = d.index.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    return d

temp_model, rain_model, columns = load_models()
history = load_history()

# ---------- sidebar controls ----------
st.sidebar.header("Settings")
city = st.sidebar.selectbox("City", list(CITIES))
use_f = st.sidebar.toggle("Show Fahrenheit (°F)")
threshold = st.sidebar.slider("Rain alert threshold", 0.1, 0.9, 0.4, 0.05,
                              help="Show a rain warning when the rain score is above this.")

if city != "Karachi":
    st.sidebar.warning("The models were trained on Karachi only, so predictions for other cities are not reliable.")

def to_unit(c, is_delta=False):
    if use_f:
        return c * 9 / 5 if is_delta else c * 9 / 5 + 32
    return c

unit = "°F" if use_f else "°C"

# ---------- fetch data ----------
try:
    recent = fetch_recent(*CITIES[city])
    feats = make_features(recent).dropna()[columns]
    latest = feats.iloc[[-1]]
except Exception as e:
    st.error(f"Could not get weather data: {e}")
    st.stop()

# ---------- page ----------
st.title(f"🌤️ {city} Weather Predictor")
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Recent data", "History explorer", "Model comparison"])

# ===== TAB 1: prediction + what-if =====
with tab1:
    temp = temp_model.predict(latest)[0]
    rain_score = float(rain_model.predict_proba(latest)[0, 1])
    today_temp = latest["temp_mean"].iloc[0]

    st.caption(f"Based on data up to {latest.index[0].date()}")

    c1, c2 = st.columns(2)
    c1.metric("Tomorrow's mean temperature",
              f"{to_unit(temp):.1f} {unit}",
              delta=f"{to_unit(temp - today_temp, True):+.1f} {unit} vs today")
    c2.metric("Rain score", f"{rain_score:.2f}")
    c2.progress(min(rain_score, 1.0))

    if rain_score >= threshold:
        st.warning("☔ Rain likely. Take an umbrella.")
    else:
        st.success("☀️ Probably dry.")
    st.caption("The rain score is a relative signal, not a calibrated probability.")

    with st.expander("🔧 What-if: change today's weather and see how the model reacts"):
        w1, w2, w3 = st.columns(3)
        new_temp = w1.slider("Today's mean temp (°C)", 5.0, 45.0, float(round(today_temp, 1)), 0.5)
        new_hum = w2.slider("Today's humidity (%)", 0.0, 100.0, float(np.clip(latest["humidity"].iloc[0], 0, 100)), 1.0)
        new_pres = w3.slider("Today's pressure (hPa)", 985.0, 1025.0,
                             float(np.clip(latest["pressure"].iloc[0], 985, 1025)), 0.5)

        what_if = latest.copy()
        what_if["temp_mean"] = new_temp
        what_if["humidity"] = new_hum
        what_if["pressure"] = new_pres

        wi_temp = temp_model.predict(what_if)[0]
        wi_rain = float(rain_model.predict_proba(what_if)[0, 1])

        r1, r2 = st.columns(2)
        r1.metric("What-if temperature", f"{to_unit(wi_temp):.1f} {unit}",
                  delta=f"{to_unit(wi_temp - temp, True):+.1f} {unit} vs real prediction")
        r2.metric("What-if rain score", f"{wi_rain:.2f}", delta=f"{wi_rain - rain_score:+.2f}")
        st.caption("Only today's raw values change. The lag and rolling features stay as they were, so this is approximate.")

# ===== TAB 2: recent data =====
with tab2:
    st.subheader("Last 2 weeks")
    chosen = st.multiselect("Variables to plot", base_cols, default=["temp_mean"])
    normalize = st.checkbox("Normalize (0 to 1) so different units can share one chart")
    if chosen:
        data = recent[chosen]
        if normalize:
            rng = (data.max() - data.min()).replace(0, 1)
            data = (data - data.min()) / rng
        st.line_chart(data)
    else:
        st.info("Pick at least one variable.")
    with st.expander("Show raw table"):
        st.dataframe(recent)

# ===== TAB 3: history explorer =====
with tab3:
    st.subheader("Karachi, 2015 to 2025")
    years = st.slider("Years", 2015, 2025, (2018, 2025))
    var = st.selectbox("Variable", base_cols, index=2)
    view = st.radio("View", ["Daily", "Monthly average", "Average by calendar month"], horizontal=True)

    data = history.loc[str(years[0]):str(years[1]), var]
    if view == "Daily":
        st.line_chart(data)
    elif view == "Monthly average":
        st.line_chart(data.resample("MS").mean())
    else:
        st.bar_chart(data.groupby(data.index.month).mean())

# ===== TAB 4: model comparison =====
with tab4:
    st.subheader("Temperature models (test set 2023 to 2025)")
    reg = pd.DataFrame({
        "MAE": [0.494, 0.497, 0.502, 0.508, 0.524, 0.524],
    }, index=["Linear Regression", "Gradient Boosting (tuned)", "Gradient Boosting",
              "Ridge", "Random Forest", "Baseline (persistence)"])
    st.bar_chart(reg)
    st.caption("Lower MAE is better. Linear Regression wins.")

    st.subheader("Rain models")
    clf = pd.DataFrame({
        "Precision": [0.372, 0.259],
        "Recall": [0.51, 0.73],
        "F1": [0.430, 0.382],
        "AUC": [0.872, 0.860],
    }, index=["Random Forest", "Logistic Regression"])
    metric = st.selectbox("Metric", clf.columns, index=2)
    st.bar_chart(clf[metric])
    st.caption("Random Forest wins on F1 and AUC. Logistic Regression catches more rain (recall).")