import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
LOOKBACK = 24  # use the previous 24 hourly readings (same as notebook)
HOURLY_FILE = "hourly_data.csv"
MODEL_FILE = "lstm_model.keras"

np.random.seed(SEED)
tf.random.set_seed(SEED)

st.set_page_config(page_title="Electricity Forecast AI", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    .main-title {font-size: 2.3rem; font-weight: 800; margin-bottom: 0;}
    .sub-title {color: #6b7280; margin-bottom: 1.2rem;}
    div[data-testid="stMetric"] {background: rgba(99,102,241,0.08);
        padding: 14px 16px; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data (same steps as notebook)
@st.cache_data(show_spinner="Cleaning data and converting to hourly averages...")
def build_hourly_from_raw(file) -> pd.DataFrame:
    df = pd.read_csv(
        file, sep=";", na_values=["?"], low_memory=False,
        usecols=["Date", "Time", "Global_active_power"],
    )
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    hourly = df["Global_active_power"].resample("h").mean().to_frame()
    hourly = hourly.rename(columns={"Global_active_power": "actual_kW"})
    hourly["actual_kW"] = hourly["actual_kW"].interpolate(method="time", limit=3)
    return hourly.dropna()


@st.cache_data
def load_hourly_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def prepare(hourly: pd.DataFrame):
    values = hourly[["actual_kW"]].values.astype("float32")
    n = len(values)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    scaler = MinMaxScaler()
    scaler.fit(values[:train_end])
    scaled_values = scaler.transform(values)

    X, y, target_positions = [], [], []
    for target_pos in range(LOOKBACK, len(scaled_values)):
        X.append(scaled_values[target_pos - LOOKBACK:target_pos])
        y.append(scaled_values[target_pos])
        target_positions.append(target_pos)
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")
    target_positions = np.array(target_positions)

    train_mask = target_positions < train_end
    val_mask = (target_positions >= train_end) & (target_positions < val_end)
    test_mask = target_positions >= val_end

    return {
        "scaler": scaler, "scaled": scaled_values,
        "X_train": X[train_mask], "y_train": y[train_mask],
        "X_val": X[val_mask], "y_val": y[val_mask],
        "X_test": X[test_mask], "y_test": y[test_mask],
        "test_dates": hourly.index[target_positions[test_mask]],
    }


def build_model() -> Sequential:
    model = Sequential([
        Input(shape=(LOOKBACK, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_model(d, epochs):
    model = build_model()
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(
        d["X_train"], d["y_train"],
        validation_data=(d["X_val"], d["y_val"]),
        epochs=epochs, batch_size=64, shuffle=False,
        callbacks=[early_stop], verbose=0,
    )
    return model, history.history


def forecast_ahead(model, scaler, scaled_values, last_ts, hours):
    window = scaled_values[-LOOKBACK:].reshape(1, LOOKBACK, 1).copy()
    preds = []
    for _ in range(hours):
        nxt = model.predict(window, verbose=0)
        preds.append(float(scaler.inverse_transform(nxt)[0, 0]))
        window = np.concatenate([window[:, 1:, :], nxt.reshape(1, 1, 1)], axis=1)
    idx = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=hours, freq="h")
    return pd.DataFrame({"forecast_kW": preds}, index=idx)


def metrics(actual, predicted, label):
    return {
        "Model": label,
        "MAE (kW)": mean_absolute_error(actual, predicted),
        "RMSE (kW)": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


# ---------------------------------------------------------------- UI
st.markdown('<p class="main-title">⚡ AI Electricity Consumption Forecaster</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Two-layer LSTM deep learning model · uses the previous 24 hours '
    "to predict household power usage</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    bundled = os.path.exists(HOURLY_FILE)
    options = (["Bundled hourly data"] if bundled else []) + ["Upload raw dataset (.txt / .csv)"]
    source = st.radio("Data source", options)
    uploaded = None
    if source.startswith("Upload"):
        uploaded = st.file_uploader("household_power_consumption.txt", type=["txt", "csv"])
    epochs = st.slider("Max training epochs", 1, 30, 15)
    horizon = st.slider("Forecast horizon (hours)", 1, 72, 24)
    days_to_show = st.slider("Days shown in test plot", 1, 14, 7)
    run = st.button("🚀 Run forecast", type="primary", width="stretch")

hourly = None
if source.startswith("Bundled"):
    hourly = load_hourly_csv(HOURLY_FILE)
elif uploaded is not None:
    hourly = build_hourly_from_raw(uploaded)

if hourly is None:
    st.info("👈 Upload the dataset in the sidebar, then click **Run forecast**.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Hourly rows", f"{len(hourly):,}")
c2.metric("First timestamp", str(hourly.index.min())[:16])
c3.metric("Last timestamp", str(hourly.index.max())[:16])

with st.expander("📈 Household electricity consumption history", expanded=True):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(hourly.index, hourly["actual_kW"], linewidth=0.8)
    ax.set_xlabel("Date and time"); ax.set_ylabel("Average active power (kW)")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    st.pyplot(fig)

if run:
    d = prepare(hourly)

    if os.path.exists(MODEL_FILE) and source.startswith("Bundled"):
        model, hist = tf.keras.models.load_model(MODEL_FILE), None
        st.success("Loaded pre-trained LSTM model.")
    else:
        with st.spinner("Training LSTM model... this can take a few minutes"):
            model, hist = train_model(d, epochs)
        st.success("Model trained.")
    st.session_state["result"] = (model, hist, d, hourly)

if "result" in st.session_state:
    model, hist, d, hourly = st.session_state["result"]

    pred_scaled = model.predict(d["X_test"], verbose=0)
    y_test_kw = d["scaler"].inverse_transform(d["y_test"].reshape(-1, 1)).ravel()
    pred_kw = d["scaler"].inverse_transform(pred_scaled).ravel()
    baseline_kw = d["scaler"].inverse_transform(d["X_test"][:, -1, 0].reshape(-1, 1)).ravel()

    metrics_df = pd.DataFrame([
        metrics(y_test_kw, pred_kw, "LSTM"),
        metrics(y_test_kw, baseline_kw, "Persistence baseline"),
    ])

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔮 Forecast", "📊 Actual vs Predicted", "🏁 Model Evaluation", "📥 Data & Download"]
    )

    with tab1:
        fc = forecast_ahead(model, d["scaler"], d["scaled"], hourly.index[-1], horizon)
        next_hour = fc["forecast_kW"].iloc[0]
        a, b, c = st.columns(3)
        a.metric("Forecast timestamp", str(fc.index[0])[:16])
        b.metric("Next-hour prediction", f"{next_hour:.3f} kW")
        c.metric(f"Average next {horizon} h", f"{fc['forecast_kW'].mean():.3f} kW")

        fig, ax = plt.subplots(figsize=(14, 4.5))
        recent = hourly["actual_kW"].tail(24 * 5)
        ax.plot(recent.index, recent.values, label="Recent actual", linewidth=1.8)
        ax.plot(fc.index, fc["forecast_kW"], label="LSTM forecast", linewidth=2, color="#ef4444", marker="o", markersize=3)
        ax.set_xlabel("Date and time"); ax.set_ylabel("Average active power (kW)")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        st.pyplot(fig)
        st.caption("Multi-hour forecast feeds each prediction back as input, so accuracy "
                   "drops the further ahead you look. The first hour is the most reliable.")

    with tab2:
        n_show = min(days_to_show * 24, len(y_test_kw))
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(d["test_dates"][:n_show], y_test_kw[:n_show], label="Actual consumption", linewidth=2)
        ax.plot(d["test_dates"][:n_show], pred_kw[:n_show], label="LSTM predicted consumption", linewidth=1.5)
        ax.plot(d["test_dates"][:n_show], baseline_kw[:n_show], label="Baseline prediction", linewidth=1, alpha=0.8)
        ax.set_xlabel("Date and time"); ax.set_ylabel("Average active power (kW)")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.dataframe(metrics_df.round(4), width="stretch", hide_index=True)
        if metrics_df.loc[0, "MAE (kW)"] < metrics_df.loc[1, "MAE (kW)"]:
            st.success("On this test split, LSTM has lower MAE than the baseline.")
        else:
            st.warning("On this test split, LSTM does not have lower MAE than the baseline.")
        if hist:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(hist["loss"], label="Training loss")
            ax.plot(hist["val_loss"], label="Validation loss")
            ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (scaled data)")
            ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
            st.pyplot(fig)

    with tab4:
        predictions_df = pd.DataFrame({
            "timestamp": d["test_dates"],
            "actual_kW": y_test_kw,
            "lstm_predicted_kW": pred_kw,
            "baseline_predicted_kW": baseline_kw,
        }).set_index("timestamp")
        st.dataframe(predictions_df.head(50).round(3), width="stretch")
        st.download_button("⬇️ Download test predictions (CSV)",
                           predictions_df.to_csv().encode(), "predictions.csv", "text/csv")
        st.download_button("⬇️ Download forecast (CSV)",
                           fc.to_csv().encode(), "forecast.csv", "text/csv")
else:
    st.info("Click **🚀 Run forecast** in the sidebar to train/load the model and see predictions.")
