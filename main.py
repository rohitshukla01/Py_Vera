#!/usr/bin/env python
"""
main.py

Core Forecasting Code for VERA
--------------------------------
This script performs:
  - Loading and preprocessing of daily target data.
  - Training an LSTM model to generate an ensemble forecast.
  - Saving the forecast output as a CSV file in the "model_output" folder.

The forecast file is saved as:
  YYYY-MM-DD-LSTM.csv
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
np.random.seed(42)
tf.random.set_seed(42)

# ---------------- Data Loading and Preprocessing ----------------
def load_and_preprocess_data():
    print("Loading daily target data...")
    targets_url = (
        "https://renc.osn.xsede.org/bio230121-bucket01/"
        "vera4cast/targets/project_id=vera4cast/"
        "duration=P1D/daily-insitu-targets.csv.gz"
    )
    targets = pd.read_csv(targets_url, compression="gzip", parse_dates=["datetime"])
    targets = targets[
        (targets["site_id"] == "fcre") &
        (targets["variable"] == "Chla_ugL_mean") &
        (targets["duration"] == "P1D") &
        (targets["depth_m"] == 1.6)
    ]
    targets["date"] = targets["datetime"].dt.floor("D")
    ref_date = datetime.today().date()  # current date as reference
    ref_date_ts = pd.Timestamp(ref_date, tz="UTC")
    idx = pd.date_range(targets["date"].min(), ref_date_ts, freq="D")
    daily = targets.set_index("date").reindex(idx).rename_axis("date")[["observation"]]
    if daily["observation"].isna().sum() > 0:
        print(f"Missing values detected. Applying KNN imputation...")
        imputer = KNNImputer(n_neighbors=10)
        df_numeric = daily.select_dtypes(include=[np.number])
        filled = imputer.fit_transform(df_numeric)
        daily[df_numeric.columns] = filled
    else:
        print("No missing values found.")
    daily = daily.reset_index().rename(columns={"index": "date"})
    daily = daily.rename(columns={"date": "datetime"})
    daily["site_id"] = "fcre"
    print("Data preprocessing complete.\n")
    return daily

# ---------------- LSTM Forecasting ----------------
def create_sequences(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

def run_lstm_ensemble_forecast(data, reference_datetime, forecast_horizon=30,
                               ensemble_size=10, window_size=30, epochs=5, batch_size=16):
    print("Running LSTM ensemble forecast (full data)...")
    df = data.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    site_ids = df["site_id"].unique()
    all_forecasts = []
    for site in site_ids:
        print(f"Processing site: {site}")
        site_data = df[df["site_id"] == site].reset_index(drop=True)
        site_data["log_obs"] = np.log(site_data["observation"] + 0.001)
        series = site_data["log_obs"].values
        if len(series) <= window_size:
            raise ValueError(f"Not enough data for site {site} with window_size {window_size}")
        X, y = create_sequences(series, window_size)
        print(f"Total training samples for site {site}: {len(X)}")
        X = X.reshape((-1, window_size, 1))
        y = y.reshape((-1, 1))
        print("Building and training LSTM model...")
        model = Sequential([
            LSTM(64, input_shape=(window_size, 1)),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(), loss="mse")
        early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
        print("Model training complete.")
        fitted = model.predict(X, verbose=0)
        residuals = y - fitted
        sd_resid = np.std(residuals)
        print("Residual standard deviation:", sd_resid)
        last_sequence = series[-window_size:].reshape((window_size, 1))
        ensemble_matrix = np.zeros((forecast_horizon, ensemble_size))
        for ens in range(ensemble_size):
            if ens % 2 == 0:
                print(f"  Ensemble iteration {ens+1}/{ensemble_size}")
            current_seq = last_sequence.copy()
            forecast_values = []
            for t in range(forecast_horizon):
                pred = model(current_seq[np.newaxis, :, :], training=True)[0, 0].numpy()
                pred += np.random.normal(loc=0.0, scale=sd_resid)
                forecast_values.append(pred)
                current_seq = np.append(current_seq[1:], [[pred]], axis=0)
            ensemble_matrix[:, ens] = forecast_values
        fc_dates = pd.date_range(start=reference_datetime, periods=forecast_horizon, freq="D")
        forecast_rows = []
        for i, fc_date in enumerate(fc_dates):
            for ens in range(ensemble_size):
                forecast_rows.append({
                    "site_id": site,
                    "datetime": fc_date,
                    "reference_datetime": pd.to_datetime(reference_datetime),
                    "family": "ensemble",
                    "variable": "Chla_ugL_mean",
                    "model_id": "LSTM",
                    "duration": "P1D",
                    "project_id": "vera4cast",
                    "depth_m": 1.6 if site == "fcre" else 1.5,
                    "parameter": str(ens+1),
                    "prediction": ensemble_matrix[i, ens]  # still in log-scale
                })
        all_forecasts.append(pd.DataFrame(forecast_rows))
    ensemble_df = pd.concat(all_forecasts, ignore_index=True)
    return ensemble_df

def save_forecast(forecast_df, daily):
    print("Saving forecast output...")
    # Convert predictions from log-scale to original
    forecast_df["prediction"] = np.exp(forecast_df["prediction"]) - 0.001
    # Format datetime fields to show only date (YYYY-MM-DD)
    forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"]).dt.strftime("%Y-%m-%d")
    forecast_df["reference_datetime"] = pd.to_datetime(forecast_df["reference_datetime"]).dt.strftime("%Y-%m-%d")
    outdir = "model_output"
    os.makedirs(outdir, exist_ok=True)
    ref_date = daily["datetime"].max().date()
    fname = f"{ref_date}-LSTM.csv"
    path = os.path.join(outdir, fname)
    forecast_df.to_csv(path, index=False)
    print(f"Forecast saved to: {path}")
    return path

def main():
    daily = load_and_preprocess_data()
    ref_datetime = daily["datetime"].max()
    print("Forecast reference date:", ref_datetime)
    forecast_df = run_lstm_ensemble_forecast(daily, ref_datetime,
                                             forecast_horizon=30, ensemble_size=10,
                                             window_size=30, epochs=5, batch_size=16)
    save_forecast(forecast_df, daily)

#if __name__ == "__main__":
#    main()
main()
