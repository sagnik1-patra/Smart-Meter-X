SmartMeterX — NILM Disaggregation, Forecasting & Anomaly Detection

SmartMeterX turns a single smart-meter (mains) signal into:

Per-appliance disaggregation (baseline multi-target regression)

Next-24h mains forecasting (seq2seq LSTM)

Unsupervised anomaly detection (autoencoder on mains)

Inference scripts & minimal UI for quick demos

All artifacts are saved to:

C:\Users\sagni\Downloads\SmartMeterX


Your AMPds dataset folder is:

C:\Users\sagni\Downloads\SmartMeterX\archive (2)

What’s included

Training / evaluation (run via notebook cells you executed):

model_disagg.h5 / model_disagg.keras — disaggregation model

preprocessor.pkl — scalers & column names for disaggregation

model_forecast.h5 — next-24h forecaster

preprocessor_forecast.pkl — scalers & config for forecasting

forecast_config.yaml — model/feature settings

model_anomaly.h5 — mains autoencoder

preprocessor_anomaly.pkl — scaler & window config for anomaly

threshold_anom.json — anomaly threshold from validation

Metrics & plots

metrics.json — disaggregation metrics (per-appliance)

accuracy_per_appliance.png — R² per appliance (test)

error_heatmap_hour.png — MAE heatmap by hour (test)

metrics_forecast.json — MAE & sMAPE for forecasting

forecast_plot.png — example 24h forecast

anomalies.csv — anomaly timeline with reconstruction error

anomaly_plot.png — anomaly spans over mains

Inference & UI

infer.py — CLI / programmatic inference

gradio_app.py — 2-tab web UI (Forecast, Anomaly)

requirements_infer.txt — runtime deps for inference

Dataset

You’re using AMPds (minutely, single home) from Kaggle.

Put all CSVs (ignore HTML files) under:
C:\Users\sagni\Downloads\SmartMeterX\archive (2)

The code auto-detects the mains column by fuzzy matching (e.g., “mains”, “power”, “energy”, “aggregate”, etc.) and resamples to 1-minute. It loads only the mains series to stay memory-safe.

Environment (Windows)
py -3.11 -m venv C:\Users\sagni\Downloads\SmartMeterX\.venv
C:\Users\sagni\Downloads\SmartMeterX\.venv\Scripts\activate

pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn tensorflow pyyaml seaborn gradio


If you only want inference:
pip install -r C:\Users\sagni\Downloads\SmartMeterX\requirements_infer.txt

Running the notebooks (summary)

You already ran cells that:

Disaggregation training & eval

Saves: model_disagg.h5 / model_disagg.keras, preprocessor.pkl

Plots: accuracy_per_appliance.png, error_heatmap_hour.png

Metrics: metrics.json

Forecast (5 epochs) & Anomaly (5 epochs)

Saves: model_forecast.h5, preprocessor_forecast.pkl, forecast_config.yaml, metrics_forecast.json, forecast_plot.png

Saves: model_anomaly.h5, preprocessor_anomaly.pkl, threshold_anom.json, anomalies.csv, anomaly_plot.png

The cells are memory-safe: they stream only mains and cap the number of windows to avoid MemoryError.

Inference (no notebook needed)
CLI
REM Forecast next 24h from the latest data
python C:\Users\sagni\Downloads\SmartMeterX\infer.py ^
  --task forecast ^
  --data_dir "C:\Users\sagni\Downloads\SmartMeterX\archive (2)" ^
  --out_dir  "C:\Users\sagni\Downloads\SmartMeterX"

REM Anomaly timeline over the dataset
python C:\Users\sagni\Downloads\SmartMeterX\infer.py ^
  --task anomaly ^
  --data_dir "C:\Users\sagni\Downloads\SmartMeterX\archive (2)" ^
  --out_dir  "C:\Users\sagni\Downloads\SmartMeterX"


Outputs:

Forecast: forecast_next24.png, forecast_next24.json

Anomaly: anomalies.csv, anomaly_timeline.png, anomaly_result.json

Web UI (Gradio)
python C:\Users\sagni\Downloads\SmartMeterX\gradio_app.py


Open: http://127.0.0.1:7860/

If Chrome shows “localhost refused to connect”, something is blocking the local server. Try:

Temporarily disable VPN/firewall for local loopback

Use server_name="0.0.0.0" in gradio_app.py and open from your machine’s LAN IP

Check that no other app uses port 7860

How it works
1) Disaggregation (NILM baseline)

Inputs: sliding windows of [mains, hour/day sin/cos]

Outputs: per-appliance power (multi-target regression)

Metrics:

accuracy_per_appliance.png: R² per appliance on test

error_heatmap_hour.png: MAE by hour of day (rows=appliances, cols=0–23)

Artifacts: model_disagg.h5/.keras, preprocessor.pkl, metrics.json

2) Forecast (next 24h)

Model: seq2seq LSTM (12h history → 24h horizon)

Features: mains + calendar (hour/day sin/cos)

Metrics: metrics_forecast.json (MAE, sMAPE)

Plot: forecast_plot.png

Artifacts: model_forecast.h5, preprocessor_forecast.pkl, forecast_config.yaml

3) Unsupervised anomaly detection

Model: 1D Conv autoencoder on mains (reconstruction error)

Threshold: 99th percentile of validation reconstruction error (threshold_anom.json)

Outputs: anomaly spans & errors → anomalies.csv, anomaly_plot.png

Artifacts: model_anomaly.h5, preprocessor_anomaly.pkl, threshold_anom.json

Files & formats
SmartMeterX/
├─ model_disagg.h5                # (optional) legacy; load with compile=False if needed
├─ model_disagg.keras             # native Keras format (preferred)
├─ preprocessor.pkl               # dict: scalers, window/stride, mains/appliance columns
├─ metrics.json                   # disaggregation summary
├─ accuracy_per_appliance.png     # R² per appliance
├─ error_heatmap_hour.png         # MAE heatmap by hour
│
├─ model_forecast.h5
├─ preprocessor_forecast.pkl      # {'hist_minutes','horizon_minutes','stride_minutes','scaler_X','scaler_Y','feature_names','mains_column_hint'}
├─ forecast_config.yaml
├─ metrics_forecast.json
├─ forecast_plot.png
│
├─ model_anomaly.h5
├─ preprocessor_anomaly.pkl       # {'window','stride','scaler','mains_column_hint'}
├─ threshold_anom.json
├─ anomalies.csv
├─ anomaly_plot.png
│
├─ infer.py                       # CLI/programmatic inference
├─ gradio_app.py                  # optional UI
├─ requirements_infer.txt
└─ archive (2)\                   # your AMPds CSVs here

Troubleshooting

MemoryError during training / loading data

You already have the memory-safe cells. If still tight on RAM:

Increase strides: STRIDE_FCST, AE_STRIDE

Reduce window caps: MAX_TR_WIN_FCST, MAX_TR_WIN_AE, etc.

Reduce horizons/history windows (e.g., 6h history, 12h horizon)

Could not locate function 'mae' when loading .h5

Load with compile=False, or map alias:

The notebook already does:

tf.keras.models.load_model(MODEL_H5, custom_objects={"mae": tf.keras.metrics.mean_absolute_error}, compile=False)


Gradio “localhost refused to connect”

Check firewall/VPN, or switch to server_name="0.0.0.0" and open http://<your-ip>:7860/.

Mains column not found

The code fuzzy-matches (“mains”, “power”, “energy”, “aggregate”, “total”, “use”) per file and picks the strongest candidate.

If nothing is found, point archive (2) to a tighter subfolder or rename the column in a copy.

Reproducibility tips

Keep a fixed random seed (already set to 42).

Train with fixed window caps and fixed strides to ensure repeatable splits.

Save both .h5 and native .keras if you move across Keras versions.

Extend this project

Richer NILM models: TCN/U-Net-1D, seq2seq with attention, multi-house transfer learning.

Forecasting: add weather/holiday features; quantile loss for prediction intervals.

Anomalies: change-point detection (e.g., Ruptures), seasonal decomposition residuals, isolation forest on features.

Edge deploy: TFLite conversion for the forecaster & autoencoder.

UI: add overlay of appliance disaggregation per day; export CSV of forecasts.

Quick checklist

 Place AMPds CSVs under archive (2)

 Run notebook cells (you did):

Disaggregation training & plots

Forecast (5 epochs) & Anomaly (5 epochs)

 Use infer.py (CLI) or gradio_app.py (UI) 
 Author
 SAGNIK PATRA
