import os, glob, csv, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# -------------------------
# Paths (artifacts live in this folder)
# -------------------------
ART_DIR = os.path.dirname(__file__)
PREPROC_DISAGG = os.path.join(ART_DIR, "preprocessor.pkl")               # only for mains name hint (optional)
PREPROC_FCST   = os.path.join(ART_DIR, "preprocessor_forecast.pkl")
MODEL_FCST     = os.path.join(ART_DIR, "model_forecast.h5")

PREPROC_AE     = os.path.join(ART_DIR, "preprocessor_anomaly.pkl")
MODEL_AE       = os.path.join(ART_DIR, "model_anomaly.h5")
THRESH_AE      = os.path.join(ART_DIR, "threshold_anom.json")

# -------------------------
# Robust CSV utils
# -------------------------
def robust_read_csv(path, expect_min_cols=2):
    encs = ["utf-8","utf-8-sig","cp1252","latin1"]
    seps = [",",";","\\t","|"]
    try:
        with open(path, "rb") as f:
            head = f.read(8192).decode("latin1", errors="ignore")
        sn = csv.Sniffer().sniff(head)
        if sn.delimiter in seps:
            seps = [sn.delimiter] + [s for s in seps if s != sn.delimiter]
    except Exception:
        pass
    last_err = None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                if df.shape[1] >= expect_min_cols:
                    return df
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Could not parse {path}. Last error: {last_err}")

def pick_time_column(cols):
    cands = ["timestamp","time","datetime","date","ts","utc","localtime","index"]
    lmap = {c.lower(): c for c in cols}
    for c in cands:
        if c in lmap: return lmap[c]
    return cols[0]

def parse_time_column(df, tcol):
    s = df[tcol]
    if pd.api.types.is_numeric_dtype(s):
        m = s.median()
        try:
            if m > 10**12:
                return pd.to_datetime(s, unit="ms", errors="coerce")
            elif m > 10**9:
                return pd.to_datetime(s, unit="s", errors="coerce")
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def looks_like_energy(colname:str):
    ln = colname.lower()
    return any(k in ln for k in ["wh","kwh","energy","consumption","power","mains","aggregate","total","house","use"])

def load_mains_series_only(data_dir, mains_hint:str|None):
    all_csvs = [p for p in glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)
                if os.path.isfile(p) and p.lower().endswith(".csv")]
    if not all_csvs:
        raise FileNotFoundError(f"No CSVs found under {data_dir}")

    best_series, best_sum = None, -1.0
    hint_low = mains_hint.lower() if mains_hint else None

    for path in all_csvs:
        try:
            df = robust_read_csv(path, expect_min_cols=2)
            tcol = pick_time_column(list(df.columns))
            df = df.dropna(subset=[tcol]).copy()
            df[tcol] = parse_time_column(df, tcol)
            df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
            num_df = df.select_dtypes(include=["number"])
            if num_df.empty:
                continue

            cand_cols = []
            if hint_low:
                cand_cols = [c for c in num_df.columns if (hint_low in c.lower()) or c.lower().endswith(hint_low)]
            if not cand_cols:
                # fallback: mains-like columns in this file
                cand_cols = [c for c in num_df.columns if looks_like_energy(c)]
                # prefer the top-5 by variance
                if len(cand_cols) > 5:
                    var = num_df[cand_cols].var().sort_values(ascending=False)
                    cand_cols = list(var.index[:5])

            for c in cand_cols:
                s = num_df[c].astype("float32")
                if looks_like_energy(c):
                    s = s.resample("1T").sum().astype("float32")
                else:
                    s = s.resample("1T").mean().astype("float32")
                tot = float(np.nansum(s.values))
                if np.isfinite(tot) and tot > best_sum:
                    best_sum = tot
                    best_series = s
        except Exception:
            continue

    if best_series is None:
        raise RuntimeError("Could not find a mains-like series. Provide a tighter folder or check column names.")
    best_series = best_series.sort_index().ffill().bfill().fillna(0.0).astype("float32")
    return best_series

def mains_hint_from_disagg():
    try:
        import pickle
        with open(PREPROC_DISAGG, "rb") as f:
            d = pickle.load(f)
        return d.get("mains_column", d.get("mains_col"))
    except Exception:
        return None

# -------------------------
# Forecast next 24h (single shot)
# -------------------------
def forecast_next24(data_dir:str, out_dir:str|None=None):
    import pickle
    if out_dir is None:
        out_dir = ART_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(PREPROC_FCST) or not os.path.exists(MODEL_FCST):
        raise FileNotFoundError("Missing forecast artifacts (preprocessor_forecast.pkl/model_forecast.h5). Train first.")

    with open(PREPROC_FCST, "rb") as f:
        pc = pickle.load(f)
    hist = int(pc["hist_minutes"]); horizon = int(pc["horizon_minutes"])
    feats = list(pc["feature_names"])
    scaler_X: StandardScaler = pc["scaler_X"]
    scaler_Y: StandardScaler = pc["scaler_Y"]

    # Load model safely
    try:
        model = tf.keras.models.load_model(MODEL_FCST, compile=False)
    except Exception as e:
        # final fallback
        model = tf.keras.models.load_model(MODEL_FCST, custom_objects={"mae": tf.keras.metrics.mean_absolute_error}, compile=False)

    mains_hint = pc.get("mains_column") or mains_hint_from_disagg()
    mains = load_mains_series_only(data_dir, mains_hint)

    # Build features from the LAST 'hist' minutes
    end = len(mains)
    if end < hist + 10:
        raise RuntimeError(f"Not enough minutes ({end}) for {hist}-minute history.")
    idx = mains.index
    df = pd.DataFrame(index=idx)
    df["mains"] = mains.values
    h = idx.hour.values; d = idx.dayofweek.values
    df["h_sin"] = np.sin(2*np.pi*h/24.0).astype("float32")
    df["h_cos"] = np.cos(2*np.pi*h/24.0).astype("float32")
    df["d_sin"] = np.sin(2*np.pi*d/7.0).astype("float32")
    df["d_cos"] = np.cos(2*np.pi*d/7.0).astype("float32")

    X_last = df[feats].iloc[-hist:].values.astype("float32")
    Xs = scaler_X.transform(X_last).reshape(1, hist, len(feats)).astype("float32")

    Y_pred_s = model.predict(Xs, verbose=0)        # (1, horizon, 1)
    Y_pred = scaler_Y.inverse_transform(Y_pred_s.reshape(-1,1)).reshape(horizon)

    # Build times for plot
    t_hist = df.index[-hist:]
    t_horz = pd.date_range(t_hist[-1] + pd.Timedelta(minutes=1), periods=horizon, freq="T")
    y_hist = df["mains"].iloc[-hist:].values

    # Plot & save
    plt.figure(figsize=(12,4))
    plt.plot(t_hist, y_hist, label="History (mains)")
    plt.plot(t_horz, Y_pred, label="Pred next 24h")
    plt.legend(); plt.title("SmartMeterX — Next-24h Forecast")
    plt.xlabel("Time"); plt.ylabel("Mains")
    plt.tight_layout()
    out_png = os.path.join(out_dir, "forecast_next24.png")
    plt.savefig(out_png, dpi=150); plt.close()

    # Return summary
    out_json = {
        "status": "ok",
        "hist_minutes": hist,
        "horizon_minutes": horizon,
        "plot_path": out_png,
    }
    with open(os.path.join(out_dir, "forecast_next24.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    return out_json

# -------------------------
# Anomaly detection (timeline)
# -------------------------
def detect_anomalies(data_dir:str, out_dir:str|None=None):
    import pickle
    if out_dir is None:
        out_dir = ART_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not (os.path.exists(PREPROC_AE) and os.path.exists(MODEL_AE) and os.path.exists(THRESH_AE)):
        raise FileNotFoundError("Missing anomaly artifacts (preprocessor_anomaly.pkl/model_anomaly.h5/threshold_anom.json). Train first.")

    with open(PREPROC_AE, "rb") as f:
        pa = pickle.load(f)
    win = int(pa["window"]); stride = int(pa["stride"])
    scaler: StandardScaler = pa["scaler"]
    mains_hint = pa.get("mains_column") or pa.get("mains_column_hint") or mains_hint_from_disagg()

    with open(THRESH_AE, "r", encoding="utf-8") as f:
        th = json.load(f)
    threshold = float(th["threshold"])

    # model
    try:
        model = tf.keras.models.load_model(MODEL_AE, compile=False)
    except Exception:
        model = tf.keras.models.load_model(MODEL_AE, compile=False)

    mains = load_mains_series_only(data_dir, mains_hint)
    vec = mains.values.astype("float32")
    N = len(vec)
    if N < win + 10:
        raise RuntimeError(f"Not enough minutes ({N}) for AE window {win}.")

    starts = np.arange(0, N - win + 1, stride, dtype=int)
    # stream in chunks for memory safety
    batch = 1024
    recon_err = np.zeros(len(starts), dtype="float32")
    for i in range(0, len(starts), batch):
        sl = starts[i:i+batch]
        X = np.stack([vec[s:s+win] for s in sl], axis=0).astype("float32")[:, :, None]
        Xs = scaler.transform(X.reshape(-1,1)).reshape(X.shape).astype("float32")
        rec = model.predict(Xs, verbose=0)
        recon_err[i:i+batch] = np.mean((rec - Xs)**2, axis=(1,2))

    is_anom = (recon_err >= threshold).astype(int)
    times = mains.index[starts]
    df = pd.DataFrame({"start_time": times, "recon_error": recon_err, "is_anomaly": is_anom})
    out_csv = os.path.join(out_dir, "anomalies.csv")
    df.to_csv(out_csv, index=False)

    # timeline plot (mark spans)
    plot_len = min(3*24*60, N)  # ~3 days or full if shorter
    t_plot = mains.index[:plot_len]
    y_plot = vec[:plot_len]
    plt.figure(figsize=(12,4))
    plt.plot(t_plot, y_plot, label="Mains")
    for st, flag in zip(starts, is_anom):
        if flag and st < plot_len:
            t0 = mains.index[st]
            t1 = mains.index[min(st + win, plot_len-1)]
            plt.axvspan(t0, t1, color="red", alpha=0.08)
    plt.title("SmartMeterX — Anomaly Timeline")
    plt.xlabel("Time"); plt.ylabel("Mains")
    plt.tight_layout()
    out_png = os.path.join(out_dir, "anomaly_timeline.png")
    plt.savefig(out_png, dpi=150); plt.close()

    out_json = {
        "status": "ok",
        "threshold": threshold,
        "csv_path": out_csv,
        "plot_path": out_png
    }
    with open(os.path.join(out_dir, "anomaly_result.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    return out_json

# -------------------------
# CLI
# -------------------------
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="SmartMeterX inference")
    p.add_argument("--task", choices=["forecast","anomaly"], required=True, help="Which task to run")
    p.add_argument("--data_dir", required=True, help="Folder containing AMPds CSVs")
    p.add_argument("--out_dir", default=ART_DIR, help="Where to save outputs")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    if args.task == "forecast":
        res = forecast_next24(args.data_dir, args.out_dir)
    else:
        res = detect_anomalies(args.data_dir, args.out_dir)
    print(json.dumps(res, indent=2, default=str))
