"""
predictor.py — Prediksi deret waktu menggunakan Prophet (Meta/Facebook)
========================================================================
Instalasi: pip install prophet
Dokumentasi: https://facebook.github.io/prophet/

run_prophet_forecast() menerima DataFrame historis dan nama parameter,
melatih model Prophet, lalu mengembalikan:
  - forecast_df : DataFrame hasil prediksi (kolom ds, yhat, yhat_lower, yhat_upper,
                  trend, yearly)
  - metrics     : dict berisi MAE, RMSE, MAPE, R²  (dihitung pada data historis)
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def run_prophet_forecast(df_hist: pd.DataFrame,
                         parameter: str,
                         horizon_months: int = 12) -> tuple:
    """
    Latih model Prophet pada data historis dan hasilkan prediksi.

    Parameters
    ----------
    df_hist       : DataFrame historis (kolom: time, <parameter>)
    parameter     : nama kolom parameter yang akan diprediksi
    horizon_months: jumlah bulan ke depan yang diprediksi

    Returns
    -------
    (forecast_df, metrics_dict)
    """
    # ── Import Prophet ──────────────────────────────────────
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError(
            "Prophet belum terpasang. Jalankan: pip install prophet\n"
            "(Mungkin perlu juga: pip install pystan==2.19.1.1)"
        )

    # ── Siapkan data untuk Prophet ──────────────────────────
    # Prophet butuh kolom 'ds' (datetime) dan 'y' (nilai)
    df_ts = df_hist.groupby("time")[parameter].mean().reset_index()
    df_ts = df_ts.rename(columns={"time": "ds", parameter: "y"})
    df_ts = df_ts.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)

    if len(df_ts) < 24:
        raise ValueError(f"Data terlalu sedikit untuk Prophet: {len(df_ts)} baris (minimum 24)")

    # ── Konfigurasi Model ───────────────────────────────────
    model = Prophet(
        yearly_seasonality=True,      # tangkap pola musiman tahunan (ENSO, monsun)
        weekly_seasonality=False,     # data bulanan, tidak ada pola mingguan
        daily_seasonality=False,
        seasonality_mode="multiplicative",   # lebih cocok untuk data oseanografi
        changepoint_prior_scale=0.05,        # fleksibilitas tren (lebih kecil = lebih konservatif)
        seasonality_prior_scale=10.0,
        interval_width=0.80,                 # interval kepercayaan 80%
        n_changepoints=25,
    )

    # Tambahkan regressor ENSO proxy (sinus 48-bulan) jika ada cukup data
    df_ts["enso_proxy"] = np.sin(2 * np.pi * np.arange(len(df_ts)) / 48)
    model.add_regressor("enso_proxy", prior_scale=0.5, standardize=True)

    # ── Latih Model ─────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_ts)

    # ── Buat DataFrame Masa Depan ───────────────────────────
    # Prophet gunakan frekuensi bulanan ('MS' = month start)
    future = model.make_future_dataframe(periods=horizon_months, freq="MS")

    # Tambahkan regressor ENSO untuk periode masa depan
    n_future = len(future)
    future["enso_proxy"] = np.sin(2 * np.pi * np.arange(n_future) / 48)

    # ── Prediksi ────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = model.predict(future)

    # ── Hitung Metrik pada Data Historis ────────────────────
    # Merge prediksi dengan data aktual (hanya periode historis)
    hist_pred = forecast[forecast["ds"].isin(df_ts["ds"])][["ds","yhat"]].merge(
        df_ts[["ds","y"]], on="ds", how="inner"
    )

    if len(hist_pred) > 0:
        y_true = hist_pred["y"].values
        y_pred = hist_pred["yhat"].values
        residuals = y_true - y_pred

        mae  = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        # MAPE — hindari pembagian nol
        nonzero = y_true != 0
        mape = float(np.mean(np.abs(residuals[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else 0.0
        # R²
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y_true - y_true.mean())**2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        mae = rmse = mape = r2 = 0.0

    metrics = {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

    return forecast, metrics


def prophet_cross_validate(df_hist: pd.DataFrame,
                            parameter: str,
                            initial_months: int = 120,
                            period_months: int = 12,
                            horizon_months: int = 12) -> pd.DataFrame:
    """
    Cross-validation Prophet untuk evaluasi lebih robust.
    Mengembalikan DataFrame metrik per fold.

    Penggunaan (opsional, bisa dipanggil dari tab analisis lanjutan):
        cv_df = prophet_cross_validate(df, "sst", initial_months=60)
    """
    try:
        from prophet import Prophet
        from prophet.diagnostics import cross_validation, performance_metrics
    except ImportError:
        raise ImportError("Prophet belum terpasang: pip install prophet")

    df_ts = df_hist.groupby("time")[parameter].mean().reset_index()
    df_ts = df_ts.rename(columns={"time": "ds", parameter: "y"}).dropna()
    df_ts["enso_proxy"] = np.sin(2 * np.pi * np.arange(len(df_ts)) / 48)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        interval_width=0.80,
    )
    model.add_regressor("enso_proxy", prior_scale=0.5, standardize=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_ts)
        df_cv = cross_validation(
            model,
            initial=f"{initial_months * 30} days",
            period=f"{period_months * 30} days",
            horizon=f"{horizon_months * 30} days",
            parallel="processes",
        )
        df_metrics = performance_metrics(df_cv)

    return df_metrics
