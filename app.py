"""
LAUTAN — Platform Intelijen Oseanografi Papua
============================================================
Real-Time  : CMEMS (arus/SST/salinitas) · NASA MODIS (klorofil-a)
             ERA5/CDS (angin) · BMKG Open (gelombang)
Prediksi   : Prophet (Facebook/Meta) — model deret waktu
Historis   : rangkuman_historis_20tahun.csv  (fallback: data sintetis)
============================================================
Instalasi dependensi:
    pip install streamlit pandas numpy plotly requests netCDF4
    pip install copernicusmarine earthaccess cdsapi prophet
    pip install global-land-mask
============================================================
Konfigurasi API (buat file .env atau isi langsung di config.py):
    CMEMS_USER, CMEMS_PASS
    NASA_USER,  NASA_PASS
    CDS_UID,    CDS_KEY
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
import json

# ── modul internal ──────────────────────────────────────────
from data_fetcher import (
    fetch_cmems, fetch_nasa_modis, fetch_era5, fetch_bmkg,
    build_realtime_dataframe,
)
from predictor import run_prophet_forecast
from spatial import build_spatial_grid, get_ocean_grid_points, normalisasi_global
from config import CMEMS_USER, CMEMS_PASS, NASA_USER, NASA_PASS, CDS_UID, CDS_KEY

# ── page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="LAUTAN — Ocean Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================
# DESIGN SYSTEM
# =========================================
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #F2F6FA; color: #0D1F33; }
[data-testid="stSidebar"] { background: #0D1F33 !important; border-right: 1px solid #1A3A5C !important; }
[data-testid="stSidebar"] * { color: #CBD8E8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: #7BAFD4 !important; font-size: 10px !important;
    text-transform: uppercase; letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace !important; font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #7BAFD4 !important; font-size: 12px !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #162A42 !important; border: 1px solid #2A4A6A !important;
    color: #CBD8E8 !important; border-radius: 6px !important;
}
.stButton > button {
    background: #1E6BB8 !important; border: none !important; color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important; font-size: 13px !important;
    font-weight: 600 !important; border-radius: 6px !important;
    transition: background 0.2s ease !important; padding: 10px 20px !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover { background: #1558A0 !important; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #D6E4F0; border-radius: 8px; padding: 16px 20px !important; }
[data-testid="stMetricLabel"] { color: #5A7FA0 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricValue"] { color: #0D1F33 !important; font-size: 24px !important; font-weight: 700 !important; }
.stTabs [data-baseweb="tab-list"] { background: #FFFFFF !important; border-bottom: 1px solid #D6E4F0 !important; border-radius: 8px 8px 0 0 !important; gap: 0; padding: 0 12px; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border: none !important; color: #5A7FA0 !important; font-family: 'Inter', sans-serif !important; font-size: 13px !important; font-weight: 500 !important; padding: 12px 18px !important; }
.stTabs [aria-selected="true"] { color: #1E6BB8 !important; border-bottom: 2px solid #1E6BB8 !important; font-weight: 600 !important; }
hr { border-color: #D6E4F0 !important; }
[data-testid="stSidebar"] .stButton > button { background: #FFFFFF !important; color: #0D1F33 !important; border: 1px solid #7BAFD4 !important; font-weight: 600 !important; }
[data-testid="stSidebar"] .stButton > button:hover { background: #E8F2FB !important; }
.page-header { border-bottom: 2px solid #1E6BB8; padding-bottom: 12px; margin-bottom: 24px; }
.page-header .eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #1E6BB8; text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 6px; }
.page-header h1 { font-size: 26px; font-weight: 700; color: #0D1F33; margin: 0; letter-spacing: -0.02em; }
.section-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #1E6BB8; text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 10px; font-weight: 500; }
.coord-tag { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #3A6080; background: #EBF3FB; padding: 3px 10px; border-radius: 4px; border: 1px solid #C0D8EE; display: inline-block; margin: 2px; }
.data-note { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #7BAFD4; background: #EBF3FB; border-left: 3px solid #1E6BB8; padding: 8px 12px; border-radius: 0 4px 4px 0; margin-top: 8px; }
.update-badge { display:inline-flex; align-items:center; gap:6px; background:#0A3620; border:1px solid #1E7A45; border-radius:20px; padding:4px 12px; font-family:'JetBrains Mono',monospace; font-size:10px; color:#4DD68A; letter-spacing:0.08em; }
.pulse { width:7px; height:7px; background:#4DD68A; border-radius:50%; animation:pulse 1.5s infinite; display:inline-block; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(1.3)} }
.api-status-ok  { color:#00895A; font-weight:700; }
.api-status-err { color:#D4811A; font-weight:700; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,1.0)",
    plot_bgcolor="rgba(242,246,250,0.8)",
    font=dict(family="Inter", color="#3A5070", size=12),
    xaxis=dict(gridcolor="#D6E4F0", zerolinecolor="#D6E4F0", color="#5A7FA0"),
    yaxis=dict(gridcolor="#D6E4F0", zerolinecolor="#D6E4F0", color="#5A7FA0"),
    title_font=dict(color="#0D1F33", size=14, family="Inter"),
)

# =========================================
# SESSION STATE
# =========================================
if "page"        not in st.session_state: st.session_state.page        = "home"
if "role"        not in st.session_state: st.session_state.role        = "akademisi"
if "last_update" not in st.session_state: st.session_state.last_update = None
if "api_status"  not in st.session_state: st.session_state.api_status  = {}
if "rt_data"     not in st.session_state: st.session_state.rt_data     = None

# =========================================
# LOAD DATA HISTORIS
# =========================================
@st.cache_data
def load_data():
    try:
        df_data = pd.read_csv("rangkuman_historis_20tahun.csv")
        df_data["time"] = pd.to_datetime(df_data["time"])
    except Exception:
        dates_fallback = pd.date_range(start="2001-01-01", end="2020-12-01", freq="MS")
        n = len(dates_fallback)
        rng = np.random.default_rng(0)
        t = np.arange(n)
        seasonal = np.sin(2 * np.pi * t / 12)
        enso     = 0.4 * np.sin(2 * np.pi * t / 48)
        trend    = -0.001 * t
        uo = -0.05 + 0.04 * seasonal + 0.02 * enso + trend + rng.normal(0, 0.008, n)
        vo = -0.01 + 0.02 * seasonal + 0.01 * enso + rng.normal(0, 0.004, n)
        df_data = pd.DataFrame({"time": dates_fallback, "uo": uo, "vo": vo})

    df_data["time"] = pd.to_datetime(df_data["time"])
    if "uo" not in df_data.columns: df_data["uo"] = -0.05
    if "vo" not in df_data.columns: df_data["vo"] = -0.01
    rng2 = np.random.default_rng(7)
    n = len(df_data)
    if "sst"       not in df_data.columns: df_data["sst"]       = 28.5 + df_data["uo"] * 6  + rng2.normal(0, 0.3,  n)
    if "ssta"      not in df_data.columns: df_data["ssta"]      = df_data["uo"] * 2.5 + rng2.normal(0, 0.1, n)
    if "ph"        not in df_data.columns: df_data["ph"]        = 8.12 + df_data["vo"] * 0.6 + rng2.normal(0, 0.01, n)
    if "do"        not in df_data.columns: df_data["do"]        = 6.2  - df_data["uo"] * 2.5 + rng2.normal(0, 0.05, n)
    if "salinitas" not in df_data.columns: df_data["salinitas"] = 34.2 + df_data["vo"] * 2.5 + rng2.normal(0, 0.08, n)
    if "chla"      not in df_data.columns: df_data["chla"]      = 0.22 + df_data["uo"] * (-0.5) + rng2.normal(0, 0.015, n)
    if "gelombang" not in df_data.columns: df_data["gelombang"] = 0.8  + abs(df_data["uo"]) * 2.5 + rng2.normal(0, 0.05, n)
    if "angin_u"   not in df_data.columns: df_data["angin_u"]   = -1.5 + df_data["uo"] * 12 + rng2.normal(0, 0.3, n)
    if "angin_v"   not in df_data.columns: df_data["angin_v"]   = -0.5 + df_data["vo"] * 6  + rng2.normal(0, 0.2, n)
    df_data["chla"]      = df_data["chla"].clip(0.05, 0.8)
    df_data["do"]        = df_data["do"].clip(4.5,  7.5)
    df_data["ph"]        = df_data["ph"].clip(7.9,  8.4)
    df_data["salinitas"] = df_data["salinitas"].clip(32.0, 36.5)
    df_data["gelombang"] = df_data["gelombang"].clip(0.2,  2.5)
    return df_data

df = load_data()
df["year"]          = df["time"].dt.year
df["month"]         = df["time"].dt.month
df["current_speed"] = np.sqrt(df["uo"]**2 + df["vo"]**2)
df["Ocean_Health_Index"] = (
    0.25 * normalisasi_global(df["do"],       4.5, 7.5) +
    0.20 * normalisasi_global(df["ph"],       7.9, 8.4) +
    0.20 * normalisasi_global(df["chla"],     0.05, 0.8) +
    0.15 * normalisasi_global(df["salinitas"],32.0, 36.5) +
    0.20 * (1 - normalisasi_global(df["gelombang"], 0.2, 2.5))
) * 100
df["Fisheries_Index"] = (
    0.35 * normalisasi_global(df["chla"],        0.05, 0.8) +
    0.25 * normalisasi_global(df["do"],           4.5, 7.5) +
    0.20 * normalisasi_global(df["current_speed"],0.0, 0.25) +
    0.20 * (1 - normalisasi_global(df["gelombang"],0.2, 2.5))
) * 100

# =========================================
# HELPERS: RENDER MAP
# =========================================
@st.cache_data(show_spinner=False)
def load_batas_provinsi():
    import urllib.request
    url = ("https://raw.githubusercontent.com/superpikar/"
           "indonesia-geojson/master/indonesia-province-simple.json")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

def render_map(df_map, z_col, colorscale, height=520):
    fig = px.scatter_mapbox(
        df_map, lat="lat", lon="lon", color=z_col,
        color_continuous_scale=colorscale, opacity=0.85,
        zoom=4.5, size_max=6,
        range_color=[float(df_map[z_col].quantile(0.03)), float(df_map[z_col].quantile(0.97))],
        mapbox_style="carto-positron",
    )
    fig.update_traces(marker=dict(size=4.5))
    map_layers = []
    geojson_prov = load_batas_provinsi()
    if geojson_prov:
        map_layers.append(dict(
            sourcetype="geojson", source=geojson_prov,
            type="line", color="#34679A", opacity=0.55, line=dict(width=1.0),
        ))
    fig.update_layout(
        mapbox=dict(center=dict(lat=-8.5, lon=137.0), layers=map_layers),
        margin={"r":0,"t":0,"l":0,"b":0}, height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            thickness=12, len=0.65,
            bgcolor="rgba(255,255,255,0.95)", bordercolor="#D6E4F0",
            tickfont=dict(color="#3A5070", size=10, family="JetBrains Mono"),
            title=dict(font=dict(color="#3A5070", size=10, family="JetBrains Mono")),
        )
    )
    return fig

# =========================================
# ROSE DIAGRAM HELPERS
# =========================================
def make_wind_rose(df_src, title="Rose Diagram Angin"):
    if df_src.empty: return go.Figure()
    speed = np.sqrt(df_src["angin_u"]**2 + df_src["angin_v"]**2)
    direction_deg = (np.degrees(np.arctan2(-df_src["angin_u"], -df_src["angin_v"])) + 360) % 360
    n_bins = 16; bin_edges = np.linspace(0, 360, n_bins + 1)
    speed_bins   = [0, 2, 4, 6, 8, 100]
    speed_labels = ["<2 m/s","2–4 m/s","4–6 m/s","6–8 m/s",">8 m/s"]
    colors_wind  = ["#A8C8E8","#5A9EC8","#1E6BB8","#0D3D6B","#031420"]
    fig = go.Figure()
    for k, (s0, s1) in enumerate(zip(speed_bins[:-1], speed_bins[1:])):
        counts, _ = np.histogram(direction_deg[(speed >= s0) & (speed < s1)], bins=bin_edges)
        fig.add_trace(go.Barpolar(r=counts, theta=bin_edges[:-1], width=[360/n_bins]*n_bins,
            name=speed_labels[k], marker_color=colors_wind[k],
            marker_line_color="white", marker_line_width=0.5, opacity=0.9))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#0D1F33", size=13, family="Inter"), x=0.5),
        polar=dict(
            radialaxis=dict(visible=True, tickfont=dict(size=9, color="#5A7FA0"), gridcolor="#D6E4F0"),
            angularaxis=dict(direction="clockwise", rotation=90,
                tickmode="array", tickvals=[0,45,90,135,180,225,270,315],
                ticktext=["U","TL","T","TG","S","BD","B","BL"],
                tickfont=dict(size=10, color="#0D1F33", family="Inter"), gridcolor="#D6E4F0"),
        ),
        legend=dict(font=dict(size=10, color="#3A5070", family="Inter"),
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#D6E4F0"),
        paper_bgcolor="#FFFFFF", height=380,
        margin=dict(l=20,r=20,t=40,b=20),
    )
    return fig

def make_wave_rose(df_src, title="Rose Diagram Gelombang"):
    if df_src.empty: return go.Figure()
    speed = df_src["gelombang"]
    direction_deg = (np.degrees(np.arctan2(df_src["uo"], df_src["vo"])) + 360) % 360
    n_bins = 16; bin_edges = np.linspace(0, 360, n_bins + 1)
    wave_bins   = [0, 0.5, 1.0, 1.5, 2.0, 10]
    wave_labels = ["<0.5 m","0.5–1 m","1–1.5 m","1.5–2 m",">2 m"]
    colors_wave = ["#C8E8D4","#6DBFA0","#1E9E6B","#0D5E40","#031F15"]
    fig = go.Figure()
    for k, (s0, s1) in enumerate(zip(wave_bins[:-1], wave_bins[1:])):
        counts, _ = np.histogram(direction_deg[(speed >= s0) & (speed < s1)], bins=bin_edges)
        fig.add_trace(go.Barpolar(r=counts, theta=bin_edges[:-1], width=[360/n_bins]*n_bins,
            name=wave_labels[k], marker_color=colors_wave[k],
            marker_line_color="white", marker_line_width=0.5, opacity=0.9))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#0D1F33", size=13, family="Inter"), x=0.5),
        polar=dict(
            radialaxis=dict(visible=True, tickfont=dict(size=9, color="#5A7FA0"), gridcolor="#D6E4F0"),
            angularaxis=dict(direction="clockwise", rotation=90,
                tickmode="array", tickvals=[0,45,90,135,180,225,270,315],
                ticktext=["U","TL","T","TG","S","BD","B","BL"],
                tickfont=dict(size=10, color="#0D1F33", family="Inter"), gridcolor="#D6E4F0"),
        ),
        legend=dict(font=dict(size=10, color="#3A5070", family="Inter"),
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#D6E4F0"),
        paper_bgcolor="#FFFFFF", height=380,
        margin=dict(l=20,r=20,t=40,b=20),
    )
    return fig

# =========================================
# FISHERIES STATUS
# =========================================
def get_fisheries_status(fsi_val, df_map):
    p25 = df_map["Fisheries_Index"].quantile(0.25)
    p75 = df_map["Fisheries_Index"].quantile(0.75)
    if fsi_val >= p75:
        return {"color":"#00895A","text":"SANGAT BAIK","icon":"✅","bg":"#EDFAF3","border":"#9FD9BE"}
    elif fsi_val >= p25:
        return {"color":"#1E6BB8","text":"NORMAL","icon":"🔵","bg":"#EBF3FB","border":"#7BAFD4"}
    else:
        return {"color":"#D4811A","text":"WASPADA","icon":"⚠️","bg":"#FEF6E8","border":"#F0C070"}

# =========================================
# HOME PAGE
# =========================================
if st.session_state.page == "home":
    st.markdown("""
<div style="background:#0D1F33;border-radius:12px;padding:64px 48px 56px;text-align:center;margin-bottom:40px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;bottom:0;background:repeating-linear-gradient(0deg,transparent,transparent 39px,rgba(30,107,184,0.08) 40px),repeating-linear-gradient(90deg,transparent,transparent 39px,rgba(30,107,184,0.08) 40px);pointer-events:none;"></div>
  <div style="display:inline-block;background:rgba(30,107,184,0.25);border:1px solid rgba(30,107,184,0.5);border-radius:4px;padding:4px 16px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#7BAFD4;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:28px;">
    SISTEM AKTIF · 4°S–12°S / 129°E–144°E · LAUT ARAFURA
  </div>
  <h1 style="font-family:'Inter',sans-serif;font-size:72px;font-weight:800;color:#FFFFFF;letter-spacing:-0.04em;margin:0 0 6px;line-height:1;">LAUTAN</h1>
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#7BAFD4;letter-spacing:0.22em;text-transform:uppercase;margin-bottom:24px;">Platform Intelijen Oseanografi Papua</div>
  <div style="color:#A8C0D8;font-size:15px;max-width:560px;margin:0 auto 40px;line-height:1.8;">
    Data real-time multi-API, klimatologi historis 20 tahun,<br>dan prediksi berbasis Prophet ML untuk kawasan Laut Arafura.
  </div>
  <div style="display:flex;justify-content:center;gap:48px;flex-wrap:wrap;">
    <div style="text-align:center;"><div style="font-size:32px;font-weight:800;color:#FFFFFF;">20+</div><div style="font-size:10px;color:#7BAFD4;font-family:'JetBrains Mono',monospace;margin-top:4px;">TAHUN DATA</div></div>
    <div style="width:1px;background:#1E3A5C;"></div>
    <div style="text-align:center;"><div style="font-size:32px;font-weight:800;color:#FFFFFF;">4</div><div style="font-size:10px;color:#7BAFD4;font-family:'JetBrains Mono',monospace;margin-top:4px;">SUMBER API</div></div>
    <div style="width:1px;background:#1E3A5C;"></div>
    <div style="text-align:center;"><div style="font-size:32px;font-weight:800;color:#FFFFFF;">Prophet</div><div style="font-size:10px;color:#7BAFD4;font-family:'JetBrains Mono',monospace;margin-top:4px;">ML FORECAST</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Sumber data badges
    st.markdown("""
<div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-bottom:32px;">
  <span style="background:#EBF3FB;border:1px solid #C0D8EE;color:#1E6BB8;font-family:'JetBrains Mono',monospace;font-size:10px;padding:5px 12px;border-radius:4px;letter-spacing:0.08em;">🌊 CMEMS · Arus/SST/Salinitas</span>
  <span style="background:#EBF3FB;border:1px solid #C0D8EE;color:#1E6BB8;font-family:'JetBrains Mono',monospace;font-size:10px;padding:5px 12px;border-radius:4px;letter-spacing:0.08em;">🛰 NASA MODIS · Klorofil-a</span>
  <span style="background:#EBF3FB;border:1px solid #C0D8EE;color:#1E6BB8;font-family:'JetBrains Mono',monospace;font-size:10px;padding:5px 12px;border-radius:4px;letter-spacing:0.08em;">💨 ERA5/ECMWF · Angin</span>
  <span style="background:#EBF3FB;border:1px solid #C0D8EE;color:#1E6BB8;font-family:'JetBrains Mono',monospace;font-size:10px;padding:5px 12px;border-radius:4px;letter-spacing:0.08em;">🌧 BMKG · Gelombang</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center;margin-bottom:24px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#7BAFD4;text-transform:uppercase;letter-spacing:0.18em;margin-bottom:8px;">Pilih Mode Tampilan</div>
  <div style="font-size:20px;font-weight:700;color:#0D1F33;">Masuk sebagai siapa?</div>
</div>
""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
<div style="background:#FFFFFF;border:1px solid #D6E4F0;border-top:3px solid #1E6BB8;border-radius:8px;padding:32px 28px;text-align:center;margin-bottom:12px;">
  <div style="font-size:40px;margin-bottom:16px;">🎣</div>
  <h3 style="color:#0D1F33;font-size:18px;font-weight:700;margin:0 0 8px;">Nelayan Lokal</h3>
  <p style="color:#5A7FA0;font-size:14px;margin:0 0 16px;line-height:1.6;">Peta zona tangkap real-time, kondisi gelombang, dan rekomendasi area melaut hari ini.</p>
  <div style="display:flex;justify-content:center;gap:6px;flex-wrap:wrap;">
    <span style="background:#EBF3FB;color:#1E6BB8;font-size:10px;padding:3px 8px;border-radius:3px;font-family:'JetBrains Mono',monospace;border:1px solid #C0D8EE;">Zona Tangkap</span>
    <span style="background:#EBF3FB;color:#1E6BB8;font-size:10px;padding:3px 8px;border-radius:3px;font-family:'JetBrains Mono',monospace;border:1px solid #C0D8EE;">Gelombang Live</span>
    <span style="background:#EBF3FB;color:#1E6BB8;font-size:10px;padding:3px 8px;border-radius:3px;font-family:'JetBrains Mono',monospace;border:1px solid #C0D8EE;">Arus Real-Time</span>
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("Masuk sebagai Nelayan →", use_container_width=True, key="btn_nelayan"):
            st.session_state.role = "nelayan"
            st.session_state.page = "dashboard"
            st.rerun()

    with c2:
        st.markdown("""
<div style="background:#FFFFFF;border:1px solid #D6E4F0;border-top:3px solid #2E7DC4;border-radius:8px;padding:32px 28px;text-align:center;margin-bottom:12px;">
  <div style="font-size:40px;margin-bottom:16px;">🔬</div>
  <h3 style="color:#0D1F33;font-size:18px;font-weight:700;margin:0 0 8px;">Akademisi / Peneliti</h3>
  <p style="color:#5A7FA0;font-size:14px;margin:0 0 16px;line-height:1.6;">12 parameter oseanografi, time series 20 tahun, prediksi Prophet, matriks korelasi, dan analisis spasial.</p>
  <div style="display:flex;justify-content:center;gap:6px;flex-wrap:wrap;">
    <span style="background:#EBF3FB;color:#1E6BB8;font-size:10px;padding:3px 8px;border-radius:3px;font-family:'JetBrains Mono',monospace;border:1px solid #C0D8EE;">Prophet Forecast</span>
    <span style="background:#EBF3FB;color:#1E6BB8;font-size:10px;padding:3px 8px;border-radius:3px;font-family:'JetBrains Mono',monospace;border:1px solid #C0D8EE;">Korelasi</span>
    <span style="background:#EBF3FB;color:#1E6BB8;font-size:10px;padding:3px 8px;border-radius:3px;font-family:'JetBrains Mono',monospace;border:1px solid #C0D8EE;">Time Series</span>
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("Masuk sebagai Akademisi →", use_container_width=True, key="btn_akademisi"):
            st.session_state.role = "akademisi"
            st.session_state.page = "dashboard"
            st.rerun()

    st.markdown("""
<div style="background:#FFFFFF;border:1px solid #D6E4F0;border-radius:8px;padding:28px 32px;text-align:center;margin-top:32px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#1E6BB8;text-transform:uppercase;letter-spacing:0.18em;margin-bottom:20px;">Tim Pengembang</div>
  <div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap;align-items:center;">
    <div style="text-align:center;"><div style="width:48px;height:48px;border-radius:50%;background:#0D1F33;margin:0 auto 10px;display:flex;align-items:center;justify-content:center;font-size:22px;">👩</div>
      <div style="font-weight:600;color:#0D1F33;font-size:13px;">Ratu Salwa Ghazalia Hade</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#5A7FA0;margin-top:2px;">12923016</div></div>
    <div style="width:1px;height:56px;background:#D6E4F0;"></div>
    <div style="text-align:center;"><div style="width:48px;height:48px;border-radius:50%;background:#0D1F33;margin:0 auto 10px;display:flex;align-items:center;justify-content:center;font-size:22px;">👩</div>
      <div style="font-weight:600;color:#0D1F33;font-size:13px;">Diandra Aulia Ramadhani</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#5A7FA0;margin-top:2px;">12923021</div></div>
    <div style="width:1px;height:56px;background:#D6E4F0;"></div>
    <div style="text-align:center;"><div style="width:48px;height:48px;border-radius:50%;background:#0D1F33;margin:0 auto 10px;display:flex;align-items:center;justify-content:center;font-size:22px;">👩</div>
      <div style="font-weight:600;color:#0D1F33;font-size:13px;">Mutiara Nurani</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#5A7FA0;margin-top:2px;">12923023</div></div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align:center;margin-top:24px;"><div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#A8C0D8;letter-spacing:0.1em;">SUMBER DATA: CMEMS · MODIS-Aqua · ERA5 · BMKG · 2001–2020</div></div>""", unsafe_allow_html=True)
    st.stop()

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.markdown("""
<div style="padding:20px 0 12px;text-align:center;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:#FFFFFF;letter-spacing:0.12em;text-transform:uppercase;">LAUTAN</div>
  <div style="font-size:11px;color:#7BAFD4;margin-top:3px;letter-spacing:0.04em;">Ocean Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("← Kembali ke Beranda", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    st.markdown("---")

    # ── Indikator update terakhir ──
    if st.session_state.last_update:
        wib = st.session_state.last_update + datetime.timedelta(hours=7)
        st.markdown(f"""
<div class="update-badge">
  <span class="pulse"></span>
  LIVE · {wib.strftime('%d %b %Y %H:%M')} WIB
</div>
""", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    mode = st.selectbox("MODE DATA", ["Historis", "Real Time", "Prediksi"])
    st.markdown("---")

    if mode == "Historis":
        tahun = st.selectbox("TAHUN", sorted(df["year"].unique(), reverse=True))
        breakdown = st.radio("RESOLUSI WAKTU", ["Bulanan", "Musiman"])
        musim_map_dict = {
            "Musim Barat":  [12,1,2], "Peralihan I": [3,4,5],
            "Musim Timur":  [6,7,8],  "Peralihan II":[9,10,11],
        }
        df_filter_base = df[df["year"] == tahun].copy()
        if breakdown == "Bulanan":
            bln_list = ["Januari","Februari","Maret","April","Mei","Juni",
                        "Juli","Agustus","September","Oktober","November","Desember"]
            bulan = st.selectbox("BULAN", bln_list)
            idx_bulan = bln_list.index(bulan) + 1
            df_filter_base = df_filter_base[df_filter_base["month"] == idx_bulan]
            waktu_label = f"{bulan} {tahun}"
        else:
            musim_pilih = st.selectbox("MUSIM", list(musim_map_dict.keys()))
            df_filter_base = df_filter_base[df_filter_base["month"].isin(musim_map_dict[musim_pilih])]
            waktu_label = f"{musim_pilih} {tahun}"

    elif mode == "Real Time":
        st.markdown("""
<div class="data-note">Data langsung dari CMEMS, NASA MODIS, ERA5, dan BMKG. Klik tombol di bawah untuk memuat data terbaru.</div>
""", unsafe_allow_html=True)
        if st.button("🔄 Muat Data Real-Time", use_container_width=True, key="btn_rt"):
            with st.spinner("Menghubungi API..."):
                result = build_realtime_dataframe(
                    cmems_user=CMEMS_USER, cmems_pass=CMEMS_PASS,
                    nasa_user=NASA_USER,   nasa_pass=NASA_PASS,
                )
                if result["data"] is not None:
                    st.session_state.rt_data    = result["data"]
                    st.session_state.api_status = result["status"]
                    st.session_state.last_update = datetime.datetime.utcnow()
                    st.success("Data berhasil dimuat!")
                else:
                    st.warning("API tidak tersedia. Menggunakan estimasi klimatologis.")
                    st.session_state.rt_data    = None
                    st.session_state.api_status = result["status"]

        # Tampilkan status tiap API
        if st.session_state.api_status:
            st.markdown("**Status API:**")
            for api_name, ok in st.session_state.api_status.items():
                icon = "🟢" if ok else "🔴"
                st.markdown(f"<span style='font-size:11px;font-family:monospace'>{icon} {api_name}</span>", unsafe_allow_html=True)

        # Fallback jika belum ada data RT
        if st.session_state.rt_data is not None:
            df_filter_base = st.session_state.rt_data
        else:
            now = datetime.datetime.utcnow()
            df_filter_base = df[df["month"] == now.month].copy()

        waktu_label = f"Real-Time · {datetime.datetime.utcnow().strftime('%d %b %Y %H:%M')} UTC"

    else:  # Prediksi
        st.markdown("""
<div class="data-note">Prediksi menggunakan model Prophet (Meta) dilatih pada data historis 2001–2020.</div>
""", unsafe_allow_html=True)
        bulan_pred = st.selectbox("TARGET PREDIKSI",
            ["Juli 2026","Agustus 2026","September 2026","Oktober 2026",
             "November 2026","Desember 2026"])
        horizon_map = {
            "Juli 2026":9,"Agustus 2026":10,"September 2026":11,
            "Oktober 2026":12,"November 2026":13,"Desember 2026":14,
        }
        horizon_months = horizon_map[bulan_pred]
        idx_p = {"Juli":7,"Agustus":8,"September":9,"Oktober":10,"November":11,"Desember":12}
        month_idx = idx_p[bulan_pred.split()[0]]
        df_filter_base = df[df["month"] == month_idx].copy()
        waktu_label = f"Proyeksi Prophet · {bulan_pred}"

    st.markdown("---")
    if st.session_state.role == "akademisi":
        PARAM_LABELS = {
            "Ocean_Health_Index":"Ocean Health Index","Fisheries_Index":"Fisheries Index",
            "sst":"Sea Surface Temp (SST)","ssta":"SST Anomali","ph":"pH Laut",
            "do":"Dissolved Oxygen","salinitas":"Salinitas","chla":"Klorofil-a",
            "current_speed":"Kecepatan Arus","gelombang":"Tinggi Gelombang",
            "angin_u":"Angin Zonal (U)","angin_v":"Angin Meridional (V)",
        }
        parameter = st.selectbox("PARAMETER RISET", list(PARAM_LABELS.keys()),
                                  format_func=lambda x: PARAM_LABELS[x])

    st.markdown("""
<div style="background:#162A42;border-radius:6px;padding:12px 14px;margin-top:12px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#7BAFD4;letter-spacing:0.08em;line-height:1.7;">
    📍 4°S – 12°S / 129°E – 144°E<br>🌏 Laut Arafura · Papua
  </div>
</div>
""", unsafe_allow_html=True)

# =========================================
# BUILD SPATIAL GRID
# =========================================
val_uo_base  = float(df_filter_base["uo"].mean()   if not df_filter_base.empty else -0.05)
val_vo_base  = float(df_filter_base["vo"].mean()   if not df_filter_base.empty else -0.01)
active_month = int(df_filter_base["month"].mean()) if not df_filter_base.empty else 6
active_year  = int(df_filter_base["year"].mean())  if not df_filter_base.empty else 2010

df_map = build_spatial_grid(val_uo_base, val_vo_base, active_month, active_year)

# =========================================
# DASHBOARD — NELAYAN
# =========================================
if st.session_state.role == "nelayan":
    mean_fsi = float(df_map["Fisheries_Index"].mean())
    status   = get_fisheries_status(mean_fsi, df_map)

    st.markdown(f"""
<div class="page-header">
  <div class="eyebrow">🎣 Dashboard Nelayan · {mode} · {waktu_label}</div>
  <h1>Peta Zona Tangkap — Laut Arafura</h1>
</div>
""", unsafe_allow_html=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns([2,1,1,1])
    with col_s1:
        st.markdown(f"""
<div style="background:{status['bg']};border:1px solid {status['border']};border-left:4px solid {status['color']};border-radius:8px;padding:18px 20px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">Status Zona Melaut</div>
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:20px;">{status['icon']}</span>
    <span style="color:{status['color']};font-size:20px;font-weight:700;">{status['text']}</span>
    <span style="font-family:'JetBrains Mono',monospace;color:#8ABDD4;font-size:12px;">FSI {mean_fsi:.1f}/100</span>
  </div>
</div>
""", unsafe_allow_html=True)
    with col_s2: st.metric("Potensi Ikan (FSI)", f"{mean_fsi:.0f}", "/ 100")
    with col_s3: st.metric("Tinggi Gelombang",   f"{df_map['gelombang'].mean():.2f}", "m")
    with col_s4: st.metric("Kec. Arus",           f"{df_map['current_speed'].mean():.3f}", "m/s")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">PETA DISTRIBUSI SPASIAL · FISHERIES INDEX</div>', unsafe_allow_html=True)
    st.plotly_chart(render_map(df_map, "Fisheries_Index", "Turbo"), use_container_width=True)

    st.markdown('<div class="section-label">ROSE DIAGRAM — ANGIN & GELOMBANG</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    df_rose_src = df_filter_base if not df_filter_base.empty else df
    with rc1:
        st.plotly_chart(make_wind_rose(df_rose_src, f"Angin · {waktu_label}"), use_container_width=True)
    with rc2:
        st.plotly_chart(make_wave_rose(df_rose_src, f"Gelombang · {waktu_label}"), use_container_width=True)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown('<div class="section-label">REKOMENDASI ZONA</div>', unsafe_allow_html=True)
        if status["text"] == "SANGAT BAIK":
            st.success("**Area oranye/merah direkomendasikan.** Nutrisi laut melimpah — turunkan jaring di perairan dalam Arafura.")
        elif status["text"] == "NORMAL":
            st.info("**Kondisi normal.** Ikan bergerak mengikuti arus — ikuti arah arus ke tenggara.")
        else:
            st.warning("**Potensi tangkapan rendah.** Disarankan memancing di pesisir dekat teluk dan muara sungai.")
    with col_r2:
        st.markdown('<div class="section-label">KONDISI PERAIRAN</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="background:#FFFFFF;border:1px solid #D6E4F0;border-radius:8px;padding:20px;">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div style="text-align:center;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Klorofil-a</div>
      <div style="font-size:22px;font-weight:700;color:#0D1F33;">{df_map['chla'].mean():.3f} <span style="font-size:12px;color:#5A7FA0;">mg/m³</span></div>
    </div>
    <div style="text-align:center;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Dissolved O₂</div>
      <div style="font-size:22px;font-weight:700;color:#0D1F33;">{df_map['do'].mean():.2f} <span style="font-size:12px;color:#5A7FA0;">mg/L</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================================
# DASHBOARD — AKADEMISI
# =========================================
else:
    PARAM_LABELS_CLEAN = {
        "Ocean_Health_Index":"Ocean Health Index","Fisheries_Index":"Fisheries Index",
        "sst":"Sea Surface Temp (SST)","ssta":"SST Anomali","ph":"pH Laut",
        "do":"Dissolved Oxygen","salinitas":"Salinitas","chla":"Klorofil-a",
        "current_speed":"Kecepatan Arus","gelombang":"Tinggi Gelombang",
        "angin_u":"Angin Zonal (U)","angin_v":"Angin Meridional (V)",
    }

    st.markdown(f"""
<div class="page-header">
  <div class="eyebrow">🔬 Portal Akademisi · {mode} · {waktu_label}</div>
  <h1>Analisis Parameter — {PARAM_LABELS_CLEAN.get(parameter, parameter)}</h1>
</div>
""", unsafe_allow_html=True)

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Rata-Rata", f"{df_map[parameter].mean():.3f}")
    col2.metric("Minimum",   f"{df_map[parameter].min():.3f}")
    col3.metric("Maksimum",  f"{df_map[parameter].max():.3f}")
    col4.metric("Std. Dev",  f"{df_map[parameter].std():.3f}")
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    PARAM_TERARAH   = ["angin_u","angin_v","gelombang"]
    tampilkan_rose  = parameter in PARAM_TERARAH
    tabs_labels = ["  Spasial  ","  Time Series  ","  Prediksi Prophet  ","  Statistik  ","  Korelasi  "]
    if tampilkan_rose: tabs_labels.append("  Rose Diagram  ")
    tabs = st.tabs(tabs_labels)

    # ── Tab 0: Spasial ──────────────────────────────────────
    with tabs[0]:
        cmap_dict = {
            'Fisheries_Index':'Turbo','chla':'Turbo','Ocean_Health_Index':'Blues',
            'do':'Blues','ph':'Viridis','salinitas':'YlOrBr','sst':'Plasma',
            'ssta':'RdBu','current_speed':'Teal',
        }
        cmap = cmap_dict.get(parameter, "Icefire")
        st.markdown(f'<div class="section-label">DISTRIBUSI SPASIAL · {parameter.upper()}</div>', unsafe_allow_html=True)
        st.plotly_chart(render_map(df_map, parameter, cmap, height=500), use_container_width=True)
        st.markdown(f"""<span class="coord-tag">4°S – 12°S</span><span class="coord-tag">129°E – 144°E</span><span class="coord-tag">Grid 100×80 · Laut Arafura</span>""", unsafe_allow_html=True)

    # ── Tab 1: Time Series ──────────────────────────────────
    with tabs[1]:
        df_ts = df.groupby("time")[parameter].mean().reset_index()
        y_vals = df_ts[parameter].to_numpy(dtype=float)
        z = np.polyfit(range(len(df_ts)), y_vals, 1)
        y_trend = np.poly1d(z)(range(len(df_ts)))
        y_lo = float(min(y_vals.min(), y_trend.min()))
        y_hi = float(max(y_vals.max(), y_trend.max()))
        span = y_hi - y_lo
        pad  = span * 0.08 if span > 0 else (abs(y_hi) * 0.08 if y_hi else 1.0)

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=df_ts["time"], y=y_vals, mode="lines",
            name=PARAM_LABELS_CLEAN.get(parameter, parameter),
            line=dict(color="#1E6BB8", width=1.8)))
        fig_ts.add_trace(go.Scatter(x=df_ts["time"], y=y_trend, mode="lines",
            name="Tren Linear", line=dict(color="#D4811A", width=2, dash="dot")))
        fig_ts.update_layout(**PLOTLY_LAYOUT,
            title=f"Tren Temporal 2001–2020 · {PARAM_LABELS_CLEAN.get(parameter,parameter)}",
            legend=dict(font=dict(color="#3A5070",size=11), bgcolor="rgba(255,255,255,0.9)", bordercolor="#D6E4F0", borderwidth=1),
            height=400)
        fig_ts.update_yaxes(range=[y_lo-pad, y_hi+pad], autorange=False)
        st.plotly_chart(fig_ts, use_container_width=True)
        st.caption(f"✓ Skala Y: {y_lo-pad:.3f} – {y_hi+pad:.3f}")

    # ── Tab 2: Prediksi Prophet ─────────────────────────────
    with tabs[2]:
        st.markdown('<div class="section-label">PREDIKSI PROPHET — FORECAST 12 BULAN KE DEPAN</div>', unsafe_allow_html=True)

        col_ph1, col_ph2 = st.columns([3, 1])
        with col_ph2:
            horizon_months_ui = st.slider("Horizon (bulan)", 3, 24, 12)
            run_prophet = st.button("▶ Jalankan Prophet", use_container_width=True)

        if run_prophet or f"prophet_{parameter}" in st.session_state:
            with st.spinner(f"Melatih model Prophet untuk {PARAM_LABELS_CLEAN.get(parameter, parameter)}..."):
                forecast_df, model_metrics = run_prophet_forecast(df, parameter, horizon_months_ui)
                st.session_state[f"prophet_{parameter}"] = (forecast_df, model_metrics)

        if f"prophet_{parameter}" in st.session_state:
            forecast_df, model_metrics = st.session_state[f"prophet_{parameter}"]

            # Plot forecast
            fig_fc = go.Figure()
            # Data historis
            df_ts2 = df.groupby("time")[parameter].mean().reset_index()
            fig_fc.add_trace(go.Scatter(x=df_ts2["time"], y=df_ts2[parameter],
                mode="lines", name="Data Historis",
                line=dict(color="#1E6BB8", width=1.5)))
            # Prediksi
            future_only = forecast_df[forecast_df["ds"] > df["time"].max()]
            fig_fc.add_trace(go.Scatter(x=future_only["ds"], y=future_only["yhat"],
                mode="lines", name="Prediksi Prophet",
                line=dict(color="#E85A0C", width=2.5)))
            # Confidence interval
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([future_only["ds"], future_only["ds"][::-1]]),
                y=pd.concat([future_only["yhat_upper"], future_only["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(232,90,12,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Interval Kepercayaan 80%"))
            fig_fc.update_layout(**PLOTLY_LAYOUT,
                title=f"Prediksi Prophet · {PARAM_LABELS_CLEAN.get(parameter,parameter)} · {horizon_months_ui} Bulan",
                legend=dict(font=dict(color="#3A5070",size=11), bgcolor="rgba(255,255,255,0.9)", bordercolor="#D6E4F0", borderwidth=1),
                height=430)
            with col_ph1:
                st.plotly_chart(fig_fc, use_container_width=True)

            # Metrik model
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("MAE",   f"{model_metrics['mae']:.4f}")
            m2.metric("RMSE",  f"{model_metrics['rmse']:.4f}")
            m3.metric("MAPE",  f"{model_metrics['mape']:.1f}%")
            m4.metric("R²",    f"{model_metrics['r2']:.4f}")

            # Tabel prediksi
            st.markdown('<div class="section-label">TABEL PREDIKSI BULANAN</div>', unsafe_allow_html=True)
            tbl = future_only[["ds","yhat","yhat_lower","yhat_upper"]].copy()
            tbl.columns = ["Periode","Prediksi","Batas Bawah (80%)","Batas Atas (80%)"]
            tbl["Periode"] = tbl["Periode"].dt.strftime("%B %Y")
            for c in ["Prediksi","Batas Bawah (80%)","Batas Atas (80%)"]:
                tbl[c] = tbl[c].round(4)
            st.dataframe(tbl, use_container_width=True, hide_index=True)

            # Komponen Prophet
            st.markdown('<div class="section-label">KOMPONEN MODEL PROPHET</div>', unsafe_allow_html=True)
            comp_cols = [c for c in ["trend","yearly","weekly"] if c in forecast_df.columns]
            if comp_cols:
                fig_comp = go.Figure()
                colors_comp = ["#1E6BB8","#E85A0C","#00895A"]
                for i, comp in enumerate(comp_cols):
                    fig_comp.add_trace(go.Scatter(
                        x=forecast_df["ds"], y=forecast_df[comp],
                        mode="lines", name=comp.capitalize(),
                        line=dict(color=colors_comp[i % len(colors_comp)], width=1.8)))
                fig_comp.update_layout(**PLOTLY_LAYOUT, height=300,
                    title="Dekomposisi Komponen Model",
                    legend=dict(font=dict(color="#3A5070",size=11)))
                st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Klik **▶ Jalankan Prophet** untuk memulai prediksi.")
            st.markdown("""
<div class="data-note">Model Prophet akan dilatih pada data historis 2001–2020 dan memproyeksikan nilai parameter ke depan. Proses sekitar 5–15 detik per parameter.</div>
""", unsafe_allow_html=True)

    # ── Tab 3: Statistik ────────────────────────────────────
    with tabs[3]:
        desc = df_map[[parameter]].describe()
        st.markdown('<div class="section-label">STATISTIK DESKRIPTIF · AREA SPASIAL AKTIF</div>', unsafe_allow_html=True)
        stats_cols = st.columns(4)
        for i, (label, key, icon) in enumerate([
            ("Count","count","N"),("Mean","mean","μ"),("Std","std","σ"),("Min","min","↓"),
        ]):
            val = f"{int(desc.loc[key, parameter]):,}" if key == "count" else f"{desc.loc[key, parameter]:.4f}"
            stats_cols[i].markdown(f"""
<div style="background:#FFFFFF;border:1px solid #D6E4F0;border-radius:8px;padding:18px;text-align:center;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;color:#1E6BB8;margin-bottom:4px;">{icon}</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;letter-spacing:0.1em;text-transform:uppercase;">{label}</div>
  <div style="font-size:18px;font-weight:700;color:#0D1F33;margin-top:4px;">{val}</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        stats_cols2 = st.columns(3)
        for i, (label, key) in enumerate([("25th Percentile","25%"),("Median (50%)","50%"),("75th Percentile","75%")]):
            stats_cols2[i].markdown(f"""
<div style="background:#FFFFFF;border:1px solid #D6E4F0;border-radius:8px;padding:18px;text-align:center;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">{label}</div>
  <div style="font-size:18px;font-weight:700;color:#0D1F33;">{desc.loc[key, parameter]:.4f}</div>
</div>
""", unsafe_allow_html=True)

    # ── Tab 4: Korelasi ─────────────────────────────────────
    with tabs[4]:
        DROP_COLS = ["year","month","lat","lon","latitude","longitude","index","time","id"]
        numeric_df = df.select_dtypes(include=np.number).drop(columns=DROP_COLS, errors="ignore")
        SHORT_CORR = {
            "uo":"UO","vo":"VO","sst":"SST","ssta":"SSTA","ph":"pH","do":"DO",
            "salinitas":"SAL","chla":"CHL-a","gelombang":"WAVE","current_speed":"CurSpd",
            "angin_u":"WindU","angin_v":"WindV","Ocean_Health_Index":"OHI","Fisheries_Index":"FSI",
        }
        corr = numeric_df.corr().rename(index=SHORT_CORR, columns=SHORT_CORR)
        labels = list(corr.columns)
        n = len(labels); vals = corr.values
        zmin, zmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        rng_z = (zmax - zmin) or 1.0
        fig_corr = go.Figure(go.Heatmap(
            z=vals, x=labels, y=labels,
            colorscale=[[0,"#EBF3FB"],[0.5,"#5A9EC8"],[1,"#0D1F33"]],
            zmin=zmin, zmax=zmax, xgap=3, ygap=3,
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(thickness=12, len=0.7, tickfont=dict(size=10, family="JetBrains Mono", color="#0D1F33")),
        ))
        annotations = []
        for i in range(n):
            for j in range(n):
                v = vals[i,j]; frac = (v-zmin)/rng_z
                annotations.append(dict(x=labels[j], y=labels[i], text=f"{v:.2f}", showarrow=False,
                    font=dict(size=10, color="#FFFFFF" if frac > 0.55 else "#0D1F33", family="JetBrains Mono")))
        chart_h = max(560, 46*n+170)
        fig_corr.update_layout(
            annotations=annotations,
            title=dict(text="Matriks Korelasi Pearson — Parameter Oseanografi",
                font=dict(color="#0D1F33",size=15,family="Inter"), x=0.01),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=chart_h,
            margin=dict(l=10,r=10,t=60,b=10),
            xaxis=dict(tickangle=45, side="bottom", tickfont=dict(size=12,color="#0D1F33",family="Inter")),
            yaxis=dict(autorange="reversed", tickfont=dict(size=12,color="#0D1F33",family="Inter")),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Tab 5: Rose Diagram (opsional) ──────────────────────
    if tampilkan_rose:
        with tabs[5]:
            st.markdown('<div class="section-label">ROSE DIAGRAM — DISTRIBUSI ARAH & INTENSITAS</div>', unsafe_allow_html=True)
            df_rose_src = df_filter_base if not df_filter_base.empty else df
            if parameter in ("angin_u","angin_v"):
                st.plotly_chart(make_wind_rose(df_rose_src, f"Angin · {waktu_label}"), use_container_width=True)
                st.markdown('<div class="data-note">Rose diagram angin menggunakan konvensi meteorologis (arah dari mana angin datang).</div>', unsafe_allow_html=True)
            else:
                st.plotly_chart(make_wave_rose(df_rose_src, f"Gelombang · {waktu_label}"), use_container_width=True)
                st.markdown('<div class="data-note">Rose diagram gelombang menggunakan tinggi gelombang dengan proxy arah arus permukaan (uo, vo).</div>', unsafe_allow_html=True)
