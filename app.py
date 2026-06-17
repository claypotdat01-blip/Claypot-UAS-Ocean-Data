"""
OCEANA — Platform Intelijen Oseanografi Papua
============================================================
Real-Time  : CMEMS (arus/SST/salinitas/klorofil-a) · ERA5/Open-Meteo (angin) · BMKG (gelombang)
            Chl-a dari CMEMS bio dataset
Historis   : rangkuman_historis_20tahun.csv  (fallback: data sintetis)
Prediksi   : Prediksi (Facebook/Meta)
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

from data_fetcher import build_realtime_dataframe
from predictor import run_prophet_forecast
from spatial import build_spatial_grid, get_ocean_grid_points, normalisasi_global
from config import CMEMS_USER, CMEMS_PASS, CDS_UID, CDS_KEY

st.set_page_config(
    page_title="OCEANA — Ocean Intelligence Platform",
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
.data-note-warn { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #8a5a00; background: #FEF6E8; border-left: 3px solid #D4811A; padding: 8px 12px; border-radius: 0 4px 4px 0; margin-top: 8px; }
.update-badge { display:inline-flex; align-items:center; gap:6px; background:#0A3620; border:1px solid #1E7A45; border-radius:20px; padding:4px 12px; font-family:'JetBrains Mono',monospace; font-size:10px; color:#4DD68A; letter-spacing:0.08em; }
.pulse { width:7px; height:7px; background:#4DD68A; border-radius:50%; animation:pulse 1.5s infinite; display:inline-block; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(1.3)} }
.mode-badge-hist { display:inline-flex;align-items:center;gap:6px;background:#EBF3FB;border:1px solid #C0D8EE;border-radius:20px;padding:4px 14px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#1E6BB8;letter-spacing:0.08em; }
.mode-badge-pred { display:inline-flex;align-items:center;gap:6px;background:#FEF6E8;border:1px solid #F0C070;border-radius:20px;padding:4px 14px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#D4811A;letter-spacing:0.08em; }
.source-pill { display:inline-block;background:#162A42;border:1px solid #2A4A6A;border-radius:4px;padding:2px 8px;font-family:'JetBrains Mono',monospace;font-size:9px;color:#7BAFD4;letter-spacing:0.08em;margin:2px; }
.rt-empty-box { background:#FFFFFF;border:2px dashed #C0D8EE;border-radius:12px;padding:56px 32px;text-align:center; }
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
if "api_errors"  not in st.session_state: st.session_state.api_errors  = {}
if "rt_data"     not in st.session_state: st.session_state.rt_data     = None
if "prev_mode"   not in st.session_state: st.session_state.prev_mode   = None

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
# HELPER: ARAH ARUS & ANGIN
# =========================================
def get_arah_arus(df_src):
    if df_src is None or (hasattr(df_src, "empty") and df_src.empty):
        return "tidak diketahui", "?"
    mean_uo = float(df_src["uo"].mean())
    mean_vo = float(df_src["vo"].mean())
    angle_math = np.degrees(np.arctan2(mean_vo, mean_uo))
    angle_nav  = (90.0 - angle_math + 360.0) % 360.0
    arah_list = [
        ("Utara","↑"),("Utara-Timur Laut","↗"),("Timur Laut","↗"),("Timur-Timur Laut","↗"),
        ("Timur","→"),("Timur-Tenggara","↘"),("Tenggara","↘"),("Selatan-Tenggara","↘"),
        ("Selatan","↓"),("Selatan-Barat Daya","↙"),("Barat Daya","↙"),("Barat-Barat Daya","↙"),
        ("Barat","←"),("Barat-Barat Laut","↖"),("Barat Laut","↖"),("Utara-Barat Laut","↖"),
    ]
    idx = int((angle_nav + 11.25) / 22.5) % 16
    return arah_list[idx]

def get_arah_angin(df_src):
    if df_src is None or (hasattr(df_src, "empty") and df_src.empty):
        return "tidak diketahui", "?"
    mean_u = float(df_src["angin_u"].mean())
    mean_v = float(df_src["angin_v"].mean())
    angle_math = np.degrees(np.arctan2(-mean_v, -mean_u))
    angle_nav  = (90.0 - angle_math + 360.0) % 360.0
    arah_list = [
        ("Utara","↓"),("Utara-Timur Laut","↙"),("Timur Laut","↙"),("Timur-Timur Laut","↙"),
        ("Timur","←"),("Timur-Tenggara","↖"),("Tenggara","↖"),("Selatan-Tenggara","↖"),
        ("Selatan","↑"),("Selatan-Barat Daya","↗"),("Barat Daya","↗"),("Barat-Barat Daya","↗"),
        ("Barat","→"),("Barat-Barat Laut","↘"),("Barat Laut","↘"),("Utara-Barat Laut","↘"),
    ]
    idx = int((angle_nav + 11.25) / 22.5) % 16
    return arah_list[idx]

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
        paper_bgcolor="#FFFFFF", height=380, margin=dict(l=20,r=20,t=40,b=20),
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
        paper_bgcolor="#FFFFFF", height=380, margin=dict(l=20,r=20,t=40,b=20),
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
  <h1 style="font-family:'Inter',sans-serif;font-size:72px;font-weight:800;color:#FFFFFF;letter-spacing:-0.04em;margin:0 0 6px;line-height:1;">OCEANA</h1>
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#7BAFD4;letter-spacing:0.22em;text-transform:uppercase;margin-bottom:24px;">Platform Intelijen Oseanografi Papua</div>
  <div style="color:#A8C0D8;font-size:15px;max-width:560px;margin:0 auto 40px;line-height:1.8;">
    Data real-time multi-API, klimatologi historis 20 tahun,<br>dan prediksi berbasis Prediksi ML untuk kawasan Laut Arafura.
  </div>
  <div style="display:flex;justify-content:center;gap:48px;flex-wrap:wrap;">
    <div style="text-align:center;"><div style="font-size:32px;font-weight:800;color:#FFFFFF;">20+</div><div style="font-size:10px;color:#7BAFD4;font-family:'JetBrains Mono',monospace;margin-top:4px;">TAHUN DATA</div></div>
    <div style="width:1px;background:#1E3A5C;"></div>
    <div style="text-align:center;"><div style="font-size:32px;font-weight:800;color:#FFFFFF;">3</div><div style="font-size:10px;color:#7BAFD4;font-family:'JetBrains Mono',monospace;margin-top:4px;">SUMBER API</div></div>
    <div style="width:1px;background:#1E3A5C;"></div>
    <div style="text-align:center;"><div style="font-size:32px;font-weight:800;color:#FFFFFF;">Historis, Real-Time, Proyeksi</div><div style="font-size:8px;color:#7BAFD4;font-family:'JetBrains Mono',monospace;margin-top:4px;">ML FORECAST</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-bottom:32px;">
  <span style="background:#EBF3FB;border:1px solid #C0D8EE;color:#1E6BB8;font-family:'JetBrains Mono',monospace;font-size:10px;padding:5px 12px;border-radius:4px;letter-spacing:0.08em;">🌊 CMEMS · Arus/SST/Salinitas/Klorofil-a</span>
  <span style="background:#EBF3FB;border:1px solid #C0D8EE;color:#1E6BB8;font-family:'JetBrains Mono',monospace;font-size:10px;padding:5px 12px;border-radius:4px;letter-spacing:0.08em;">💨 ERA5/Open-Meteo · Angin</span>
  <span style="background:#EBF3FB;border:1px solid #C0D8EE;color:#1E6BB8;font-family:'JetBrains Mono',monospace;font-size:10px;padding:5px 12px;border-radius:4px;letter-spacing:0.08em;">🌧 BMKG/Open-Meteo · Gelombang</span>
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
  <p style="color:#5A7FA0;font-size:14px;margin:0 0 16px;line-height:1.6;">12 parameter oseanografi, time series 20 tahun, Prediksi, matriks korelasi, dan analisis spasial.</p>
  <div style="display:flex;justify-content:center;gap:6px;flex-wrap:wrap;">
    <span style="background:#EBF3FB;color:#1E6BB8;font-size:10px;padding:3px 8px;border-radius:3px;font-family:'JetBrains Mono',monospace;border:1px solid #C0D8EE;">Prediksi Forecast</span>
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
    st.markdown("""<div style="text-align:center;margin-top:24px;"><div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#A8C0D8;letter-spacing:0.1em;">SUMBER DATA: CMEMS · ERA5 · BMKG · 2001–2020</div></div>""", unsafe_allow_html=True)
    st.stop()

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.markdown("""
<div style="padding:20px 0 12px;text-align:center;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:#FFFFFF;letter-spacing:0.12em;text-transform:uppercase;">OCEANA</div>
  <div style="font-size:11px;color:#7BAFD4;margin-top:3px;letter-spacing:0.04em;">Ocean Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("← Kembali ke Beranda", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    st.markdown("---")

    mode = st.selectbox("MODE DATA", ["Historis", "Real Time", "Prediksi"])

    if st.session_state.prev_mode != mode:
        keys_to_del = [k for k in st.session_state.keys() if k.startswith("Prediksi_")]
        for k in keys_to_del:
            del st.session_state[k]
        st.session_state.prev_mode = mode

    st.markdown("---")

    # ── SIDEBAR: Historis ────────────────────────────────────
    if mode == "Historis":
        st.markdown("""<div class="data-note">📂 Data klimatologi 2001–2020</div>""", unsafe_allow_html=True)
        tahun = st.selectbox("TAHUN", sorted(df["year"].unique(), reverse=True))
        breakdown = st.radio("RESOLUSI WAKTU", ["Bulanan", "Musiman"])
        musim_map_dict = {
            "Musim Barat":  [12,1,2], "Peralihan I": [3,4,5],
            "Musim Timur":  [6,7,8],  "Peralihan II":[9,10,11],
        }
        df_hist = df[df["year"] == tahun].copy()
        if breakdown == "Bulanan":
            bln_list = ["Januari","Februari","Maret","April","Mei","Juni",
                        "Juli","Agustus","September","Oktober","November","Desember"]
            bulan = st.selectbox("BULAN", bln_list)
            idx_bulan = bln_list.index(bulan) + 1
            df_hist = df_hist[df_hist["month"] == idx_bulan]
            waktu_label = f"{bulan} {tahun}"
        else:
            musim_pilih = st.selectbox("MUSIM", list(musim_map_dict.keys()))
            df_hist = df_hist[df_hist["month"].isin(musim_map_dict[musim_pilih])]
            waktu_label = f"{musim_pilih} {tahun}"

    # ── SIDEBAR: Real Time ───────────────────────────────────
    elif mode == "Real Time":
        st.markdown("""
<div class="data-note">🛰 Data langsung dari:<br>
· CMEMS (arus, SST, salinitas, <b>klorofil-a</b>)<br>
· ERA5/Open-Meteo (angin)<br>
· BMKG/Open-Meteo (gelombang)<br>
<br>
<i>Chl-a kini dari CMEMS Ocean Colour,<br>
""", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.button("🔄 Muat Data Real-Time Sekarang", use_container_width=True, key="btn_rt"):
            with st.spinner("Menghubungi semua API..."):
                result = build_realtime_dataframe(
                    cmems_user=CMEMS_USER,
                    cmems_pass=CMEMS_PASS,
                    cds_uid=CDS_UID,
                    cds_key=CDS_KEY,
                )
                # data_fetcher baru return {"data", "status", "errors"}
                st.session_state.rt_data     = result.get("data")       # DataFrame atau None
                st.session_state.api_status  = result.get("status", {})
                st.session_state.api_errors  = result.get("errors", {})
                st.session_state.last_update = datetime.datetime.utcnow()

        # Tampilkan status koneksi API
        if st.session_state.api_status:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown("**Status Koneksi API:**")
            # Tampilkan semua kecuali NASA MODIS (selalu False, sudah dinonaktifkan)
            for api_name, ok in st.session_state.api_status.items():
                icon  = "🟢" 
                label = "Terhubung" 

        if st.session_state.last_update:
            wib = st.session_state.last_update + datetime.timedelta(hours=7)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(f"""
<div class="update-badge">
  <span class="pulse"></span>
  TERAKHIR: {wib.strftime('%d %b %Y %H:%M')} WIB
</div>
""", unsafe_allow_html=True)

    # ── SIDEBAR: Prediksi ────────────────────────────────────
    else:
        st.markdown("""
<div class="data-note">🤖 Prediksi menggunakan Prediksi (Meta/Facebook) dilatih pada data historis 2001–2020.</div>
""", unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        bulan_pred = st.selectbox("TARGET BULAN PREDIKSI",
            ["Juli 2026","Agustus 2026","September 2026","Oktober 2026",
             "November 2026","Desember 2026"])
        idx_p = {"Juli":7,"Agustus":8,"September":9,"Oktober":10,"November":11,"Desember":12}
        month_idx_pred = idx_p[bulan_pred.split()[0]]

    st.markdown("---")

    PARAM_LABELS = {
        "Ocean_Health_Index":"Ocean Health Index","Fisheries_Index":"Fisheries Index",
        "sst":"Sea Surface Temp (SST)","ssta":"SST Anomali","ph":"pH Laut",
        "do":"Dissolved Oxygen","salinitas":"Salinitas","chla":"Klorofil-a",
        "current_speed":"Kecepatan Arus","gelombang":"Tinggi Gelombang",
        "angin_u":"Angin Zonal (U)","angin_v":"Angin Meridional (V)",
    }
    if st.session_state.role == "akademisi":
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
# BUILD SPATIAL GRID dari data sumber
# =========================================
def build_map_from_df(df_src):
    if df_src is None or (hasattr(df_src, 'empty') and df_src.empty):
        val_uo = -0.05; val_vo = -0.01
        active_month = datetime.datetime.utcnow().month
        active_year  = datetime.datetime.utcnow().year
    else:
        val_uo = float(df_src["uo"].mean())
        val_vo = float(df_src["vo"].mean())
        active_month = int(df_src["month"].mean()) if "month" in df_src.columns else datetime.datetime.utcnow().month
        active_year  = int(df_src["year"].mean())  if "year"  in df_src.columns else datetime.datetime.utcnow().year
    return build_spatial_grid(val_uo, val_vo, active_month, active_year)


# =========================================
#   MODE: HISTORIS
# =========================================
if mode == "Historis":

    df_map = build_map_from_df(df_hist)

    # ── NELAYAN ─────────────────────────────────────────────
    if st.session_state.role == "nelayan":
        mean_fsi = float(df_map["Fisheries_Index"].mean())
        status   = get_fisheries_status(mean_fsi, df_map)
        df_rose_src = df_hist if not df_hist.empty else df
        arah_arus,  ikon_arus  = get_arah_arus(df_rose_src)
        arah_angin, ikon_angin = get_arah_angin(df_rose_src)

        st.markdown(f"""
<div class="page-header">
  <div class="eyebrow">🎣 Nelayan · <span class="mode-badge-hist">📂 HISTORIS</span> &nbsp;{waktu_label}</div>
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

        tabs_n = st.tabs(["  Peta Zona Tangkap  ", "  Rose Diagram  ", "  Rekomendasi  "])

        with tabs_n[0]:
            st.markdown('<div class="section-label">DISTRIBUSI SPASIAL · FISHERIES INDEX · DATA HISTORIS</div>', unsafe_allow_html=True)
            st.plotly_chart(render_map(df_map, "Fisheries_Index", "Turbo"), use_container_width=True)
            st.markdown(f"""<span class="source-pill">📂 Sumber: Data Historis 2001–2020</span><span class="source-pill">📅 Periode: {waktu_label}</span>""", unsafe_allow_html=True)

        with tabs_n[1]:
            st.markdown('<div class="section-label">ROSE DIAGRAM — ANGIN & GELOMBANG</div>', unsafe_allow_html=True)
            rc1, rc2 = st.columns(2)
            with rc1: st.plotly_chart(make_wind_rose(df_rose_src, f"Angin · {waktu_label}"), use_container_width=True)
            with rc2: st.plotly_chart(make_wave_rose(df_rose_src, f"Gelombang · {waktu_label}"), use_container_width=True)

        with tabs_n[2]:
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown('<div class="section-label">REKOMENDASI ZONA</div>', unsafe_allow_html=True)
                if status["text"] == "SANGAT BAIK":
                    st.success(f"**Area oranye/merah direkomendasikan.** Nutrisi laut melimpah — turunkan jaring di perairan dalam Arafura. Arus dominan menuju **{arah_arus}** {ikon_arus}, angin bertiup dari **{arah_angin}** {ikon_angin}.")
                elif status["text"] == "NORMAL":
                    st.info(f"**Kondisi normal.** Ikan bergerak mengikuti arus — ikuti arah arus ke **{arah_arus}** {ikon_arus}. Angin bertiup dari **{arah_angin}** {ikon_angin}, sesuaikan posisi perahu agar tidak melawan arus.")
                else:
                    st.warning(f"**Potensi tangkapan rendah.** Disarankan memancing di pesisir dekat teluk dan muara sungai. Waspadai arus ke **{arah_arus}** {ikon_arus} dan angin dari **{arah_angin}** {ikon_angin}.")
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
    <div style="text-align:center;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Arah Arus</div>
      <div style="font-size:18px;font-weight:700;color:#0D1F33;">{ikon_arus} {arah_arus}</div>
    </div>
    <div style="text-align:center;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Angin Dari</div>
      <div style="font-size:18px;font-weight:700;color:#0D1F33;">{ikon_angin} {arah_angin}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── AKADEMISI ────────────────────────────────────────────
    else:
        st.markdown(f"""
<div class="page-header">
  <div class="eyebrow">🔬 Akademisi · <span class="mode-badge-hist">📂 HISTORIS 2001–2020</span> · {waktu_label}</div>
  <h1>Analisis Parameter — {PARAM_LABELS.get(parameter, parameter)}</h1>
</div>
""", unsafe_allow_html=True)

        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Rata-Rata", f"{df_map[parameter].mean():.3f}")
        col2.metric("Minimum",   f"{df_map[parameter].min():.3f}")
        col3.metric("Maksimum",  f"{df_map[parameter].max():.3f}")
        col4.metric("Std. Dev",  f"{df_map[parameter].std():.3f}")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        PARAM_TERARAH = ["angin_u","angin_v","gelombang"]
        tampilkan_rose = parameter in PARAM_TERARAH
        tabs_labels = ["  Spasial  ","  Time Series  ","  Statistik  ","  Korelasi  "]
        if tampilkan_rose: tabs_labels.append("  Rose Diagram  ")
        tabs = st.tabs(tabs_labels)

        with tabs[0]:
            cmap_dict = {
                'Fisheries_Index':'Turbo','chla':'Turbo','Ocean_Health_Index':'Blues',
                'do':'Blues','ph':'Viridis','salinitas':'YlOrBr','sst':'Plasma',
                'ssta':'RdBu','current_speed':'Teal',
            }
            cmap = cmap_dict.get(parameter, "Icefire")
            st.markdown(f'<div class="section-label">DISTRIBUSI SPASIAL · {parameter.upper()} · {waktu_label}</div>', unsafe_allow_html=True)
            st.plotly_chart(render_map(df_map, parameter, cmap, height=500), use_container_width=True)
            st.markdown(f"""<span class="coord-tag">4°S – 12°S</span><span class="coord-tag">129°E – 144°E</span><span class="coord-tag">Grid 100×80 · Laut Arafura</span>""", unsafe_allow_html=True)

        with tabs[1]:
            df_ts = df.groupby("time")[parameter].mean().reset_index()
            y_vals  = df_ts[parameter].to_numpy(dtype=float)
            z       = np.polyfit(range(len(df_ts)), y_vals, 1)
            y_trend = np.poly1d(z)(range(len(df_ts)))
            y_lo = float(min(y_vals.min(), y_trend.min()))
            y_hi = float(max(y_vals.max(), y_trend.max()))
            span = y_hi - y_lo
            pad  = span * 0.08 if span > 0 else (abs(y_hi) * 0.08 if y_hi else 1.0)

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=df_ts["time"], y=y_vals, mode="lines",
                name=PARAM_LABELS.get(parameter, parameter),
                line=dict(color="#1E6BB8", width=1.8)))
            fig_ts.add_trace(go.Scatter(x=df_ts["time"], y=y_trend, mode="lines",
                name="Tren Linear", line=dict(color="#D4811A", width=2, dash="dot")))
            fig_ts.update_layout(**PLOTLY_LAYOUT,
                title=f"Tren Temporal 2001–2020 · {PARAM_LABELS.get(parameter,parameter)}",
                legend=dict(font=dict(color="#3A5070",size=11), bgcolor="rgba(255,255,255,0.9)", bordercolor="#D6E4F0", borderwidth=1),
                height=400)
            fig_ts.update_yaxes(range=[y_lo-pad, y_hi+pad], autorange=False)
            st.plotly_chart(fig_ts, use_container_width=True)

        with tabs[2]:
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

        with tabs[3]:
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
                title=dict(text="Matriks Korelasi Pearson — Parameter Oseanografi (Data Historis 2001–2020)",
                    font=dict(color="#0D1F33",size=15,family="Inter"), x=0.01),
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=chart_h,
                margin=dict(l=10,r=10,t=60,b=10),
                xaxis=dict(tickangle=45, side="bottom", tickfont=dict(size=12,color="#0D1F33",family="Inter")),
                yaxis=dict(autorange="reversed", tickfont=dict(size=12,color="#0D1F33",family="Inter")),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        if tampilkan_rose:
            with tabs[4]:
                st.markdown('<div class="section-label">ROSE DIAGRAM — DISTRIBUSI ARAH & INTENSITAS</div>', unsafe_allow_html=True)
                df_rose_src = df_hist if not df_hist.empty else df
                if parameter in ("angin_u","angin_v"):
                    st.plotly_chart(make_wind_rose(df_rose_src, f"Angin · {waktu_label}"), use_container_width=True)
                else:
                    st.plotly_chart(make_wave_rose(df_rose_src, f"Gelombang · {waktu_label}"), use_container_width=True)


# =========================================
#   MODE: REAL TIME
# =========================================
elif mode == "Real Time":

    now_utc = datetime.datetime.utcnow()
    now_wib = now_utc + datetime.timedelta(hours=7)

    # ── Belum ada data sama sekali ───────────────────────────
    if st.session_state.last_update is None:
        st.markdown("""
<div class="page-header">
  <div class="eyebrow">🛰 Mode Real-Time — Laut Arafura</div>
  <h1>Data Oseanografi Terkini</h1>
</div>
""", unsafe_allow_html=True)
        st.markdown("""
<div class="rt-empty-box">
  <div style="font-size:48px;margin-bottom:16px;">🛰</div>
  <div style="font-size:18px;font-weight:700;color:#0D1F33;margin-bottom:8px;">Belum Ada Data Real-Time</div>
  <div style="color:#5A7FA0;font-size:14px;margin-bottom:24px;line-height:1.6;">
    Klik tombol <b>🔄 Muat Data Real-Time Sekarang</b> di sidebar kiri<br>
    untuk mengambil data terbaru dari semua sumber API.
  </div>
  <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;">
    <span style="background:#EBF3FB;border:1px solid #C0D8EE;color:#1E6BB8;font-family:'JetBrains Mono',monospace;font-size:10px;padding:4px 12px;border-radius:4px;">🌊 CMEMS · Arus/SST/Salinitas/Klorofil-a</span>
    <span style="background:#EBF3FB;border:1px solid #C0D8EE;color:#1E6BB8;font-family:'JetBrains Mono',monospace;font-size:10px;padding:4px 12px;border-radius:4px;">💨 ERA5/Open-Meteo · Angin</span>
    <span style="background:#EBF3FB;border:1px solid #C0D8EE;color:#1E6BB8;font-family:'JetBrains Mono',monospace;font-size:10px;padding:4px 12px;border-radius:4px;">🌧 BMKG/Open-Meteo · Gelombang</span>
  </div>
</div>
""", unsafe_allow_html=True)
        st.stop()

    update_wib = (st.session_state.last_update + datetime.timedelta(hours=7)).strftime('%d %b %Y %H:%M') + " WIB"

    # ── Tentukan sumber data yang akan dipakai ───────────────
    # data_fetcher baru: rt_data adalah DataFrame (sudah lengkap) atau None
    df_rt_raw = st.session_state.rt_data   # DataFrame 1 baris atau None

    if df_rt_raw is not None and not df_rt_raw.empty:
        df_rt      = df_rt_raw.copy()
        is_live    = True
        # Hitung berapa API yang berhasil (kecuali NASA MODIS)
        active_apis = {k: v for k, v in st.session_state.api_status.items() if k != "NASA MODIS"}
        n_ok  = sum(1 for v in active_apis.values() if v)
        n_all = len(active_apis)
        sumber_badge = f"🟢 Data langsung dari API ({n_ok}/{n_all} aktif)"
    else:
        # Semua API gagal → fallback klimatologis dari data historis
        df_rt      = df[df["month"] == now_utc.month].copy()
        is_live    = False
        n_ok, n_all = 0, 3
        sumber_badge = "🟡 Estimasi klimatologis (semua API gagal)"

    # ── Pastikan kolom turunan ada ───────────────────────────
    if "current_speed" not in df_rt.columns:
        df_rt["current_speed"] = np.sqrt(df_rt["uo"]**2 + df_rt["vo"]**2)
    if "Ocean_Health_Index" not in df_rt.columns:
        df_rt["Ocean_Health_Index"] = (
            0.25 * normalisasi_global(df_rt["do"],        4.5,  7.5) +
            0.20 * normalisasi_global(df_rt["ph"],        7.9,  8.4) +
            0.20 * normalisasi_global(df_rt["chla"],      0.05, 0.8) +
            0.15 * normalisasi_global(df_rt["salinitas"], 32.0, 36.5) +
            0.20 * (1 - normalisasi_global(df_rt["gelombang"], 0.2, 2.5))
        ) * 100
    if "Fisheries_Index" not in df_rt.columns:
        df_rt["Fisheries_Index"] = (
            0.35 * normalisasi_global(df_rt["chla"],          0.05, 0.8) +
            0.25 * normalisasi_global(df_rt["do"],            4.5,  7.5) +
            0.20 * normalisasi_global(df_rt["current_speed"], 0.0,  0.25) +
            0.20 * (1 - normalisasi_global(df_rt["gelombang"], 0.2, 2.5))
        ) * 100

    # ── Build spatial grid dari nilai rata-rata RT ───────────
    df_map_rt = build_map_from_df(df_rt)

    # ── Hitung arah dari data RT ─────────────────────────────
    arah_arus_rt,  ikon_arus_rt  = get_arah_arus(df_rt)
    arah_angin_rt, ikon_angin_rt = get_arah_angin(df_rt)

    # ── Banner status API (mana yang live, mana fallback) ────
    def _rt_banner(is_live, n_ok, n_all, update_wib, api_status, api_errors):
        """Tampilkan banner hijau (live) atau kuning (fallback klimatologis)."""
        if is_live:
            detail_parts = []
            for api_name, ok in api_status.items():
                if api_name == "NASA MODIS":
                    continue
                icon = "✓" if ok else "✗"
                err = api_errors.get(api_name, "")
                hint = f" ({err[:40]})" if (not ok and err) else ""
                detail_parts.append(f"{icon} {api_name}{hint}")
            detail = " &nbsp;|&nbsp; ".join(detail_parts)
            st.markdown(f"""
<div style="background:#EDFAF3;border:1px solid #9FD9BE;border-left:4px solid #00895A;border-radius:6px;padding:10px 16px;margin-bottom:16px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#00895A;">
  🟢 <b>{n_ok}/{n_all} API aktif</b> · Diperbarui: {update_wib}<br>
  <span style="font-size:10px;color:#2A7050;">{detail}</span><br>
</div>
""", unsafe_allow_html=True)
        else:
            err_hints = []
            for api_name, err in api_errors.items():
                if api_name == "NASA MODIS": continue
                if err: err_hints.append(f"{api_name}: {err[:50]}")
            err_str = " | ".join(err_hints) if err_hints else "—"
            st.markdown(f"""
<div style="background:#FEF6E8;border:1px solid #F0C070;border-left:4px solid #D4811A;border-radius:6px;padding:10px 16px;margin-bottom:16px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#8a5a00;">
  🟡 <b>Semua API tidak merespons</b> — menampilkan estimasi klimatologis bulan {now_utc.strftime('%B')}.<br>
  <span style="font-size:10px;">{err_str}</span><br>
  Tekan 🔄 di sidebar untuk mencoba lagi.
</div>
""", unsafe_allow_html=True)

    # =========================================================
    # REAL TIME — NELAYAN
    # =========================================================
    if st.session_state.role == "nelayan":
        mean_fsi = float(df_map_rt["Fisheries_Index"].mean())
        status   = get_fisheries_status(mean_fsi, df_map_rt)

        st.markdown(f"""
<div class="page-header">
  <div class="eyebrow">
    🎣 Nelayan · <span class="update-badge"><span class="pulse"></span> REAL-TIME</span>
    &nbsp; Diperbarui: {update_wib}
  </div>
  <h1>Kondisi Laut Sekarang — Laut Arafura</h1>
</div>
""", unsafe_allow_html=True)

        _rt_banner(is_live, n_ok, n_all, update_wib,
                   st.session_state.api_status, st.session_state.api_errors)

        col_s1, col_s2, col_s3, col_s4 = st.columns([2,1,1,1])
        with col_s1:
            st.markdown(f"""
<div style="background:{status['bg']};border:1px solid {status['border']};border-left:4px solid {status['color']};border-radius:8px;padding:18px 20px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">Status Zona Melaut Hari Ini</div>
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:20px;">{status['icon']}</span>
    <span style="color:{status['color']};font-size:20px;font-weight:700;">{status['text']}</span>
    <span style="font-family:'JetBrains Mono',monospace;color:#8ABDD4;font-size:12px;">FSI {mean_fsi:.1f}/100</span>
  </div>
</div>
""", unsafe_allow_html=True)
        with col_s2: st.metric("Potensi Ikan (FSI)", f"{mean_fsi:.0f}", "/ 100")
        with col_s3: st.metric("Tinggi Gelombang",   f"{df_map_rt['gelombang'].mean():.2f}", "m")
        with col_s4: st.metric("Kec. Arus",           f"{df_map_rt['current_speed'].mean():.3f}", "m/s")

        tabs_rt = st.tabs(["  Peta Kondisi Terkini  ", "  Detail Parameter  ", "  Rose Diagram  ", "  Rekomendasi  "])

        with tabs_rt[0]:
            st.markdown(f'<div class="section-label">FISHERIES INDEX · {update_wib}</div>', unsafe_allow_html=True)
            st.plotly_chart(render_map(df_map_rt, "Fisheries_Index", "Turbo"), use_container_width=True)
            st.markdown(f"""
<span class="source-pill">{sumber_badge}</span>
<span class="source-pill">📅 {update_wib}</span>
<span class="source-pill">🌊 4°S–12°S / 129°E–144°E</span>
""", unsafe_allow_html=True)

        with tabs_rt[1]:
            st.markdown('<div class="section-label">KONDISI OSEANOGRAFI TERKINI</div>', unsafe_allow_html=True)
            # Tentukan label sumber per parameter berdasarkan status API
            cmems_ok   = st.session_state.api_status.get("CMEMS", False)
            era5_ok    = st.session_state.api_status.get("ERA5/Open-Meteo", False)
            bmkg_ok    = st.session_state.api_status.get("BMKG", False)

            def _src_label(ok, name_live, name_fallback="Klimatologis"):
                return (name_live if ok else name_fallback), ("🟢" if ok else "🟡")

            src_cmems, ic_cmems   = _src_label(cmems_ok, "CMEMS")
            src_era5,  ic_era5   = _src_label(era5_ok,  "ERA5/Open-Meteo")
            src_bmkg,  ic_bmkg   = _src_label(bmkg_ok,  "BMKG/Open-Meteo")

            # Ambil nilai dari df_rt (1 baris rata-rata spasial)
            val_sst  = float(df_rt["sst"].mean())
            val_sal  = float(df_rt["salinitas"].mean())
            val_chla = float(df_rt["chla"].mean())
            val_do   = float(df_rt["do"].mean())
            val_wave = float(df_rt["gelombang"].mean())
            val_wu   = float(df_rt["angin_u"].mean())

            param_rt_list = [
                ("🌡 SST",           f"{val_sst:.2f} °C",    src_cmems, ic_cmems),
                ("🧪 Salinitas",     f"{val_sal:.2f} PSU",   src_cmems, ic_cmems),
                ("🌿 Klorofil-a",    f"{val_chla:.3f} mg/m³",f"{src_cmems} (Ocean Colour)", ic_cmems),
                ("💧 Dissolved O₂",  f"{val_do:.2f} mg/L",   "Derivasi", "⚪"),
                ("🌊 Gelombang",     f"{val_wave:.2f} m",    src_bmkg, ic_bmkg),
                ("💨 Angin U",       f"{val_wu:.2f} m/s",    src_era5, ic_era5),
            ]
            cols_rt = st.columns(3)
            for idx, (label, val, src, ic) in enumerate(param_rt_list):
                with cols_rt[idx % 3]:
                    st.markdown(f"""
<div style="background:#FFFFFF;border:1px solid #D6E4F0;border-radius:8px;padding:16px;margin-bottom:12px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;text-transform:uppercase;letter-spacing:0.1em;">{label}</div>
  <div style="font-size:20px;font-weight:700;color:#0D1F33;margin:6px 0 4px;">{val}</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#5A7FA0;">{ic} {src}</div>
</div>
""", unsafe_allow_html=True)

        with tabs_rt[2]:
            st.markdown('<div class="section-label">ROSE DIAGRAM — ANGIN & GELOMBANG TERKINI</div>', unsafe_allow_html=True)
            rc1, rc2 = st.columns(2)
            with rc1: st.plotly_chart(make_wind_rose(df_rt, f"Angin · {update_wib}"), use_container_width=True)
            with rc2: st.plotly_chart(make_wave_rose(df_rt, f"Gelombang · {update_wib}"), use_container_width=True)

        with tabs_rt[3]:
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown('<div class="section-label">REKOMENDASI ZONA</div>', unsafe_allow_html=True)
                if status["text"] == "SANGAT BAIK":
                    st.success(f"**Area oranye/merah direkomendasikan.** Nutrisi laut melimpah — turunkan jaring di perairan dalam Arafura. Arus dominan menuju **{arah_arus_rt}** {ikon_arus_rt}, angin bertiup dari **{arah_angin_rt}** {ikon_angin_rt}.")
                elif status["text"] == "NORMAL":
                    st.info(f"**Kondisi normal.** Ikan bergerak mengikuti arus — ikuti arah arus ke **{arah_arus_rt}** {ikon_arus_rt}. Angin bertiup dari **{arah_angin_rt}** {ikon_angin_rt}, sesuaikan posisi perahu agar tidak melawan arus.")
                else:
                    st.warning(f"**Potensi tangkapan rendah.** Disarankan memancing di pesisir dekat teluk dan muara sungai. Waspadai arus ke **{arah_arus_rt}** {ikon_arus_rt} dan angin dari **{arah_angin_rt}** {ikon_angin_rt}.")
            with col_r2:
                st.markdown('<div class="section-label">SST TERKINI</div>', unsafe_allow_html=True)
                st.plotly_chart(render_map(df_map_rt, "sst", "Plasma", height=300), use_container_width=True)

    # =========================================================
    # REAL TIME — AKADEMISI
    # =========================================================
    else:
        st.markdown(f"""
<div class="page-header">
  <div class="eyebrow">
    🔬 Akademisi · <span class="update-badge"><span class="pulse"></span> REAL-TIME</span>
    &nbsp; Diperbarui: {update_wib}
  </div>
  <h1>Data Terkini — {PARAM_LABELS.get(parameter, parameter)}</h1>
</div>
""", unsafe_allow_html=True)

        _rt_banner(is_live, n_ok, n_all, update_wib,
                   st.session_state.api_status, st.session_state.api_errors)

        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Rata-Rata", f"{df_map_rt[parameter].mean():.3f}")
        col2.metric("Minimum",   f"{df_map_rt[parameter].min():.3f}")
        col3.metric("Maksimum",  f"{df_map_rt[parameter].max():.3f}")
        col4.metric("Std. Dev",  f"{df_map_rt[parameter].std():.3f}")
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        PARAM_TERARAH = ["angin_u","angin_v","gelombang"]
        tampilkan_rose = parameter in PARAM_TERARAH
        tabs_labels_rt = ["  Spasial Terkini  ","  Perbandingan Historis  ","  Statistik  "]
        if tampilkan_rose: tabs_labels_rt.append("  Rose Diagram  ")
        tabs_rt_ak = st.tabs(tabs_labels_rt)

        with tabs_rt_ak[0]:
            cmap_dict = {
                'Fisheries_Index':'Turbo','chla':'Turbo','Ocean_Health_Index':'Blues',
                'do':'Blues','ph':'Viridis','salinitas':'YlOrBr','sst':'Plasma',
                'ssta':'RdBu','current_speed':'Teal',
            }
            cmap = cmap_dict.get(parameter, "Icefire")
            st.markdown(f'<div class="section-label">DISTRIBUSI SPASIAL · {parameter.upper()} · {update_wib}</div>', unsafe_allow_html=True)
            st.plotly_chart(render_map(df_map_rt, parameter, cmap, height=500), use_container_width=True)
            st.markdown(f"""<span class="source-pill">{sumber_badge}</span><span class="source-pill">📅 {update_wib}</span>""", unsafe_allow_html=True)

        with tabs_rt_ak[1]:
            hist_same_month = df[df["month"] == now_utc.month][parameter]
            rt_val   = float(df_map_rt[parameter].mean())
            hist_val = float(hist_same_month.mean())
            delta    = rt_val - hist_val
            delta_pct = (delta / hist_val * 100) if hist_val != 0 else 0

            st.markdown(f'<div class="section-label">NILAI TERKINI vs RATA-RATA HISTORIS BULAN {now_utc.strftime("%B").upper()}</div>', unsafe_allow_html=True)
            c1c, c2c, c3c = st.columns(3)
            c1c.metric("Nilai Terkini",      f"{rt_val:.4f}",   f"{'+' if delta >= 0 else ''}{delta:.4f} vs historis")
            c2c.metric("Rata-Rata Historis", f"{hist_val:.4f}", f"Bulan {now_utc.strftime('%B')} (2001–2020)")
            c3c.metric("Deviasi (%)",        f"{delta_pct:+.2f}%", "terhadap klimatologi")

            df_ts_rt = df.groupby("time")[parameter].mean().reset_index()
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(
                x=df_ts_rt["time"], y=df_ts_rt[parameter].to_numpy(dtype=float),
                mode="lines", name="Historis 2001–2020",
                line=dict(color="#A8C8E8", width=1.5)))
            fig_compare.add_hline(
                y=rt_val, line_color="#E85A0C", line_width=2.5, line_dash="dash",
                annotation_text=f"Nilai RT: {rt_val:.3f}",
                annotation_font=dict(color="#E85A0C", size=11))
            fig_compare.update_layout(**PLOTLY_LAYOUT,
                title=f"Posisi Nilai Terkini vs Klimatologi · {PARAM_LABELS.get(parameter,parameter)}",
                height=380)
            st.plotly_chart(fig_compare, use_container_width=True)

        with tabs_rt_ak[2]:
            desc = df_map_rt[[parameter]].describe()
            st.markdown('<div class="section-label">STATISTIK DESKRIPTIF · DATA TERKINI</div>', unsafe_allow_html=True)
            sc = st.columns(4)
            for i, (label, key, icon) in enumerate([
                ("Count","count","N"),("Mean","mean","μ"),("Std","std","σ"),("Min","min","↓"),
            ]):
                val = f"{int(desc.loc[key, parameter]):,}" if key == "count" else f"{desc.loc[key, parameter]:.4f}"
                sc[i].markdown(f"""
<div style="background:#FFFFFF;border:1px solid #D6E4F0;border-radius:8px;padding:18px;text-align:center;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;color:#1E6BB8;margin-bottom:4px;">{icon}</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;letter-spacing:0.1em;text-transform:uppercase;">{label}</div>
  <div style="font-size:18px;font-weight:700;color:#0D1F33;margin-top:4px;">{val}</div>
</div>
""", unsafe_allow_html=True)

        if tampilkan_rose:
            with tabs_rt_ak[3]:
                st.markdown('<div class="section-label">ROSE DIAGRAM — DATA TERKINI</div>', unsafe_allow_html=True)
                rc1, rc2 = st.columns(2)
                with rc1: st.plotly_chart(make_wind_rose(df_rt, f"Angin · {update_wib}"), use_container_width=True)
                with rc2: st.plotly_chart(make_wave_rose(df_rt, f"Gelombang · {update_wib}"), use_container_width=True)


# =========================================
#   MODE: PREDIKSI
# =========================================
elif mode == "Prediksi":

    if st.session_state.role == "nelayan":
        st.markdown(f"""
<div class="page-header">
  <div class="eyebrow">🎣 Nelayan · <span class="mode-badge-pred">🤖 Prediksi</span> &nbsp;· {bulan_pred}</div>
  <h1>Proyeksi Kondisi Laut — {bulan_pred}</h1>
</div>
""", unsafe_allow_html=True)

        st.markdown("""<div class="data-note">Model Prediksi dilatih pada 20 tahun data historis (2001–2020).</div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        run_pred_nelayan = st.button("▶ Jalankan Prediksi untuk Bulan Ini", use_container_width=False)
        KEY_PARAMS_NELAYAN = ["Fisheries_Index","chla","gelombang","current_speed"]
        KEY_LABELS_NELAYAN = ["Fisheries Index","Klorofil-a","Tinggi Gelombang","Kecepatan Arus"]

        if run_pred_nelayan:
            with st.spinner("Melatih model Prediksi..."):
                for kp in KEY_PARAMS_NELAYAN:
                    fc, metrics = run_prophet_forecast(df, kp, 12)
                    st.session_state[f"Prediksi_nelayan_{kp}"] = (fc, metrics)

        results_ready = all(f"Prediksi_nelayan_{kp}" in st.session_state for kp in KEY_PARAMS_NELAYAN)

        if results_ready:
            cols_pred = st.columns(4)
            for idx, kp in enumerate(KEY_PARAMS_NELAYAN):
                fc, metrics = st.session_state[f"Prediksi_nelayan_{kp}"]
                future_only = fc[fc["ds"] > df["time"].max()]
                target_row = future_only[future_only["ds"].dt.month == month_idx_pred]
                if not target_row.empty:
                    pred_val = float(target_row["yhat"].iloc[0])
                    lo_val   = float(target_row["yhat_lower"].iloc[0])
                    hi_val   = float(target_row["yhat_upper"].iloc[0])
                    cols_pred[idx].metric(KEY_LABELS_NELAYAN[idx], f"{pred_val:.3f}", f"± {(hi_val - lo_val)/2:.3f}")

            fc_fsi, _ = st.session_state["Prediksi_nelayan_Fisheries_Index"]
            future_only = fc_fsi[fc_fsi["ds"] > df["time"].max()]
            fig_fsi_pred = go.Figure()
            df_ts_fsi = df.groupby("time")["Fisheries_Index"].mean().reset_index()
            fig_fsi_pred.add_trace(go.Scatter(x=df_ts_fsi["time"], y=df_ts_fsi["Fisheries_Index"].to_numpy(dtype=float),
                mode="lines", name="Historis", line=dict(color="#A8C8E8", width=1.5)))
            fig_fsi_pred.add_trace(go.Scatter(x=future_only["ds"], y=future_only["yhat"],
                mode="lines", name="Prediksi", line=dict(color="#E85A0C", width=2.5)))
            fig_fsi_pred.add_trace(go.Scatter(
                x=pd.concat([future_only["ds"], future_only["ds"][::-1]]),
                y=pd.concat([future_only["yhat_upper"], future_only["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(232,90,12,0.12)",
                line=dict(color="rgba(255,255,255,0)"), name="Interval 80%"))
            fig_fsi_pred.update_layout(**PLOTLY_LAYOUT, title=f"Prediksi Fisheries Index · {bulan_pred}", height=380)
            st.plotly_chart(fig_fsi_pred, use_container_width=True)

            df_bulan_pred = df[df["month"] == month_idx_pred]
            arah_arus_pred,  ikon_arus_pred  = get_arah_arus(df_bulan_pred)
            arah_angin_pred, ikon_angin_pred = get_arah_angin(df_bulan_pred)

            fsi_pred_rows = fc_fsi[fc_fsi["ds"].dt.month == month_idx_pred]
            fsi_pred = float(fsi_pred_rows["yhat"].iloc[0]) if not fsi_pred_rows.empty else 50.0
            fsi_p25  = df["Fisheries_Index"].quantile(0.25)
            fsi_p75  = df["Fisheries_Index"].quantile(0.75)

            st.markdown('<div class="section-label">PROYEKSI REKOMENDASI</div>', unsafe_allow_html=True)
            if fsi_pred >= fsi_p75:
                st.success(f"**Proyeksi {bulan_pred}: SANGAT BAIK (FSI {fsi_pred:.1f}/100).** Kondisi laut diprediksi mendukung penangkapan optimal. Arus diprediksi menuju **{arah_arus_pred}** {ikon_arus_pred}, angin dari **{arah_angin_pred}** {ikon_angin_pred}.")
            elif fsi_pred >= fsi_p25:
                st.info(f"**Proyeksi {bulan_pred}: NORMAL (FSI {fsi_pred:.1f}/100).** Tangkapan diperkirakan rata-rata — ikuti arah arus ke **{arah_arus_pred}** {ikon_arus_pred}. Angin diprediksi dari **{arah_angin_pred}** {ikon_angin_pred}.")
            else:
                st.warning(f"**Proyeksi {bulan_pred}: WASPADA (FSI {fsi_pred:.1f}/100).** Pertimbangkan mengurangi aktivitas melaut jauh. Waspadai arus ke **{arah_arus_pred}** {ikon_arus_pred} dan angin dari **{arah_angin_pred}** {ikon_angin_pred}.")
        else:
            st.info("Klik **▶ Jalankan Prediksi** untuk melihat proyeksi kondisi laut.")

    else:
        st.markdown(f"""
<div class="page-header">
  <div class="eyebrow">🔬 Akademisi · <span class="mode-badge-pred">🤖 Prediksi</span> &nbsp;· {bulan_pred}</div>
  <h1>Model Prediksi — {PARAM_LABELS.get(parameter, parameter)}</h1>
</div>
""", unsafe_allow_html=True)

        st.markdown("""<div class="data-note">Model Prediksi (Meta/Facebook) dilatih pada data historis 2001–2020. Prediksi mencakup tren, seasonalitas tahunan, dan interval kepercayaan 80%.</div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        col_ph1, col_ph2 = st.columns([3, 1])
        with col_ph2:
            horizon_months_ui = st.slider("Horizon (bulan)", 3, 24, 12)
            run_Prediksi = st.button("▶ Jalankan Prediksi", use_container_width=True)

        Prediksi_key = f"Prediksi_{parameter}_{horizon_months_ui}"

        if run_Prediksi:
            with st.spinner(f"Melatih model Prediksi untuk {PARAM_LABELS.get(parameter, parameter)}..."):
                forecast_df, model_metrics = run_prophet_forecast(df, parameter, horizon_months_ui)
                st.session_state[Prediksi_key] = (forecast_df, model_metrics)

        if Prediksi_key in st.session_state:
            forecast_df, model_metrics = st.session_state[Prediksi_key]

            tabs_pred = st.tabs(["  Grafik Prediksi  ","  Komponen Model  ","  Tabel Prediksi  ","  Metrik Akurasi  "])

            with tabs_pred[0]:
                fig_fc = go.Figure()
                df_ts2 = df.groupby("time")[parameter].mean().reset_index()
                fig_fc.add_trace(go.Scatter(x=df_ts2["time"], y=df_ts2[parameter].to_numpy(dtype=float),
                    mode="lines", name="Data Historis (2001–2020)", line=dict(color="#1E6BB8", width=1.5)))
                future_only = forecast_df[forecast_df["ds"] > df["time"].max()]
                fig_fc.add_trace(go.Scatter(x=future_only["ds"], y=future_only["yhat"],
                    mode="lines", name="Prediksi", line=dict(color="#E85A0C", width=2.5)))
                fig_fc.add_trace(go.Scatter(
                    x=pd.concat([future_only["ds"], future_only["ds"][::-1]]),
                    y=pd.concat([future_only["yhat_upper"], future_only["yhat_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(232,90,12,0.12)",
                    line=dict(color="rgba(255,255,255,0)"), name="Interval Kepercayaan 80%"))
                target_rows = future_only[future_only["ds"].dt.month == month_idx_pred]
                if not target_rows.empty:
                    fig_fc.add_trace(go.Scatter(x=target_rows["ds"], y=target_rows["yhat"],
                        mode="markers", name=f"Target: {bulan_pred}",
                        marker=dict(color="#E85A0C", size=10, symbol="star")))
                fig_fc.update_layout(**PLOTLY_LAYOUT,
                    title=f"Prediksi · {PARAM_LABELS.get(parameter,parameter)} · Horizon {horizon_months_ui} Bulan",
                    legend=dict(font=dict(color="#3A5070",size=11), bgcolor="rgba(255,255,255,0.9)", bordercolor="#D6E4F0", borderwidth=1),
                    height=430)
                with col_ph1:
                    st.plotly_chart(fig_fc, use_container_width=True)

                if not target_rows.empty:
                    tv = float(target_rows["yhat"].iloc[0])
                    tl = float(target_rows["yhat_lower"].iloc[0])
                    tu = float(target_rows["yhat_upper"].iloc[0])
                    st.markdown(f"""
<div style="background:#FEF6E8;border:1px solid #F0C070;border-left:4px solid #D4811A;border-radius:6px;padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:12px;color:#8a5a00;">
  ⭐ Nilai Prediksi untuk <b>{bulan_pred}</b>: <b>{tv:.4f}</b> &nbsp;|&nbsp; Interval 80%: [{tl:.4f} — {tu:.4f}]
</div>
""", unsafe_allow_html=True)

            with tabs_pred[1]:
                comp_cols = [c for c in ["trend","yearly","weekly"] if c in forecast_df.columns]
                if comp_cols:
                    fig_comp = go.Figure()
                    colors_comp = ["#1E6BB8","#E85A0C","#00895A"]
                    for i, comp in enumerate(comp_cols):
                        fig_comp.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df[comp],
                            mode="lines", name=comp.capitalize(),
                            line=dict(color=colors_comp[i % len(colors_comp)], width=1.8)))
                    fig_comp.update_layout(**PLOTLY_LAYOUT, height=340,
                        title=f"Dekomposisi Komponen Prediksi · {PARAM_LABELS.get(parameter,parameter)}",
                        legend=dict(font=dict(color="#3A5070",size=11)))
                    st.plotly_chart(fig_comp, use_container_width=True)

            with tabs_pred[2]:
                st.markdown('<div class="section-label">TABEL PREDIKSI BULANAN</div>', unsafe_allow_html=True)
                tbl = future_only[["ds","yhat","yhat_lower","yhat_upper"]].copy()
                tbl.columns = ["Periode","Prediksi","Batas Bawah (80%)","Batas Atas (80%)"]
                tbl["Periode"] = tbl["Periode"].dt.strftime("%B %Y")
                for c in ["Prediksi","Batas Bawah (80%)","Batas Atas (80%)"]:
                    tbl[c] = tbl[c].round(4)
                st.dataframe(tbl, use_container_width=True, hide_index=True)

            with tabs_pred[3]:
                st.markdown('<div class="section-label">METRIK AKURASI MODEL</div>', unsafe_allow_html=True)
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("MAE",  f"{model_metrics['mae']:.4f}",  "Mean Absolute Error")
                m2.metric("RMSE", f"{model_metrics['rmse']:.4f}", "Root Mean Sq Error")
                m3.metric("MAPE", f"{model_metrics['mape']:.1f}%","Mean Abs % Error")
                m4.metric("R²",   f"{model_metrics['r2']:.4f}",   "Coefficient of Det.")
        else:
            st.info("Klik **▶ Jalankan Prediksi** di atas untuk memulai prediksi.")
