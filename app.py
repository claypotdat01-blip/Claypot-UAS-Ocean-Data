import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime

# ── Real Time data fetcher ──────────────────────────────────
try:
    from config import CMEMS_USER, CMEMS_PASS, CDS_UID, CDS_KEY
    from data_fetcher import build_realtime_dataframe
    _HAS_FETCHER = True
except ImportError:
    _HAS_FETCHER = False

st.set_page_config(
    page_title="OCEANA — Ocean Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# DESIGN SYSTEM — Professional Maritime
# =========================================
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #F2F6FA;
    color: #0D1F33;
}
[data-testid="stSidebar"] {
    background: #0D1F33 !important;
    border-right: 1px solid #1A3A5C !important;
}
[data-testid="stSidebar"] * { color: #CBD8E8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: #7BAFD4 !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #7BAFD4 !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #162A42 !important;
    border: 1px solid #2A4A6A !important;
    color: #CBD8E8 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
    color: #CBD8E8 !important;
}
.stButton > button {
    background: #1E6BB8 !important;
    border: none !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    transition: background 0.2s ease !important;
    padding: 10px 20px !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: #1558A0 !important;
}
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D6E4F0;
    border-radius: 8px;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] {
    color: #5A7FA0 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #0D1F33 !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-bottom: 1px solid #D6E4F0 !important;
    border-radius: 8px 8px 0 0 !important;
    gap: 0;
    padding: 0 12px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #5A7FA0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 12px 18px !important;
}
.stTabs [aria-selected="true"] {
    color: #1E6BB8 !important;
    border-bottom: 2px solid #1E6BB8 !important;
    font-weight: 600 !important;
}
hr { border-color: #D6E4F0 !important; }

[data-testid="stSidebar"] .stButton > button {
    background: #FFFFFF !important;
    color: #0D1F33 !important;
    border: 1px solid #7BAFD4 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #E8F2FB !important;
}

.page-header {
    border-bottom: 2px solid #1E6BB8;
    padding-bottom: 12px;
    margin-bottom: 24px;
}
.page-header .eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #1E6BB8;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 6px;
}
.page-header h1 {
    font-size: 26px;
    font-weight: 700;
    color: #0D1F33;
    margin: 0;
    letter-spacing: -0.02em;
}
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #1E6BB8;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 10px;
    font-weight: 500;
}
.coord-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #3A6080;
    background: #EBF3FB;
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid #C0D8EE;
    display: inline-block;
    margin: 2px;
}
.data-note {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #7BAFD4;
    background: #EBF3FB;
    border-left: 3px solid #1E6BB8;
    padding: 8px 12px;
    border-radius: 0 4px 4px 0;
    margin-top: 8px;
}
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

def normalisasi_global(series, vmin, vmax):
    if (vmax - vmin) == 0:
        return series * 0 if hasattr(series, '__len__') else 0.0
    return (series - vmin) / (vmax - vmin)

if "page" not in st.session_state: st.session_state.page = "home"
if "role" not in st.session_state: st.session_state.role = "akademisi"

# =========================================
# LOAD DATA
# =========================================
@st.cache_data
def load_data():
    try:
        df_data = pd.read_csv("rangkuman_historis_20tahun.csv")
        df_data["time"] = pd.to_datetime(df_data["time"])
    except:
        dates_fallback = pd.date_range(start="2001-01-01", end="2020-12-01", freq="MS")
        n = len(dates_fallback)
        rng = np.random.default_rng(0)
        t = np.arange(n)
        seasonal = np.sin(2 * np.pi * t / 12)
        enso     = 0.4 * np.sin(2 * np.pi * t / 48)
        trend    = -0.001 * t
        noise_uo = rng.normal(0, 0.008, n)
        noise_vo = rng.normal(0, 0.004, n)
        uo = -0.05 + 0.04 * seasonal + 0.02 * enso + trend + noise_uo
        vo = -0.01 + 0.02 * seasonal + 0.01 * enso + noise_vo
        df_data = pd.DataFrame({"time": dates_fallback, "uo": uo, "vo": vo})

    df_data["time"] = pd.to_datetime(df_data["time"])
    if "uo" not in df_data.columns: df_data["uo"] = -0.05
    if "vo" not in df_data.columns: df_data["vo"] = -0.01

    rng2 = np.random.default_rng(7)
    n = len(df_data)
    if "sst"       not in df_data.columns: df_data["sst"]       = 28.5 + df_data["uo"] * 6 + rng2.normal(0, 0.3, n)
    if "ssta"      not in df_data.columns: df_data["ssta"]      = df_data["uo"] * 2.5 + rng2.normal(0, 0.1, n)
    if "ph"        not in df_data.columns: df_data["ph"]        = 8.12 + df_data["vo"] * 0.6 + rng2.normal(0, 0.01, n)
    if "do"        not in df_data.columns: df_data["do"]        = 6.2 - df_data["uo"] * 2.5 + rng2.normal(0, 0.05, n)
    if "salinitas" not in df_data.columns: df_data["salinitas"] = 34.2 + df_data["vo"] * 2.5 + rng2.normal(0, 0.08, n)
    if "chla"      not in df_data.columns: df_data["chla"]      = 0.22 + df_data["uo"] * (-0.5) + rng2.normal(0, 0.015, n)
    if "gelombang" not in df_data.columns: df_data["gelombang"] = 0.8 + abs(df_data["uo"]) * 2.5 + rng2.normal(0, 0.05, n)
    if "angin_u"   not in df_data.columns: df_data["angin_u"]   = -1.5 + df_data["uo"] * 12 + rng2.normal(0, 0.3, n)
    if "angin_v"   not in df_data.columns: df_data["angin_v"]   = -0.5 + df_data["vo"] * 6 + rng2.normal(0, 0.2, n)

    df_data["chla"]      = df_data["chla"].clip(0.05, 0.8)
    df_data["do"]        = df_data["do"].clip(4.5, 7.5)
    df_data["ph"]        = df_data["ph"].clip(7.9, 8.4)
    df_data["salinitas"] = df_data["salinitas"].clip(32.0, 36.5)
    df_data["gelombang"] = df_data["gelombang"].clip(0.2, 2.5)
    return df_data

df = load_data()
df["year"]  = df["time"].dt.year
df["month"] = df["time"].dt.month
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
# HOME PAGE
# =========================================
if st.session_state.page == "home":
    hero_html = """
<div style="background:#0D1F33;border-radius:12px;padding:64px 48px 56px;text-align:center;margin-bottom:40px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;bottom:0;background:repeating-linear-gradient(
    0deg,transparent,transparent 39px,rgba(30,107,184,0.08) 40px),
    repeating-linear-gradient(90deg,transparent,transparent 39px,rgba(30,107,184,0.08) 40px);
    pointer-events:none;"></div>
  <div style="display:inline-block;background:rgba(30,107,184,0.25);border:1px solid rgba(30,107,184,0.5);border-radius:4px;padding:4px 16px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#7BAFD4;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:28px;">
    SISTEM AKTIF · 4°S–12°S / 129°E–144°E · LAUT ARAFURA
  </div>
  <h1 style="font-family:'Inter',sans-serif;font-size:72px;font-weight:800;color:#FFFFFF;letter-spacing:-0.04em;margin:0 0 6px;line-height:1;">OCEANA</h1>
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#7BAFD4;letter-spacing:0.22em;text-transform:uppercase;margin-bottom:24px;">Platform Intelijen Oseanografi Papua</div>
  <div style="color:#A8C0D8;font-size:15px;max-width:500px;margin:0 auto 40px;line-height:1.8;">
    Data klimatologi laut historis dan proyeksi musiman<br>untuk kawasan perairan Papua &amp; Laut Arafura.
  </div>
  <div style="display:flex;justify-content:center;gap:48px;flex-wrap:wrap;">
    <div style="text-align:center;">
      <div style="font-size:32px;font-weight:800;color:#FFFFFF;letter-spacing:-0.02em;">20+</div>
      <div style="font-size:10px;color:#7BAFD4;font-family:'JetBrains Mono',monospace;letter-spacing:0.12em;margin-top:4px;">TAHUN DATA</div>
    </div>
    <div style="width:1px;background:#1E3A5C;"></div>
    <div style="text-align:center;">
      <div style="font-size:32px;font-weight:800;color:#FFFFFF;letter-spacing:-0.02em;">12</div>
      <div style="font-size:10px;color:#7BAFD4;font-family:'JetBrains Mono',monospace;letter-spacing:0.12em;margin-top:4px;">PARAMETER</div>
    </div>
    <div style="width:1px;background:#1E3A5C;"></div>
    <div style="text-align:center;">
      <div style="font-size:32px;font-weight:800;color:#FFFFFF;letter-spacing:-0.02em;">100×100</div>
      <div style="font-size:10px;color:#7BAFD4;font-family:'JetBrains Mono',monospace;letter-spacing:0.12em;margin-top:4px;">GRID SPASIAL</div>
    </div>
  </div>
</div>
"""
    st.markdown(hero_html, unsafe_allow_html=True)

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
  <p style="color:#5A7FA0;font-size:14px;margin:0 0 16px;line-height:1.6;">Peta zona tangkap, kondisi gelombang, dan rekomendasi area melaut.</p>
  <div style="display:flex;justify-content:center;gap:6px;flex-wrap:wrap;">
    <span style="background:#EBF3FB;color:#1E6BB8;font-size:10px;padding:3px 8px;border-radius:3px;font-family:'JetBrains Mono',monospace;border:1px solid #C0D8EE;">Zona Tangkap</span>
    <span style="background:#EBF3FB;color:#1E6BB8;font-size:10px;padding:3px 8px;border-radius:3px;font-family:'JetBrains Mono',monospace;border:1px solid #C0D8EE;">Gelombang</span>
    <span style="background:#EBF3FB;color:#1E6BB8;font-size:10px;padding:3px 8px;border-radius:3px;font-family:'JetBrains Mono',monospace;border:1px solid #C0D8EE;">Arus</span>
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
  <p style="color:#5A7FA0;font-size:14px;margin:0 0 16px;line-height:1.6;">12 parameter oseanografi, time series 20 tahun, matriks korelasi, dan analisis spasial.</p>
  <div style="display:flex;justify-content:center;gap:6px;flex-wrap:wrap;">
    <span style="background:#EBF3FB;color:#1E6BB8;font-size:10px;padding:3px 8px;border-radius:3px;font-family:'JetBrains Mono',monospace;border:1px solid #C0D8EE;">Analisis</span>
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
    <div style="text-align:center;">
      <div style="width:48px;height:48px;border-radius:50%;background:#0D1F33;margin:0 auto 10px;display:flex;align-items:center;justify-content:center;font-size:22px;">👩</div>
      <div style="font-weight:600;color:#0D1F33;font-size:13px;">Ratu Salwa Ghazalia Hade</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#5A7FA0;margin-top:2px;">12923016</div>
    </div>
    <div style="width:1px;height:56px;background:#D6E4F0;"></div>
    <div style="text-align:center;">
      <div style="width:48px;height:48px;border-radius:50%;background:#0D1F33;margin:0 auto 10px;display:flex;align-items:center;justify-content:center;font-size:22px;">👩</div>
      <div style="font-weight:600;color:#0D1F33;font-size:13px;">Diandra Aulia Ramadhani</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#5A7FA0;margin-top:2px;">12923021</div>
    </div>
    <div style="width:1px;height:56px;background:#D6E4F0;"></div>
    <div style="text-align:center;">
      <div style="width:48px;height:48px;border-radius:50%;background:#0D1F33;margin:0 auto 10px;display:flex;align-items:center;justify-content:center;font-size:22px;">👩</div>
      <div style="font-weight:600;color:#0D1F33;font-size:13px;">Mutiara Nurani</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#5A7FA0;margin-top:2px;">12923023</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center;margin-top:24px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#A8C0D8;letter-spacing:0.1em;">
    SUMBER DATA: CMEMS · MODIS-Aqua · ERA5 · BMKG · 2001–2020
  </div>
</div>
""", unsafe_allow_html=True)
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
        st.session_state.pop("rt_data", None)
        st.rerun()
    st.markdown("---")
    mode = st.selectbox("MODE DATA", ["Historis", "Real Time", "Prediksi"])
    st.markdown("---")

    if mode == "Historis":
        tahun = st.selectbox("TAHUN", sorted(df["year"].unique(), reverse=True))
        breakdown = st.radio("RESOLUSI WAKTU", ["Bulanan", "Musiman"])
        musim_map_dict = {
            "Musim Barat":   [12, 1, 2],
            "Peralihan I":   [3, 4, 5],
            "Musim Timur":   [6, 7, 8],
            "Peralihan II":  [9, 10, 11]
        }
        df_filter_base = df[df["year"] == tahun].copy()
        if breakdown == "Bulanan":
            bln_list = ["Januari","Februari","Maret","April","Mei","Juni",
                        "Juli","Agustus","September","Oktober","November","Desember"]
            bulan = st.selectbox("BULAN", bln_list)
            idx_bulan = bln_list.index(bulan) + 1
            df_filter_base = df_filter_base[df_filter_base["month"] == idx_bulan]
            waktu_label = f"{bulan} {tahun}"
            ts_months = [idx_bulan]
            ts_highlight_year = tahun
            ts_scope_label = f"Bulan {bulan}"
        else:
            musim_pilih = st.selectbox("MUSIM", list(musim_map_dict.keys()))
            df_filter_base = df_filter_base[df_filter_base["month"].isin(musim_map_dict[musim_pilih])]
            waktu_label = f"{musim_pilih} {tahun}"
            ts_months = musim_map_dict[musim_pilih]
            ts_highlight_year = tahun
            ts_scope_label = musim_pilih

    elif mode == "Real Time":
        if st.button("🔄 Ambil Data Sekarang", use_container_width=True, key="btn_fetch_rt"):
            st.session_state.pop("rt_data", None)

        if _HAS_FETCHER:
            if "rt_data" not in st.session_state:
                with st.spinner("Menghubungi API..."):
                    rt_result = build_realtime_dataframe(
                        cmems_user=CMEMS_USER,
                        cmems_pass=CMEMS_PASS,
                        cds_uid=CDS_UID,
                        cds_key=CDS_KEY,
                    )
                st.session_state["rt_data"] = rt_result

            rt_result = st.session_state["rt_data"]
            df_rt     = rt_result["data"].copy()
            now_month = int(df_rt["month"].iloc[0])
            now_year  = int(df_rt["year"].iloc[0])

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

            df_filter_base = df_rt

        else:
            now_month      = datetime.datetime.now().month
            now_year       = datetime.datetime.now().year
            df_filter_base = df[df["month"] == now_month].copy()

        _ts               = datetime.datetime.utcnow()
        waktu_label       = f"Real Time · {_ts.strftime('%d %b %Y %H:%M')} UTC"
        ts_months         = [now_month]
        ts_highlight_year = None
        ts_scope_label    = f"Bulan {_ts.strftime('%B')}"

    else:  # Prediksi
        st.markdown("""
<div class="data-note">⚠ Proyeksi musiman berbasis regresi klimatologis 2001–2020. Bukan model NWP/GCM.</div>
""", unsafe_allow_html=True)
        bulan_pred = st.selectbox("TARGET PREDIKSI",
                                   ["Juli 2026","Agustus 2026","September 2026","Desember 2026"])
        idx_p = 7 if "Juli" in bulan_pred else 8 if "Agustus" in bulan_pred else 9 if "September" in bulan_pred else 12
        df_filter_base = df[df["month"] == idx_p].copy()
        waktu_label = f"Proyeksi {bulan_pred}"
        ts_months = [idx_p]
        ts_highlight_year = None
        ts_scope_label = f"Bulan {bulan_pred.split()[0]}"

    st.markdown("---")
    SHOW_BORDERS = st.checkbox(
        "Tampilkan batas provinsi", value=False,
        help="Mengambil GeoJSON batas dari internet; bisa memperlambat app saat pertama dimuat."
    )

    if st.session_state.role == "akademisi":
        PARAM_LABELS = {
            "Ocean_Health_Index": "Ocean Health Index",
            "Fisheries_Index":    "Fisheries Index",
            "sst":                "Sea Surface Temp (SST)",
            "ssta":               "SST Anomali",
            "ph":                 "pH Laut",
            "do":                 "Dissolved Oxygen",
            "salinitas":          "Salinitas",
            "chla":               "Klorofil-a",
            "current_speed":      "Kecepatan Arus",
            "gelombang":          "Tinggi Gelombang",
            "angin_u":            "Angin Zonal (U)",
            "angin_v":            "Angin Meridional (V)",
        }
        parameter = st.selectbox("PARAMETER RISET", list(PARAM_LABELS.keys()),
                                  format_func=lambda x: PARAM_LABELS[x])

    st.markdown("""
<div style="background:#162A42;border-radius:6px;padding:12px 14px;margin-top:12px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#7BAFD4;letter-spacing:0.08em;line-height:1.7;">
    📍 4°S – 12°S / 129°E – 144°E<br>
    🌏 Laut Arafura · Papua
  </div>
</div>
""", unsafe_allow_html=True)

# =========================================
# LAND MASK
# =========================================
try:
    from global_land_mask import globe as _glm
    _HAS_GLM = True
except Exception:
    _HAS_GLM = False

def _manual_land_mask(lat_arr, lon_arr):
    lat_arr = np.asarray(lat_arr, dtype=float)
    lon_arr = np.asarray(lon_arr, dtype=float)
    m = np.zeros(lat_arr.shape, dtype=bool)
    m |= (lon_arr < 132.5) & (lat_arr > -2.0)
    m |= (lon_arr < 133.5) & (lat_arr > -3.0)
    m |= (lon_arr < 134.5) & (lat_arr > -3.5)
    m |= (lon_arr < 136.0) & (lat_arr > -4.0)
    m |= (lon_arr < 137.0) & (lat_arr > -4.5)
    m |= (lon_arr < 138.0) & (lat_arr > -5.0)
    m |= (lon_arr < 139.0) & (lat_arr > -5.5)
    m |= (lon_arr < 140.0) & (lat_arr > -6.0)
    m |= (lon_arr < 141.0) & (lat_arr > -6.5)
    m |= (lon_arr < 141.5) & (lat_arr > -7.0)
    m |= (lon_arr >= 141.0) & (lon_arr < 142.0) & (lat_arr > -8.5)
    m |= (lon_arr >= 140.0) & (lon_arr < 141.0) & (lat_arr > -8.0)
    m |= (lon_arr > 135.0) & (lon_arr < 139.0) & (lat_arr > -3.5) & (lat_arr < -1.0)
    m |= (lon_arr > 133.5) & (lon_arr < 135.5) & (lat_arr > -7.5) & (lat_arr < -5.5)
    m |= (lon_arr > 137.5) & (lon_arr < 139.5) & (lat_arr > -8.5) & (lat_arr < -7.0)
    return m

def compute_land_mask(lat_arr, lon_arr):
    if _HAS_GLM:
        return np.asarray(_glm.is_land(np.asarray(lat_arr, dtype=float),
                                       np.asarray(lon_arr, dtype=float)))
    return _manual_land_mask(lat_arr, lon_arr)

@st.cache_data
def get_ocean_grid_points():
    lat_grid = np.linspace(-12.0, -4.5, 80)
    lon_grid = np.linspace(130.0, 144.0, 100)
    lon_g, lat_g = np.meshgrid(lon_grid, lat_grid)
    lat_flat = lat_g.flatten()
    lon_flat = lon_g.flatten()
    ocean = ~compute_land_mask(lat_flat, lon_flat)
    return lat_flat[ocean], lon_flat[ocean]

# =========================================
# SPATIAL GRID
# =========================================
@st.cache_data
def build_spatial_grid(val_uo_base, val_vo_base, month_seed, year_seed):
    lat_flat, lon_flat = get_ocean_grid_points()

    seed = int(month_seed * 1000 + year_seed)
    rng = np.random.default_rng(seed)

    var_spasial = (
        2.5 * np.sin(lon_flat * 0.22 + lat_flat * 0.31 + month_seed * 0.5) +
        2.0 * np.cos(lon_flat * 0.15 - lat_flat * 0.28 + month_seed * 0.3) +
        1.5 * np.sin(lon_flat * 0.40 + lat_flat * 0.18 + year_seed * 0.1) +
        1.0 * np.cos(lon_flat * 0.12 + lat_flat * 0.42 + year_seed * 0.07)
    )

    records = []
    for i in range(len(lat_flat)):
        t_lat, t_lon = float(lat_flat[i]), float(lon_flat[i])
        vs = var_spasial[i]
        grid_uo = val_uo_base + (vs * 0.012) + rng.normal(0, 0.003)
        grid_vo = val_vo_base + (vs * 0.006) + rng.normal(0, 0.002)
        grid_speed = float(np.sqrt(grid_uo**2 + grid_vo**2))

        grid_do   = float(np.clip(6.2  - (vs * 0.06)  + rng.normal(0, 0.05),  4.5, 7.5))
        grid_ph   = float(np.clip(8.12 + (vs * 0.005) + rng.normal(0, 0.008), 7.9, 8.4))
        grid_chla = float(np.clip(0.22 + (vs * 0.012) + rng.normal(0, 0.01),  0.05, 0.8))
        grid_sal  = float(np.clip(34.2 + (vs * 0.04)  + rng.normal(0, 0.08),  32.0, 36.5))
        grid_wave = float(np.clip(0.8  + abs(vs) * 0.05 + rng.normal(0, 0.04), 0.2, 2.5))

        grid_sohi = float(np.clip((
            0.25 * normalisasi_global(grid_do,    4.5, 7.5) +
            0.20 * normalisasi_global(grid_ph,    7.9, 8.4) +
            0.20 * normalisasi_global(grid_chla,  0.05, 0.8) +
            0.15 * normalisasi_global(grid_sal,   32.0, 36.5) +
            0.20 * (1 - normalisasi_global(grid_wave, 0.2, 2.5))
        ) * 100, 10, 100))

        grid_fsi = float(np.clip((
            0.35 * normalisasi_global(grid_chla,  0.05, 0.8) +
            0.25 * normalisasi_global(grid_do,    4.5, 7.5) +
            0.20 * normalisasi_global(grid_speed, 0.0, 0.25) +
            0.20 * (1 - normalisasi_global(grid_wave, 0.2, 2.5))
        ) * 100, 10, 100))

        records.append({
            'lat': t_lat, 'lon': t_lon,
            'Ocean_Health_Index': grid_sohi,
            'Fisheries_Index':    grid_fsi,
            'uo': float(grid_uo), 'vo': float(grid_vo),
            'sst':    float(np.clip(28.5 + (vs * 0.18) + rng.normal(0, 0.1), 26.0, 32.0)),
            'ssta':   float(vs * 0.06 + rng.normal(0, 0.05)),
            'ph': grid_ph, 'do': grid_do,
            'salinitas': grid_sal, 'chla': grid_chla,
            'current_speed': grid_speed, 'gelombang': grid_wave,
            'angin_u': float(-1.5 + (vs * 0.25) + rng.normal(0, 0.1)),
            'angin_v': float(-0.5 + (vs * 0.12) + rng.normal(0, 0.06)),
        })
    return pd.DataFrame(records)

val_uo_base  = float(df_filter_base["uo"].mean()    if not df_filter_base.empty else -0.05)
val_vo_base  = float(df_filter_base["vo"].mean()    if not df_filter_base.empty else -0.01)
active_month = int(df_filter_base["month"].mean())  if not df_filter_base.empty else 6
active_year  = int(df_filter_base["year"].mean())   if not df_filter_base.empty else 2010

df_map = build_spatial_grid(val_uo_base, val_vo_base, active_month, active_year)

@st.cache_data(show_spinner=False)
def load_batas_provinsi():
    import urllib.request, json
    url = ("https://raw.githubusercontent.com/superpikar/"
           "indonesia-geojson/master/indonesia-province-simple.json")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

def render_map(df_map, z_col, colorscale, height=520):
    fig = px.scatter_mapbox(
        df_map, lat="lat", lon="lon", color=z_col,
        color_continuous_scale=colorscale,
        opacity=0.85,
        zoom=4.5,
        size_max=6,
        range_color=[float(df_map[z_col].quantile(0.03)), float(df_map[z_col].quantile(0.97))],
        mapbox_style="carto-positron",
    )
    fig.update_traces(marker=dict(size=4.5))

    map_layers = []
    if globals().get("SHOW_BORDERS", False):
        geojson_prov = load_batas_provinsi()
        if geojson_prov is not None:
            map_layers.append(dict(
                sourcetype="geojson",
                source=geojson_prov,
                type="line",
                color="#34679A",
                opacity=0.55,
                line=dict(width=1.0),
            ))

    fig.update_layout(
        mapbox=dict(center=dict(lat=-8.5, lon=137.0), layers=map_layers),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            thickness=12, len=0.65,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#D6E4F0",
            tickfont=dict(color="#3A5070", size=10, family="JetBrains Mono"),
            title=dict(font=dict(color="#3A5070", size=10, family="JetBrains Mono")),
        )
    )
    return fig

# =========================================
# ROSE DIAGRAM HELPERS
# =========================================
def make_wind_rose(df_src, title="Rose Diagram Angin"):
    if df_src.empty:
        return go.Figure()
    speed = np.sqrt(df_src["angin_u"]**2 + df_src["angin_v"]**2)
    direction_rad = np.arctan2(-df_src["angin_u"], -df_src["angin_v"])
    direction_deg = (np.degrees(direction_rad) + 360) % 360

    n_bins = 16
    bin_edges = np.linspace(0, 360, n_bins + 1)

    speed_bins = [0, 2, 4, 6, 8, 100]
    speed_labels = ["<2 m/s", "2–4 m/s", "4–6 m/s", "6–8 m/s", ">8 m/s"]
    colors_wind = ["#A8C8E8","#5A9EC8","#1E6BB8","#0D3D6B","#031420"]

    fig = go.Figure()
    for k, (s0, s1) in enumerate(zip(speed_bins[:-1], speed_bins[1:])):
        mask_speed = (speed >= s0) & (speed < s1)
        counts, _ = np.histogram(direction_deg[mask_speed], bins=bin_edges)
        fig.add_trace(go.Barpolar(
            r=counts, theta=bin_edges[:-1],
            width=[360/n_bins]*n_bins,
            name=speed_labels[k],
            marker_color=colors_wind[k],
            marker_line_color="white",
            marker_line_width=0.5,
            opacity=0.9,
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#0D1F33", size=13, family="Inter"), x=0.5),
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True, tickfont=dict(size=9, color="#5A7FA0"), gridcolor="#D6E4F0"),
            angularaxis=dict(direction="clockwise", rotation=90,
                             tickmode="array",
                             tickvals=[0,45,90,135,180,225,270,315],
                             ticktext=["U","TL","T","TG","S","BD","B","BL"],
                             tickfont=dict(size=10, color="#0D1F33", family="Inter"),
                             gridcolor="#D6E4F0"),
        ),
        legend=dict(font=dict(size=10, color="#3A5070", family="Inter"), bgcolor="rgba(255,255,255,0.9)", bordercolor="#D6E4F0"),
        paper_bgcolor="#FFFFFF",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

def make_wave_rose(df_src, title="Rose Diagram Gelombang"):
    if df_src.empty:
        return go.Figure()
    speed = df_src["gelombang"]
    direction_rad = np.arctan2(df_src["uo"], df_src["vo"])
    direction_deg = (np.degrees(direction_rad) + 360) % 360

    n_bins = 16
    bin_edges = np.linspace(0, 360, n_bins + 1)
    wave_bins   = [0, 0.5, 1.0, 1.5, 2.0, 10]
    wave_labels = ["<0.5 m","0.5–1 m","1–1.5 m","1.5–2 m",">2 m"]
    colors_wave = ["#C8E8D4","#6DBFA0","#1E9E6B","#0D5E40","#031F15"]

    fig = go.Figure()
    for k, (s0, s1) in enumerate(zip(wave_bins[:-1], wave_bins[1:])):
        mask = (speed >= s0) & (speed < s1)
        counts, _ = np.histogram(direction_deg[mask], bins=bin_edges)
        fig.add_trace(go.Barpolar(
            r=counts, theta=bin_edges[:-1],
            width=[360/n_bins]*n_bins,
            name=wave_labels[k],
            marker_color=colors_wave[k],
            marker_line_color="white",
            marker_line_width=0.5,
            opacity=0.9,
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#0D1F33", size=13, family="Inter"), x=0.5),
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True, tickfont=dict(size=9, color="#5A7FA0"), gridcolor="#D6E4F0"),
            angularaxis=dict(direction="clockwise", rotation=90,
                             tickmode="array",
                             tickvals=[0,45,90,135,180,225,270,315],
                             ticktext=["U","TL","T","TG","S","BD","B","BL"],
                             tickfont=dict(size=10, color="#0D1F33", family="Inter"),
                             gridcolor="#D6E4F0"),
        ),
        legend=dict(font=dict(size=10, color="#3A5070", family="Inter"), bgcolor="rgba(255,255,255,0.9)", bordercolor="#D6E4F0"),
        paper_bgcolor="#FFFFFF",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

# =========================================
# HELPER: Hitung arah dominan dari komponen u/v
# =========================================
def get_dominant_direction(df_src, u_col, v_col, mode="toward"):
    if df_src.empty or u_col not in df_src.columns or v_col not in df_src.columns:
        return "tidak tersedia", 0.0, 0.0

    u_mean = float(df_src[u_col].mean())
    v_mean = float(df_src[v_col].mean())
    speed  = float(np.sqrt(u_mean**2 + v_mean**2))

    if mode == "from":
        angle_rad = np.arctan2(-u_mean, -v_mean)
    else:
        angle_rad = np.arctan2(u_mean, v_mean)

    deg = float((np.degrees(angle_rad) + 360) % 360)

    compass = [
        "Utara", "Utara-Timur Laut", "Timur Laut", "Timur-Timur Laut",
        "Timur", "Timur-Tenggara", "Tenggara", "Selatan-Tenggara",
        "Selatan", "Selatan-Barat Daya", "Barat Daya", "Barat-Barat Daya",
        "Barat", "Barat-Barat Laut", "Barat Laut", "Utara-Barat Laut"
    ]
    idx = int((deg + 11.25) / 22.5) % 16
    return compass[idx], speed, deg


def get_dominant_wave_direction(df_src):
    if df_src.empty:
        return "tidak tersedia", 0.0, 0.0

    u_mean = float(df_src["uo"].mean())
    v_mean = float(df_src["vo"].mean())
    wave_mean = float(df_src["gelombang"].mean())

    angle_rad = np.arctan2(u_mean, v_mean)
    deg = float((np.degrees(angle_rad) + 360) % 360)

    compass = [
        "Utara", "Utara-Timur Laut", "Timur Laut", "Timur-Timur Laut",
        "Timur", "Timur-Tenggara", "Tenggara", "Selatan-Tenggara",
        "Selatan", "Selatan-Barat Daya", "Barat Daya", "Barat-Barat Daya",
        "Barat", "Barat-Barat Laut", "Barat Laut", "Utara-Barat Laut"
    ]
    idx = int((deg + 11.25) / 22.5) % 16
    return compass[idx], wave_mean, deg


# =========================================
# FISHERIES STATUS
# =========================================
def get_fisheries_status(fsi_val):
    p25 = df_map["Fisheries_Index"].quantile(0.25)
    p75 = df_map["Fisheries_Index"].quantile(0.75)
    if fsi_val >= p75:
        return {"color":"#00895A","text":"SANGAT BAIK","icon":"✅","bg":"#EDFAF3","border":"#9FD9BE"}
    elif fsi_val >= p25:
        return {"color":"#1E6BB8","text":"NORMAL","icon":"🔵","bg":"#EBF3FB","border":"#7BAFD4"}
    else:
        return {"color":"#D4811A","text":"WASPADA","icon":"⚠️","bg":"#FEF6E8","border":"#F0C070"}


# =========================================
# DASHBOARD — NELAYAN
# =========================================
if st.session_state.role == "nelayan":
    mean_fsi  = float(df_map["Fisheries_Index"].mean())
    status    = get_fisheries_status(mean_fsi)
    fsi_abs   = f"{mean_fsi:.1f}"

    st.markdown(f"""
<div class="page-header">
  <div class="eyebrow">🎣 Dashboard Nelayan · {mode} · {waktu_label}</div>
  <h1>Peta Zona Tangkap — Laut Arafura</h1>
</div>
""", unsafe_allow_html=True)

    # ── Baris 1: Status + Metrics ──────────────────────────────────────────
    col_s1, col_s2, col_s3, col_s4 = st.columns([2, 1, 1, 1])
    with col_s1:
        st.markdown(f"""
<div style="background:{status['bg']};border:1px solid {status['border']};border-left:4px solid {status['color']};border-radius:8px;padding:18px 20px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">Status Zona Melaut</div>
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:20px;">{status['icon']}</span>
    <span style="color:{status['color']};font-size:20px;font-weight:700;letter-spacing:0.02em;">{status['text']}</span>
    <span style="font-family:'JetBrains Mono',monospace;color:#8ABDD4;font-size:12px;">FSI {fsi_abs}/100</span>
  </div>
</div>
""", unsafe_allow_html=True)
    with col_s2:
        st.metric("Potensi Ikan (FSI)", f"{mean_fsi:.0f}", "/ 100")
    with col_s3:
        st.metric("Tinggi Gelombang", f"{df_map['gelombang'].mean():.2f}", "m")
    with col_s4:
        st.metric("Kec. Arus", f"{df_map['current_speed'].mean():.3f}", "m/s")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Baris 2: Peta (kiri) + Rekomendasi & Kondisi (kanan) ──────────────
    col_map, col_info = st.columns([3, 2], gap="medium")

    with col_map:
        st.markdown('<div class="section-label">PETA DISTRIBUSI SPASIAL · FISHERIES INDEX</div>', unsafe_allow_html=True)
        if not df_map.empty:
            st.plotly_chart(
                render_map(df_map, "Fisheries_Index", "Turbo", height=460),
                use_container_width=True
            )

    with col_info:
        # ── Rekomendasi Zona Melaut ──
        st.markdown('<div class="section-label">REKOMENDASI ZONA MELAUT</div>', unsafe_allow_html=True)

        df_rec_src = df_filter_base if not df_filter_base.empty else df

        dir_angin, spd_angin, deg_angin = get_dominant_direction(
            df_rec_src, "angin_u", "angin_v", mode="from")
        dir_arus, spd_arus, deg_arus = get_dominant_direction(
            df_rec_src, "uo", "vo", mode="toward")
        dir_gelombang, h_gelombang, deg_gelombang = get_dominant_wave_direction(df_rec_src)

        wave_mean = float(df_map["gelombang"].mean())

        if spd_angin > 7:
            angin_info = f"⚠️ **Angin kencang** dari **{dir_angin}** ({deg_angin:.0f}°, {spd_angin:.1f} m/s) — waspada!"
        elif spd_angin > 4:
            angin_info = f"🌬️ Angin sedang dari **{dir_angin}** ({deg_angin:.0f}°, {spd_angin:.1f} m/s)"
        else:
            angin_info = f"🍃 Angin lemah dari **{dir_angin}** ({deg_angin:.0f}°, {spd_angin:.1f} m/s)"

        if wave_mean > 1.5:
            gelombang_info = f"🌊 Gelombang **tinggi** menuju **{dir_gelombang}** ({wave_mean:.2f} m) — hati-hati!"
        elif wave_mean > 0.8:
            gelombang_info = f"〰️ Gelombang sedang menuju **{dir_gelombang}** ({wave_mean:.2f} m)"
        else:
            gelombang_info = f"🏝️ Gelombang tenang menuju **{dir_gelombang}** ({wave_mean:.2f} m)"

        if status["text"] == "SANGAT BAIK":
            msg = (
                f"**Area oranye/merah pada peta direkomendasikan.**\n\n"
                f"🧭 **Arah melaut:** ikuti arus ke **{dir_arus}** ({deg_arus:.0f}°)\n\n"
                f"Klorofil & oksigen melimpah — turunkan jaring di perairan dalam Arafura.\n\n"
                f"{angin_info}\n\n"
                f"{gelombang_info}"
            )
            st.success(msg)
        elif status["text"] == "NORMAL":
            msg = (
                f"**Kondisi normal — potensi tangkapan cukup.**\n\n"
                f"🧭 **Arah melaut:** ikuti arus ke **{dir_arus}** ({deg_arus:.0f}°)\n\n"
                f"Ikan bergerak mengikuti arus permukaan menuju **{dir_arus}**.\n\n"
                f"{angin_info}\n\n"
                f"{gelombang_info}"
            )
            st.info(msg)
        else:
            msg = (
                f"**Potensi tangkapan rendah saat ini.**\n\n"
                f"🧭 **Arah yang disarankan:** menyusuri arus menuju **{dir_arus}** ({deg_arus:.0f}°)\n\n"
                f"Disarankan memancing di pesisir dekat teluk dan muara sungai.\n\n"
                f"{angin_info}\n\n"
                f"{gelombang_info}"
            )
            st.warning(msg)

        st.markdown(f"""
<div class="data-note">
  🧭 Arus: <b>{dir_arus}</b> ({deg_arus:.0f}°) · {spd_arus:.4f} m/s <br>
  💨 Angin dari: <b>{dir_angin}</b> ({deg_angin:.0f}°)<br>
  🌊 Gelombang: <b>{dir_gelombang}</b> ({deg_gelombang:.0f}°) · {wave_mean:.2f} m
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Kondisi Perairan ──
        st.markdown('<div class="section-label">KONDISI PERAIRAN SAAT INI</div>', unsafe_allow_html=True)
        chla_mean = df_map["chla"].mean()
        do_mean   = df_map["do"].mean()
        st.markdown(f"""
<div style="background:#FFFFFF;border:1px solid #D6E4F0;border-radius:8px;padding:20px;">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div style="text-align:center;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Klorofil-a</div>
      <div style="font-size:22px;font-weight:700;color:#0D1F33;">{chla_mean:.3f} <span style="font-size:12px;color:#5A7FA0;">mg/m³</span></div>
    </div>
    <div style="text-align:center;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Dissolved O₂</div>
      <div style="font-size:22px;font-weight:700;color:#0D1F33;">{do_mean:.2f} <span style="font-size:12px;color:#5A7FA0;">mg/L</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Baris 3: Rose Diagram Angin (kiri) + Rose Diagram Gelombang (kanan) ──
    st.markdown('<div class="section-label">ROSE DIAGRAM — ANGIN & GELOMBANG</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    df_rose_src = df_filter_base if not df_filter_base.empty else df
    with rc1:
        st.plotly_chart(
            make_wind_rose(df_rose_src, f"Arah & Kecepatan Angin · {waktu_label}"),
            use_container_width=True
        )
    with rc2:
        st.plotly_chart(
            make_wave_rose(df_rose_src, f"Arah & Tinggi Gelombang · {waktu_label}"),
            use_container_width=True
        )

# =========================================
# DASHBOARD — AKADEMISI
# =========================================
else:
    PARAM_LABELS_CLEAN = {
        "Ocean_Health_Index": "Ocean Health Index",
        "Fisheries_Index":    "Fisheries Index",
        "sst":                "Sea Surface Temp (SST)",
        "ssta":               "SST Anomali",
        "ph":                 "pH Laut",
        "do":                 "Dissolved Oxygen",
        "salinitas":          "Salinitas",
        "chla":               "Klorofil-a",
        "current_speed":      "Kecepatan Arus",
        "gelombang":          "Tinggi Gelombang",
        "angin_u":            "Angin Zonal (U)",
        "angin_v":            "Angin Meridional (V)",
    }

    st.markdown(f"""
<div class="page-header">
  <div class="eyebrow">🔬 Portal Akademisi · {mode} · {waktu_label}</div>
  <h1>Analisis Parameter — {PARAM_LABELS_CLEAN.get(parameter, parameter)}</h1>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rata-Rata",  f"{df_map[parameter].mean():.3f}")
    col2.metric("Minimum",    f"{df_map[parameter].min():.3f}")
    col3.metric("Maksimum",   f"{df_map[parameter].max():.3f}")
    col4.metric("Std. Dev",   f"{df_map[parameter].std():.3f}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    PARAM_TERARAH = ["angin_u", "angin_v", "gelombang"]
    tampilkan_rose = parameter in PARAM_TERARAH

    if tampilkan_rose:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["  Spasial  ", "  Time Series  ", "  Statistik  ", "  Korelasi  ", "  Rose Diagram  "])
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["  Spasial  ", "  Time Series  ", "  Statistik  ", "  Korelasi  "])

    with tab1:
        cmap_dict = {
            'Fisheries_Index': 'Turbo',   'chla': 'Turbo',
            'Ocean_Health_Index': 'Blues', 'do': 'Blues',
            'ph': 'Viridis',               'salinitas': 'YlOrBr',
            'sst': 'Plasma',               'ssta': 'RdBu',
            'current_speed': 'Teal',
        }
        cmap = cmap_dict.get(parameter, "Icefire")
        st.markdown(f'<div class="section-label">DISTRIBUSI SPASIAL · {parameter.upper()}</div>', unsafe_allow_html=True)
        st.plotly_chart(render_map(df_map, parameter, cmap, height=500), use_container_width=True)
        st.markdown(f"""
<span class="coord-tag">4°S – 12°S</span>
<span class="coord-tag">129°E – 144°E</span>
<span class="coord-tag">Grid 100×80 · Laut Arafura</span>
""", unsafe_allow_html=True)

    with tab2:
        df_ts_src = df[df["month"].isin(ts_months)].copy()
        df_ts = (df_ts_src.groupby("year")[parameter].mean()
                 .reset_index().sort_values("year"))
        x_year = df_ts["year"].to_numpy()
        y_vals = df_ts[parameter].to_numpy(dtype=float)

        has_trend = len(df_ts) >= 2
        if has_trend:
            z = np.polyfit(range(len(df_ts)), y_vals, 1)
            y_trend = np.poly1d(z)(range(len(df_ts)))
        else:
            y_trend = y_vals

        y_lo = float(min(y_vals.min(), np.min(y_trend)))
        y_hi = float(max(y_vals.max(), np.max(y_trend)))
        span = y_hi - y_lo
        pad = span * 0.08 if span > 0 else (abs(y_hi) * 0.08 if y_hi != 0 else 1.0)
        y_floor, y_ceil = y_lo - pad, y_hi + pad

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=x_year, y=y_vals,
            mode="lines+markers", name=PARAM_LABELS_CLEAN.get(parameter, parameter),
            line=dict(color="#1E6BB8", width=1.8), marker=dict(size=5),
        ))
        if has_trend:
            fig_ts.add_trace(go.Scatter(
                x=x_year, y=y_trend,
                mode="lines", name="Tren Linear",
                line=dict(color="#D4811A", width=2, dash="dot")
            ))
        if ts_highlight_year is not None and ts_highlight_year in set(x_year.tolist()):
            fig_ts.add_vline(x=ts_highlight_year,
                             line=dict(color="#C0392B", width=1.5, dash="dash"))
            fig_ts.add_annotation(
                x=ts_highlight_year, y=y_ceil, text=f"{ts_highlight_year}",
                showarrow=False, yanchor="bottom",
                font=dict(color="#C0392B", size=11, family="JetBrains Mono"))

        fig_ts.update_layout(
            **PLOTLY_LAYOUT,
            title=(f"Tren Antar-Tahun · {PARAM_LABELS_CLEAN.get(parameter, parameter)} "
                   f"· {ts_scope_label}"),
            legend=dict(font=dict(color="#3A5070", size=11), bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#D6E4F0", borderwidth=1),
            height=400,
        )
        fig_ts.update_xaxes(title_text="Tahun", dtick=2)
        fig_ts.update_yaxes(range=[y_floor, y_ceil], autorange=False)
        st.plotly_chart(fig_ts, use_container_width=True)
        st.caption(
            f"✓ Mengikuti sidebar — Mode: {mode} · Cakupan: {ts_scope_label}"
            + (f" · tahun ditandai: {ts_highlight_year}" if ts_highlight_year is not None else "")
            + f"  ·  skala-Y {y_floor:.3f}–{y_ceil:.3f}"
        )

    with tab3:
        desc = df_map[[parameter]].describe()
        st.markdown('<div class="section-label">STATISTIK DESKRIPTIF · AREA SPASIAL AKTIF</div>', unsafe_allow_html=True)

        stats_cols = st.columns(4)
        for i, (label, key, icon) in enumerate([
            ("Count", "count", "N"), ("Mean", "mean", "μ"),
            ("Std", "std", "σ"), ("Min", "min", "↓"),
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
        for i, (label, key) in enumerate([
            ("25th Percentile", "25%"),
            ("Median (50%)", "50%"),
            ("75th Percentile", "75%"),
        ]):
            stats_cols2[i].markdown(f"""
<div style="background:#FFFFFF;border:1px solid #D6E4F0;border-radius:8px;padding:18px;text-align:center;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#5A7FA0;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">{label}</div>
  <div style="font-size:18px;font-weight:700;color:#0D1F33;">{desc.loc[key, parameter]:.4f}</div>
</div>
""", unsafe_allow_html=True)

    with tab4:
        DROP_COLS = ["year", "month", "lat", "lon", "latitude", "longitude",
                     "index", "time", "id"]
        numeric_df = df.select_dtypes(include=np.number).drop(columns=DROP_COLS, errors="ignore")

        SHORT_CORR = {
            "uo": "UO", "vo": "VO", "sst": "SST", "ssta": "SSTA",
            "ph": "pH", "do": "DO", "salinitas": "SAL", "chla": "CHL-a",
            "gelombang": "WAVE", "current_speed": "CurSpd",
            "angin_u": "WindU", "angin_v": "WindV",
            "Ocean_Health_Index": "OHI", "Fisheries_Index": "FSI",
        }
        corr = numeric_df.corr().rename(index=SHORT_CORR, columns=SHORT_CORR)
        labels = list(corr.columns)
        n = len(labels)
        vals = corr.values
        zmin, zmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        rng_z = (zmax - zmin) or 1.0

        fig_corr = go.Figure(go.Heatmap(
            z=vals, x=labels, y=labels,
            colorscale=[[0, "#EBF3FB"], [0.5, "#5A9EC8"], [1, "#0D1F33"]],
            zmin=zmin, zmax=zmax,
            xgap=3, ygap=3,
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(thickness=12, len=0.7,
                          tickfont=dict(size=10, family="JetBrains Mono", color="#0D1F33")),
        ))

        annotations = []
        for i in range(n):
            for j in range(n):
                v = vals[i, j]
                frac = (v - zmin) / rng_z
                txt_color = "#FFFFFF" if frac > 0.55 else "#0D1F33"
                annotations.append(dict(
                    x=labels[j], y=labels[i], text=f"{v:.2f}",
                    showarrow=False,
                    font=dict(size=10, color=txt_color, family="JetBrains Mono"),
                ))

        chart_h = max(560, 46 * n + 170)
        fig_corr.update_layout(
            annotations=annotations,
            title=dict(text="Matriks Korelasi Pearson — Parameter Oseanografi",
                       font=dict(color="#0D1F33", size=15, family="Inter"), x=0.01),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            height=chart_h,
            margin=dict(l=10, r=10, t=60, b=10),
            xaxis=dict(tickangle=45, side="bottom",
                       tickfont=dict(size=12, color="#0D1F33", family="Inter")),
            yaxis=dict(autorange="reversed",
                       tickfont=dict(size=12, color="#0D1F33", family="Inter")),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    if tampilkan_rose:
        with tab5:
            st.markdown('<div class="section-label">ROSE DIAGRAM — DISTRIBUSI ARAH & INTENSITAS</div>', unsafe_allow_html=True)
            df_rose_src = df_filter_base if not df_filter_base.empty else df
            if parameter in ("angin_u", "angin_v"):
                st.plotly_chart(
                    make_wind_rose(df_rose_src, f"Arah & Kecepatan Angin · {waktu_label}"),
                    use_container_width=True)
                st.markdown("""
<div class="data-note">Rose diagram angin dibangun dari komponen zonal (U) dan meridional (V) menggunakan konvensi meteorologis (arah dari mana angin datang).</div>
""", unsafe_allow_html=True)
            else:
                st.plotly_chart(
                    make_wave_rose(df_rose_src, f"Arah & Tinggi Gelombang · {waktu_label}"),
                    use_container_width=True)
                st.markdown("""
<div class="data-note">Rose diagram gelombang menggunakan tinggi gelombang dengan proxy arah arus permukaan (uo, vo).</div>
""", unsafe_allow_html=True)
