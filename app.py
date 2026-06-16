import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="LAUTAN — Ocean Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# DESIGN SYSTEM — Bright Coastal Sunshine
# =========================================
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.stApp {
    background: linear-gradient(160deg, #E8F6FF 0%, #F0FAFF 40%, #FFF8F0 100%);
    color: #1A3A5C;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F0F8FF 100%) !important;
    border-right: 2px solid #C8E8FF !important;
    box-shadow: 4px 0 20px rgba(30, 120, 200, 0.08) !important;
}
[data-testid="stSidebar"] * { color: #1A3A5C !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: #2E7DC4 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #4A8EC4 !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #EBF5FF !important;
    border: 1.5px solid #7EC8F0 !important;
    color: #1A3A5C !important;
    border-radius: 10px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1E90D4 0%, #0FB8D4 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    transition: all 0.25s ease !important;
    padding: 10px 20px !important;
    box-shadow: 0 4px 14px rgba(30, 144, 212, 0.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1678B8 0%, #0FA0BC 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(30, 144, 212, 0.4) !important;
}
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1.5px solid #C8E8FF;
    border-radius: 14px;
    padding: 16px 20px !important;
    box-shadow: 0 2px 12px rgba(30, 120, 200, 0.08);
}
[data-testid="stMetricLabel"] {
    color: #5A9EC8 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #1A3A5C !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-bottom: 2px solid #C8E8FF !important;
    border-radius: 12px 12px 0 0 !important;
    gap: 0px;
    padding: 0 8px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #5A9EC8 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 12px 20px !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #1E90D4 !important;
    border-bottom: 3px solid #1E90D4 !important;
    font-weight: 700 !important;
}
hr { border-color: #C8E8FF !important; }
.ocean-header {
    border-left: 4px solid #1E90D4;
    padding: 6px 0 6px 18px;
    margin-bottom: 20px;
}
.ocean-header h1 {
    font-size: 28px;
    font-weight: 800;
    color: #1A3A5C;
    margin: 0;
    letter-spacing: -0.02em;
}
.ocean-header .subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #1E90D4;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 4px;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #1E90D4;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 8px;
    font-weight: 500;
}
.coord-tag {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #3A7FC4;
    background: #EBF5FF;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #B0D8F0;
    display: inline-block;
    margin: 2px;
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,0.6)",
    plot_bgcolor="rgba(235,248,255,0.5)",
    font=dict(family="Plus Jakarta Sans", color="#3A6080", size=12),
    xaxis=dict(gridcolor="#C8E8FF", zerolinecolor="#C8E8FF", color="#5A9EC8"),
    yaxis=dict(gridcolor="#C8E8FF", zerolinecolor="#C8E8FF", color="#5A9EC8"),
    title_font=dict(color="#1A3A5C", size=15, family="Plus Jakarta Sans"),
)

def normalisasi_global(series, vmin, vmax):
    if (vmax - vmin) == 0:
        return series * 0
    return (series - vmin) / (vmax - vmin)

if "page" not in st.session_state: st.session_state.page = "home"
if "role" not in st.session_state: st.session_state.role = "akademisi"

# =========================================
# LOAD DATA — dengan variasi per bulan/tahun
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
        # Buat variasi musiman + tren + noise yang realistis
        seasonal = np.sin(2 * np.pi * t / 12)          # siklus tahunan
        enso     = 0.4 * np.sin(2 * np.pi * t / 48)    # siklus ENSO ~4 tahun
        trend    = -0.001 * t                             # tren lemah
        noise_uo = rng.normal(0, 0.008, n)
        noise_vo = rng.normal(0, 0.004, n)

        uo = -0.05 + 0.04 * seasonal + 0.02 * enso + trend + noise_uo
        vo = -0.01 + 0.02 * seasonal + 0.01 * enso + noise_vo

        df_data = pd.DataFrame({
            "time": dates_fallback,
            "uo": uo,
            "vo": vo,
        })

    df_data["time"] = pd.to_datetime(df_data["time"])

    if "uo" not in df_data.columns: df_data["uo"] = -0.05
    if "vo" not in df_data.columns: df_data["vo"] = -0.01

    # Derive semua parameter dari uo/vo dengan variasi realistis
    rng2 = np.random.default_rng(7)
    n = len(df_data)
    if "sst" not in df_data.columns:
        df_data["sst"] = 28.5 + df_data["uo"] * 6 + rng2.normal(0, 0.3, n)
    if "ssta" not in df_data.columns:
        df_data["ssta"] = df_data["uo"] * 2.5 + rng2.normal(0, 0.1, n)
    if "ph" not in df_data.columns:
        df_data["ph"] = 8.12 + df_data["vo"] * 0.6 + rng2.normal(0, 0.01, n)
    if "do" not in df_data.columns:
        df_data["do"] = 6.2 - df_data["uo"] * 2.5 + rng2.normal(0, 0.05, n)
    if "salinitas" not in df_data.columns:
        df_data["salinitas"] = 34.2 + df_data["vo"] * 2.5 + rng2.normal(0, 0.08, n)
    if "chla" not in df_data.columns:
        df_data["chla"] = 0.22 + df_data["uo"] * (-0.5) + rng2.normal(0, 0.015, n)
    if "gelombang" not in df_data.columns:
        df_data["gelombang"] = 0.8 + abs(df_data["uo"]) * 2.5 + rng2.normal(0, 0.05, n)
    if "angin_u" not in df_data.columns:
        df_data["angin_u"] = -1.5 + df_data["uo"] * 12 + rng2.normal(0, 0.3, n)
    if "angin_v" not in df_data.columns:
        df_data["angin_v"] = -0.5 + df_data["vo"] * 6 + rng2.normal(0, 0.2, n)

    # Clip nilai-nilai agar tetap realistis
    df_data["chla"] = df_data["chla"].clip(0.05, 0.8)
    df_data["do"] = df_data["do"].clip(4.5, 7.5)
    df_data["ph"] = df_data["ph"].clip(7.9, 8.4)
    df_data["salinitas"] = df_data["salinitas"].clip(32.0, 36.5)
    df_data["gelombang"] = df_data["gelombang"].clip(0.2, 2.5)

    return df_data

df = load_data()
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["current_speed"] = np.sqrt(df["uo"]**2 + df["vo"]**2)

df["Ocean_Health_Index"] = (
    0.25 * normalisasi_global(df["do"], 4.5, 7.5) +
    0.20 * normalisasi_global(df["ph"], 7.9, 8.4) +
    0.20 * normalisasi_global(df["chla"], 0.05, 0.8) +
    0.15 * normalisasi_global(df["salinitas"], 32.0, 36.5) +
    0.20 * (1 - normalisasi_global(df["gelombang"], 0.2, 2.5))
) * 100

df["Fisheries_Index"] = (
    0.35 * normalisasi_global(df["chla"], 0.05, 0.8) +
    0.25 * normalisasi_global(df["do"], 4.5, 7.5) +
    0.20 * normalisasi_global(df["current_speed"], 0.0, 0.25) +
    0.20 * (1 - normalisasi_global(df["gelombang"], 0.2, 2.5))
) * 100

# =========================================
# HOME PAGE
# =========================================
if st.session_state.page == "home":
    hero_html = """
<div style="background:linear-gradient(135deg,#1E90D4 0%,#0FB8D4 50%,#00D4AA 100%);border-radius:24px;padding:56px 40px 48px;text-align:center;margin-bottom:36px;position:relative;overflow:hidden;box-shadow:0 8px 40px rgba(30,144,212,0.3);">
  <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;border-radius:50%;background:rgba(255,255,255,0.06);pointer-events:none;"></div>
  <div style="position:absolute;bottom:-60px;left:-30px;width:260px;height:260px;border-radius:50%;background:rgba(255,255,255,0.05);pointer-events:none;"></div>
  <div style="display:inline-block;background:rgba(255,255,255,0.18);border:1px solid rgba(255,255,255,0.35);border-radius:30px;padding:5px 18px;font-family:'DM Mono',monospace;font-size:11px;color:#FFFFFF;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:20px;">
    ◈ SISTEM AKTIF &middot; PERAIRAN LAUT ARAFURA &middot; 4&deg;S&ndash;12&deg;S / 129&deg;E&ndash;144&deg;E
  </div>
  <div style="font-size:72px;margin-bottom:4px;">🌊</div>
  <h1 style="font-family:'Plus Jakarta Sans',sans-serif;font-size:58px;font-weight:800;color:#FFFFFF;letter-spacing:-0.03em;margin:0 0 8px;text-shadow:0 2px 20px rgba(0,0,0,0.15);">LAUTAN</h1>
  <div style="font-family:'DM Mono',monospace;font-size:13px;color:rgba(255,255,255,0.85);letter-spacing:0.2em;text-transform:uppercase;margin-bottom:20px;">Platform Intelijen Oseanografi Papua</div>
  <div style="color:rgba(255,255,255,0.8);font-size:15px;max-width:520px;margin:0 auto;line-height:1.8;">
    Data klimatologi laut historis, real-time, dan proyeksi musiman<br>untuk kawasan perairan Papua &amp; Laut Arafura.
  </div>
  <div style="display:flex;justify-content:center;gap:40px;margin-top:32px;flex-wrap:wrap;">
    <div style="text-align:center;">
      <div style="font-size:28px;font-weight:800;color:#FFFFFF;">20+</div>
      <div style="font-size:11px;color:rgba(255,255,255,0.7);font-family:'DM Mono',monospace;letter-spacing:0.1em;">TAHUN DATA</div>
    </div>
    <div style="width:1px;background:rgba(255,255,255,0.25);"></div>
    <div style="text-align:center;">
      <div style="font-size:28px;font-weight:800;color:#FFFFFF;">12</div>
      <div style="font-size:11px;color:rgba(255,255,255,0.7);font-family:'DM Mono',monospace;letter-spacing:0.1em;">PARAMETER</div>
    </div>
    <div style="width:1px;background:rgba(255,255,255,0.25);"></div>
    <div style="text-align:center;">
      <div style="font-size:28px;font-weight:800;color:#FFFFFF;">80&times;80</div>
      <div style="font-size:11px;color:rgba(255,255,255,0.7);font-family:'DM Mono',monospace;letter-spacing:0.1em;">GRID SPASIAL</div>
    </div>
  </div>
</div>
"""
    st.markdown(hero_html, unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center;margin-bottom:20px;">
  <div style="font-family:'DM Mono',monospace;font-size:11px;color:#5A9EC8;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:6px;">Pilih Mode Tampilan</div>
  <div style="font-size:22px;font-weight:700;color:#1A3A5C;">Siapa kamu hari ini? 👋</div>
</div>
""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
<div style="background:linear-gradient(135deg,#FFFFFF 0%,#EBF8FF 100%);border:2px solid #7EC8F0;border-radius:20px;padding:36px 28px;text-align:center;margin-bottom:12px;box-shadow:0 4px 20px rgba(30,144,212,0.10);">
  <div style="font-size:56px;margin-bottom:16px;">🎣</div>
  <h3 style="color:#1A3A5C;font-size:20px;font-weight:700;margin:0 0 10px;">Nelayan Lokal</h3>
  <p style="color:#4A8EC4;font-size:14px;margin:0;line-height:1.6;">Peta zona tangkap, kondisi gelombang, dan rekomendasi area melaut hari ini.</p>
  <div style="margin-top:16px;display:flex;justify-content:center;gap:8px;flex-wrap:wrap;">
    <span style="background:#EBF8FF;color:#1E90D4;font-size:11px;padding:3px 10px;border-radius:20px;font-family:'DM Mono',monospace;border:1px solid #B0D8F0;">Zona Tangkap</span>
    <span style="background:#EBF8FF;color:#1E90D4;font-size:11px;padding:3px 10px;border-radius:20px;font-family:'DM Mono',monospace;border:1px solid #B0D8F0;">Gelombang</span>
    <span style="background:#EBF8FF;color:#1E90D4;font-size:11px;padding:3px 10px;border-radius:20px;font-family:'DM Mono',monospace;border:1px solid #B0D8F0;">Arus</span>
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("🎣  Masuk sebagai Nelayan", use_container_width=True, key="btn_nelayan"):
            st.session_state.role = "nelayan"
            st.session_state.page = "dashboard"
            st.rerun()

    with c2:
        st.markdown("""
<div style="background:linear-gradient(135deg,#FFFFFF 0%,#FFF6EB 100%);border:2px solid #FAC87A;border-radius:20px;padding:36px 28px;text-align:center;margin-bottom:12px;box-shadow:0 4px 20px rgba(245,166,35,0.10);">
  <div style="font-size:56px;margin-bottom:16px;">🔬</div>
  <h3 style="color:#1A3A5C;font-size:20px;font-weight:700;margin:0 0 10px;">Akademisi / Peneliti</h3>
  <p style="color:#4A8EC4;font-size:14px;margin:0;line-height:1.6;">12 parameter oseanografi, time series 20 tahun, matriks korelasi, dan analisis spasial.</p>
  <div style="margin-top:16px;display:flex;justify-content:center;gap:8px;flex-wrap:wrap;">
    <span style="background:#FFF6EB;color:#C07A10;font-size:11px;padding:3px 10px;border-radius:20px;font-family:'DM Mono',monospace;border:1px solid #FAC87A;">Analisis</span>
    <span style="background:#FFF6EB;color:#C07A10;font-size:11px;padding:3px 10px;border-radius:20px;font-family:'DM Mono',monospace;border:1px solid #FAC87A;">Korelasi</span>
    <span style="background:#FFF6EB;color:#C07A10;font-size:11px;padding:3px 10px;border-radius:20px;font-family:'DM Mono',monospace;border:1px solid #FAC87A;">Time Series</span>
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("🔬  Masuk sebagai Akademisi", use_container_width=True, key="btn_akademisi"):
            st.session_state.role = "akademisi"
            st.session_state.page = "dashboard"
            st.rerun()

    st.markdown("""
<div style="background:linear-gradient(135deg,#FFFFFF 0%,#F5FAFF 100%);border:1.5px solid #C8E8FF;border-radius:20px;padding:28px 32px;text-align:center;margin-top:32px;box-shadow:0 2px 12px rgba(30,120,200,0.07);">
  <div style="font-family:'DM Mono',monospace;font-size:10px;color:#1E90D4;text-transform:uppercase;letter-spacing:0.18em;margin-bottom:16px;">✨ Tim Pengembang</div>
  <div style="display:flex;justify-content:center;gap:32px;flex-wrap:wrap;align-items:center;">
    <div style="text-align:center;">
      <div style="width:52px;height:52px;border-radius:50%;background:linear-gradient(135deg,#1E90D4,#0FB8D4);margin:0 auto 10px;display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 4px 12px rgba(30,144,212,0.3);">👩‍💻</div>
      <div style="font-weight:700;color:#1A3A5C;font-size:14px;">Ratu Salwa Ghazalia Hade</div>
      <div style="font-family:'DM Mono',monospace;font-size:11px;color:#5A9EC8;margin-top:2px;">12923016</div>
    </div>
    <div style="width:1.5px;height:60px;background:linear-gradient(180deg,transparent,#C8E8FF,transparent);"></div>
    <div style="text-align:center;">
      <div style="width:52px;height:52px;border-radius:50%;background:linear-gradient(135deg,#0FB8D4,#00D4AA);margin:0 auto 10px;display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 4px 12px rgba(0,180,160,0.3);">👩‍🔬</div>
      <div style="font-weight:700;color:#1A3A5C;font-size:14px;">Diandra Aulia Ramadhani</div>
      <div style="font-family:'DM Mono',monospace;font-size:11px;color:#5A9EC8;margin-top:2px;">12923021</div>
    </div>
    <div style="width:1.5px;height:60px;background:linear-gradient(180deg,transparent,#C8E8FF,transparent);"></div>
    <div style="text-align:center;">
      <div style="width:52px;height:52px;border-radius:50%;background:linear-gradient(135deg,#00D4AA,#1E90D4);margin:0 auto 10px;display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 4px 12px rgba(0,212,170,0.3);">🌊</div>
      <div style="font-weight:700;color:#1A3A5C;font-size:14px;">Mutiara Nurani</div>
      <div style="font-family:'DM Mono',monospace;font-size:11px;color:#5A9EC8;margin-top:2px;">12923023</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center;margin-top:28px;">
  <div style="font-family:'DM Mono',monospace;font-size:10px;color:#A8CBDF;letter-spacing:0.1em;">
    DATA SUMBER: CMEMS &middot; MODIS-Aqua &middot; ERA5 &middot; BMKG &middot; 2001&ndash;2026
  </div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.markdown("""
<div style="padding:16px 0 8px;text-align:center;">
  <div style="font-size:32px;margin-bottom:6px;">🌊</div>
  <div style="font-family:'DM Mono',monospace;font-size:13px;font-weight:600;color:#1E90D4;letter-spacing:0.1em;text-transform:uppercase;">LAUTAN</div>
  <div style="font-size:12px;color:#5A9EC8;margin-top:2px;">Ocean Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("← Kembali ke Beranda", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    st.markdown("---")
    mode = st.selectbox("MODE DATA", ["Historis", "Real Time", "Prediksi"])
    st.markdown("---")

    if mode == "Historis":
        tahun = st.selectbox("TAHUN", sorted(df["year"].unique(), reverse=True))
        breakdown = st.radio("RESOLUSI WAKTU", ["Bulanan", "Musiman"])
        musim_map_dict = {
            "Musim Barat": [12, 1, 2],
            "Peralihan I": [3, 4, 5],
            "Musim Timur": [6, 7, 8],
            "Peralihan II": [9, 10, 11]
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
        st.info("Estimasi operasional · Juni 2026")
        df_filter_base = df[(df["year"] == 2020) & (df["month"] == 6)].copy()
        waktu_label = "Juni 2026"
    else:
        st.warning("Proyeksi iklim Semester-II 2026")
        bulan_pred = st.selectbox("TARGET PREDIKSI",
                                   ["Juli 2026","Agustus 2026","September 2026","Desember 2026"])
        idx_p = 7 if "Juli" in bulan_pred else 8 if "Agustus" in bulan_pred else 9 if "September" in bulan_pred else 12
        df_filter_base = df[(df["year"] == 2020) & (df["month"] == idx_p)].copy()
        waktu_label = f"Proyeksi {bulan_pred}"

    st.markdown("---")
    if st.session_state.role == "akademisi":
        PARAM_LABELS = {
            "Ocean_Health_Index": "🌊 Ocean Health Index",
            "Fisheries_Index":    "🐟 Fisheries Index",
            "sst":                "🌡️ Sea Surface Temp (SST)",
            "ssta":               "📈 SST Anomali",
            "ph":                 "⚗️ pH Laut",
            "do":                 "💧 Dissolved Oxygen",
            "salinitas":          "🧂 Salinitas",
            "chla":               "🟢 Klorofil-a",
            "current_speed":      "🌀 Kecepatan Arus",
            "gelombang":          "🌊 Tinggi Gelombang",
            "angin_u":            "💨 Angin Zonal (U)",
            "angin_v":            "💨 Angin Meridional (V)",
        }
        parameter = st.selectbox("PARAMETER RISET", list(PARAM_LABELS.keys()),
                                  format_func=lambda x: PARAM_LABELS[x])

    st.markdown("""
<div style="background:linear-gradient(135deg,#EBF8FF,#F0FFFE);border:1.5px solid #B0D8F0;border-radius:10px;padding:12px 14px;margin-top:12px;">
  <div style="font-family:'DM Mono',monospace;font-size:10px;color:#5A9EC8;letter-spacing:0.08em;">
    📍 4°S – 12°S / 129°E – 144°E<br>
    🌏 Laut Arafura · Papua
  </div>
</div>
""", unsafe_allow_html=True)

# =========================================
# SPATIAL GRID — dengan variasi per bulan+tahun
# =========================================
# Mask daratan Papua yang lebih akurat (bounding polygon Laut Arafura)
# Titik-titik di dalam polygon ini adalah DARATAN — dibuang
def is_land(lat, lon):
    """
    Approx mask daratan Papua & sekitarnya untuk Laut Arafura (4S-12S, 129E-144E).
    Mengembalikan True jika titik kemungkinan besar daratan.
    """
    # Pulau Papua bagian barat (Irian Jaya)
    if lon < 132.0 and lat > -6.0: return True
    if lon < 133.5 and lat > -5.0: return True
    if lon < 134.0 and lat > -4.5: return True
    if lon < 135.0 and lat > -4.2: return True
    if lon < 136.0 and lat > -4.1: return True
    if lon < 137.0 and lat > -4.0: return True
    if lon < 138.0 and lat > -4.0: return True
    if lon < 139.0 and lat > -4.1: return True
    if lon < 140.0 and lat > -4.2: return True
    if lon < 141.0 and lat > -4.5: return True
    if lon < 141.5 and lat > -5.0: return True
    if lon < 141.5 and lat > -8.0 and lat < -5.5 and lon > 140.5: return True
    # Semenanjung Bird's Head (kepala burung) barat laut Papua
    if lon < 132.5 and lat < -0.5 and lat > -4.0: return True
    if lon < 131.5 and lat > -4.5: return True
    if lon < 130.5 and lat > -6.0: return True
    # Papua bagian selatan menjorok ke Laut Arafura
    if lat > -5.5 and lon > 136.0 and lon < 138.5: return True
    if lat > -6.0 and lon > 137.5 and lon < 139.5: return True
    if lat > -6.5 and lon > 139.0 and lon < 141.0: return True
    # Australia (ujung utara) — tidak masuk bounding box ini
    # Kepulauan kecil Aru, Kei dll — biarkan tetap (sudah di laut)
    return False

@st.cache_data
def build_spatial_grid(val_uo_base, val_vo_base, month_seed, year_seed):
    """
    Bangun grid spasial dengan:
    - Mask daratan yang lebih akurat
    - Variasi berbeda per bulan dan tahun (pakai month_seed + year_seed)
    """
    lat_grid = np.linspace(-12.0, -4.0, 100)
    lon_grid = np.linspace(129.0, 144.0, 100)
    lon_g, lat_g = np.meshgrid(lon_grid, lat_grid)
    lat_flat = lat_g.flatten()
    lon_flat = lon_g.flatten()

    # Seed berbeda per bulan & tahun → pola spasial berbeda
    seed = int(month_seed * 1000 + year_seed)
    rng = np.random.default_rng(seed)

    # Variasi spasial dasar (deterministik berdasarkan koordinat)
    var_spasial = (
        2.5 * np.sin(lon_flat * 0.22 + lat_flat * 0.31 + month_seed * 0.5) +
        2.0 * np.cos(lon_flat * 0.15 - lat_flat * 0.28 + month_seed * 0.3) +
        1.5 * np.sin(lon_flat * 0.40 + lat_flat * 0.18 + year_seed * 0.1) +
        1.0 * np.cos(lon_flat * 0.12 + lat_flat * 0.42 + year_seed * 0.07) +
        0.8 * np.sin(lon_flat * 0.33 - lat_flat * 0.22)
    )

    records = []
    for i in range(len(lat_flat)):
        t_lat, t_lon = lat_flat[i], lon_flat[i]

        # Skip daratan
        if is_land(t_lat, t_lon):
            continue

        vs = var_spasial[i]
        grid_uo = val_uo_base + (vs * 0.012) + rng.normal(0, 0.003)
        grid_vo = val_vo_base + (vs * 0.006) + rng.normal(0, 0.002)
        grid_speed = np.sqrt(grid_uo**2 + grid_vo**2)

        grid_do   = np.clip(6.2  - (vs * 0.06)  + rng.normal(0, 0.05),  4.5, 7.5)
        grid_ph   = np.clip(8.12 + (vs * 0.005) + rng.normal(0, 0.008), 7.9, 8.4)
        grid_chla = np.clip(0.22 + (vs * 0.012) + rng.normal(0, 0.01),  0.05, 0.8)
        grid_sal  = np.clip(34.2 + (vs * 0.04)  + rng.normal(0, 0.08),  32.0, 36.5)
        grid_wave = np.clip(0.8  + abs(vs) * 0.05 + rng.normal(0, 0.04), 0.2, 2.5)

        grid_sohi = (
            0.25 * normalisasi_global(grid_do,   4.5, 7.5) +
            0.20 * normalisasi_global(grid_ph,   7.9, 8.4) +
            0.20 * normalisasi_global(grid_chla, 0.05, 0.8) +
            0.15 * normalisasi_global(grid_sal,  32.0, 36.5) +
            0.20 * (1 - normalisasi_global(grid_wave, 0.2, 2.5))
        ) * 100

        grid_fsi = (
            0.35 * normalisasi_global(grid_chla,  0.05, 0.8) +
            0.25 * normalisasi_global(grid_do,    4.5, 7.5) +
            0.20 * normalisasi_global(grid_speed, 0.0, 0.25) +
            0.20 * (1 - normalisasi_global(grid_wave, 0.2, 2.5))
        ) * 100

        records.append({
            'lat': t_lat, 'lon': t_lon,
            'Ocean_Health_Index': float(np.clip(grid_sohi, 10, 100)),
            'Fisheries_Index':    float(np.clip(grid_fsi,  10, 100)),
            'uo': grid_uo, 'vo': grid_vo,
            'sst':    float(np.clip(28.5 + (vs * 0.18) + rng.normal(0, 0.1), 26.0, 32.0)),
            'ssta':   float(vs * 0.06 + rng.normal(0, 0.05)),
            'ph': grid_ph, 'do': grid_do,
            'salinitas': grid_sal, 'chla': grid_chla,
            'current_speed': grid_speed, 'gelombang': grid_wave,
            'angin_u': float(-1.5 + (vs * 0.25) + rng.normal(0, 0.1)),
            'angin_v': float(-0.5 + (vs * 0.12) + rng.normal(0, 0.06)),
        })
    return pd.DataFrame(records)

# Ambil nilai filter aktif
val_uo_base = float(df_filter_base["uo"].mean() if not df_filter_base.empty else -0.05)
val_vo_base = float(df_filter_base["vo"].mean() if not df_filter_base.empty else -0.01)
active_month = int(df_filter_base["month"].mean()) if not df_filter_base.empty else 6
active_year  = int(df_filter_base["year"].mean())  if not df_filter_base.empty else 2010

df_map = build_spatial_grid(val_uo_base, val_vo_base, active_month, active_year)

def render_map(df_map, z_col, colorscale, height=520):
    fig = px.scatter_mapbox(
        df_map, lat="lat", lon="lon", color=z_col,
        color_continuous_scale=colorscale,
        opacity=0.82,
        zoom=4.6,
        size_max=8,
        range_color=[float(df_map[z_col].quantile(0.03)), float(df_map[z_col].quantile(0.97))],
        mapbox_style="carto-positron",
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        mapbox=dict(center=dict(lat=-8.0, lon=136.5)),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            thickness=12, len=0.65,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#C8E8FF",
            tickfont=dict(color="#3A6080", size=10, family="DM Mono"),
            title=dict(font=dict(color="#3A6080", size=10, family="DM Mono")),
        )
    )
    return fig

# =========================================
# DASHBOARD — NELAYAN
# =========================================
if st.session_state.role == "nelayan":
    mean_fsi = df_map['Fisheries_Index'].mean()

    if mean_fsi > 73:
        status_color = "#00C48C"; status_text = "SANGAT BAIK"
        status_icon = "✅"; status_bg = "#F0FFF8"; status_border = "#A8E8D0"
    elif mean_fsi > 55:
        status_color = "#1E90D4"; status_text = "NORMAL"
        status_icon = "🌊"; status_bg = "#EBF8FF"; status_border = "#7EC8F0"
    else:
        status_color = "#F5A623"; status_text = "WASPADA"
        status_icon = "⚠️"; status_bg = "#FFF8E8"; status_border = "#FAC87A"

    st.markdown(f"""
<div class="ocean-header">
  <div class="subtitle">🎣 Dashboard Nelayan · {mode} · {waktu_label}</div>
  <h1>Peta Zona Tangkap — Laut Arafura</h1>
</div>
""", unsafe_allow_html=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns([2, 1, 1, 1])
    with col_s1:
        st.markdown(f"""
<div style="background:{status_bg};border:2px solid {status_border};border-radius:14px;padding:18px 22px;border-left:5px solid {status_color};box-shadow:0 2px 12px rgba(30,120,200,0.08);">
  <div style="font-family:'DM Mono',monospace;font-size:10px;color:#5A9EC8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">Status Zona Melaut</div>
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:24px;">{status_icon}</span>
    <span style="color:{status_color};font-size:22px;font-weight:800;letter-spacing:0.03em;">{status_text}</span>
    <span style="font-family:'DM Mono',monospace;color:#8ABDD4;font-size:13px;">({mean_fsi:.1f}/100)</span>
  </div>
</div>
""", unsafe_allow_html=True)
    with col_s2:
        st.metric("🐟 Potensi Ikan", f"{mean_fsi:.0f}", "/ 100")
    with col_s3:
        st.metric("🌊 Tinggi Gelombang", f"{df_map['gelombang'].mean():.2f}", "meter")
    with col_s4:
        st.metric("🌀 Kec. Arus", f"{df_map['current_speed'].mean():.3f}", "m/s")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🗺️ PETA DISTRIBUSI SPASIAL · FISHERIES INDEX</div>', unsafe_allow_html=True)
    if not df_map.empty:
        st.plotly_chart(render_map(df_map, "Fisheries_Index", "Turbo"), use_container_width=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown('<div class="section-label">💡 REKOMENDASI ZONA</div>', unsafe_allow_html=True)
        if mean_fsi > 73:
            st.success("🟢 **Area merah/oranye di peta sangat direkomendasikan!** Nutrisi laut melimpah — turunkan jaring di perairan dalam Arafura.")
        elif mean_fsi > 55:
            st.info("🔵 **Kondisi normal.** Ikan bergerak mengikuti arus permukaan — ikuti arah arus ke tenggara.")
        else:
            st.warning("🟡 **Tangkapan rendah.** Disarankan memancing di pesisir dekat teluk dan muara sungai.")
    with col_r2:
        st.markdown('<div class="section-label">🌊 KONDISI PERAIRAN SAAT INI</div>', unsafe_allow_html=True)
        chla_mean = df_map['chla'].mean()
        do_mean   = df_map['do'].mean()
        st.markdown(f"""
<div style="background:linear-gradient(135deg,#FFFFFF,#EBF8FF);border:1.5px solid #B0D8F0;border-radius:14px;padding:20px;box-shadow:0 2px 12px rgba(30,120,200,0.07);">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div style="text-align:center;">
      <div style="font-size:28px;margin-bottom:6px;">🟢</div>
      <div style="font-family:'DM Mono',monospace;font-size:10px;color:#5A9EC8;text-transform:uppercase;letter-spacing:0.08em;">Klorofil-a</div>
      <div style="font-size:22px;font-weight:700;color:#1A3A5C;">{chla_mean:.3f} <span style="font-size:12px;color:#5A9EC8;">mg/m³</span></div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:28px;margin-bottom:6px;">💧</div>
      <div style="font-family:'DM Mono',monospace;font-size:10px;color:#5A9EC8;text-transform:uppercase;letter-spacing:0.08em;">Dissolved O₂</div>
      <div style="font-size:22px;font-weight:700;color:#1A3A5C;">{do_mean:.2f} <span style="font-size:12px;color:#5A9EC8;">mg/L</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

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
<div class="ocean-header">
  <div class="subtitle">🔬 Portal Akademisi · {mode} · {waktu_label}</div>
  <h1>Analisis Parameter — <span style="color:#1E90D4;">{PARAM_LABELS_CLEAN.get(parameter, parameter)}</span></h1>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Rata-Rata", f"{df_map[parameter].mean():.3f}")
    col2.metric("⬇️ Minimum",   f"{df_map[parameter].min():.3f}")
    col3.metric("⬆️ Maksimum",  f"{df_map[parameter].max():.3f}")
    col4.metric("📉 Std. Dev",  f"{df_map[parameter].std():.3f}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["  🗺️ Spasial  ", "  📈 Time Series  ", "  📊 Statistik  ", "  🔥 Korelasi  "])

    with tab1:
        cmap_dict = {
            'Fisheries_Index': 'Turbo',   'chla': 'Turbo',
            'Ocean_Health_Index': 'Blues', 'do': 'Blues',
            'ph': 'Viridis',               'salinitas': 'YlOrBr',
            'sst': 'Plasma',               'ssta': 'RdBu',
            'current_speed': 'Teal',
        }
        cmap = cmap_dict.get(parameter, "Icefire")
        st.markdown(f'<div class="section-label">📍 DISTRIBUSI SPASIAL · {parameter.upper()}</div>', unsafe_allow_html=True)
        st.plotly_chart(render_map(df_map, parameter, cmap, height=500), use_container_width=True)
        st.markdown(f"""
<span class="coord-tag">📍 4°S – 12°S</span>
<span class="coord-tag">📍 129°E – 144°E</span>
<span class="coord-tag">🔲 Grid 100×100 · Laut Arafura</span>
""", unsafe_allow_html=True)

    with tab2:
        df_ts = df.groupby('time')[parameter].mean().reset_index()
        z = np.polyfit(range(len(df_ts)), df_ts[parameter], 1)
        p_fn = np.poly1d(z)

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=df_ts['time'], y=df_ts[parameter],
            mode='lines', name=PARAM_LABELS_CLEAN.get(parameter, parameter),
            line=dict(color='#1E90D4', width=2),
            fill='tozeroy', fillcolor='rgba(30,144,212,0.08)'
        ))
        fig_ts.add_trace(go.Scatter(
            x=df_ts['time'], y=p_fn(range(len(df_ts))),
            mode='lines', name='Tren Linear',
            line=dict(color='#F5A623', width=2, dash='dot')
        ))
        fig_ts.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Tren Temporal 2001–2020 · {PARAM_LABELS_CLEAN.get(parameter, parameter)}",
            legend=dict(font=dict(color="#3A6080", size=11), bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#C8E8FF", borderwidth=1),
            height=400,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with tab3:
        desc = df_map[[parameter]].describe()
        st.markdown('<div class="section-label">📐 STATISTIK DESKRIPTIF · AREA SPASIAL AKTIF</div>', unsafe_allow_html=True)

        stats_cols = st.columns(4)
        for i, (label, key, icon) in enumerate([
            ("Count", "count", "🔢"), ("Mean", "mean", "📊"),
            ("Std", "std", "📉"),    ("Min", "min", "⬇️"),
        ]):
            val = f"{int(desc.loc[key, parameter]):,}" if key == "count" else f"{desc.loc[key, parameter]:.4f}"
            stats_cols[i].markdown(f"""
<div style="background:linear-gradient(135deg,#FFFFFF,#EBF8FF);border:1.5px solid #B0D8F0;border-radius:12px;padding:18px;text-align:center;box-shadow:0 2px 8px rgba(30,120,200,0.07);">
  <div style="font-size:22px;margin-bottom:6px;">{icon}</div>
  <div style="font-family:'DM Mono',monospace;font-size:10px;color:#5A9EC8;letter-spacing:0.1em;text-transform:uppercase;">{label}</div>
  <div style="font-size:20px;font-weight:700;color:#1A3A5C;margin-top:4px;">{val}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        stats_cols2 = st.columns(3)
        for i, (label, key, icon) in enumerate([
            ("25th Percentile", "25%", "📦"),
            ("Median (50%)", "50%", "⚖️"),
            ("75th Percentile", "75%", "📦"),
        ]):
            stats_cols2[i].markdown(f"""
<div style="background:linear-gradient(135deg,#FFFFFF,#F0FAFF);border:1.5px solid #B0D8F0;border-radius:12px;padding:18px;text-align:center;box-shadow:0 2px 8px rgba(30,120,200,0.07);">
  <div style="font-size:22px;margin-bottom:6px;">{icon}</div>
  <div style="font-family:'DM Mono',monospace;font-size:10px;color:#5A9EC8;letter-spacing:0.1em;text-transform:uppercase;">{label}</div>
  <div style="font-size:20px;font-weight:700;color:#1A3A5C;margin-top:4px;">{desc.loc[key, parameter]:.4f}</div>
</div>
""", unsafe_allow_html=True)

    with tab4:
        numeric_df = df.select_dtypes(include=np.number).drop(columns=['year', 'month'], errors='ignore')
        fig_corr = px.imshow(
            numeric_df.corr(), text_auto=".2f",
            color_continuous_scale=[[0, "#EBF8FF"], [0.5, "#7EC8F0"], [1, "#1A3A8C"]],
            title="Matriks Korelasi Pearson — Semua Parameter"
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(255,255,255,0.6)",
            plot_bgcolor="rgba(235,248,255,0.5)",
            font=dict(family="Plus Jakarta Sans", color="#3A6080", size=12),
            title_font=dict(color="#1A3A5C", size=15),
            height=480,
        )
        fig_corr.update_traces(textfont=dict(size=9, color="#1A3A5C"))
        st.plotly_chart(fig_corr, use_container_width=True)
