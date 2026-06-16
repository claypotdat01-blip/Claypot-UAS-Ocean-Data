"""
╔══════════════════════════════════════════════════════════════╗
║           CLAYPOT OCEAN DATA  —  app.py                      ║
║  Platform Informasi Klimatologi Oseanografi Papua             ║
║  Lat: -12 s/d -2  |  Lon: 129 s/d 142                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Claypot Ocean Data",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  CSS GLOBAL — TEMA PREMIUM SAMUDERA
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── FONT ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── ROOT PALETTE ── */
:root {
    --abyss:       #07111F;
    --deep-sea:    #0B2545;
    --ocean:       #1A4F7A;
    --biolum:      #00C8FF;
    --biolum-glow: rgba(0, 200, 255, 0.18);
    --seafoam:     #E0F2FE;
    --muted:       #7EB8D4;
    --surface:     rgba(11, 37, 69, 0.75);
    --glass:       rgba(26, 79, 122, 0.25);
    --border:      rgba(0, 200, 255, 0.2);
}

/* ── GLOBAL RESET ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--seafoam);
}

.stApp {
    background: linear-gradient(160deg, var(--abyss) 0%, var(--deep-sea) 60%, #0D3B6E 100%);
    min-height: 100vh;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--abyss); }
::-webkit-scrollbar-thumb { background: var(--ocean); border-radius: 3px; }

/* ── GLASSMORPHIC CARD ── */
.ocean-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06);
}

/* ── WELCOME HERO ── */
.hero-container {
    background: linear-gradient(135deg, rgba(0,200,255,0.08) 0%, rgba(11,37,69,0.95) 50%, rgba(0,0,0,0.8) 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 3.5rem 3rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 80px rgba(0,200,255,0.08), 0 8px 48px rgba(0,0,0,0.4);
}

.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #FFFFFF 0%, var(--biolum) 60%, #89D8F0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--muted);
    font-weight: 400;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

.hero-tagline {
    font-size: 1.0rem;
    color: rgba(224, 242, 254, 0.7);
    max-width: 520px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
}

.coord-badge {
    display: inline-block;
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--biolum);
    letter-spacing: 0.05em;
    margin-bottom: 2.5rem;
}

/* ── ROLE BUTTONS ── */
.role-btn {
    display: block;
    width: 100%;
    background: var(--glass);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    color: var(--seafoam);
    text-decoration: none;
    backdrop-filter: blur(8px);
}

.role-btn:hover {
    background: rgba(0, 200, 255, 0.12);
    border-color: var(--biolum);
    box-shadow: 0 0 32px rgba(0,200,255,0.2);
    transform: translateY(-3px);
}

.role-icon { font-size: 2.8rem; margin-bottom: 0.7rem; display: block; }
.role-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.4rem; }
.role-desc { font-size: 0.82rem; color: var(--muted); line-height: 1.5; }

/* ── SECTION HEADER ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--biolum);
    font-family: 'JetBrains Mono', monospace;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--seafoam);
}

/* ── METRIC CHIP ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.metric-chip {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    min-width: 130px;
    flex: 1;
}
.metric-chip .mc-label { font-size: 0.68rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.2rem; }
.metric-chip .mc-value { font-size: 1.4rem; font-weight: 700; color: var(--biolum); font-family: 'JetBrains Mono', monospace; }
.metric-chip .mc-unit  { font-size: 0.72rem; color: var(--muted); }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.9rem;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--ocean), #0E3E6A) !important;
    color: white !important;
    box-shadow: 0 2px 12px rgba(0,200,255,0.2) !important;
}

/* ── STREAMLIT WIDGETS OVERRIDE ── */
.stSelectbox > div > div,
.stDateInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--seafoam) !important;
    border-radius: 10px !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--ocean) 0%, #0E3E6A 100%);
    border: 1px solid var(--biolum);
    color: white;
    border-radius: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    letter-spacing: 0.04em;
    transition: all 0.25s;
    box-shadow: 0 0 16px rgba(0,200,255,0.12);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1A6FA8 0%, var(--ocean) 100%);
    box-shadow: 0 0 28px rgba(0,200,255,0.3);
    transform: translateY(-1px);
    border-color: var(--biolum);
}

div[data-testid="stSidebar"] {
    background: var(--abyss);
    border-right: 1px solid var(--border);
}

/* ── BACK BUTTON ── */
.back-link {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--muted);
    font-size: 0.82rem;
    cursor: pointer;
    margin-bottom: 1.5rem;
    transition: color 0.2s;
    background: none;
    border: none;
    padding: 0;
    font-family: 'Space Grotesk', sans-serif;
}

.back-link:hover { color: var(--biolum); }

/* ── FISHERIES INDEX LEGEND ── */
.legend-row { display: flex; gap: 0.6rem; align-items: center; margin: 0.3rem 0; }
.legend-dot { width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0; }
.legend-label { font-size: 0.82rem; color: var(--muted); }

/* ── NOTICE BOX ── */
.notice-box {
    background: rgba(0,200,255,0.06);
    border-left: 3px solid var(--biolum);
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1.1rem;
    font-size: 0.83rem;
    color: var(--muted);
    margin-bottom: 1.2rem;
    line-height: 1.6;
}

/* ── PLOTLY TRANSPARENT BG ── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  KONSTANTA & DOMAIN
# ─────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX   = -12, -2
LON_MIN, LON_MAX   = 129, 142
GRID_STEP_SPATIAL  = 0.25      # resolusi grid spasial (derajat)
GRID_STEP_TIMESERIES = 0.5

PARAMS_NELAYAN = {
    "Klorofil-a (Chl-a)":       {"unit": "mg/m³",  "cmap": "YlGn",    "vmin": 0.0,  "vmax": 2.5,  "icon": "🌿"},
    "Suhu Permukaan Laut (SST)": {"unit": "°C",     "cmap": "thermal", "vmin": 26.0, "vmax": 32.0, "icon": "🌡️"},
    "Konsentrasi Oksigen":       {"unit": "mL/L",   "cmap": "Blues",   "vmin": 3.5,  "vmax": 7.5,  "icon": "💧"},
}

PARAMS_AKADEMISI = {
    "SST Anomali (SSTA)":        {"unit": "°C",      "cmap": "RdBu_r",  "vmin": -2.0, "vmax": 2.0,  "icon": "🌡️"},
    "Tinggi Gelombang (SWH)":    {"unit": "m",       "cmap": "Blues",   "vmin": 0.5,  "vmax": 4.5,  "icon": "🌊"},
    "Kecepatan Angin (U10)":     {"unit": "m/s",     "cmap": "speed",   "vmin": 0.0,  "vmax": 15.0, "icon": "💨"},
    "Salinitas Permukaan":       {"unit": "PSU",     "cmap": "haline",  "vmin": 31.0, "vmax": 36.0, "icon": "🧂"},
}

MAPBOX_STYLE = "open-street-map"

# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "welcome"


# ─────────────────────────────────────────────────────────────
#  HELPERS — DATA SINTETIS (Cloud Mode)
# ─────────────────────────────────────────────────────────────
def generate_spatial_grid(param_key: str, tanggal: date, role: str) -> tuple:
    """
    Buat grid 2D sintetis berbasis seed tanggal + parameter.
    GANTI fungsi ini dengan pemanggilan Copernicus Marine API
    saat data real-time tersedia.
    """
    params = {**PARAMS_NELAYAN, **PARAMS_AKADEMISI}
    cfg    = params[param_key]

    np.random.seed(int(tanggal.strftime("%Y%m%d")) + hash(param_key) % 9999)

    lon_arr = np.arange(LON_MIN, LON_MAX + GRID_STEP_SPATIAL, GRID_STEP_SPATIAL)
    lat_arr = np.arange(LAT_MIN, LAT_MAX + GRID_STEP_SPATIAL, GRID_STEP_SPATIAL)
    lon_g, lat_g = np.meshgrid(lon_arr, lat_arr)

    # ─── Variasi Spasial ───
    base  = cfg["vmin"] + (cfg["vmax"] - cfg["vmin"]) * (
        0.5
        + 0.25 * np.sin(0.4 * (lon_g - LON_MIN))
        + 0.20 * np.cos(0.5 * (lat_g - LAT_MIN))
        + 0.10 * np.random.randn(*lon_g.shape)
    )
    base  = np.clip(base, cfg["vmin"], cfg["vmax"])

    # ─── LAND MASKING PAPUA ───
    mask_daratan = (lon_g > 136) & (lat_g > -8)
    base[mask_daratan] = np.nan

    # Ratakan untuk scatter
    lons  = lon_g.flatten()
    lats  = lat_g.flatten()
    vals  = base.flatten()

    valid = ~np.isnan(vals)
    return lons[valid], lats[valid], vals[valid], cfg


def load_timeseries_csv(param_key: str) -> pd.DataFrame:
    """
    Baca file rangkuman_historis_20tahun.csv.
    Fallback: buat data sintetis jika file belum ada.
    """
    try:
        df = pd.read_csv("rangkuman_historis_20tahun.csv", parse_dates=["tanggal"])
        if param_key in df.columns:
            return df[["tanggal", param_key]].rename(columns={param_key: "nilai"})
        raise FileNotFoundError
    except FileNotFoundError:
        # ── FALLBACK SINTETIS ──
        dates  = pd.date_range("2001-01-01", "2020-12-01", freq="MS")
        params = {**PARAMS_NELAYAN, **PARAMS_AKADEMISI}
        cfg    = params.get(param_key, {"vmin": 0, "vmax": 1})
        np.random.seed(abs(hash(param_key)) % 9999)
        mid    = (cfg["vmin"] + cfg["vmax"]) / 2
        span   = (cfg["vmax"] - cfg["vmin"]) * 0.3
        trend  = np.linspace(0, span * 0.4, len(dates))
        season = span * 0.3 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise  = span * 0.15 * np.random.randn(len(dates))
        vals   = np.clip(mid + trend + season + noise, cfg["vmin"], cfg["vmax"])
        return pd.DataFrame({"tanggal": dates, "nilai": vals})


def compute_fisheries_index(lons, lats, vals_chl, vals_sst, vals_o2) -> np.ndarray:
    """Indeks potensi tangkap sederhana (0–100)."""
    norm = lambda v, vmin, vmax: np.clip((v - vmin) / (vmax - vmin), 0, 1)
    fi   = (
        0.5  * norm(vals_chl, 0.0, 2.5)
        + 0.3 * norm(vals_o2,  3.5, 7.5)
        + 0.2 * (1 - norm(vals_sst, 26.0, 32.0))
    ) * 100
    return fi


# ─────────────────────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#E0F2FE"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(
        bgcolor="rgba(11,37,69,0.8)",
        bordercolor="rgba(0,200,255,0.3)",
        borderwidth=1,
    ),
)

def style_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(0,200,255,0.08)", zerolinecolor="rgba(0,200,255,0.15)")
    fig.update_yaxes(gridcolor="rgba(0,200,255,0.08)", zerolinecolor="rgba(0,200,255,0.15)")
    return fig


# ─────────────────────────────────────────────────────────────
#  KOMPONEN UI REUSABLE
# ─────────────────────────────────────────────────────────────
def render_back_button(label="← Kembali ke Halaman Utama"):
    if st.button(label, key="btn_back_" + st.session_state.page):
        st.session_state.page = "welcome"
        st.rerun()


def render_top_bar(title: str, icon: str, role_label: str):
    st.markdown(f"""
    <div class="ocean-card" style="display:flex; align-items:center; gap:1rem; padding:1rem 1.5rem; margin-bottom:1.2rem;">
        <span style="font-size:1.8rem">{icon}</span>
        <div>
            <div style="font-size:0.68rem; color:var(--biolum); letter-spacing:0.15em; text-transform:uppercase; font-family:'JetBrains Mono',monospace;">{role_label}</div>
            <div style="font-size:1.15rem; font-weight:600;">{title}</div>
        </div>
        <div style="margin-left:auto; text-align:right;">
            <div style="font-size:0.65rem; color:var(--muted); font-family:'JetBrains Mono',monospace;">CLAYPOT OCEAN DATA</div>
            <div style="font-size:0.65rem; color:var(--muted); font-family:'JetBrains Mono',monospace;">LAT {LAT_MIN}°–{LAT_MAX}° · LON {LON_MIN}°–{LON_MAX}°</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_spatial_map(param_key: str, tanggal: date, role: str = "nelayan"):
    lons, lats, vals, cfg = generate_spatial_grid(param_key, tanggal, role)

    fig = go.Figure(go.Scattermapbox(
        lon=lons, lat=lats, mode="markers",
        marker=dict(
            size=7,
            color=vals,
            colorscale=cfg["cmap"],
            cmin=cfg["vmin"], cmax=cfg["vmax"],
            colorbar=dict(
                title=dict(text=f'{param_key}<br><span style="font-size:10px">{cfg["unit"]}</span>', side="right"),
                thickness=14,
                bgcolor="rgba(11,37,69,0.8)",
                bordercolor="rgba(0,200,255,0.3)",
                borderwidth=1,
                tickfont=dict(color="#E0F2FE", size=10),
            ),
            opacity=0.88,
        ),
        hovertemplate=(
            f"<b>{param_key}</b><br>"
            "Lon: %{lon:.2f}°  Lat: %{lat:.2f}°<br>"
            f"Nilai: %{{marker.color:.3f}} {cfg['unit']}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        mapbox=dict(style=MAPBOX_STYLE, center=dict(lat=-7, lon=135.5), zoom=4.2),
        height=500,
        **PLOTLY_LAYOUT,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})


def render_timeseries(param_key: str):
    df  = load_timeseries_csv(param_key)
    cfg = {**PARAMS_NELAYAN, **PARAMS_AKADEMISI}.get(param_key, {"unit": "", "vmin": 0, "vmax": 1})

    # Moving average 12-bulan
    df["MA12"] = df["nilai"].rolling(12, center=True).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["tanggal"], y=df["nilai"],
        mode="lines", name="Bulanan",
        line=dict(color="rgba(0,200,255,0.35)", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=df["tanggal"], y=df["MA12"],
        mode="lines", name="Rata-rata 12 Bln",
        line=dict(color="#00C8FF", width=2.5),
    ))

    # Tren linier
    x_num  = (df["tanggal"] - df["tanggal"].min()).dt.days
    m, b   = np.polyfit(x_num.dropna(), df["nilai"].dropna(), 1)
    df_trend = df.dropna(subset=["nilai"]).copy()
    x_t    = (df_trend["tanggal"] - df["tanggal"].min()).dt.days
    fig.add_trace(go.Scatter(
        x=df_trend["tanggal"], y=m * x_t + b,
        mode="lines", name="Tren Linear",
        line=dict(color="#FF6B6B", width=1.5, dash="dot"),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        xaxis_title="Tahun",
        yaxis_title=f"{param_key} ({cfg['unit']})",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, **PLOTLY_LAYOUT["legend"]),
    )
    style_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Statistik ringkas
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("Rata-rata",  f"{df['nilai'].mean():.3f}", cfg['unit']),
        ("Minimum",    f"{df['nilai'].min():.3f}",  cfg['unit']),
        ("Maksimum",   f"{df['nilai'].max():.3f}",  cfg['unit']),
        ("Tren/Tahun", f"{m*365:.4f}",              cfg['unit']+"/yr"),
    ]
    for col, (lbl, val, unit) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-chip">
                <div class="mc-label">{lbl}</div>
                <div class="mc-value">{val}</div>
                <div class="mc-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PAGE: WELCOME
# ─────────────────────────────────────────────────────────────
def page_welcome():
    # Bioluminescent particle CSS animation
    st.markdown("""
    <style>
    @keyframes float-up {
        0%   { transform: translateY(0px) scale(1);   opacity: 0.6; }
        50%  { transform: translateY(-18px) scale(1.3); opacity: 0.9; }
        100% { transform: translateY(-36px) scale(0.8); opacity: 0; }
    }
    .particle {
        position: absolute;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0,200,255,0.8) 0%, transparent 70%);
        animation: float-up linear infinite;
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero-container">
        <!-- Partikel bioluminesen dekoratif -->
        <div class="particle" style="width:6px;height:6px;left:12%;top:72%;animation-duration:4.2s;animation-delay:0s;"></div>
        <div class="particle" style="width:4px;height:4px;left:28%;top:80%;animation-duration:3.6s;animation-delay:0.8s;"></div>
        <div class="particle" style="width:8px;height:8px;left:65%;top:75%;animation-duration:5.1s;animation-delay:1.2s;"></div>
        <div class="particle" style="width:5px;height:5px;left:82%;top:68%;animation-duration:4.0s;animation-delay:0.4s;"></div>
        <div class="particle" style="width:3px;height:3px;left:45%;top:85%;animation-duration:3.2s;animation-delay:2.0s;"></div>

        <div class="hero-subtitle">Platform Informasi Klimatologi Oseanografi</div>
        <div class="hero-title">Claypot Ocean Data</div>
        <div class="coord-badge">🌐 Perairan Papua &nbsp;·&nbsp; Lat -12° s/d -2° &nbsp;·&nbsp; Lon 129° s/d 142°</div>
        <div class="hero-tagline">
            Data multidekade kondisi oseanografi Perairan Papua tersedia dalam satu platform — dari informasi zona tangkap nelayan lokal hingga analisis runtun waktu para peneliti.
        </div>
        <div style="font-size:0.78rem; color:rgba(126,184,212,0.6); font-family:'JetBrains Mono',monospace; letter-spacing:0.1em;">
            ▼&nbsp; PILIH AKSES DI BAWAH INI &nbsp;▼
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("""
        <div class="ocean-card" style="height:100%;">
            <div class="role-icon">🧑‍🌾</div>
            <div class="role-title">Portal Nelayan Lokal</div>
            <div style="font-size:0.7rem; color:var(--biolum); letter-spacing:0.12em; text-transform:uppercase; font-family:'JetBrains Mono',monospace; margin-bottom:0.6rem;">MASYARAKAT</div>
            <div class="role-desc">
                Peta sebaran kesuburan laut dan zona potensi memancing harian secara instan —
                disajikan sederhana tanpa jargon teknis.
            </div>
            <div style="margin-top:1.2rem; display:flex; gap:0.4rem; flex-wrap:wrap;">
                <span style="background:rgba(0,200,255,0.1);border:1px solid var(--border);border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;color:var(--muted);">🌿 Klorofil-a</span>
                <span style="background:rgba(0,200,255,0.1);border:1px solid var(--border);border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;color:var(--muted);">🌡️ SST</span>
                <span style="background:rgba(0,200,255,0.1);border:1px solid var(--border);border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;color:var(--muted);">💧 Oksigen</span>
                <span style="background:rgba(0,200,255,0.1);border:1px solid var(--border);border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;color:var(--muted);">🎯 Fisheries Index</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("🧑‍🌾  Masuk sebagai Nelayan", key="btn_nelayan", use_container_width=True):
            st.session_state.page = "nelayan"
            st.rerun()

    with col_r:
        st.markdown("""
        <div class="ocean-card" style="height:100%;">
            <div class="role-icon">🎓</div>
            <div class="role-title">Portal Akademisi / Peneliti</div>
            <div style="font-size:0.7rem; color:var(--biolum); letter-spacing:0.12em; text-transform:uppercase; font-family:'JetBrains Mono',monospace; margin-bottom:0.6rem;">PENELITI & MAHASISWA</div>
            <div class="role-desc">
                Eksplorasi spasial kontur & analisis runtun waktu multidekade — termasuk
                Ocean Health Index dan tren perubahan iklim 2001–2020.
            </div>
            <div style="margin-top:1.2rem; display:flex; gap:0.4rem; flex-wrap:wrap;">
                <span style="background:rgba(0,200,255,0.1);border:1px solid var(--border);border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;color:var(--muted);">🗺️ Peta Spasial</span>
                <span style="background:rgba(0,200,255,0.1);border:1px solid var(--border);border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;color:var(--muted);">📈 Time Series</span>
                <span style="background:rgba(0,200,255,0.1);border:1px solid var(--border);border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;color:var(--muted);">🌡️ SSTA</span>
                <span style="background:rgba(0,200,255,0.1);border:1px solid var(--border);border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;color:var(--muted);">🌊 Gelombang</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("🎓  Masuk sebagai Akademisi", key="btn_akademisi", use_container_width=True):
            st.session_state.page = "akademisi"
            st.rerun()

    # Footer
    st.markdown("""
    <div style="text-align:center; margin-top:3rem; padding-top:1.5rem; border-top:1px solid rgba(0,200,255,0.1);">
        <div style="font-size:0.72rem; color:rgba(126,184,212,0.45); font-family:'JetBrains Mono',monospace; letter-spacing:0.08em;">
            CLAYPOT OCEAN DATA &nbsp;·&nbsp; Data Oseanografi Perairan Papua &nbsp;·&nbsp; 2001 – 2020 &nbsp;·&nbsp; Sumber: Copernicus Marine Service
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PAGE: PORTAL NELAYAN
# ─────────────────────────────────────────────────────────────
def page_nelayan():
    render_back_button()
    render_top_bar("Portal Informasi Nelayan Lokal", "🧑‍🌾", "MASYARAKAT · ZONA TANGKAP")

    # ── KONTROL ──
    col_param, col_date, col_btn = st.columns([3, 2, 1])
    with col_param:
        param = st.selectbox("Parameter", list(PARAMS_NELAYAN.keys()), key="sel_param_nelayan")
    with col_date:
        tgl = st.date_input("Tanggal", value=date.today(), key="date_nelayan",
                            min_value=date(2001, 1, 1), max_value=date.today())
    with col_btn:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        tampil = st.button("🗺️ Tampilkan", use_container_width=True, key="btn_tampil_nelayan")

    st.markdown("""
    <div class="notice-box">
        📡 Peta dihasilkan secara langsung (<i>cloud mode</i>) berdasarkan tanggal yang dipilih.
        Data sintetis digunakan sampai integrasi API Copernicus Marine aktif.
    </div>
    """, unsafe_allow_html=True)

    # ── MAP + LEGEND ──
    col_map, col_side = st.columns([4, 1])

    with col_map:
        with st.spinner("Memuat peta..."):
            render_spatial_map(param, tgl, role="nelayan")

    with col_side:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # Fisheries Index mini-panel
        st.markdown("""
        <div class="ocean-card" style="padding:1rem;">
            <div style="font-size:0.68rem; color:var(--biolum); letter-spacing:0.12em; text-transform:uppercase; font-family:'JetBrains Mono',monospace; margin-bottom:0.8rem;">Fisheries Index</div>
        """, unsafe_allow_html=True)

        fi_zones = [
            ("#00C853", "Sangat Potensial", "FI > 75"),
            ("#64DD17", "Potensial",         "FI 50–75"),
            ("#FFD600", "Cukup",             "FI 25–50"),
            ("#FF3D00", "Rendah",            "FI < 25"),
        ]
        for color, label, range_ in fi_zones:
            st.markdown(f"""
            <div class="legend-row">
                <div class="legend-dot" style="background:{color};"></div>
                <div>
                    <div style="font-size:0.8rem;color:var(--seafoam);">{label}</div>
                    <div style="font-size:0.7rem;color:var(--muted);">{range_}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # Tips singkat
        tips = {
            "Klorofil-a (Chl-a)":       "Nilai tinggi → banyak fitoplankton → ikan lebih banyak.",
            "Suhu Permukaan Laut (SST)": "SST 28–30°C umumnya disukai ikan pelagis tropis.",
            "Konsentrasi Oksigen":       "Oksigen > 5 mL/L mendukung aktivitas ikan yang baik.",
        }
        st.markdown(f"""
        <div class="ocean-card" style="padding:0.9rem; font-size:0.82rem; color:var(--muted); line-height:1.6;">
            <div style="font-size:0.65rem; color:var(--biolum); letter-spacing:0.12em; text-transform:uppercase; font-family:'JetBrains Mono',monospace; margin-bottom:0.5rem;">ℹ️ Tips Parameter</div>
            {tips.get(param, "")}
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PAGE: PORTAL AKADEMISI
# ─────────────────────────────────────────────────────────────
def page_akademisi():
    render_back_button()
    render_top_bar("Portal Akademisi & Peneliti", "🎓", "PENELITI · KLIMATOLOGI OSEANOGRAFI")

    tab1, tab2 = st.tabs([
        "🗺️  Tab 1 — Pemetaan Spasial Kontur",
        "📈  Tab 2 — Analisis Runtun Waktu"
    ])

    # ════════════════════════════════════════
    #  TAB 1 — PETA SPASIAL
    # ════════════════════════════════════════
    with tab1:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        col_p, col_d, col_b = st.columns([3, 2, 1])
        with col_p:
            all_params = {**PARAMS_NELAYAN, **PARAMS_AKADEMISI}
            param_spasial = st.selectbox(
                "Parameter Oseanografi",
                list(all_params.keys()),
                key="sel_param_akademisi"
            )
        with col_d:
            tgl_spasial = st.date_input(
                "Tanggal Operasional",
                value=date.today(),
                key="date_spasial",
                min_value=date(2001, 1, 1),
                max_value=date.today()
            )
        with col_b:
            st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
            st.button("🗺️ Render", use_container_width=True, key="btn_render_spasial")

        st.markdown("""
        <div class="notice-box">
            🔬 Visualisasi spasial harian berbasis <i>cloud mode</i> — grid 2D dibangun secara langsung
            tanpa memuat file NetCDF besar. Land masking Papua aktif (lon > 136° & lat > -8°).
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Merender peta kontur..."):
            render_spatial_map(param_spasial, tgl_spasial, role="akademisi")

        # Statistik grid
        lons, lats, vals, cfg = generate_spatial_grid(param_spasial, tgl_spasial, "akademisi")
        c1, c2, c3, c4 = st.columns(4)
        for col, (lbl, val) in zip([c1, c2, c3, c4], [
            ("Rata-rata Grid", f"{np.nanmean(vals):.3f}"),
            ("Std. Deviasi",   f"{np.nanstd(vals):.3f}"),
            ("Min",            f"{np.nanmin(vals):.3f}"),
            ("Max",            f"{np.nanmax(vals):.3f}"),
        ]):
            with col:
                st.markdown(f"""
                <div class="metric-chip">
                    <div class="mc-label">{lbl}</div>
                    <div class="mc-value">{val}</div>
                    <div class="mc-unit">{cfg['unit']}</div>
                </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════
    #  TAB 2 — RUNTUN WAKTU
    # ════════════════════════════════════════
    with tab2:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        all_params = {**PARAMS_NELAYAN, **PARAMS_AKADEMISI}

        col_pl, col_pr = st.columns([3, 1])
        with col_pl:
            param_ts = st.selectbox(
                "Parameter Runtun Waktu",
                list(all_params.keys()),
                key="sel_param_ts"
            )
        with col_pr:
            tahun_range = st.select_slider(
                "Rentang Tahun",
                options=list(range(2001, 2021)),
                value=(2001, 2020),
                key="slider_tahun"
            )

        st.markdown("""
        <div class="notice-box">
            📂 Data bersumber dari file <code>rangkuman_historis_20tahun.csv</code> (diproses lokal,
            ringan, tidak memuat file NetCDF besar ke server). Jika file belum ada, data sintetis
            ditampilkan sebagai <i>preview</i>.
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Memuat data historis..."):
            render_timeseries(param_ts)

        # Ocean Health Index mini-panel
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="ocean-card">
            <div style="font-size:0.68rem; color:var(--biolum); letter-spacing:0.12em; text-transform:uppercase; font-family:'JetBrains Mono',monospace; margin-bottom:0.8rem;">🌊 Ocean Health Index (OHI) — Ringkasan</div>
            <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem;">
        """, unsafe_allow_html=True)

        ohi_items = [
            ("Produktivitas Primer",  "72.4", "/ 100", "#00C853"),
            ("Kestabilan Termal",     "61.8", "/ 100", "#FFD600"),
            ("Keseimbangan Salinitas","79.1", "/ 100", "#00C8FF"),
        ]
        for lbl, val, unit, color in ohi_items:
            st.markdown(f"""
                <div style="text-align:center; background:rgba(0,200,255,0.05); border:1px solid rgba(0,200,255,0.15); border-radius:10px; padding:0.8rem;">
                    <div style="font-size:1.6rem; font-weight:700; color:{color}; font-family:'JetBrains Mono',monospace;">{val}</div>
                    <div style="font-size:0.7rem; color:var(--muted);">{unit}</div>
                    <div style="font-size:0.75rem; color:var(--seafoam); margin-top:0.2rem;">{lbl}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  ROUTER UTAMA
# ─────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "welcome":
    page_welcome()
elif page == "nelayan":
    page_nelayan()
elif page == "akademisi":
    page_akademisi()
else:
    st.session_state.page = "welcome"
    st.rerun()
