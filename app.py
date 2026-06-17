"""
app.py — Dashboard Laut Arafura (Streamlit)
===========================================
Jalankan dengan:
    streamlit run app.py

Tiap layer data diambil terpisah dan punya penanganan error sendiri, jadi kalau
satu API gagal, layer lain tetap tampil. Klik "Tes Koneksi API" di sidebar untuk
mendiagnosis sumber mana yang bermasalah.
"""

import datetime as dt

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import config
import data_sources as ds


st.set_page_config(page_title="Dashboard Laut Arafura", page_icon="🌊", layout="wide")


# ─────────────────────────────────────────────────────────────────────────────
# Caching: bungkus fetcher agar hasilnya di-cache 30 menit
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def get_cmems(d): return ds.fetch_cmems(d)

@st.cache_data(ttl=1800, show_spinner=False)
def get_nasa(d): return ds.fetch_nasa_chlorophyll(d)

@st.cache_data(ttl=1800, show_spinner=False)
def get_era5(d): return ds.fetch_era5_wind(d)

@st.cache_data(ttl=1800, show_spinner=False)
def get_bmkg(code): return ds.fetch_bmkg(code)


# ─────────────────────────────────────────────────────────────────────────────
# Helper plot (matplotlib sederhana, tanpa cartopy supaya gampang dipasang)
# ─────────────────────────────────────────────────────────────────────────────
def _base_extent(ax):
    ax.set_xlim(config.LON_MIN, config.LON_MAX)
    ax.set_ylim(config.LAT_MIN, config.LAT_MAX)
    ax.set_xlabel("Bujur (°E)")
    ax.set_ylabel("Lintang (°)")
    ax.set_aspect("equal", adjustable="box")


def plot_scalar(lon, lat, field, title, label, cmap="viridis", log=False):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    LON, LAT = np.meshgrid(lon, lat)
    data = np.array(field, dtype=float)
    kw = {}
    if log:
        data = np.where(data > 0, data, np.nan)
        from matplotlib.colors import LogNorm
        valid = data[np.isfinite(data)]
        if valid.size:
            kw["norm"] = LogNorm(vmin=max(valid.min(), 1e-3), vmax=valid.max())
    pcm = ax.pcolormesh(LON, LAT, data, cmap=cmap, shading="auto", **kw)
    fig.colorbar(pcm, ax=ax, label=label)
    _base_extent(ax)
    ax.set_title(title)
    return fig


def plot_vectors(lon, lat, u, v, title, label, cmap="plasma"):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    LON, LAT = np.meshgrid(lon, lat)
    speed = np.sqrt(np.array(u, float) ** 2 + np.array(v, float) ** 2)
    pcm = ax.pcolormesh(LON, LAT, speed, cmap=cmap, shading="auto")
    fig.colorbar(pcm, ax=ax, label=label)
    # Jarangkan panah biar tidak penuh
    step = max(1, min(LON.shape) // 18)
    ax.quiver(LON[::step, ::step], LAT[::step, ::step],
              np.array(u, float)[::step, ::step], np.array(v, float)[::step, ::step],
              color="white", scale=20, width=0.003)
    _base_extent(ax)
    ax.set_title(title)
    return fig


def err_box(sumber, msg):
    st.error(f"**{sumber} gagal.** {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Pengaturan")

target_date = st.sidebar.date_input(
    "Tanggal data",
    value=dt.date.today() - dt.timedelta(days=1),
    max_value=dt.date.today(),
)
st.sidebar.caption("Catatan: ERA5 (angin) telat ±5 hari, klorofil MODIS telat 1-3 hari. "
                   "Kalau kosong, mundurkan tanggalnya.")

layers = st.sidebar.multiselect(
    "Tampilkan layer",
    ["Suhu Permukaan Laut (CMEMS)", "Arus Permukaan (CMEMS)",
     "Klorofil-a (NASA)", "Angin 10 m (ERA5)", "Cuaca BMKG"],
    default=["Suhu Permukaan Laut (CMEMS)", "Arus Permukaan (CMEMS)"],
)

st.sidebar.divider()
run_test = st.sidebar.button("🔌 Tes Koneksi API")

# Peringatan kredensial yang belum diisi
miss = config.missing_credentials()
if any(miss.values()):
    belum = ", ".join([k for k, v in miss.items() if v])
    st.sidebar.warning(f"Belum dikonfigurasi: {belum}")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🌊 Dashboard Laut Arafura")
st.caption(f"Wilayah: lintang {config.LAT_MIN}° s/d {config.LAT_MAX}°, "
           f"bujur {config.LON_MIN}°E s/d {config.LON_MAX}°E  •  Tanggal: {target_date}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel diagnostik
# ─────────────────────────────────────────────────────────────────────────────
if run_test:
    st.subheader("🔌 Hasil Tes Koneksi")
    with st.spinner("Mengecek tiap API..."):
        results = ds.test_connections()
    for r in results:
        icon = "✅" if r["ok"] else "❌"
        (st.success if r["ok"] else st.error)(f"{icon} **{r['sumber']}** — {r['pesan']}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Layer: SST
# ─────────────────────────────────────────────────────────────────────────────
if "Suhu Permukaan Laut (CMEMS)" in layers:
    st.subheader("🌡️ Suhu Permukaan Laut (CMEMS)")
    with st.spinner("Mengambil data SST dari CMEMS..."):
        r = get_cmems(target_date)
    if r["ok"]:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.pyplot(plot_scalar(r["lon"], r["lat"], r["sst"],
                                  f"SST {r.get('time','')}", "°C", cmap="turbo"))
        with c2:
            vals = np.array(r["sst"], float)
            st.metric("Rata-rata", f"{np.nanmean(vals):.2f} °C")
            st.metric("Min / Maks", f"{np.nanmin(vals):.1f} / {np.nanmax(vals):.1f} °C")
    else:
        err_box("CMEMS (SST)", r["error"])


# ─────────────────────────────────────────────────────────────────────────────
# Layer: Arus
# ─────────────────────────────────────────────────────────────────────────────
if "Arus Permukaan (CMEMS)" in layers:
    st.subheader("🧭 Arus Permukaan (CMEMS)")
    with st.spinner("Mengambil data arus dari CMEMS..."):
        r = get_cmems(target_date)  # cache → tidak fetch ulang
    if r["ok"]:
        st.pyplot(plot_vectors(r["lon"], r["lat"], r["u"], r["v"],
                               f"Arus permukaan {r.get('time','')}", "Kecepatan (m/s)"))
    else:
        err_box("CMEMS (Arus)", r["error"])


# ─────────────────────────────────────────────────────────────────────────────
# Layer: Klorofil
# ─────────────────────────────────────────────────────────────────────────────
if "Klorofil-a (NASA)" in layers:
    st.subheader("🟢 Klorofil-a (NASA MODIS Aqua)")
    with st.spinner("Mencari & mengunduh granul klorofil dari NASA..."):
        r = get_nasa(target_date)
    if r["ok"]:
        st.pyplot(plot_scalar(r["lon"], r["lat"], r["chl"],
                              "Klorofil-a", "mg/m³", cmap="YlGn", log=True))
        st.caption(f"Granul: {r.get('time','')}")
    else:
        err_box("NASA (Klorofil)", r["error"])


# ─────────────────────────────────────────────────────────────────────────────
# Layer: Angin
# ─────────────────────────────────────────────────────────────────────────────
if "Angin 10 m (ERA5)" in layers:
    st.subheader("💨 Angin 10 m (ERA5 / Copernicus CDS)")
    era_date = target_date if target_date <= dt.date.today() - dt.timedelta(days=6) else dt.date.today() - dt.timedelta(days=6)
    if era_date != target_date:
        st.caption(f"ERA5 telat ±5 hari → memakai tanggal {era_date}.")
    with st.spinner("Mengambil data angin dari CDS (bisa antre beberapa menit)..."):
        r = get_era5(era_date)
    if r["ok"]:
        st.pyplot(plot_vectors(r["lon"], r["lat"], r["u10"], r["v10"],
                               f"Angin 10 m {r.get('time','')}", "Kecepatan (m/s)", cmap="cividis"))
    else:
        err_box("ERA5/CDS (Angin)", r["error"])


# ─────────────────────────────────────────────────────────────────────────────
# Layer: BMKG
# ─────────────────────────────────────────────────────────────────────────────
if "Cuaca BMKG" in layers:
    st.subheader("☁️ Prakiraan Cuaca BMKG")
    with st.spinner("Mengambil prakiraan dari BMKG..."):
        r = get_bmkg(config.BMKG_ADM4)
    if r["ok"]:
        st.markdown(f"**Lokasi:** {r['location']}")
        df = r["forecast"]
        st.dataframe(df, use_container_width=True, hide_index=True)
        if "suhu_C" in df and df["waktu"].notna().any():
            chart_df = df.dropna(subset=["waktu"]).set_index("waktu")[["suhu_C", "kelembapan_%"]]
            st.line_chart(chart_df)
    else:
        err_box("BMKG", r["error"])


if not layers:
    st.info("Pilih minimal satu layer di sidebar untuk mulai.")

st.divider()
st.caption("Sumber data: Copernicus Marine Service, NASA OB.DAAC, ECMWF/Copernicus CDS, BMKG.")
