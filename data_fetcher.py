LAUTAN — Data Fetcher
=====================================================================
Mengambil data oseanografi real-time dari:
  1. CMEMS  → arus (uo, vo), SST, salinitas, klorofil-a (CHL)
  2. ERA5   → angin (angin_u, angin_v)
  3. BMKG   → gelombang
  4. NASA MODIS → TIDAK DIGUNAKAN (diganti CMEMS untuk klorofil-a)

Perubahan utama:
  - fetch_nasa_modis() diganti fetch_cmems_chla() yang mengambil
    klorofil-a dari CMEMS dataset Ocean Colour (CHL).
  - build_realtime_dataframe() tidak lagi memanggil NASA/MODIS;
    semua parameter laut kini dari satu sumber: CMEMS.
=====================================================================
"""

import datetime
import numpy as np
import pandas as pd

# ── Lazy-import API clients ────────────────────────────────────────
def _import_copernicusmarine():
    try:
        import copernicusmarine as cm
        return cm
    except ImportError:
        return None

def _import_cdsapi():
    try:
        import cdsapi
        return cdsapi
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────
# 1. CMEMS — Arus, SST, Salinitas
# ─────────────────────────────────────────────────────────────────
def fetch_cmems(cmems_user: str, cmems_pass: str,
                lat_min=-12.0, lat_max=-4.0,
                lon_min=129.0, lon_max=144.0) -> dict:
    """
    Ambil data arus (uo, vo), SST, salinitas dari CMEMS.
    Dataset: cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m (Physics)

    Returns dict dengan key: uo, vo, sst, salinitas
    Nilai adalah rata-rata spasial area Laut Arafura.
    Mengembalikan None jika gagal.
    """
    cm = _import_copernicusmarine()
    if cm is None:
        print("[CMEMS] copernicusmarine tidak terinstal")
        return None

    now   = datetime.datetime.utcnow()
    today = now.strftime("%Y-%m-%dT00:00:00")
    yest  = (now - datetime.timedelta(days=2)).strftime("%Y-%m-%dT00:00:00")

    try:
        ds = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            variables=["uo", "vo"],
            minimum_latitude=lat_min,
            maximum_latitude=lat_max,
            minimum_longitude=lon_min,
            maximum_longitude=lon_max,
            start_datetime=yest,
            end_datetime=today,
            username=cmems_user,
            password=cmems_pass,
        )
        uo_val = float(ds["uo"].mean(skipna=True).values)
        vo_val = float(ds["vo"].mean(skipna=True).values)
        ds.close()
    except Exception as e:
        print(f"[CMEMS] Gagal ambil arus: {e}")
        return None

    # SST & Salinitas dari dataset terpisah
    sst_val = sal_val = None
    try:
        ds2 = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
            variables=["thetao"],
            minimum_latitude=lat_min,
            maximum_latitude=lat_max,
            minimum_longitude=lon_min,
            maximum_longitude=lon_max,
            start_datetime=yest,
            end_datetime=today,
            username=cmems_user,
            password=cmems_pass,
        )
        sst_val = float(ds2["thetao"].isel(depth=0).mean(skipna=True).values)
        ds2.close()
    except Exception as e:
        print(f"[CMEMS] Gagal ambil SST: {e}")

    try:
        ds3 = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
            variables=["so"],
            minimum_latitude=lat_min,
            maximum_latitude=lat_max,
            minimum_longitude=lon_min,
            maximum_longitude=lon_max,
            start_datetime=yest,
            end_datetime=today,
            username=cmems_user,
            password=cmems_pass,
        )
        sal_val = float(ds3["so"].isel(depth=0).mean(skipna=True).values)
        ds3.close()
    except Exception as e:
        print(f"[CMEMS] Gagal ambil salinitas: {e}")

    return {
        "uo":        uo_val,
        "vo":        vo_val,
        "sst":       sst_val,
        "salinitas": sal_val,
    }


# ─────────────────────────────────────────────────────────────────
# 2. CMEMS — Klorofil-a (Ocean Colour, menggantikan NASA MODIS)
# ─────────────────────────────────────────────────────────────────
def fetch_cmems_chla(cmems_user: str, cmems_pass: str,
                     lat_min=-12.0, lat_max=-4.0,
                     lon_min=129.0, lon_max=144.0) -> float | None:
    """
    Ambil klorofil-a dari CMEMS Ocean Colour (menggantikan NASA MODIS).

    Dataset utama (dicoba pertama):
      cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D
      Variabel: CHL  (mg/m³)

    Fallback dataset (jika utama gagal):
      cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D
      Variabel: CHL

    Mengembalikan float rata-rata spasial (mg/m³), atau None jika gagal.
    """
    cm = _import_copernicusmarine()
    if cm is None:
        print("[CMEMS-CHL] copernicusmarine tidak terinstal")
        return None

    now   = datetime.datetime.utcnow()
    # Dataset MY (multi-year) tersedia hingga ~5 hari sebelum hari ini
    end_dt   = (now - datetime.timedelta(days=5)).strftime("%Y-%m-%dT00:00:00")
    start_dt = (now - datetime.timedelta(days=8)).strftime("%Y-%m-%dT00:00:00")

    datasets_to_try = [
        # Multi-year reanalysis (lebih stabil)
        "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
        # Near-real-time (lebih baru tapi kadang ada gap)
        "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D",
    ]

    for dataset_id in datasets_to_try:
        try:
            ds = cm.open_dataset(
                dataset_id=dataset_id,
                variables=["CHL"],
                minimum_latitude=lat_min,
                maximum_latitude=lat_max,
                minimum_longitude=lon_min,
                maximum_longitude=lon_max,
                start_datetime=start_dt,
                end_datetime=end_dt,
                username=cmems_user,
                password=cmems_pass,
            )
            chla_val = float(ds["CHL"].mean(skipna=True).values)
            ds.close()
            print(f"[CMEMS-CHL] ✓ Berhasil dari {dataset_id}: CHL = {chla_val:.4f} mg/m³")
            return chla_val
        except Exception as e:
            print(f"[CMEMS-CHL] Gagal dari {dataset_id}: {e}")
            continue

    print("[CMEMS-CHL] Semua dataset gagal.")
    return None


# ─────────────────────────────────────────────────────────────────
# 3. NASA MODIS — TIDAK DIGUNAKAN (diganti CMEMS)
#    Fungsi ini dipertahankan hanya untuk kompatibilitas impor.
# ─────────────────────────────────────────────────────────────────
def fetch_nasa_modis(nasa_user: str, nasa_pass: str,
                     lat_min=-12.0, lat_max=-4.0,
                     lon_min=129.0, lon_max=144.0) -> float | None:
    """
    [DEPRECATED] NASA MODIS untuk klorofil-a.
    Digantikan oleh fetch_cmems_chla().
    Fungsi ini mengembalikan None dan mencetak peringatan.
    """
    print("[NASA MODIS] Fungsi ini sudah digantikan oleh fetch_cmems_chla(). "
          "Klorofil-a sekarang diambil dari CMEMS Ocean Colour.")
    return None


# ─────────────────────────────────────────────────────────────────
# 4. ERA5 / CDS — Angin
# ─────────────────────────────────────────────────────────────────
def fetch_era5(cds_uid: str, cds_key: str,
               lat_min=-12.0, lat_max=-4.0,
               lon_min=129.0, lon_max=144.0) -> dict | None:
    """
    Ambil komponen angin (u10, v10) dari ERA5 via CDS API.
    Mengembalikan dict {angin_u, angin_v} atau None jika gagal.
    """
    cdsapi = _import_cdsapi()
    if cdsapi is None:
        print("[ERA5] cdsapi tidak terinstal")
        return None

    if not cds_key:
        print("[ERA5] CDS_KEY kosong, skip.")
        return None

    import os, tempfile
    now  = datetime.datetime.utcnow()
    # ERA5 tersedia dengan delay ~5 hari
    date = (now - datetime.timedelta(days=5)).strftime("%Y-%m-%d")

    try:
        # Tulis ~/.cdsapirc sementara jika belum ada
        rc_path = os.path.expanduser("~/.cdsapirc")
        if not os.path.exists(rc_path):
            with open(rc_path, "w") as f:
                f.write(f"url: https://cds.climate.copernicus.eu/api/v2\n")
                f.write(f"key: {cds_uid}:{cds_key}\n")

        c = cdsapi.Client(quiet=True)
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable":     ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "date":         date,
                "time":         "12:00",
                "area":         [lat_max, lon_min, lat_min, lon_max],
                "format":       "netcdf",
            },
            tmp_path,
        )

        import netCDF4 as nc
        ds = nc.Dataset(tmp_path)
        u10 = float(np.nanmean(ds.variables["u10"][:]))
        v10 = float(np.nanmean(ds.variables["v10"][:]))
        ds.close()
        os.remove(tmp_path)
        print(f"[ERA5] ✓ u10={u10:.3f}, v10={v10:.3f} m/s")
        return {"angin_u": u10, "angin_v": v10}

    except Exception as e:
        print(f"[ERA5] Gagal: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# 5. BMKG — Gelombang
# ─────────────────────────────────────────────────────────────────
def fetch_bmkg(bmkg_base_url: str = "https://inaoc.bmkg.go.id",
               adm4: str = "81.71.01.1001") -> float | None:
    """
    Ambil tinggi gelombang signifikan (Hs) dari BMKG InaOC.
    Mengembalikan float (meter) atau None jika gagal.
    """
    import requests

    endpoints = [
        f"{bmkg_base_url}/api/v1/wave?adm4={adm4}",
        f"{bmkg_base_url}/api/wave/latest?adm4={adm4}",
        f"{bmkg_base_url}/data/wave/{adm4}",
    ]

    headers = {"User-Agent": "LAUTAN-Platform/1.0", "Accept": "application/json"}

    for url in endpoints:
        try:
            r = requests.get(url, headers=headers, timeout=12)
            if r.status_code == 200:
                data = r.json()
                # Coba beberapa struktur respons yang mungkin
                for key in ["wave_height", "hs", "significant_wave_height", "Hs", "tinggi_gelombang"]:
                    if key in data:
                        val = float(data[key])
                        print(f"[BMKG] ✓ Gelombang = {val:.2f} m (key: {key})")
                        return val
                # Jika nested
                if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                    row = data["data"][0]
                    for key in ["wave_height", "hs", "Hs", "significant_wave_height"]:
                        if key in row:
                            val = float(row[key])
                            print(f"[BMKG] ✓ Gelombang = {val:.2f} m")
                            return val
        except Exception as e:
            print(f"[BMKG] Gagal {url}: {e}")
            continue

    print("[BMKG] Semua endpoint gagal.")
    return None


# ─────────────────────────────────────────────────────────────────
# 6. BUILD REALTIME DATAFRAME
# ─────────────────────────────────────────────────────────────────
def build_realtime_dataframe(
    cmems_user: str, cmems_pass: str,
    nasa_user:  str = "",   # Tidak dipakai, dipertahankan untuk kompatibilitas
    nasa_pass:  str = "",   # Tidak dipakai
    cds_uid:    str = "",
    cds_key:    str = "",
    lat_min: float = -12.0,
    lat_max: float =  -4.0,
    lon_min: float = 129.0,
    lon_max: float = 144.0,
) -> dict:
    """
    Gabungkan semua sumber data real-time menjadi satu DataFrame.

    Alur pengambilan data:
      - CMEMS → uo, vo, sst, salinitas           (satu sumber)
      - CMEMS Ocean Colour → chla                 (GANTI dari NASA MODIS)
      - ERA5  → angin_u, angin_v
      - BMKG  → gelombang

    Nilai yang gagal diambil akan diisi dengan estimasi klimatologis
    berdasarkan bulan saat ini dari rata-rata historis yang di-hardcode.

    Returns:
        {
          "data":   pd.DataFrame (1 baris per grid point / rata-rata spasial),
          "status": dict {nama_api: bool}
        }
    """
    status = {
        "CMEMS (Fisika)":      False,
        "CMEMS (Klorofil-a)":  False,
        "ERA5 (Angin)":        False,
        "BMKG (Gelombang)":    False,
    }

    now_month = datetime.datetime.utcnow().month

    # ── Nilai fallback klimatologis per bulan ──────────────────────
    # Estimasi kasar untuk Laut Arafura jika API gagal
    FALLBACK = {
        "uo":        [-0.06,-0.07,-0.05,-0.04,-0.03,-0.02,-0.02,-0.03,-0.04,-0.05,-0.06,-0.06],
        "vo":        [-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01],
        "sst":       [28.8,29.0,29.1,29.0,28.5,27.8,27.2,27.0,27.5,28.2,28.7,28.9],
        "salinitas": [34.1,34.0,34.0,34.1,34.2,34.4,34.5,34.5,34.3,34.2,34.1,34.1],
        "chla":      [0.18,0.17,0.18,0.20,0.22,0.26,0.28,0.30,0.27,0.23,0.20,0.19],
        "angin_u":   [-1.2,-1.0,-0.8,-0.5,-0.3,-0.5,-0.8,-1.0,-1.2,-1.5,-1.8,-1.5],
        "angin_v":   [-0.4,-0.3,-0.2,-0.1,-0.1,-0.2,-0.3,-0.4,-0.5,-0.5,-0.4,-0.4],
        "gelombang": [0.9, 0.9, 0.8, 0.7, 0.7, 0.9, 1.1, 1.2, 1.1, 0.9, 0.8, 0.9],
    }
    mi = now_month - 1  # index 0-based

    # ── Ambil data ─────────────────────────────────────────────────
    # 1. CMEMS Fisika (arus, SST, salinitas)
    cmems_result = fetch_cmems(
        cmems_user, cmems_pass, lat_min, lat_max, lon_min, lon_max
    )

    if cmems_result is not None:
        uo_val        = cmems_result.get("uo",        FALLBACK["uo"][mi])
        vo_val        = cmems_result.get("vo",        FALLBACK["vo"][mi])
        sst_val       = cmems_result.get("sst",       FALLBACK["sst"][mi])
        sal_val       = cmems_result.get("salinitas", FALLBACK["salinitas"][mi])
        # Handle None untuk sub-variabel yang mungkin gagal
        if uo_val  is None: uo_val  = FALLBACK["uo"][mi]
        if vo_val  is None: vo_val  = FALLBACK["vo"][mi]
        if sst_val is None: sst_val = FALLBACK["sst"][mi]
        if sal_val is None: sal_val = FALLBACK["salinitas"][mi]
        status["CMEMS (Fisika)"] = True
    else:
        uo_val  = FALLBACK["uo"][mi]
        vo_val  = FALLBACK["vo"][mi]
        sst_val = FALLBACK["sst"][mi]
        sal_val = FALLBACK["salinitas"][mi]

    # 2. CMEMS Ocean Colour (klorofil-a) — GANTI dari NASA MODIS
    chla_val = fetch_cmems_chla(
        cmems_user, cmems_pass, lat_min, lat_max, lon_min, lon_max
    )
    if chla_val is not None and not np.isnan(chla_val):
        status["CMEMS (Klorofil-a)"] = True
    else:
        chla_val = FALLBACK["chla"][mi]
        print(f"[CHL] Menggunakan fallback klimatologis: {chla_val:.3f} mg/m³")

    # 3. ERA5 (angin)
    era5_result = fetch_era5(cds_uid, cds_key, lat_min, lat_max, lon_min, lon_max)
    if era5_result is not None:
        angin_u = era5_result.get("angin_u", FALLBACK["angin_u"][mi])
        angin_v = era5_result.get("angin_v", FALLBACK["angin_v"][mi])
        status["ERA5 (Angin)"] = True
    else:
        angin_u = FALLBACK["angin_u"][mi]
        angin_v = FALLBACK["angin_v"][mi]

    # 4. BMKG (gelombang)
    gelombang_val = fetch_bmkg()
    if gelombang_val is not None and not np.isnan(gelombang_val):
        status["BMKG (Gelombang)"] = True
    else:
        gelombang_val = FALLBACK["gelombang"][mi]

    # ── Derive parameter tambahan ──────────────────────────────────
    # pH dan DO diestimasi dari SST dan klorofil-a (relasi empiris)
    ph_val = 8.12 - 0.005 * (sst_val - 28.5) - 0.01 * (chla_val - 0.22)
    ph_val = float(np.clip(ph_val, 7.9, 8.4))

    do_val = 6.2 - 0.05 * (sst_val - 28.5) + 0.3 * (chla_val - 0.22)
    do_val = float(np.clip(do_val, 4.5, 7.5))

    ssta_val = float(sst_val - 28.5)  # Anomali SST sederhana terhadap mean klimatologis

    # ── Bangun DataFrame ───────────────────────────────────────────
    current_speed = float(np.sqrt(uo_val**2 + vo_val**2))

    # Buat 1 baris ringkasan untuk dipakai di seluruh app
    df_rt = pd.DataFrame([{
        "time":          datetime.datetime.utcnow(),
        "month":         now_month,
        "year":          datetime.datetime.utcnow().year,
        "uo":            uo_val,
        "vo":            vo_val,
        "sst":           sst_val,
        "ssta":          ssta_val,
        "salinitas":     sal_val,
        "chla":          chla_val,
        "ph":            ph_val,
        "do":            do_val,
        "gelombang":     gelombang_val,
        "angin_u":       angin_u,
        "angin_v":       angin_v,
        "current_speed": current_speed,
    }])

    # Hitung indeks turunan
    def norm(val, vmin, vmax):
        return float(np.clip((val - vmin) / (vmax - vmin + 1e-9), 0, 1))

    df_rt["Ocean_Health_Index"] = (
        0.25 * norm(do_val,        4.5,  7.5) +
        0.20 * norm(ph_val,        7.9,  8.4) +
        0.20 * norm(chla_val,      0.05, 0.8) +
        0.15 * norm(sal_val,       32.0, 36.5) +
        0.20 * (1 - norm(gelombang_val, 0.2,  2.5))
    ) * 100

    df_rt["Fisheries_Index"] = (
        0.35 * norm(chla_val,      0.05, 0.8) +
        0.25 * norm(do_val,        4.5,  7.5) +
        0.20 * norm(current_speed, 0.0,  0.25) +
        0.20 * (1 - norm(gelombang_val, 0.2,  2.5))
    ) * 100

    print("\n=== LAUTAN Real-Time Summary ===")
    print(f"  SST:       {sst_val:.2f} °C")
    print(f"  Salinitas: {sal_val:.2f} PSU")
    print(f"  Chl-a:     {chla_val:.4f} mg/m³  ← CMEMS Ocean Colour")
    print(f"  pH:        {ph_val:.3f}")
    print(f"  DO:        {do_val:.3f} mg/L")
    print(f"  Arus:      uo={uo_val:.4f}  vo={vo_val:.4f}  spd={current_speed:.4f} m/s")
    print(f"  Angin:     u={angin_u:.2f}  v={angin_v:.2f} m/s")
    print(f"  Gelombang: {gelombang_val:.2f} m")
    print(f"  OHI:       {df_rt['Ocean_Health_Index'].iloc[0]:.1f}/100")
    print(f"  FSI:       {df_rt['Fisheries_Index'].iloc[0]:.1f}/100")
    print("================================\n")

    return {"data": df_rt, "status": status}
ENDOFFILE
echo "Done"
