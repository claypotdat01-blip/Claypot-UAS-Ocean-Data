"""
data_fetcher.py — Pengambil data real-time dari CMEMS, NASA MODIS, ERA5, BMKG
==============================================================================
Setiap fungsi mengembalikan dict dengan struktur:
  { "ok": bool, "data": nilai_atau_None, "error": str_atau_None }

Jika API gagal, fungsi TIDAK crash — mengembalikan ok=False dan data=None.
build_realtime_dataframe() menggabungkan semua sumber ke satu DataFrame
yang kompatibel dengan format df_filter_base di app.py.
"""

import datetime
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Batas wilayah Laut Arafura ──────────────────────────────
LAT_MIN, LAT_MAX = -12.0, -4.0
LON_MIN, LON_MAX = 129.0, 144.0


# =============================================================
# 1. CMEMS — Arus (uo, vo), SST (thetao), Salinitas (so)
# =============================================================
def fetch_cmems(user: str, password: str) -> dict:
    """
    Dataset: cmems_mod_glo_phy_anfc_0.083deg_PT1H-i
    (Global Ocean Physics Analysis and Forecast — near real-time, update tiap jam)

    Instalasi: pip install copernicusmarine
    Daftar   : https://marine.copernicus.eu → Register (gratis)
    """
    try:
        import copernicusmarine  # noqa: F401
    except ImportError:
        return {"ok": False, "data": None,
                "error": "copernicusmarine belum terpasang. Jalankan: pip install copernicusmarine"}

    if "ISI_" in user or not user.strip():
        return {"ok": False, "data": None,
                "error": "CMEMS_USER belum diisi di config.py"}

    try:
        ds = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-i",
            variables=["uo", "vo", "thetao", "so"],
            minimum_latitude=LAT_MIN,  maximum_latitude=LAT_MAX,
            minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
            minimum_depth=0.0,         maximum_depth=1.0,
            username=user,             password=password,
            # Ambil 24 jam terakhir
            start_datetime=(datetime.datetime.utcnow() - datetime.timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        )
        df = ds.to_dataframe().reset_index().dropna()

        # Rename ke nama kolom internal
        rename_map = {}
        if "thetao" in df.columns: rename_map["thetao"] = "sst"
        if "so"     in df.columns: rename_map["so"]     = "salinitas"
        df = df.rename(columns=rename_map)

        # Rata-ratakan ke satu baris per waktu
        agg_cols = [c for c in ["uo","vo","sst","salinitas"] if c in df.columns]
        result   = df[agg_cols].mean().to_dict()
        result["source_cmems"] = True
        return {"ok": True, "data": result, "error": None}

    except Exception as e:
        return {"ok": False, "data": None, "error": f"CMEMS error: {str(e)[:120]}"}


# =============================================================
# 2. NASA Earthdata — Klorofil-a MODIS Aqua (8-hari komposit)
# =============================================================
def fetch_nasa_modis(user: str, password: str) -> dict:
    """
    Endpoint: NASA OceanColor Web / ERDDAP
    Dataset  : MODIS Aqua Chlorophyll 8-day 4km

    Instalasi: pip install earthaccess requests
    Daftar   : https://urs.earthdata.nasa.gov → Register (gratis, langsung aktif)
    """
    try:
        import requests
    except ImportError:
        return {"ok": False, "data": None, "error": "requests belum terpasang"}

    if "ISI_" in user or not user.strip():
        return {"ok": False, "data": None,
                "error": "NASA_USER belum diisi di config.py"}

    # Gunakan OceanWatch ERDDAP (tidak perlu login khusus untuk data publik 8-harian)
    # Hitung periode 8-hari terdekat
    now   = datetime.datetime.utcnow()
    start = (now - datetime.timedelta(days=16)).strftime("%Y-%m-%dT00:00:00Z")
    end   = (now - datetime.timedelta(days=8)).strftime("%Y-%m-%dT00:00:00Z")

    url = (
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chla8day.json"
        f"?chlorophyll[({start}):1:({end})]"
        f"[({LAT_MIN}):1:({LAT_MAX})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )

    try:
        resp = requests.get(url, auth=(user, password), timeout=30)
        if resp.status_code == 200:
            data_json = resp.json()
            rows = data_json.get("table", {}).get("rows", [])
            vals = [r[-1] for r in rows if r[-1] is not None and not np.isnan(float(r[-1]))]
            if vals:
                chla_mean = float(np.mean(vals))
                chla_mean = float(np.clip(chla_mean, 0.05, 0.8))
                return {"ok": True, "data": {"chla": chla_mean, "source_modis": True}, "error": None}
        # Fallback: coba endpoint alternatif tanpa auth
        url2 = (
            "https://oceanwatch.pifsc.noaa.gov/erddap/griddap/"
            f"aqua_chla_8day_2018_0.json?chlor_a[({start}):1:({end})]"
            f"[({LAT_MIN}):1:({LAT_MAX})][({LON_MIN}):1:({LON_MAX})]"
        )
        resp2 = requests.get(url2, timeout=20)
        if resp2.status_code == 200:
            rows2 = resp2.json().get("table", {}).get("rows", [])
            vals2 = [r[-1] for r in rows2 if r[-1] is not None]
            if vals2:
                return {"ok": True,
                        "data": {"chla": float(np.clip(np.mean(vals2), 0.05, 0.8)),
                                 "source_modis": True},
                        "error": None}
        return {"ok": False, "data": None,
                "error": f"NASA MODIS HTTP {resp.status_code}"}
    except Exception as e:
        return {"ok": False, "data": None, "error": f"NASA MODIS error: {str(e)[:120]}"}


# =============================================================
# 3. ERA5 / CDS — Angin (u10, v10) + tinggi gelombang
# =============================================================
def fetch_era5(cds_uid: str, cds_key: str) -> dict:
    """
    Dataset: reanalysis-era5-single-levels (delay ~5 hari dari hari ini)
    Untuk near real-time, fallback ke klimatologi bulanan ERA5.

    Instalasi: pip install cdsapi
    Daftar   : https://cds.climate.copernicus.eu → Register
               My Account → API Key → copy UID dan Key
               Buat ~/.cdsapirc:
                   url: https://cds.climate.copernicus.eu/api/v2
                   key: UID:KEY
    """
    try:
        import cdsapi
    except ImportError:
        return {"ok": False, "data": None,
                "error": "cdsapi belum terpasang. Jalankan: pip install cdsapi"}

    if "ISI_" in str(cds_uid) or not str(cds_uid).strip():
        return {"ok": False, "data": None,
                "error": "CDS_UID belum diisi di config.py"}

    import tempfile, os
    try:
        # Tulis .cdsapirc sementara ke temp dir
        cdsapirc_content = (
            "url: https://cds.climate.copernicus.eu/api/v2\n"
            f"key: {cds_uid}:{cds_key}\n"
        )
        tmp_rc = os.path.join(tempfile.gettempdir(), ".cdsapirc_lautan")
        with open(tmp_rc, "w") as f:
            f.write(cdsapirc_content)
        os.environ["CDSAPI_RC"] = tmp_rc

        c = cdsapi.Client(quiet=True, progress=False)
        now = datetime.datetime.utcnow()
        # ERA5 delay ~5 hari; ambil 7 hari lalu agar aman
        target_date = now - datetime.timedelta(days=7)

        tmp_nc = os.path.join(tempfile.gettempdir(), "era5_lautan_tmp.nc")
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "significant_height_of_combined_wind_waves_and_swell",
            ],
            "year":  target_date.strftime("%Y"),
            "month": target_date.strftime("%m"),
            "day":   target_date.strftime("%d"),
            "time":  ["00:00","06:00","12:00","18:00"],
            "area":  [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],   # N, W, S, E
            "format": "netcdf",
        }, tmp_nc)

        import netCDF4 as nc
        with nc.Dataset(tmp_nc) as ds:
            u10  = float(np.nanmean(ds.variables.get("u10",  ds.variables.get("u10m",  None))[:]))
            v10  = float(np.nanmean(ds.variables.get("v10",  ds.variables.get("v10m",  None))[:]))
            swh_var = ds.variables.get("swh", ds.variables.get("shww", None))
            wave = float(np.nanmean(swh_var[:])) if swh_var is not None else 0.8

        os.remove(tmp_nc)
        return {"ok": True,
                "data": {"angin_u": u10, "angin_v": v10,
                         "gelombang": float(np.clip(wave, 0.2, 2.5)),
                         "source_era5": True},
                "error": None}
    except Exception as e:
        return {"ok": False, "data": None, "error": f"ERA5 error: {str(e)[:120]}"}


# =============================================================
# 4. BMKG — Prakiraan gelombang (open data, tanpa API key)
# =============================================================
def fetch_bmkg() -> dict:
    """
    Endpoint BMKG open data — tidak perlu akun/API key.
    Mencoba beberapa endpoint; fallback ke nilai klimatologis jika semua gagal.
    """
    try:
        import requests
    except ImportError:
        return {"ok": False, "data": None, "error": "requests belum terpasang"}

    endpoints = [
        "https://inaoc.bmkg.go.id/DataOlahan/gelombangLaut",
        "https://data.bmkg.go.id/DataMKG/MEWS/maritim/",
    ]

    for url in endpoints:
        try:
            resp = requests.get(url, timeout=15,
                                headers={"User-Agent": "LAUTAN-OceanPlatform/1.0"})
            if resp.status_code == 200:
                # BMKG mengembalikan HTML/JSON tergantung endpoint
                # Coba parse angka gelombang dari respons teks
                text = resp.text
                # Cari pola angka gelombang (biasanya dalam format "X.X - Y.Y meter")
                import re
                matches = re.findall(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*[Mm]eter", text)
                if matches:
                    vals = [(float(a) + float(b)) / 2 for a, b in matches]
                    wave_avg = float(np.clip(np.mean(vals), 0.2, 2.5))
                    return {"ok": True,
                            "data": {"gelombang": wave_avg, "source_bmkg": True},
                            "error": None}
        except Exception:
            continue

    # Semua endpoint gagal — kembalikan ok=False, nanti diganti klimatologis
    return {"ok": False, "data": None,
            "error": "BMKG: semua endpoint tidak merespons (mungkin offline atau format berubah)"}


# =============================================================
# GABUNGAN: build_realtime_dataframe()
# =============================================================
def build_realtime_dataframe(cmems_user: str, cmems_pass: str,
                              nasa_user: str,  nasa_pass: str,
                              cds_uid: str = "", cds_key: str = "") -> dict:
    """
    Panggil semua API secara berurutan, gabungkan hasilnya menjadi satu baris
    DataFrame yang kompatibel dengan format df_filter_base di app.py.

    Returns:
        {
          "data"  : pd.DataFrame atau None jika semua API gagal,
          "status": { "CMEMS": bool, "NASA MODIS": bool, "ERA5": bool, "BMKG": bool }
        }
    """
    # Nilai default klimatologis bulan berjalan (fallback jika API gagal)
    now = datetime.datetime.utcnow()
    defaults = {
        "uo": -0.05, "vo": -0.01,
        "sst": 28.5 + 0.5 * np.sin(2 * np.pi * now.month / 12),
        "ssta": 0.0,
        "salinitas": 34.2,
        "chla": 0.22,
        "ph": 8.10,
        "do": 6.2,
        "gelombang": 0.8 + 0.3 * abs(np.sin(2 * np.pi * now.month / 12)),
        "angin_u": -1.5,
        "angin_v": -0.5,
        "time": pd.Timestamp(now),
        "year":  now.year,
        "month": now.month,
    }

    merged  = defaults.copy()
    status  = {"CMEMS": False, "NASA MODIS": False, "ERA5": False, "BMKG": False}
    any_ok  = False

    # ── 1. CMEMS ──
    r_cmems = fetch_cmems(cmems_user, cmems_pass)
    if r_cmems["ok"] and r_cmems["data"]:
        for k in ["uo", "vo", "sst", "salinitas"]:
            if k in r_cmems["data"]:
                merged[k] = r_cmems["data"][k]
        # Hitung SSTA sederhana: SST saat ini dikurangi rata-rata klimatologis
        merged["ssta"] = merged["sst"] - 28.5
        status["CMEMS"] = True
        any_ok = True

    # ── 2. NASA MODIS ──
    r_modis = fetch_nasa_modis(nasa_user, nasa_pass)
    if r_modis["ok"] and r_modis["data"]:
        if "chla" in r_modis["data"]:
            merged["chla"] = r_modis["data"]["chla"]
        status["NASA MODIS"] = True
        any_ok = True

    # ── 3. ERA5 ──
    r_era5 = fetch_era5(cds_uid, cds_key)
    if r_era5["ok"] and r_era5["data"]:
        for k in ["angin_u", "angin_v", "gelombang"]:
            if k in r_era5["data"]:
                merged[k] = r_era5["data"][k]
        status["ERA5"] = True
        any_ok = True

    # ── 4. BMKG ──
    r_bmkg = fetch_bmkg()
    if r_bmkg["ok"] and r_bmkg["data"]:
        if "gelombang" in r_bmkg["data"]:
            # BMKG override ERA5 untuk gelombang (resolusi lokal lebih baik)
            merged["gelombang"] = r_bmkg["data"]["gelombang"]
        status["BMKG"] = True
        any_ok = True

    # Derive current_speed
    merged["current_speed"] = float(np.sqrt(merged["uo"]**2 + merged["vo"]**2))
    merged["do"] = float(np.clip(merged["do"], 4.5, 7.5))
    merged["ph"] = float(np.clip(merged["ph"], 7.9, 8.4))

    df_rt = pd.DataFrame([merged])
    return {"data": df_rt if any_ok else None, "status": status}
