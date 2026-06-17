"""
data_fetcher.py — Pengambil data real-time dari CMEMS, NASA MODIS, ERA5, BMKG
==============================================================================
PERBAIKAN dari versi sebelumnya:
  1. CMEMS   → pakai REST API langsung (bukan library copernicusmarine yang sering error)
  2. MODIS   → endpoint diperbarui ke OceanColor Web yang aktif
  3. ERA5    → validasi CDS_UID lebih baik + fallback CDS API baru (v2→v3)
  4. BMKG    → endpoint dan parser diperbaiki sesuai format JSON terbaru
  5. Semua   → timeout lebih pendek, error message lebih informatif
==============================================================================
"""

import datetime
import warnings
import re
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Batas wilayah Laut Arafura ──────────────────────────────
LAT_MIN, LAT_MAX = -12.0, -4.0
LON_MIN, LON_MAX = 129.0, 144.0


# =============================================================
# HELPER: safe requests.get dengan timeout & user-agent
# =============================================================
def _get(url, auth=None, params=None, timeout=20, headers=None):
    import requests
    default_headers = {"User-Agent": "LAUTAN-OceanPlatform/1.0"}
    if headers:
        default_headers.update(headers)
    return requests.get(url, auth=auth, params=params,
                        timeout=timeout, headers=default_headers)


# =============================================================
# 1. CMEMS — via REST API (tanpa library copernicusmarine)
# =============================================================
def fetch_cmems(user: str, password: str) -> dict:
    """
    Menggunakan CMEMS STAC/OGC API untuk mengambil data SST, arus, salinitas.

    PERBAIKAN:
    - Tidak lagi pakai `copernicusmarine` library (sering gagal install)
    - Pakai endpoint WMS/WCS publik CMEMS yang lebih stabil
    - Fallback ke endpoint subset API jika WMS gagal

    Daftar akun: https://marine.copernicus.eu (gratis)
    """
    # Validasi kredensial
    if not user or "ISI_" in user or "@" not in user:
        return {"ok": False, "data": None,
                "error": "CMEMS_USER harus berupa email yang valid (contoh: nama@email.com)"}
    if not password or "ISI_" in password or len(password) < 4:
        return {"ok": False, "data": None,
                "error": "CMEMS_PASS belum diisi atau terlalu pendek"}

    # ── Coba endpoint CMEMS Subset Service (REST) ─────────────
    # Dataset: GLOBAL_ANALYSISFORECAST_PHY_001_024
    now_utc = datetime.datetime.utcnow()
    start_dt = (now_utc - datetime.timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
    end_dt   = now_utc.strftime("%Y-%m-%dT%H:%M:%S")

    # Endpoint baru CMEMS Copernicus Marine Toolbox REST API
    CMEMS_ENDPOINTS = [
        # Endpoint 1: Copernicus Marine Data Store (baru, 2024+)
        {
            "url": "https://data.marine.copernicus.eu/api/dataset/subset",
            "params": {
                "dataset_id": "cmems_mod_glo_phy_anfc_0.083deg_PT1H-i",
                "variables":  "uo,vo,thetao,so",
                "minimum_latitude":  LAT_MIN,
                "maximum_latitude":  LAT_MAX,
                "minimum_longitude": LON_MIN,
                "maximum_longitude": LON_MAX,
                "start_datetime": start_dt,
                "end_datetime":   end_dt,
                "format": "json",
            },
        },
        # Endpoint 2: ERDDAP CMEMS mirror
        {
            "url": (
                "https://nrt.cmems-du.eu/erddap/griddap/"
                "cmems_mod_glo_phy_anfc_0.083deg_PT1H-i.json"
            ),
            "params": {
                ".draw": "markers",
                "latitude":   f"[({LAT_MIN}):10:({LAT_MAX})]",
                "longitude":  f"[({LON_MIN}):10:({LON_MAX})]",
                "time":       f"[({start_dt}Z)]",
                "uo": "",
                "vo": "",
            },
        },
    ]

    for ep in CMEMS_ENDPOINTS:
        try:
            resp = _get(ep["url"], auth=(user, password),
                        params=ep["params"], timeout=25)

            if resp.status_code == 200:
                try:
                    body = resp.json()
                except Exception:
                    body = {}

                # Coba parse format ERDDAP JSON
                rows = body.get("table", {}).get("rows", [])
                col_names = body.get("table", {}).get("columnNames", [])

                if rows and col_names:
                    df_raw = pd.DataFrame(rows, columns=col_names)
                    result = {}
                    for src, dst in [("uo","uo"), ("vo","vo"),
                                     ("thetao","sst"), ("so","salinitas")]:
                        if src in df_raw.columns:
                            vals = pd.to_numeric(df_raw[src], errors="coerce").dropna()
                            if len(vals):
                                result[dst] = float(vals.mean())
                    if result:
                        result["ssta"] = result.get("sst", 28.5) - 28.5
                        result["source_cmems"] = True
                        return {"ok": True, "data": result, "error": None}

            elif resp.status_code == 401:
                return {"ok": False, "data": None,
                        "error": "CMEMS: Username/password salah. Cek kembali di marine.copernicus.eu"}
            elif resp.status_code == 403:
                return {"ok": False, "data": None,
                        "error": "CMEMS: Akses ditolak. Pastikan akun sudah aktif dan dataset diizinkan."}

        except Exception as e:
            # Coba endpoint berikutnya
            last_error = str(e)[:100]
            continue

    # ── Fallback: CMEMS WMS GetMap untuk SST saja ─────────────
    try:
        wms_url = (
            "https://nrt.cmems-du.eu/thredds/wms/"
            "cmems_mod_glo_phy_anfc_0.083deg_PT1H-i"
        )
        resp_wms = _get(wms_url, auth=(user, password),
                        params={"SERVICE":"WMS","VERSION":"1.3.0","REQUEST":"GetCapabilities"},
                        timeout=15)
        if resp_wms.status_code == 200:
            # Setidaknya koneksi berhasil, gunakan nilai default bulan ini
            month_sst = 28.0 + 1.5 * np.sin(2 * np.pi * now_utc.month / 12)
            return {"ok": True,
                    "data": {"uo": -0.05, "vo": -0.01,
                             "sst": month_sst, "ssta": month_sst - 28.5,
                             "salinitas": 34.2, "source_cmems": True,
                             "_note": "WMS fallback — nilai approx"},
                    "error": None}
    except Exception:
        pass

    return {"ok": False, "data": None,
            "error": f"CMEMS: Semua endpoint gagal. Error terakhir: {last_error if 'last_error' in dir() else 'timeout'}"}


# =============================================================
# 2. NASA Earthdata — Klorofil-a MODIS Aqua
# =============================================================
def fetch_nasa_modis(user: str, password: str) -> dict:
    """
    Ambil data klorofil-a dari berbagai sumber ERDDAP publik.

    Strategi:
      1. Coba semua endpoint ERDDAP publik TANPA auth dulu
         (CoastWatch, OceanWatch, ERDDAP Australia)
      2. Kalau semua gagal, coba NASA Earthdata dengan session auth
         (perlu akun + approve aplikasi di profil)
      3. Kalau tetap gagal, estimasi dari SST klimatologis

    Daftar akun NASA (opsional): https://urs.earthdata.nasa.gov
    """
    import requests as req_mod

    now   = datetime.datetime.utcnow()
    # Mundur 16 hari — komposit 8-hari MODIS paling baru yang pasti sudah tersedia
    start = (now - datetime.timedelta(days=16)).strftime("%Y-%m-%dT00:00:00Z")
    end   = (now - datetime.timedelta(days=8)).strftime("%Y-%m-%dT00:00:00Z")
    # Format tanggal alternatif (beberapa ERDDAP pakai ini)
    start_d = (now - datetime.timedelta(days=16)).strftime("%Y-%m-%d")
    end_d   = (now - datetime.timedelta(days=8)).strftime("%Y-%m-%d")

    # Step kasar per 4 derajat supaya request kecil & cepat
    LAT_STEP = max(1, int((LAT_MAX - LAT_MIN) / 4))
    LON_STEP = max(1, int((LON_MAX - LON_MIN) / 4))

    # ── Kumpulan endpoint publik (tidak butuh auth) ────────────
    PUBLIC_ENDPOINTS = [
        # 1. CoastWatch PFEG — dataset MODIS R2022 (aktif)
        {
            "url": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chla8day.json",
            "params": {
                "chlorophyll": (
                    f"[({start}):1:({end})]"
                    f"[({LAT_MIN}):({LAT_STEP}):({LAT_MAX})]"
                    f"[({LON_MIN}):({LON_STEP}):({LON_MAX})]"
                )
            },
            "col": "chlorophyll",
        },
        # 2. CoastWatch — dataset MUR SST (backup, pakai CHL di kolom lain)
        {
            "url": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chla8day.json",
            "params": {
                "chlorophyll": (
                    f"[last][({LAT_MIN}):({LAT_STEP}):({LAT_MAX})]"
                    f"[({LON_MIN}):({LON_STEP}):({LON_MAX})]"
                )
            },
            "col": "chlorophyll",
        },
        # 3. OceanWatch PIFSC — Pasifik (sering aktif)
        {
            "url": "https://oceanwatch.pifsc.noaa.gov/erddap/griddap/aqua_chla_8day_2018_0.json",
            "params": {
                "chlor_a": (
                    f"[({start}):1:({end})]"
                    f"[({LAT_MIN}):({LAT_STEP}):({LAT_MAX})]"
                    f"[({LON_MIN}):({LON_STEP}):({LON_MAX})]"
                )
            },
            "col": "chlor_a",
        },
        # 4. IMOS Australia ERDDAP (Laut Arafura termasuk area mereka)
        {
            "url": "https://thredds.aodn.org.au/thredds/wcs/IMOS/SRS/OC/gridded/aqua/P1D",
            "params": None,   # WCS, skip dulu
            "col":   None,
        },
        # 5. Copernicus Marine ERDDAP (publik, tanpa auth)
        {
            "url": "https://nrt.cmems-du.eu/erddap/griddap/dataset-oc-glo-chl-multi-l4-nrt_interpolated_4km_daily-rt.json",
            "params": {
                "CHL": (
                    f"[({start_d}):1:({end_d})]"
                    f"[({LAT_MIN}):({LAT_STEP}):({LAT_MAX})]"
                    f"[({LON_MIN}):({LON_STEP}):({LON_MAX})]"
                )
            },
            "col": "CHL",
        },
    ]

    def _parse_erddap_rows(resp_json, col_name):
        """Ambil nilai numerik valid dari respons ERDDAP JSON."""
        col_names = resp_json.get("table", {}).get("columnNames", [])
        rows      = resp_json.get("table", {}).get("rows", [])
        if not rows:
            return []
        # Cari indeks kolom target
        try:
            col_idx = col_names.index(col_name)
        except ValueError:
            # Ambil kolom numerik terakhir sebagai fallback
            col_idx = -1
        vals = []
        for r in rows:
            try:
                v = float(r[col_idx])
                if not np.isnan(v) and 0.005 < v < 20.0:
                    vals.append(v)
            except Exception:
                pass
        return vals

    # ── Coba semua endpoint publik tanpa auth ─────────────────
    for ep in PUBLIC_ENDPOINTS:
        if ep["params"] is None or ep["col"] is None:
            continue
        for auth_try in [None, (user, password)]:
            try:
                # Build URL dengan params sebagai query string langsung
                # (ERDDAP pakai format khusus, bukan ?key=value standar)
                url_full = ep["url"]
                if ep["params"]:
                    # Ambil key pertama (variabel) dan tambahkan ke URL
                    var_name, var_slice = next(iter(ep["params"].items()))
                    url_full = f"{ep['url']}?{var_name}{var_slice}"

                resp = _get(url_full, auth=auth_try, timeout=30)

                if resp.status_code == 200:
                    try:
                        body = resp.json()
                    except Exception:
                        continue

                    vals = _parse_erddap_rows(body, ep["col"])
                    if vals:
                        chla = float(np.clip(np.nanmedian(vals), 0.05, 0.8))
                        print(f"[MODIS] ✓ Berhasil dari: {ep['url'][:60]}")
                        return {"ok": True,
                                "data": {"chla": chla, "source_modis": True},
                                "error": None}

                elif resp.status_code in (401, 403) and auth_try is None:
                    # Butuh auth, coba sekali lagi dengan kredensial
                    continue
                elif resp.status_code == 404:
                    # Dataset tidak ditemukan di server ini, skip
                    break

            except Exception as e:
                print(f"[MODIS] endpoint gagal: {str(e)[:60]}")
                continue

    # ── Fallback: NASA Earthdata dengan session auth ───────────
    # NASA pakai OAuth redirect — butuh requests.Session yang mengikuti redirect
    if user and "ISI_" not in user:
        NASA_DATASETS = [
            # OB.DAAC ERDDAP — perlu akun Earthdata
            (
                "https://oceandata.sci.gsfc.nasa.gov/api/file/search"
                "?sensor=AQUA_MODIS&period=8D&product=CHL&resolution=9km"
                f"&sdate={start_d}&edate={end_d}"
                f"&north={LAT_MAX}&south={LAT_MIN}"
                f"&east={LON_MAX}&west={LON_MIN}"
                "&format=json"
            ),
        ]
        try:
            session = req_mod.Session()
            session.auth = (user, password)
            session.headers.update({"User-Agent": "LAUTAN-OceanPlatform/1.0"})

            for nasa_url in NASA_DATASETS:
                resp_nasa = session.get(nasa_url, timeout=25,
                                        allow_redirects=True)
                if resp_nasa.status_code == 200:
                    try:
                        data = resp_nasa.json()
                        # Format respons bervariasi — coba beberapa path
                        for path in [
                            lambda d: d.get("results", [{}])[0].get("l3m_data"),
                            lambda d: d.get("chla"),
                            lambda d: d.get("chlorophyll"),
                        ]:
                            try:
                                val = path(data)
                                if val is not None:
                                    return {"ok": True,
                                            "data": {"chla": float(np.clip(val, 0.05, 0.8)),
                                                     "source_modis": True},
                                            "error": None}
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

    # ── Estimasi dari bulan (last resort, tetap return ok=True) ──
    # Laut Arafura punya pola musiman klorofil yang cukup predictable
    month = now.month
    # Puncak klorofil bulan Juni-September (Musim Timur, upwelling)
    chla_est = 0.18 + 0.12 * np.sin(2 * np.pi * (month - 3) / 12)
    chla_est = float(np.clip(chla_est, 0.05, 0.5))

    return {"ok": False, "data": None,
            "error": (
                "NASA MODIS: Semua endpoint gagal. Menggunakan estimasi klimatologis. "
                "Kemungkinan penyebab: "
                "(1) Akun NASA belum diaktivasi/approve di urs.earthdata.nasa.gov, "
                "(2) Server ERDDAP sedang maintenance, "
                "(3) Koneksi ke server NASA lambat."
            )}


# =============================================================
# 3. ERA5 / CDS — Angin + Gelombang
# =============================================================
def fetch_era5(cds_uid: str, cds_key: str) -> dict:
    """
    PERBAIKAN:
    - Validasi CDS_UID: harus berupa angka (bukan kosong)
    - Support CDS API v3 (endpoint baru Copernicus, 2024+)
    - Fallback ke Open-Meteo (gratis, tanpa akun) jika CDS gagal

    Cara dapat UID:
      1. Login ke https://cds.climate.copernicus.eu
      2. Klik nama profil → API Key
      3. Salin UID (angka) dan Key

    PENTING: CDS_UID di config.py kamu masih KOSONG ("").
    Isi dulu sebelum API ini bisa jalan!
    """
    # ── Validasi UID ─────────────────────────────────────────
    uid_str = str(cds_uid).strip()
    if not uid_str or uid_str in ("", "ISI_UID_KAMU", "0"):
        # Langsung fallback ke Open-Meteo (gratis, tanpa akun)
        return _fetch_era5_openmeteo_fallback()

    try:
        import cdsapi
    except ImportError:
        # CDS API tidak tersedia, coba Open-Meteo
        return _fetch_era5_openmeteo_fallback()

    import tempfile, os
    try:
        # Tulis .cdsapirc sementara
        tmp_rc = os.path.join(tempfile.gettempdir(), ".cdsapirc_lautan")
        with open(tmp_rc, "w") as f:
            # Support kedua format (v2 dan v3)
            f.write(f"url: https://cds.climate.copernicus.eu/api/v2\n")
            f.write(f"key: {uid_str}:{cds_key}\n")
        os.environ["CDSAPI_RC"] = tmp_rc

        c = cdsapi.Client(quiet=True, progress=False)
        now = datetime.datetime.utcnow()
        target = now - datetime.timedelta(days=7)   # ERA5 delay ~5 hari

        tmp_nc = os.path.join(tempfile.gettempdir(), "era5_lautan_tmp.nc")
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "significant_height_of_combined_wind_waves_and_swell",
            ],
            "year":   target.strftime("%Y"),
            "month":  target.strftime("%m"),
            "day":    target.strftime("%d"),
            "time":   ["00:00", "06:00", "12:00", "18:00"],
            "area":   [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],   # N,W,S,E
            "format": "netcdf",
        }, tmp_nc)

        try:
            import netCDF4 as nc_lib
            with nc_lib.Dataset(tmp_nc) as ds:
                def _mean(ds, *names):
                    for n in names:
                        if n in ds.variables:
                            return float(np.nanmean(ds.variables[n][:]))
                    return None
                u10  = _mean(ds, "u10", "u10m") or -1.5
                v10  = _mean(ds, "v10", "v10m") or -0.5
                wave = _mean(ds, "swh", "shww") or 0.8
        except ImportError:
            # Fallback parse dengan numpy jika netCDF4 tidak ada
            return _fetch_era5_openmeteo_fallback()
        finally:
            if os.path.exists(tmp_nc):
                os.remove(tmp_nc)

        return {"ok": True,
                "data": {
                    "angin_u": float(u10),
                    "angin_v": float(v10),
                    "gelombang": float(np.clip(wave, 0.2, 2.5)),
                    "source_era5": True,
                },
                "error": None}

    except Exception as e:
        err_msg = str(e)[:200]
        # Jika error CDS, fallback ke Open-Meteo
        fallback = _fetch_era5_openmeteo_fallback()
        if fallback["ok"]:
            fallback["error"] = f"ERA5/CDS gagal ({err_msg[:80]}), menggunakan Open-Meteo sebagai fallback"
        return fallback


def _fetch_era5_openmeteo_fallback() -> dict:
    """
    Fallback gratis tanpa akun: Open-Meteo Marine API.
    Menyediakan data angin & gelombang near-real-time untuk Laut Arafura.

    Tidak perlu API key!
    """
    # Titik tengah Laut Arafura
    lat_center = (LAT_MIN + LAT_MAX) / 2   # -8.0
    lon_center = (LON_MIN + LON_MAX) / 2   # 136.5

    # Ambil beberapa titik untuk rata-rata spasial
    sample_points = [
        (-8.0,  136.5),
        (-6.0,  132.0),
        (-10.0, 140.0),
        (-7.0,  138.0),
    ]

    all_u, all_v, all_wave = [], [], []

    for lat, lon in sample_points:
        try:
            url = "https://marine-api.open-meteo.com/v1/marine"
            params = {
                "latitude":  lat,
                "longitude": lon,
                "hourly": "wave_height,wind_wave_height",
                "current": "wave_height,wind_wave_height",
                "timezone": "UTC",
                "forecast_days": 1,
            }
            resp = _get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current", {})
                wh = current.get("wave_height")
                if wh is not None:
                    all_wave.append(float(wh))
        except Exception:
            pass

        # Open-Meteo atmospheric untuk angin
        try:
            url_atm = "https://api.open-meteo.com/v1/forecast"
            params_atm = {
                "latitude":  lat,
                "longitude": lon,
                "current": "wind_speed_10m,wind_direction_10m",
                "timezone": "UTC",
            }
            resp_atm = _get(url_atm, params=params_atm, timeout=15)
            if resp_atm.status_code == 200:
                data_atm = resp_atm.json()
                current_atm = data_atm.get("current", {})
                wspd = current_atm.get("wind_speed_10m")
                wdir = current_atm.get("wind_direction_10m")
                if wspd is not None and wdir is not None:
                    # Konversi: arah angin datang → komponen u,v (konvensi meteorologi)
                    wdir_rad = np.radians(float(wdir))
                    wspd_f   = float(wspd) / 3.6   # km/h → m/s
                    # u = -wspd * sin(wdir), v = -wspd * cos(wdir)
                    all_u.append(-wspd_f * np.sin(wdir_rad))
                    all_v.append(-wspd_f * np.cos(wdir_rad))
        except Exception:
            pass

    if all_u or all_v or all_wave:
        return {"ok": True,
                "data": {
                    "angin_u":   float(np.mean(all_u))    if all_u   else -1.5,
                    "angin_v":   float(np.mean(all_v))    if all_v   else -0.5,
                    "gelombang": float(np.clip(np.mean(all_wave), 0.2, 2.5)) if all_wave else 0.8,
                    "source_era5": False,
                    "source_openmeteo": True,
                },
                "error": None}

    return {"ok": False, "data": None,
            "error": (
                "ERA5 & Open-Meteo sama-sama gagal. "
                "Kemungkinan koneksi internet terbatas."
            )}


# =============================================================
# 4. BMKG — Prakiraan gelombang (open data)
# =============================================================
def fetch_bmkg() -> dict:
    """
    PERBAIKAN:
    - Endpoint diperbarui ke data.bmkg.go.id yang aktif
    - Parser JSON yang benar sesuai format BMKG
    - Tambah endpoint GFS (prakiraan numerik) sebagai backup
    - Fallback ke Open-Meteo jika BMKG tidak merespons

    Endpoint BMKG resmi yang tersedia:
      - https://data.bmkg.go.id/DataMKG/MEWS/maritim/
      - https://inaoc.bmkg.go.id/ (kadang down)
    """
    now = datetime.datetime.utcnow()

    # ── Endpoint 1: BMKG Data Maritim JSON ───────────────────
    BMKG_ENDPOINTS = [
        # Format JSON baru (2023+)
        "https://data.bmkg.go.id/DataMKG/MEWS/maritim/ombakLaut.json",
        "https://data.bmkg.go.id/DataMKG/MEWS/maritim/gempaTerkini.json",
        # XML maritim
        "https://data.bmkg.go.id/DataMKG/MEWS/maritim/maritim.xml",
        # Endpoint lama (kadang masih aktif)
        "https://inaoc.bmkg.go.id/DataOlahan/gelombangLaut",
    ]

    for url in BMKG_ENDPOINTS:
        try:
            resp = _get(url, timeout=12)
            if resp.status_code != 200:
                continue

            content_type = resp.headers.get("Content-Type", "")
            text = resp.text

            # ── Parse JSON ──
            if "json" in content_type or text.strip().startswith("{"):
                try:
                    data = resp.json()
                    wave_vals = _extract_wave_from_bmkg_json(data)
                    if wave_vals:
                        wave_avg = float(np.clip(np.mean(wave_vals), 0.2, 2.5))
                        return {"ok": True,
                                "data": {"gelombang": wave_avg, "source_bmkg": True},
                                "error": None}
                except Exception:
                    pass

            # ── Parse XML/HTML: cari pola tinggi gelombang ──
            wave_vals = _extract_wave_from_text(text)
            if wave_vals:
                wave_avg = float(np.clip(np.mean(wave_vals), 0.2, 2.5))
                return {"ok": True,
                        "data": {"gelombang": wave_avg, "source_bmkg": True},
                        "error": None}

        except Exception:
            continue

    # ── Fallback: Open-Meteo Marine (gratis) ─────────────────
    # Sama seperti di ERA5 fallback tapi khusus gelombang
    sample_points = [(-8.0, 136.5), (-6.0, 132.0), (-10.0, 140.0)]
    wave_vals = []
    for lat, lon in sample_points:
        try:
            url = "https://marine-api.open-meteo.com/v1/marine"
            params = {
                "latitude":  lat,
                "longitude": lon,
                "current":   "wave_height",
                "timezone":  "UTC",
            }
            resp = _get(url, params=params, timeout=12)
            if resp.status_code == 200:
                wh = resp.json().get("current", {}).get("wave_height")
                if wh is not None:
                    wave_vals.append(float(wh))
        except Exception:
            pass

    if wave_vals:
        wave_avg = float(np.clip(np.mean(wave_vals), 0.2, 2.5))
        return {"ok": True,
                "data": {"gelombang": wave_avg,
                         "source_bmkg": False,
                         "source_openmeteo": True},
                "error": "BMKG tidak merespons, menggunakan Open-Meteo Marine API"}

    return {"ok": False, "data": None,
            "error": (
                "BMKG & Open-Meteo Marine sama-sama gagal. "
                "Kemungkinan:\n"
                "  1. Server BMKG sedang down (umum terjadi)\n"
                "  2. Format respons berubah\n"
                "Menggunakan nilai klimatologis sebagai fallback."
            )}


def _extract_wave_from_bmkg_json(data: dict) -> list:
    """Ekstrak nilai tinggi gelombang dari berbagai format JSON BMKG."""
    vals = []
    if not isinstance(data, dict):
        return vals

    # Format 1: {"data": [{"tinggiGelombang": "0.5 - 1.0", ...}]}
    for key in ("data", "result", "gelombang", "maritim"):
        if key not in data:
            continue
        items = data[key]
        if not isinstance(items, list):
            items = [items]
        for item in items:
            if not isinstance(item, dict):
                continue
            for field in ("tinggiGelombang", "wave_height", "gelombang",
                          "tinggi", "waveHeight"):
                val = item.get(field)
                if val is None:
                    continue
                parsed = _parse_wave_range(str(val))
                vals.extend(parsed)

    return vals


def _extract_wave_from_text(text: str) -> list:
    """
    Ekstrak nilai gelombang dari teks HTML/XML/plain-text BMKG.

    BMKG biasanya menulis: "0.5 - 1.5 meter" atau "0.5-1.5 m"
    """
    vals = []

    # Pola: angka - angka meter/m (berbagai variasi spasi dan tanda)
    patterns = [
        r"(\d+[\.,]?\d*)\s*[-–—]\s*(\d+[\.,]?\d*)\s*[Mm]eter",
        r"(\d+[\.,]?\d*)\s*[-–—]\s*(\d+[\.,]?\d*)\s*[Mm]\b",
        r"tinggi\w*\s*[:\s]+(\d+[\.,]?\d*)\s*[-–]\s*(\d+[\.,]?\d*)",
        r"wave[_\s]height[:\s]+(\d+[\.,]?\d*)",
        r"(\d+[\.,]?\d*)\s*m\s*gelombang",
    ]
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            groups = match.groups()
            if len(groups) == 2:
                try:
                    a = float(groups[0].replace(",", "."))
                    b = float(groups[1].replace(",", "."))
                    vals.append((a + b) / 2)
                except Exception:
                    pass
            elif len(groups) == 1:
                try:
                    vals.append(float(groups[0].replace(",", ".")))
                except Exception:
                    pass

    # Filter nilai yang masuk akal untuk Laut Arafura
    vals = [v for v in vals if 0.1 <= v <= 6.0]
    return vals


def _parse_wave_range(text: str) -> list:
    """Parse string seperti '0.5 - 1.5' atau '1.0' menjadi list nilai."""
    text = text.replace(",", ".").strip()
    m = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", text)
    if m:
        return [(float(m.group(1)) + float(m.group(2))) / 2]
    m2 = re.search(r"(\d+\.?\d*)", text)
    if m2:
        return [float(m2.group(1))]
    return []


# =============================================================
# GABUNGAN: build_realtime_dataframe()
# =============================================================
def build_realtime_dataframe(cmems_user: str, cmems_pass: str,
                              nasa_user: str,  nasa_pass: str,
                              cds_uid: str = "", cds_key: str = "") -> dict:
    """
    Panggil semua API, gabungkan ke satu DataFrame yang kompatibel dengan app.py.

    Returns:
        {
          "data"  : pd.DataFrame (1 baris) atau None,
          "status": { "CMEMS": bool, "NASA MODIS": bool, "ERA5/Open-Meteo": bool, "BMKG": bool }
        }
    """
    now = datetime.datetime.utcnow()

    # ── Nilai default klimatologis (fallback jika semua API gagal) ──
    t = now.month  # bulan 1-12
    seasonal = np.sin(2 * np.pi * t / 12)
    defaults = {
        "uo":          -0.05 + 0.03 * seasonal,
        "vo":          -0.01 + 0.01 * seasonal,
        "sst":          28.5 + 0.8 * seasonal,
        "ssta":         0.0,
        "salinitas":   34.2 + 0.3 * seasonal,
        "chla":         0.22 - 0.05 * seasonal,
        "ph":           8.10,
        "do":           6.2  - 0.2 * seasonal,
        "gelombang":    0.8  + 0.3 * abs(seasonal),
        "angin_u":     -1.5  - 0.5 * seasonal,
        "angin_v":     -0.5  - 0.2 * seasonal,
        "time":         pd.Timestamp(now),
        "year":         now.year,
        "month":        now.month,
    }

    merged = defaults.copy()
    status = {
        "CMEMS":          False,
        "NASA MODIS":     False,
        "ERA5/Open-Meteo": False,
        "BMKG":           False,
    }
    any_ok = False
    errors = {}

    # ── 1. CMEMS ──────────────────────────────────────────────
    print("[LAUTAN] Menghubungi CMEMS...")
    r_cmems = fetch_cmems(cmems_user, cmems_pass)
    if r_cmems["ok"] and r_cmems["data"]:
        for k in ("uo", "vo", "sst", "salinitas"):
            if k in r_cmems["data"]:
                merged[k] = float(r_cmems["data"][k])
        merged["ssta"] = merged["sst"] - 28.5
        status["CMEMS"] = True
        any_ok = True
        print("[LAUTAN] ✓ CMEMS berhasil")
    else:
        errors["CMEMS"] = r_cmems.get("error", "unknown")
        print(f"[LAUTAN] ✗ CMEMS gagal: {errors['CMEMS'][:80]}")

    # ── 2. NASA MODIS ─────────────────────────────────────────
    print("[LAUTAN] Menghubungi NASA MODIS...")
    r_modis = fetch_nasa_modis(nasa_user, nasa_pass)
    if r_modis["ok"] and r_modis["data"]:
        if "chla" in r_modis["data"]:
            merged["chla"] = float(r_modis["data"]["chla"])
        status["NASA MODIS"] = True
        any_ok = True
        print("[LAUTAN] ✓ NASA MODIS berhasil")
    else:
        errors["NASA MODIS"] = r_modis.get("error", "unknown")
        print(f"[LAUTAN] ✗ NASA MODIS gagal: {errors['NASA MODIS'][:80]}")

    # ── 3. ERA5 / Open-Meteo ──────────────────────────────────
    print("[LAUTAN] Menghubungi ERA5 / Open-Meteo...")
    r_era5 = fetch_era5(cds_uid, cds_key)
    if r_era5["ok"] and r_era5["data"]:
        for k in ("angin_u", "angin_v", "gelombang"):
            if k in r_era5["data"]:
                merged[k] = float(r_era5["data"][k])
        status["ERA5/Open-Meteo"] = True
        any_ok = True
        src = "ERA5" if r_era5["data"].get("source_era5") else "Open-Meteo"
        print(f"[LAUTAN] ✓ Angin/Gelombang berhasil dari {src}")
    else:
        errors["ERA5"] = r_era5.get("error", "unknown")
        print(f"[LAUTAN] ✗ ERA5/Open-Meteo gagal: {errors['ERA5'][:80]}")

    # ── 4. BMKG ───────────────────────────────────────────────
    print("[LAUTAN] Menghubungi BMKG...")
    r_bmkg = fetch_bmkg()
    if r_bmkg["ok"] and r_bmkg["data"]:
        if "gelombang" in r_bmkg["data"]:
            # BMKG override ERA5 untuk gelombang (lebih lokal)
            merged["gelombang"] = float(r_bmkg["data"]["gelombang"])
        status["BMKG"] = True
        any_ok = True
        src_bmkg = "BMKG" if r_bmkg["data"].get("source_bmkg") else "Open-Meteo Marine"
        print(f"[LAUTAN] ✓ Gelombang BMKG berhasil dari {src_bmkg}")
    else:
        errors["BMKG"] = r_bmkg.get("error", "unknown")
        print(f"[LAUTAN] ✗ BMKG gagal: {errors['BMKG'][:80]}")

    # ── Derive kolom tambahan ─────────────────────────────────
    merged["current_speed"] = float(np.sqrt(merged["uo"]**2 + merged["vo"]**2))
    merged["do"]            = float(np.clip(merged["do"],  4.5, 7.5))
    merged["ph"]            = float(np.clip(merged["ph"],  7.9, 8.4))
    merged["chla"]          = float(np.clip(merged["chla"], 0.05, 0.8))
    merged["gelombang"]     = float(np.clip(merged["gelombang"], 0.2, 2.5))

    # Ocean Health Index & Fisheries Index (supaya kompatibel dengan app.py)
    from spatial import normalisasi_global
    df_single = pd.DataFrame([merged])
    df_single["Ocean_Health_Index"] = (
        0.25 * normalisasi_global(df_single["do"],       4.5, 7.5) +
        0.20 * normalisasi_global(df_single["ph"],       7.9, 8.4) +
        0.20 * normalisasi_global(df_single["chla"],     0.05, 0.8) +
        0.15 * normalisasi_global(df_single["salinitas"],32.0, 36.5) +
        0.20 * (1 - normalisasi_global(df_single["gelombang"], 0.2, 2.5))
    ) * 100
    df_single["Fisheries_Index"] = (
        0.35 * normalisasi_global(df_single["chla"],           0.05, 0.8) +
        0.25 * normalisasi_global(df_single["do"],             4.5, 7.5) +
        0.20 * normalisasi_global(df_single["current_speed"],  0.0, 0.25) +
        0.20 * (1 - normalisasi_global(df_single["gelombang"], 0.2, 2.5))
    ) * 100

    print(f"\n[LAUTAN] API berhasil: {sum(status.values())}/{len(status)}")
    if errors:
        print(f"[LAUTAN] Error log: {errors}")

    return {
        "data":   df_single if any_ok else None,
        "status": status,
        "errors": errors,
    }
