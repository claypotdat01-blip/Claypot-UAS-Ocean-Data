"""
data_fetcher.py — Pengambil data real-time LAUTAN
==============================================================================
Sumber data:
  1. CMEMS  → Arus (uo/vo), SST, Salinitas, Klorofil-a (dataset bio) ← UTAMA
  2. ERA5   → Angin (u10/v10) via CDS API; fallback Open-Meteo (tanpa akun)
  3. BMKG   → Gelombang; fallback Open-Meteo Marine (tanpa akun)

CATATAN PERUBAHAN:
  - NASA MODIS DINONAKTIFKAN — Klorofil-a sepenuhnya dari CMEMS bio dataset
  - Prioritas Chl-a: CMEMS bio dataset → fallback klimatologis

Strategi fallback berlapis:
  - Jika satu endpoint gagal → coba endpoint berikutnya
  - Jika semua API gagal → gunakan estimasi klimatologis musiman
  - App TIDAK pernah crash karena API gagal

Instalasi:
  pip install requests cdsapi
  pip install copernicusmarine   ← opsional, untuk CMEMS subset
==============================================================================
"""

import datetime
import warnings
import re
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Batas wilayah Laut Arafura ──────────────────────────────────────────────
LAT_MIN, LAT_MAX = -12.0, -4.0
LON_MIN, LON_MAX = 129.0, 144.0

# ── Titik sampel spasial untuk rata-rata ────────────────────────────────────
SAMPLE_POINTS = [
    (-6.0,  132.0),
    (-8.0,  136.5),
    (-10.0, 140.0),
    (-7.0,  138.0),
    (-9.0,  134.0),
]


# =============================================================================
# HELPER UMUM
# =============================================================================
def _get(url, auth=None, params=None, timeout=20, headers=None):
    """requests.get dengan header default dan error handling."""
    import requests
    h = {"User-Agent": "LAUTAN-OceanPlatform/1.0", "Accept": "application/json"}
    if headers:
        h.update(headers)
    return requests.get(url, auth=auth, params=params, timeout=timeout, headers=h)


def _klimatologi_bulan(month: int) -> dict:
    """
    Nilai klimatologis rata-rata Laut Arafura per bulan.
    Digunakan sebagai fallback terakhir jika semua API gagal.
    """
    t = 2 * np.pi * (month - 1) / 12
    return {
        "uo":        -0.06 + 0.04 * np.sin(t + 1.0),
        "vo":        -0.01 + 0.02 * np.cos(t),
        "sst":        28.5 - 1.2  * np.sin(t + 0.5),
        "ssta":      -0.1  + 0.3  * np.sin(t),
        "salinitas":  34.2 + 0.4  * np.cos(t + 0.3),
        "chla":        0.20 + 0.10 * np.sin(t + 1.5),
        "ph":          8.10 - 0.02 * np.sin(t),
        "do":          6.20 - 0.15 * np.sin(t + 0.5),
        "gelombang":   0.85 + 0.40 * np.sin(t + 1.0),
        "angin_u":    -1.5  - 1.0  * np.sin(t + 1.0),
        "angin_v":    -0.5  - 0.3  * np.cos(t),
    }


# =============================================================================
# 1. CMEMS — Arus, SST, Salinitas, Klorofil-a
# =============================================================================
def fetch_cmems(user: str, password: str) -> dict:
    """
    Ambil data oseanografi fisik dan biologis dari CMEMS.

    Dataset fisik  : cmems_mod_glo_phy_anfc_0.083deg_PT1H-i  (arus, SST, salinitas)
    Dataset biologi: cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D
                     (klorofil-a, near-real-time, tanpa celah data)

    Endpoint yang dicoba (berurutan):
      A. CMEMS ERDDAP NRT  — nrt.cmems-du.eu/erddap
      B. CMEMS Copernicus Marine Data Store REST API
      C. CMEMS WMS (GetCapabilities) — minimal untuk verifikasi koneksi
    """
    if not user or "@" not in user:
        return {"ok": False, "data": None,
                "error": "CMEMS_USER harus berupa email (contoh: nama@email.com)"}
    if not password or len(password) < 4:
        return {"ok": False, "data": None,
                "error": "CMEMS_PASS terlalu pendek atau kosong"}

    now      = datetime.datetime.utcnow()
    # CMEMS NRT biasanya tersedia dengan delay ~3 jam
    end_dt   = now.strftime("%Y-%m-%dT%H:00:00")
    start_dt = (now - datetime.timedelta(hours=6)).strftime("%Y-%m-%dT%H:00:00")

    result = {}

    # ── A. Coba CMEMS ERDDAP untuk data fisik ─────────────────────────────
    ERDDAP_PHY_DATASETS = [
        "cmems_mod_glo_phy_anfc_0.083deg_PT1H-i",
        "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
        "global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh",
    ]

    for dataset_id in ERDDAP_PHY_DATASETS:
        url = f"https://nrt.cmems-du.eu/erddap/griddap/{dataset_id}.json"
        lat_c = (LAT_MIN + LAT_MAX) / 2   # -8.0
        lon_c = (LON_MIN + LON_MAX) / 2   # 136.5
        query = (
            f"?uo[({start_dt}Z):1:({end_dt}Z)][0:1:0]"
            f"[({lat_c - 2}):4:({lat_c + 2})]"
            f"[({lon_c - 2}):4:({lon_c + 2})]"
            f",vo[({start_dt}Z):1:({end_dt}Z)][0:1:0]"
            f"[({lat_c - 2}):4:({lat_c + 2})]"
            f"[({lon_c - 2}):4:({lon_c + 2})]"
        )
        try:
            resp = _get(url + query, auth=(user, password), timeout=30)
            if resp.status_code == 200:
                body      = resp.json()
                col_names = body.get("table", {}).get("columnNames", [])
                rows      = body.get("table", {}).get("rows", [])
                if rows and col_names:
                    df_raw = pd.DataFrame(rows, columns=col_names)
                    for src, dst in [("uo","uo"),("vo","vo"),
                                     ("thetao","sst"),("so","salinitas")]:
                        if src in df_raw.columns:
                            v = pd.to_numeric(df_raw[src], errors="coerce").dropna()
                            if len(v):
                                result[dst] = float(v.mean())
                    if "uo" in result:
                        print(f"[CMEMS] ✓ Fisik dari dataset: {dataset_id}")
                        break
        except Exception as e:
            print(f"[CMEMS] dataset {dataset_id} gagal: {str(e)[:60]}")
            continue

    # ── B. Coba CMEMS untuk klorofil-a (dataset biologi) ──────────────────
    ERDDAP_BIO_DATASETS = [
        # Dataset utama NRT global 4km (nama aktif per 2024-2025)
        ("cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D", "CHL"),
        # Nama lama (kadang masih aktif)
        ("cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1D", "CHL"),
        # Dataset multi-sensor alternatif
        ("cmems_obs-oc_glo_bgc-optics_nrt_l3-multi-4km_P1D", "CHL"),
        # Dataset tambahan — OLCI Sentinel-3
        ("cmems_obs-oc_glo_bgc-plankton_nrt_l3-olci-4km_P1D", "CHL"),
        # Dataset MY (multi-year) sebagai fallback terakhir CMEMS
        ("cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D", "CHL"),
    ]

    # Coba mundur sampai 7 hari — data NRT butuh waktu proses 2-3 hari
    date_candidates = [
        (now - datetime.timedelta(days=d)).strftime("%Y-%m-%dT00:00:00")
        for d in range(2, 9)   # coba D-2 sampai D-8
    ]

    for dataset_id, var_name in ERDDAP_BIO_DATASETS:
        url = f"https://nrt.cmems-du.eu/erddap/griddap/{dataset_id}.json"

        for date_str in date_candidates:
            query = (
                f"?{var_name}[({date_str}Z):1:({date_str}Z)]"
                f"[({LAT_MIN}):4:({LAT_MAX})]"
                f"[({LON_MIN}):4:({LON_MAX})]"
            )
            try:
                resp = _get(url + query, auth=(user, password), timeout=40)

                if resp.status_code == 404:
                    # Dataset tidak ada di server ini, langsung skip ke dataset berikutnya
                    break

                if resp.status_code == 400:
                    # Tanggal belum tersedia, coba mundur lagi
                    continue

                if resp.status_code == 200:
                    body  = resp.json()
                    col_n = body.get("table", {}).get("columnNames", [])
                    rows  = body.get("table", {}).get("rows", [])

                    if not rows or var_name not in col_n:
                        # Respons kosong, coba tanggal sebelumnya
                        continue

                    df_raw = pd.DataFrame(rows, columns=col_n)
                    v = pd.to_numeric(df_raw[var_name], errors="coerce").dropna()
                    v = v[(v > 0.001) & (v < 20)]

                    if len(v):
                        result["chla"] = float(np.clip(v.median(), 0.05, 0.8))
                        print(
                            f"[CMEMS] ✓ Klorofil-a dari {dataset_id} "
                            f"(tgl {date_str[:10]}): {result['chla']:.3f} mg/m³ "
                            f"(n={len(v)})"
                        )
                        break  # tanggal ini berhasil, keluar dari loop tanggal

            except Exception as e:
                print(f"[CMEMS] bio {dataset_id} @ {date_str[:10]}: {str(e)[:60]}")
                break  # connection error, skip ke dataset berikutnya

        if "chla" in result:
            break  # sudah dapat klorofil, tidak perlu coba dataset lain

    # ── Jika Chl-a belum dapat dari ERDDAP, coba endpoint alternatif ──────
    if "chla" not in result:
        print("[CMEMS] ⚠ Chl-a belum dapat dari ERDDAP, coba endpoint alternatif...")
        chla_alt = _fetch_cmems_chla_alternative(user, password, now)
        if chla_alt is not None:
            result["chla"] = chla_alt

    # ── C. Fallback WMS — minimal verifikasi koneksi + nilai approx ───────
    if not result:
        try:
            wms_url = (
                "https://nrt.cmems-du.eu/thredds/wms/"
                "cmems_mod_glo_phy_anfc_0.083deg_PT1H-i"
            )
            resp = _get(wms_url, auth=(user, password),
                        params={"SERVICE":"WMS","VERSION":"1.3.0","REQUEST":"GetCapabilities"},
                        timeout=15)
            if resp.status_code == 200:
                klim = _klimatologi_bulan(now.month)
                result.update(klim)
                result["_source"] = "wms_klimatologi"
                print("[CMEMS] ✓ Koneksi WMS OK, menggunakan klimatologis")
        except Exception:
            pass

    if result:
        if "ssta" not in result and "sst" in result:
            result["ssta"] = result["sst"] - 28.5
        result["source_cmems"] = True
        return {"ok": True, "data": result, "error": None}

    return {"ok": False, "data": None,
            "error": (
                "CMEMS gagal di semua endpoint. "
                "Pastikan email/password benar di marine.copernicus.eu"
            )}


def _fetch_cmems_chla_alternative(user: str, password: str,
                                   now: datetime.datetime):
    """
    Endpoint alternatif untuk Chl-a CMEMS jika ERDDAP utama gagal.
    Mencoba:
      1. CMEMS Copernicus Marine Toolbox REST API (subset service)
      2. CMEMS THREDDS OPeNDAP
    Mengembalikan nilai float Chl-a atau None jika semua gagal.
    """
    # ── Coba Copernicus Marine Toolbox REST (subset) ──────────────────────
    SUBSET_DATASETS = [
        "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D",
        "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1D",
    ]

    date_str = (now - datetime.timedelta(days=3)).strftime("%Y-%m-%d")

    for dataset_id in SUBSET_DATASETS:
        try:
            url = (
                "https://marine.copernicus.eu/services/auth/aai"
                "/downloadChunkDataset"
            )
            params = {
                "datasetId":   dataset_id,
                "variableList": "CHL",
                "minimumDepth":  0,
                "maximumDepth":  1,
                "minimumLat":   LAT_MIN,
                "maximumLat":   LAT_MAX,
                "minimumLon":   LON_MIN,
                "maximumLon":   LON_MAX,
                "startDate":    date_str,
                "endDate":      date_str,
                "outputFormat": "json",
            }
            resp = _get(url, auth=(user, password), params=params, timeout=30)
            if resp.status_code == 200:
                body = resp.json()
                # Format response bervariasi, cari nilai CHL di mana saja
                vals = _deep_find_numeric(body, ["CHL","chl","chlorophyll"])
                vals = [v for v in vals if 0.001 < v < 20]
                if vals:
                    chla = float(np.clip(np.median(vals), 0.05, 0.8))
                    print(f"[CMEMS-ALT] ✓ Chl-a dari subset API: {chla:.3f} mg/m³")
                    return chla
        except Exception as e:
            print(f"[CMEMS-ALT] Subset API {dataset_id}: {str(e)[:60]}")

    # ── Coba THREDDS OPeNDAP ──────────────────────────────────────────────
    try:
        thredds_url = (
            "https://nrt.cmems-du.eu/thredds/dodsC/"
            "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D"
            ".ascii?CHL[0:1:0][0:10:100][0:10:100]"
        )
        resp = _get(thredds_url, auth=(user, password), timeout=20)
        if resp.status_code == 200:
            # Parse ASCII DDS response
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", resp.text)
            vals = []
            for n in nums:
                try:
                    v = float(n)
                    if 0.001 < v < 20:
                        vals.append(v)
                except Exception:
                    pass
            if vals:
                chla = float(np.clip(np.median(vals), 0.05, 0.8))
                print(f"[CMEMS-ALT] ✓ Chl-a dari THREDDS: {chla:.3f} mg/m³")
                return chla
    except Exception as e:
        print(f"[CMEMS-ALT] THREDDS error: {str(e)[:60]}")

    return None


def _deep_find_numeric(obj, keys: list) -> list:
    """Cari nilai numerik dari dict/list secara rekursif berdasarkan nama key."""
    vals = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(key.lower() in k.lower() for key in keys):
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, (int, float)):
                            vals.append(float(item))
            vals.extend(_deep_find_numeric(v, keys))
    elif isinstance(obj, list):
        for item in obj:
            vals.extend(_deep_find_numeric(item, keys))
    return vals


# =============================================================================
# 2. NASA MODIS — DINONAKTIFKAN
#    Klorofil-a sepenuhnya diambil dari CMEMS bio dataset.
#    Fungsi ini dipertahankan untuk kompatibilitas tapi tidak dipanggil.
# =============================================================================
def fetch_nasa_modis(user: str, password: str) -> dict:
    """
    [DINONAKTIFKAN] — Klorofil-a sekarang diambil dari CMEMS.
    Fungsi ini tidak dipanggil oleh build_realtime_dataframe().
    Tetap ada untuk kompatibilitas mundur jika diperlukan.
    """
    return {
        "ok":    False,
        "data":  None,
        "error": "NASA MODIS dinonaktifkan. Klorofil-a diambil dari CMEMS bio dataset.",
    }


# =============================================================================
# 3. ERA5 / Open-Meteo — Angin 10m
# =============================================================================
def fetch_era5(cds_uid: str, cds_key: str) -> dict:
    """
    Ambil data angin 10m dan tinggi gelombang.

    Urutan prioritas:
      1. Open-Meteo (gratis, tanpa akun, near-real-time) ← utama
      2. ERA5 via CDS API (butuh akun + CDS_UID terisi)  ← jika UID ada
    """
    print("[ERA5] Mencoba Open-Meteo...")
    result_om = _fetch_openmeteo_wind()
    if result_om["ok"]:
        return result_om

    uid_str = str(cds_uid).strip()
    if uid_str and uid_str not in ("", "0", "ISI_UID_KAMU"):
        print(f"[ERA5] Mencoba CDS API dengan UID: {uid_str[:6]}...")
        result_cds = _fetch_era5_cds(uid_str, str(cds_key).strip())
        if result_cds["ok"]:
            return result_cds

    return {"ok": False, "data": None,
            "error": "ERA5 & Open-Meteo keduanya gagal. Cek koneksi internet."}


def _fetch_openmeteo_wind() -> dict:
    """
    Ambil angin real-time dari Open-Meteo API (gratis, tanpa akun).
    Rata-ratakan dari beberapa titik di Laut Arafura.
    """
    all_u, all_v, all_wave = [], [], []

    for lat, lon in SAMPLE_POINTS:
        # ── Angin dari Open-Meteo Forecast ──
        try:
            resp = _get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":  lat,
                    "longitude": lon,
                    "current":   "wind_speed_10m,wind_direction_10m",
                    "wind_speed_unit": "ms",
                    "timezone":  "UTC",
                    "forecast_days": 1,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                cur = resp.json().get("current", {})
                wspd = cur.get("wind_speed_10m")
                wdir = cur.get("wind_direction_10m")
                if wspd is not None and wdir is not None:
                    rad = np.radians(float(wdir))
                    spd = float(wspd)
                    all_u.append(-spd * np.sin(rad))
                    all_v.append(-spd * np.cos(rad))
        except Exception:
            pass

        # ── Gelombang dari Open-Meteo Marine ──
        try:
            resp = _get(
                "https://marine-api.open-meteo.com/v1/marine",
                params={
                    "latitude":  lat,
                    "longitude": lon,
                    "current":   "wave_height,wind_wave_height,swell_wave_height",
                    "timezone":  "UTC",
                    "forecast_days": 1,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                cur = resp.json().get("current", {})
                wh = cur.get("wave_height") or cur.get("wind_wave_height")
                if wh is not None:
                    all_wave.append(float(wh))
        except Exception:
            pass

    if all_u or all_v or all_wave:
        return {
            "ok": True,
            "data": {
                "angin_u":   float(np.mean(all_u))   if all_u   else -1.5,
                "angin_v":   float(np.mean(all_v))   if all_v   else -0.5,
                "gelombang": float(np.clip(np.mean(all_wave), 0.2, 2.5))
                             if all_wave else 0.8,
                "source_era5":      False,
                "source_openmeteo": True,
            },
            "error": None,
        }

    return {"ok": False, "data": None, "error": "Open-Meteo tidak merespons"}


def _fetch_era5_cds(uid: str, key: str) -> dict:
    """Ambil angin dari ERA5 via CDS API (butuh akun dan UID terisi)."""
    try:
        import cdsapi
    except ImportError:
        return {"ok": False, "data": None,
                "error": "cdsapi belum terpasang. Jalankan: pip install cdsapi"}

    import tempfile, os
    tmp_rc = os.path.join(tempfile.gettempdir(), ".cdsapirc_lautan")
    tmp_nc = os.path.join(tempfile.gettempdir(), "era5_lautan.nc")

    try:
        with open(tmp_rc, "w") as f:
            f.write(f"url: https://cds.climate.copernicus.eu/api/v2\n")
            f.write(f"key: {uid}:{key}\n")
        os.environ["CDSAPI_RC"] = tmp_rc

        c      = cdsapi.Client(quiet=True, progress=False)
        now    = datetime.datetime.utcnow()
        target = now - datetime.timedelta(days=7)

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
            "time":   ["00:00","06:00","12:00","18:00"],
            "area":   [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],
            "format": "netcdf",
        }, tmp_nc)

        try:
            import netCDF4 as nc_lib
            with nc_lib.Dataset(tmp_nc) as ds:
                def _m(ds, *names):
                    for n in names:
                        if n in ds.variables:
                            return float(np.nanmean(ds.variables[n][:]))
                    return None
                u10  = _m(ds, "u10", "u10m") or -1.5
                v10  = _m(ds, "v10", "v10m") or -0.5
                wave = _m(ds, "swh", "shww") or 0.8
        except ImportError:
            try:
                import xarray as xr
                ds = xr.open_dataset(tmp_nc)
                u10  = float(ds["u10"].mean())  if "u10"  in ds else -1.5
                v10  = float(ds["v10"].mean())  if "v10"  in ds else -0.5
                wave = float(ds["swh"].mean())  if "swh"  in ds else 0.8
                ds.close()
            except Exception:
                return {"ok": False, "data": None,
                        "error": "netCDF4/xarray belum terpasang untuk baca file ERA5"}
        finally:
            if os.path.exists(tmp_nc):
                os.remove(tmp_nc)

        return {
            "ok": True,
            "data": {
                "angin_u":   float(u10),
                "angin_v":   float(v10),
                "gelombang": float(np.clip(wave, 0.2, 2.5)),
                "source_era5":      True,
                "source_openmeteo": False,
            },
            "error": None,
        }

    except Exception as e:
        return {"ok": False, "data": None,
                "error": f"ERA5/CDS error: {str(e)[:150]}"}


# =============================================================================
# 4. BMKG — Tinggi Gelombang
# =============================================================================
def fetch_bmkg() -> dict:
    """
    Ambil prakiraan tinggi gelombang dari BMKG.

    Endpoint yang dicoba (berurutan):
      1. data.bmkg.go.id — JSON maritim resmi
      2. inaoc.bmkg.go.id — sistem peringatan dini gelombang
      3. Open-Meteo Marine — fallback gratis jika BMKG down
    """
    BMKG_URLS = [
        "https://data.bmkg.go.id/DataMKG/MEWS/maritim/ombakLaut.json",
        "https://data.bmkg.go.id/DataMKG/MEWS/maritim/maritim.json",
        "https://data.bmkg.go.id/DataMKG/MEWS/maritim/maritim.xml",
        "https://inaoc.bmkg.go.id/DataOlahan/gelombangLaut",
    ]

    for url in BMKG_URLS:
        try:
            resp = _get(url, timeout=12)
            if resp.status_code != 200:
                continue

            text = resp.text.strip()

            if text.startswith("{") or text.startswith("["):
                try:
                    data = resp.json()
                    vals = _bmkg_extract_json(data)
                    if vals:
                        wave = float(np.clip(np.mean(vals), 0.2, 2.5))
                        print(f"[BMKG] ✓ Gelombang dari {url.split('/')[-1]}: {wave:.2f} m")
                        return {"ok": True,
                                "data": {"gelombang": wave, "source_bmkg": True},
                                "error": None}
                except Exception:
                    pass

            vals = _bmkg_extract_text(text)
            if vals:
                wave = float(np.clip(np.mean(vals), 0.2, 2.5))
                print(f"[BMKG] ✓ Gelombang (text parse): {wave:.2f} m")
                return {"ok": True,
                        "data": {"gelombang": wave, "source_bmkg": True},
                        "error": None}

        except Exception as e:
            print(f"[BMKG] {url.split('/')[-1]} gagal: {str(e)[:50]}")
            continue

    # ── Fallback: Open-Meteo Marine ──────────────────────────────────────────
    print("[BMKG] Server tidak merespons, mencoba Open-Meteo Marine...")
    wave_vals = []
    for lat, lon in SAMPLE_POINTS:
        try:
            resp = _get(
                "https://marine-api.open-meteo.com/v1/marine",
                params={
                    "latitude":  lat,
                    "longitude": lon,
                    "current":   "wave_height",
                    "timezone":  "UTC",
                },
                timeout=12,
            )
            if resp.status_code == 200:
                wh = resp.json().get("current", {}).get("wave_height")
                if wh is not None:
                    wave_vals.append(float(wh))
        except Exception:
            pass

    if wave_vals:
        wave = float(np.clip(np.mean(wave_vals), 0.2, 2.5))
        print(f"[BMKG] ✓ Gelombang dari Open-Meteo Marine: {wave:.2f} m")
        return {"ok": True,
                "data": {"gelombang": wave,
                         "source_bmkg": False,
                         "source_openmeteo": True},
                "error": "BMKG down, menggunakan Open-Meteo Marine"}

    return {"ok": False, "data": None,
            "error": "BMKG dan Open-Meteo Marine keduanya tidak merespons"}


def _bmkg_extract_json(data) -> list:
    """Ekstrak nilai gelombang dari berbagai format JSON BMKG."""
    vals = []
    if isinstance(data, list):
        for item in data:
            vals.extend(_bmkg_extract_json(item))
        return vals
    if not isinstance(data, dict):
        return vals
    for field in ("tinggiGelombang","wave_height","gelombang","tinggi",
                  "waveHeight","tinggiGelombangMax","tinggiGelombangMin"):
        val = data.get(field)
        if val is not None:
            parsed = _parse_wave_str(str(val))
            vals.extend(parsed)
    for v in data.values():
        if isinstance(v, (dict, list)):
            vals.extend(_bmkg_extract_json(v))
    return vals


def _bmkg_extract_text(text: str) -> list:
    """Ekstrak nilai gelombang dari teks HTML/XML/plain BMKG via regex."""
    vals = []
    patterns = [
        r"(\d+[\.,]\d*)\s*[-–—]\s*(\d+[\.,]\d*)\s*[Mm]eter",
        r"(\d+[\.,]\d*)\s*[-–—]\s*(\d+[\.,]\d*)\s*[Mm]\b",
        r"tinggi\w*\s*[=:]\s*(\d+[\.,]\d*)\s*[-–]\s*(\d+[\.,]\d*)",
        r"wave[_\s]?height[=:\s]+(\d+[\.,]\d*)",
        r">(\d+[\.,]\d*)\s*[-–]\s*(\d+[\.,]\d*)<",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            groups = m.groups()
            try:
                if len(groups) == 2:
                    a = float(groups[0].replace(",", "."))
                    b = float(groups[1].replace(",", "."))
                    if 0.1 <= (a+b)/2 <= 8.0:
                        vals.append((a + b) / 2)
                elif len(groups) == 1:
                    v = float(groups[0].replace(",", "."))
                    if 0.1 <= v <= 8.0:
                        vals.append(v)
            except Exception:
                pass
    return vals


def _parse_wave_str(text: str) -> list:
    """Parse string seperti '0.5 - 1.5' atau '1.0' → list nilai."""
    text = text.replace(",", ".").strip()
    m = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return [(a + b) / 2] if 0.1 <= (a+b)/2 <= 8.0 else []
    m2 = re.search(r"(\d+\.?\d*)", text)
    if m2:
        v = float(m2.group(1))
        return [v] if 0.1 <= v <= 8.0 else []
    return []


# =============================================================================
# GABUNGAN — build_realtime_dataframe()
# =============================================================================
def build_realtime_dataframe(cmems_user: str, cmems_pass: str,
                              nasa_user:  str, nasa_pass:  str,
                              cds_uid: str = "", cds_key: str = "") -> dict:
    """
    Panggil semua API, gabungkan ke satu DataFrame kompatibel dengan app.py.

    Urutan:
      1. CMEMS  → uo, vo, sst, salinitas, chla (fisik + bio dataset)
      2. ERA5   → angin_u, angin_v, gelombang (via Open-Meteo atau CDS)
      3. BMKG   → gelombang (override ERA5 jika berhasil)

    CATATAN: NASA MODIS tidak lagi dipanggil. Chl-a sepenuhnya dari CMEMS.

    Returns:
        {
          "data"  : pd.DataFrame (1 baris) | None jika semua API gagal,
          "status": { "CMEMS": bool, "NASA MODIS": bool,
                      "ERA5/Open-Meteo": bool, "BMKG": bool },
          "errors": { nama_api: pesan_error }
        }
    """
    now      = datetime.datetime.utcnow()
    klim     = _klimatologi_bulan(now.month)

    merged = {
        **klim,
        "time":  pd.Timestamp(now),
        "year":  now.year,
        "month": now.month,
        "ph":    8.10,
        "do":    6.20,
    }

    status = {
        "CMEMS":           False,
        "NASA MODIS":      False,   # selalu False — dinonaktifkan
        "ERA5/Open-Meteo": False,
        "BMKG":            False,
    }
    errors = {
        # Langsung isi pesan info untuk MODIS supaya UI tidak tampil merah tanpa sebab
        "NASA MODIS": "Dinonaktifkan — Klorofil-a diambil dari CMEMS bio dataset.",
    }
    any_ok = False

    # ── 1. CMEMS (fisik + klorofil-a) ────────────────────────────────────────
    print("\n[LAUTAN] ── Menghubungi CMEMS...")
    r = fetch_cmems(cmems_user, cmems_pass)
    if r["ok"] and r["data"]:
        for k in ("uo","vo","sst","ssta","salinitas","chla"):
            if k in r["data"]:
                merged[k] = float(r["data"][k])
        status["CMEMS"] = True
        any_ok = True
        chla_src = "CMEMS bio" if "chla" in r["data"] else "klimatologi"
        print(f"[LAUTAN] ✓ CMEMS berhasil (Chl-a dari {chla_src})")
    else:
        errors["CMEMS"] = r.get("error","unknown")
        print(f"[LAUTAN] ✗ CMEMS: {errors['CMEMS'][:80]}")

    # ── 2. NASA MODIS — DILEWATI ─────────────────────────────────────────────
    # Chl-a sudah diambil dari CMEMS di blok 1.
    # Status NASA MODIS dibiarkan False dan tidak mempengaruhi data.
    print("[LAUTAN] ── NASA MODIS dilewati (Chl-a dari CMEMS)")

    # ── 3. ERA5 / Open-Meteo (angin) ─────────────────────────────────────────
    print("[LAUTAN] ── Menghubungi ERA5/Open-Meteo...")
    r = fetch_era5(cds_uid, cds_key)
    if r["ok"] and r["data"]:
        for k in ("angin_u","angin_v","gelombang"):
            if k in r["data"]:
                merged[k] = float(r["data"][k])
        status["ERA5/Open-Meteo"] = True
        any_ok = True
        src = "ERA5" if r["data"].get("source_era5") else "Open-Meteo"
        print(f"[LAUTAN] ✓ Angin dari {src}")
    else:
        errors["ERA5"] = r.get("error","unknown")
        print(f"[LAUTAN] ✗ ERA5: {errors['ERA5'][:80]}")

    # ── 4. BMKG (gelombang) ──────────────────────────────────────────────────
    print("[LAUTAN] ── Menghubungi BMKG...")
    r = fetch_bmkg()
    if r["ok"] and r["data"]:
        merged["gelombang"] = float(r["data"]["gelombang"])
        status["BMKG"] = True
        any_ok = True
        src = "BMKG" if r["data"].get("source_bmkg") else "Open-Meteo Marine"
        print(f"[LAUTAN] ✓ Gelombang dari {src}")
    else:
        errors["BMKG"] = r.get("error","unknown")
        print(f"[LAUTAN] ✗ BMKG: {errors['BMKG'][:80]}")

    # ── Derive & clip ─────────────────────────────────────────────────────────
    merged["current_speed"] = float(np.sqrt(merged["uo"]**2 + merged["vo"]**2))
    merged["ssta"]    = merged.get("sst", 28.5) - 28.5
    merged["chla"]    = float(np.clip(merged.get("chla",   0.20), 0.05, 0.8))
    merged["do"]      = float(np.clip(merged.get("do",     6.20), 4.5,  7.5))
    merged["ph"]      = float(np.clip(merged.get("ph",     8.10), 7.9,  8.4))
    merged["salinitas"] = float(np.clip(merged.get("salinitas", 34.2), 32.0, 36.5))
    merged["gelombang"] = float(np.clip(merged.get("gelombang",  0.85), 0.2,  2.5))

    # ── Hitung indeks (kompatibel dengan app.py) ──────────────────────────────
    try:
        from spatial import normalisasi_global
        df_out = pd.DataFrame([merged])
        df_out["Ocean_Health_Index"] = (
            0.25 * normalisasi_global(df_out["do"],        4.5,  7.5) +
            0.20 * normalisasi_global(df_out["ph"],        7.9,  8.4) +
            0.20 * normalisasi_global(df_out["chla"],      0.05, 0.8) +
            0.15 * normalisasi_global(df_out["salinitas"], 32.0, 36.5) +
            0.20 * (1 - normalisasi_global(df_out["gelombang"], 0.2, 2.5))
        ) * 100
        df_out["Fisheries_Index"] = (
            0.35 * normalisasi_global(df_out["chla"],          0.05, 0.8) +
            0.25 * normalisasi_global(df_out["do"],            4.5,  7.5) +
            0.20 * normalisasi_global(df_out["current_speed"], 0.0,  0.25) +
            0.20 * (1 - normalisasi_global(df_out["gelombang"], 0.2, 2.5))
        ) * 100
    except Exception as e:
        df_out = pd.DataFrame([merged])
        df_out["Ocean_Health_Index"] = 50.0
        df_out["Fisheries_Index"]    = 50.0
        print(f"[LAUTAN] ⚠ Indeks dihitung dengan fallback: {e}")

    # Hitung berapa API yang berhasil (NASA MODIS dikecualikan dari hitungan)
    active_apis = ["CMEMS", "ERA5/Open-Meteo", "BMKG"]
    n_ok  = sum(status[k] for k in active_apis)
    n_all = len(active_apis)
    print(f"\n[LAUTAN] Selesai: {n_ok}/{n_all} API aktif berhasil")
    if errors:
        active_errors = {k:v for k,v in errors.items() if k != "NASA MODIS"}
        if active_errors:
            print(f"[LAUTAN] Error: { {k:v[:60] for k,v in active_errors.items()} }")

    return {
        "data":   df_out if any_ok else None,
        "status": status,
        "errors": errors,
    }
