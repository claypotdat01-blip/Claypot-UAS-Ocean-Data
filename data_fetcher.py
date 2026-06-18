"""
data_fetcher.py — Pengambil data real-time LAUTAN
Sumber: CMEMS (arus/SST/salinitas/klorofil-a), ERA5/Open-Meteo (angin), BMKG (gelombang)
NASA MODIS: DINONAKTIFKAN
"""

import datetime
import warnings
import re
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LAT_MIN, LAT_MAX = -12.0, -4.0
LON_MIN, LON_MAX = 129.0, 144.0

SAMPLE_POINTS = [
    (-6.0,  132.0),
    (-8.0,  136.5),
    (-10.0, 140.0),
    (-7.0,  138.0),
    (-9.0,  134.0),
]


def _get(url, auth=None, params=None, timeout=15, headers=None):
    import requests
    h = {"User-Agent": "LAUTAN-OceanPlatform/1.0", "Accept": "application/json"}
    if headers:
        h.update(headers)
    return requests.get(url, auth=auth, params=params, timeout=timeout, headers=h)


def _klimatologi_bulan(month):
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
# 1. CMEMS
# =============================================================================
def fetch_cmems(user, password):
    if not user or "@" not in str(user):
        return {"ok": False, "data": None,
                "error": "CMEMS_USER harus berupa email (contoh: nama@email.com)"}
    if not password or len(str(password)) < 4:
        return {"ok": False, "data": None,
                "error": "CMEMS_PASS terlalu pendek atau kosong"}

    now      = datetime.datetime.utcnow()
    end_dt   = now.strftime("%Y-%m-%dT%H:00:00")
    start_dt = (now - datetime.timedelta(hours=6)).strftime("%Y-%m-%dT%H:00:00")
    result   = {}
    lat_c    = (LAT_MIN + LAT_MAX) / 2
    lon_c    = (LON_MIN + LON_MAX) / 2

    # Fisik: arus, SST, salinitas
    PHY_DATASETS = [
        "cmems_mod_glo_phy_anfc_0.083deg_PT1H-i",
        "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
    ]
    for dataset_id in PHY_DATASETS:
        url = f"https://nrt.cmems-du.eu/erddap/griddap/{dataset_id}.json"
        query = (
            f"?uo[({start_dt}Z):1:({end_dt}Z)][0:1:0]"
            f"[({lat_c - 2}):4:({lat_c + 2})]"
            f"[({lon_c - 2}):4:({lon_c + 2})]"
            f",vo[({start_dt}Z):1:({end_dt}Z)][0:1:0]"
            f"[({lat_c - 2}):4:({lat_c + 2})]"
            f"[({lon_c - 2}):4:({lon_c + 2})]"
        )
        try:
            resp = _get(url + query, auth=(user, password), timeout=20)
            if resp.status_code == 200:
                body      = resp.json()
                col_names = body.get("table", {}).get("columnNames", [])
                rows      = body.get("table", {}).get("rows", [])
                if rows and col_names:
                    df_raw = pd.DataFrame(rows, columns=col_names)
                    for src, dst in [("uo", "uo"), ("vo", "vo"),
                                     ("thetao", "sst"), ("so", "salinitas")]:
                        if src in df_raw.columns:
                            v = pd.to_numeric(df_raw[src], errors="coerce").dropna()
                            if len(v):
                                result[dst] = float(v.mean())
                    if "uo" in result:
                        break
        except Exception as e:
            print(f"[CMEMS] Fisik {dataset_id} gagal: {str(e)[:60]}")
            continue

    # Klorofil-a dari bio dataset (2 dataset, 3 hari kandidat)
    BIO_DATASETS = [
        ("cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D", "CHL"),
        ("cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1D",          "CHL"),
    ]
    date_candidates = [
        (now - datetime.timedelta(days=d)).strftime("%Y-%m-%dT00:00:00")
        for d in range(2, 5)
    ]
    for dataset_id, var_name in BIO_DATASETS:
        url = f"https://nrt.cmems-du.eu/erddap/griddap/{dataset_id}.json"
        for date_str in date_candidates:
            query = (
                f"?{var_name}[({date_str}Z):1:({date_str}Z)]"
                f"[({LAT_MIN}):4:({LAT_MAX})]"
                f"[({LON_MIN}):4:({LON_MAX})]"
            )
            try:
                resp = _get(url + query, auth=(user, password), timeout=15)
                if resp.status_code == 404:
                    break
                if resp.status_code == 400:
                    continue
                if resp.status_code == 200:
                    body  = resp.json()
                    col_n = body.get("table", {}).get("columnNames", [])
                    rows  = body.get("table", {}).get("rows", [])
                    if not rows or var_name not in col_n:
                        continue
                    df_raw = pd.DataFrame(rows, columns=col_n)
                    v = pd.to_numeric(df_raw[var_name], errors="coerce").dropna()
                    v = v[(v > 0.001) & (v < 20)]
                    if len(v):
                        result["chla"] = float(np.clip(v.median(), 0.05, 0.8))
                        break
            except Exception as e:
                print(f"[CMEMS] Bio {dataset_id}: {str(e)[:60]}")
                break
        if "chla" in result:
            break

    if result:
        if "ssta" not in result and "sst" in result:
            result["ssta"] = result["sst"] - 28.5
        result["source_cmems"] = True
        return {"ok": True, "data": result, "error": None}

    return {"ok": False, "data": None,
            "error": "CMEMS gagal di semua endpoint. Cek email/password di marine.copernicus.eu"}


# =============================================================================
# 2. NASA MODIS — DINONAKTIFKAN
# =============================================================================
def fetch_nasa_modis(user="", password=""):
    return {
        "ok":    False,
        "data":  None,
        "error": "NASA MODIS dinonaktifkan — Chl-a diambil dari CMEMS bio dataset.",
    }


# =============================================================================
# 3. ERA5 / Open-Meteo
# =============================================================================
def fetch_era5(cds_uid="", cds_key=""):
    result_om = _fetch_openmeteo_wind()
    if result_om["ok"]:
        return result_om
    uid_str = str(cds_uid).strip()
    if uid_str and uid_str not in ("", "0", "ISI_UID_KAMU"):
        result_cds = _fetch_era5_cds(uid_str, str(cds_key).strip())
        if result_cds["ok"]:
            return result_cds
    return {"ok": False, "data": None,
            "error": "ERA5 & Open-Meteo keduanya gagal. Cek koneksi internet."}


def _fetch_openmeteo_wind():
    all_u, all_v, all_wave = [], [], []
    for lat, lon in SAMPLE_POINTS:
        try:
            resp = _get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "current":  "wind_speed_10m,wind_direction_10m",
                    "wind_speed_unit": "ms",
                    "timezone": "UTC", "forecast_days": 1,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                cur  = resp.json().get("current", {})
                wspd = cur.get("wind_speed_10m")
                wdir = cur.get("wind_direction_10m")
                if wspd is not None and wdir is not None:
                    rad = np.radians(float(wdir))
                    spd = float(wspd)
                    all_u.append(-spd * np.sin(rad))
                    all_v.append(-spd * np.cos(rad))
        except Exception:
            pass
        try:
            resp = _get(
                "https://marine-api.open-meteo.com/v1/marine",
                params={
                    "latitude": lat, "longitude": lon,
                    "current":  "wave_height,wind_wave_height",
                    "timezone": "UTC", "forecast_days": 1,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                cur = resp.json().get("current", {})
                wh  = cur.get("wave_height") or cur.get("wind_wave_height")
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


def _fetch_era5_cds(uid, key):
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
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind",
                         "significant_height_of_combined_wind_waves_and_swell"],
            "year":   target.strftime("%Y"),
            "month":  target.strftime("%m"),
            "day":    target.strftime("%d"),
            "time":   ["00:00", "06:00", "12:00", "18:00"],
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
            import xarray as xr
            ds   = xr.open_dataset(tmp_nc)
            u10  = float(ds["u10"].mean()) if "u10" in ds else -1.5
            v10  = float(ds["v10"].mean()) if "v10" in ds else -0.5
            wave = float(ds["swh"].mean())  if "swh"  in ds else 0.8
            ds.close()
        finally:
            if os.path.exists(tmp_nc):
                os.remove(tmp_nc)
        return {
            "ok": True,
            "data": {
                "angin_u": float(u10), "angin_v": float(v10),
                "gelombang": float(np.clip(wave, 0.2, 2.5)),
                "source_era5": True, "source_openmeteo": False,
            },
            "error": None,
        }
    except Exception as e:
        return {"ok": False, "data": None,
                "error": f"ERA5/CDS error: {str(e)[:150]}"}


# =============================================================================
# 4. BMKG
# =============================================================================
def fetch_bmkg():
    BMKG_URLS = [
        "https://data.bmkg.go.id/DataMKG/MEWS/maritim/ombakLaut.json",
        "https://data.bmkg.go.id/DataMKG/MEWS/maritim/maritim.json",
        "https://data.bmkg.go.id/DataMKG/MEWS/maritim/maritim.xml",
    ]
    for url in BMKG_URLS:
        try:
            resp = _get(url, timeout=10)
            if resp.status_code != 200:
                continue
            text = resp.text.strip()
            if text.startswith("{") or text.startswith("["):
                try:
                    vals = _bmkg_extract_json(resp.json())
                    if vals:
                        wave = float(np.clip(np.mean(vals), 0.2, 2.5))
                        return {"ok": True,
                                "data": {"gelombang": wave, "source_bmkg": True},
                                "error": None}
                except Exception:
                    pass
            vals = _bmkg_extract_text(text)
            if vals:
                wave = float(np.clip(np.mean(vals), 0.2, 2.5))
                return {"ok": True,
                        "data": {"gelombang": wave, "source_bmkg": True},
                        "error": None}
        except Exception as e:
            print(f"[BMKG] {url.split('/')[-1]} gagal: {str(e)[:50]}")
            continue

    # Fallback Open-Meteo Marine
    wave_vals = []
    for lat, lon in SAMPLE_POINTS:
        try:
            resp = _get(
                "https://marine-api.open-meteo.com/v1/marine",
                params={"latitude": lat, "longitude": lon,
                        "current": "wave_height", "timezone": "UTC"},
                timeout=10,
            )
            if resp.status_code == 200:
                wh = resp.json().get("current", {}).get("wave_height")
                if wh is not None:
                    wave_vals.append(float(wh))
        except Exception:
            pass

    if wave_vals:
        wave = float(np.clip(np.mean(wave_vals), 0.2, 2.5))
        return {"ok": True,
                "data": {"gelombang": wave,
                         "source_bmkg": False, "source_openmeteo": True},
                "error": "BMKG down, pakai Open-Meteo Marine"}

    return {"ok": False, "data": None,
            "error": "BMKG dan Open-Meteo Marine keduanya tidak merespons"}


def _bmkg_extract_json(data):
    vals = []
    if isinstance(data, list):
        for item in data:
            vals.extend(_bmkg_extract_json(item))
        return vals
    if not isinstance(data, dict):
        return vals
    for field in ("tinggiGelombang", "wave_height", "gelombang", "tinggi",
                  "waveHeight", "tinggiGelombangMax", "tinggiGelombangMin"):
        val = data.get(field)
        if val is not None:
            vals.extend(_parse_wave_str(str(val)))
    for v in data.values():
        if isinstance(v, (dict, list)):
            vals.extend(_bmkg_extract_json(v))
    return vals


def _bmkg_extract_text(text):
    vals = []
    patterns = [
        r"(\d+[\.,]\d*)\s*[-\u2013\u2014]\s*(\d+[\.,]\d*)\s*[Mm]eter",
        r"(\d+[\.,]\d*)\s*[-\u2013\u2014]\s*(\d+[\.,]\d*)\s*[Mm]\b",
        r"wave[_\s]?height[=:\s]+(\d+[\.,]\d*)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            groups = m.groups()
            try:
                if len(groups) == 2:
                    a = float(groups[0].replace(",", "."))
                    b = float(groups[1].replace(",", "."))
                    if 0.1 <= (a + b) / 2 <= 8.0:
                        vals.append((a + b) / 2)
                elif len(groups) == 1:
                    v = float(groups[0].replace(",", "."))
                    if 0.1 <= v <= 8.0:
                        vals.append(v)
            except Exception:
                pass
    return vals


def _parse_wave_str(text):
    text = text.replace(",", ".").strip()
    m = re.search(r"(\d+\.?\d*)\s*[-\u2013]\s*(\d+\.?\d*)", text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return [(a + b) / 2] if 0.1 <= (a + b) / 2 <= 8.0 else []
    m2 = re.search(r"(\d+\.?\d*)", text)
    if m2:
        v = float(m2.group(1))
        return [v] if 0.1 <= v <= 8.0 else []
    return []


# =============================================================================
# GABUNGAN
# =============================================================================
def build_realtime_dataframe(cmems_user, cmems_pass, cds_uid="", cds_key=""):
    now  = datetime.datetime.utcnow()
    klim = _klimatologi_bulan(now.month)

    merged = {
        **klim,
        "time":  pd.Timestamp(now),
        "year":  int(now.year),
        "month": int(now.month),
        "ph":    8.10,
        "do":    6.20,
    }

    status = {
        "CMEMS":           False,
        "NASA MODIS":      False,
        "ERA5/Open-Meteo": False,
        "BMKG":            False,
    }
    errors = {
        "NASA MODIS": "Dinonaktifkan — Chl-a diambil dari CMEMS bio dataset.",
    }
    any_ok = False

    # 1. CMEMS
    print("\n[LAUTAN] CMEMS...")
    r = fetch_cmems(cmems_user, cmems_pass)
    if r["ok"] and r["data"]:
        for k in ("uo", "vo", "sst", "ssta", "salinitas", "chla"):
            if k in r["data"]:
                merged[k] = float(r["data"][k])
        status["CMEMS"] = True
        any_ok = True
        print("[LAUTAN] CMEMS OK")
    else:
        errors["CMEMS"] = r.get("error", "unknown")
        print(f"[LAUTAN] CMEMS gagal: {errors['CMEMS'][:80]}")

    # 2. ERA5 / Open-Meteo
    print("[LAUTAN] ERA5/Open-Meteo...")
    r = fetch_era5(cds_uid, cds_key)
    if r["ok"] and r["data"]:
        for k in ("angin_u", "angin_v", "gelombang"):
            if k in r["data"]:
                merged[k] = float(r["data"][k])
        status["ERA5/Open-Meteo"] = True
        any_ok = True
        print("[LAUTAN] ERA5/Open-Meteo OK")
    else:
        errors["ERA5"] = r.get("error", "unknown")
        print(f"[LAUTAN] ERA5 gagal: {errors.get('ERA5','')[:80]}")

    # 3. BMKG
    print("[LAUTAN] BMKG...")
    r = fetch_bmkg()
    if r["ok"] and r["data"]:
        merged["gelombang"] = float(r["data"]["gelombang"])
        status["BMKG"] = True
        any_ok = True
        print("[LAUTAN] BMKG OK")
    else:
        errors["BMKG"] = r.get("error", "unknown")
        print(f"[LAUTAN] BMKG gagal: {errors.get('BMKG','')[:80]}")

    # Derive & clip
    merged["current_speed"] = float(np.sqrt(merged["uo"] ** 2 + merged["vo"] ** 2))
    merged["ssta"]      = float(merged.get("sst", 28.5)) - 28.5
    merged["chla"]      = float(np.clip(merged.get("chla",      0.20), 0.05, 0.8))
    merged["do"]        = float(np.clip(merged.get("do",        6.20), 4.5,  7.5))
    merged["ph"]        = float(np.clip(merged.get("ph",        8.10), 7.9,  8.4))
    merged["salinitas"] = float(np.clip(merged.get("salinitas", 34.2), 32.0, 36.5))
    merged["gelombang"] = float(np.clip(merged.get("gelombang", 0.85), 0.2,  2.5))

    df_out = pd.DataFrame([merged])

    def _norm(s, vmin, vmax):
        """Normalisasi linear 0..1 (makin tinggi makin baik)."""
        return (s - vmin) / (vmax - vmin) if (vmax - vmin) != 0 else s * 0

    def _suit(s, lo, opt_lo, opt_hi, hi):
        """
        Skor kesesuaian trapesium 0..1:
          0  bila s <= lo  atau s >= hi
          1  bila opt_lo <= s <= opt_hi
          naik/turun linear di antaranya.
        Identik dengan suitabilitas_optimal() di app.py.
        """
        s = np.asarray(s, dtype=float)
        naik  = np.clip((s - lo)   / max(opt_lo - lo,  1e-9), 0.0, 1.0)
        turun = np.clip((hi - s)   / max(hi - opt_hi,  1e-9), 0.0, 1.0)
        return np.minimum(naik, turun)

    # =================================================================
    # OCEAN HEALTH INDEX — sama persis dengan app.py
    # bobot: DO 0.30 · pH 0.25 · chl-a 0.20 (trapesium) ·
    #        salinitas 0.15 (trapesium) · SST 0.10 (trapesium)
    # =================================================================
    df_out["Ocean_Health_Index"] = (
        0.30 * _norm(df_out["do"],  4.5,  7.5) +
        0.25 * _norm(df_out["ph"],  7.9,  8.4) +
        0.20 * _suit(df_out["chla"],     0.05, 0.10, 0.40, 0.80) +
        0.15 * _suit(df_out["salinitas"],32.0, 33.5, 35.0, 36.5) +
        0.10 * _suit(df_out["sst"],      22.0, 26.0, 30.0, 32.0)
    ) * 100

    # =================================================================
    # FISHERIES INDEX — sama persis dengan app.py
    # bobot: chl-a 0.35 · SST 0.25 (trapesium) · DO 0.20 ·
    #        arus 0.10 · gelombang 0.10 (negatif)
    # =================================================================
    df_out["Fisheries_Index"] = (
        0.35 * _norm(df_out["chla"],          0.05, 0.8) +
        0.25 * _suit(df_out["sst"],           24.0, 28.0, 30.0, 33.0) +
        0.20 * _norm(df_out["do"],            4.5,  7.5) +
        0.10 * _norm(df_out["current_speed"], 0.0,  0.25) +
        0.10 * (1 - _norm(df_out["gelombang"], 0.2, 2.5))
    ) * 100

    return {
        "data":   df_out if any_ok else None,
        "status": status,
        "errors": errors,
    }
