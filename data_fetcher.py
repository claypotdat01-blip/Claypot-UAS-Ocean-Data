"""
data_fetcher.py — Pengambil data real-time OCEANA
==============================================================================
Sumber data aktif:
  1. CMEMS  → Arus (uo/vo), SST, Salinitas (via ERDDAP NRT + fallback REST)
              Klorofil-a (bio dataset ocean colour)
  2. ERA5/Open-Meteo → Angin (u10/v10); Open-Meteo tanpa akun, ERA5 CDS opsional
  3. BMKG   → Gelombang; fallback Open-Meteo Marine (tanpa akun)

Parameter TANPA sumber API langsung (derivasi/klimatologi):
  - pH        : estimasi klimatologis musiman (tidak ada API publik gratis real-time)
  - DO        : derivasi dari SST menggunakan hubungan empiris (oksigen menurun saat SST naik)
  - SSTA      : dihitung dari SST - baseline klimatologi (28.5°C)
  - current_speed : dihitung dari sqrt(uo² + vo²)

Strategi fallback berlapis:
  - Jika ERDDAP gagal → coba Copernicus Marine REST API
  - Jika satu endpoint gagal → coba endpoint berikutnya
  - Jika semua API gagal → gunakan estimasi klimatologis musiman
  - App TIDAK pernah crash karena API gagal
==============================================================================
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

# ── Rentang fisik realistis Laut Arafura ─────────────────────────────────────
# Berdasarkan literatur: arus permukaan Arafura umumnya 0.03–0.20 m/s,
# maksimum saat musim muson bisa mencapai ~0.35 m/s.
UO_MIN, UO_MAX   = -0.40, 0.40   # m/s — komponen zonal arus
VO_MIN, VO_MAX   = -0.30, 0.30   # m/s — komponen meridional arus
CS_MAX           =  0.45          # m/s — current_speed maksimum realistis


def _get(url, auth=None, params=None, timeout=15, headers=None):
    import requests
    h = {"User-Agent": "OCEANA-OceanPlatform/1.0", "Accept": "application/json"}
    if headers:
        h.update(headers)
    return requests.get(url, auth=auth, params=params, timeout=timeout, headers=h)


def _klimatologi_bulan(month):
    """
    Estimasi klimatologis bulanan untuk Laut Arafura.
    Dipakai sebagai nilai awal (prior) dan fallback jika semua API gagal.
    Semua parameter kecuali pH dan DO punya sumber API — nilai ini hanya
    digunakan saat koneksi gagal.
    pH dan DO selalu klimatologis karena tidak ada API publik gratis real-time
    untuk kedua parameter ini di wilayah Arafura.
    """
    t = 2 * np.pi * (month - 1) / 12
    return {
        # === DARI CMEMS (jika CMEMS OK, nilai ini ditimpa) ===
        "uo":        -0.06 + 0.04 * np.sin(t + 1.0),
        "vo":        -0.01 + 0.02 * np.cos(t),
        "sst":        28.5 - 1.2  * np.sin(t + 0.5),
        "salinitas":  34.2 + 0.4  * np.cos(t + 0.3),
        "chla":        0.20 + 0.10 * np.sin(t + 1.5),
        # === DARI ERA5/Open-Meteo (jika OK, nilai ini ditimpa) ===
        "angin_u":   -1.5  - 1.0  * np.sin(t + 1.0),
        "angin_v":   -0.5  - 0.3  * np.cos(t),
        # === DARI BMKG/Open-Meteo Marine (jika OK, nilai ini ditimpa) ===
        "gelombang":  0.85 + 0.40 * np.sin(t + 1.0),
        # === SELALU KLIMATOLOGIS — tidak ada API real-time gratis ===
        # pH: tidak ada API publik untuk pH laut real-time di Arafura.
        #     Nilai diestimasi dari tren penurunan pH global ~8.1 ± 0.02.
        "ph":         8.10 - 0.02 * np.sin(t),
        # DO: diderivasikan dari SST via hubungan empiris Garcia & Gordon (1992).
        #     DO menurun ~0.15 mg/L per +1°C SST di perairan tropis.
        "do":         6.20 - 0.15 * np.sin(t + 0.5),
        # SSTA dihitung ulang di bawah setelah SST final diketahui.
        "ssta":      -0.1  + 0.3  * np.sin(t),
    }


# =============================================================================
# 1. CMEMS — Arus, SST, Salinitas, Klorofil-a
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

    # ── A. Fisik: arus (uo/vo), SST, salinitas via ERDDAP ────────────────
    PHY_DATASETS = [
        "cmems_mod_glo_phy_anfc_0.083deg_PT1H-i",
        "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
        "cmems_mod_glo_phy_anfc_merged-sl_PT1H-i",
    ]
    for dataset_id in PHY_DATASETS:
        url   = f"https://nrt.cmems-du.eu/erddap/griddap/{dataset_id}.json"
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
                                print(f"{dst} = {float(v.mean())}")
                    if "uo" in result:
                        print(f"[CMEMS] ✓ Fisik ERDDAP: {dataset_id}")
                        break
            elif resp.status_code == 401:
                return {"ok": False, "data": None,
                        "error": "CMEMS 401 Unauthorized — email/password salah atau belum daftar di marine.copernicus.eu"}
        except Exception as e:
            print(f"[CMEMS] ERDDAP {dataset_id}: {str(e)[:60]}")
            continue

    # ── A2. Fallback fisik: Copernicus Marine REST subset API ─────────────
    # [FIX] OPeNDAP fallback diperbaiki: filter angka lebih ketat agar tidak
    # menyedot nilai dari header/koordinat yang menyebabkan uo/vo ekstrem.
    if "uo" not in result:
        print("[CMEMS] ERDDAP fisik gagal, coba Copernicus Marine subset API...")
        try:
            subset_url = "https://nrt.cmems-du.eu/thredds/dodsC/cmems_mod_glo_phy_anfc_0.083deg_PT1H-i"
            opendap_url = (
                f"{subset_url}.ascii"
                f"?uo[0][0][{int((lat_c+2-LAT_MIN)/0.083)}:{int((lat_c-2-LAT_MIN)/0.083)}]"
                f"[{int((lon_c-2-LON_MIN)/0.083)}:{int((lon_c+2-LON_MIN)/0.083)}]"
            )
            resp2 = _get(opendap_url, auth=(user, password), timeout=15)
            if resp2.status_code == 200:
                # [FIX] Filter jauh lebih ketat: hanya nilai dalam rentang fisik
                # arus Arafura yang realistis (~±0.4 m/s), bukan ±5.0 yang
                # sebelumnya menyebabkan angka koordinat/timestamp ikut terambil.
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", resp2.text)
                floats = []
                for x in nums:
                    try:
                        val = float(x)
                        # Hanya ambil nilai yang masuk akal sebagai komponen arus (m/s)
                        if UO_MIN <= val <= UO_MAX:
                            floats.append(val)
                    except ValueError:
                        pass
                if len(floats) >= 4:
                    half = len(floats) // 2
                    uo_val = float(np.mean(floats[:half]))
                    vo_val = float(np.mean(floats[half:]))
                    # Clip sekali lagi untuk keamanan
                    result["uo"] = float(np.clip(uo_val, UO_MIN, UO_MAX))
                    result["vo"] = float(np.clip(vo_val, VO_MIN, VO_MAX))
                    print(f"[CMEMS] ✓ Fisik OPeNDAP fallback: uo={result['uo']:.4f}, vo={result['vo']:.4f}")
                else:
                    print("[CMEMS] OPeNDAP: tidak cukup nilai arus yang valid, skip.")
        except Exception as e:
            print(f"[CMEMS] OPeNDAP fallback: {str(e)[:60]}")

    # ── B. Klorofil-a dari CMEMS Ocean Colour bio dataset ────────────────
    BIO_DATASETS = [
        ("cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D", "CHL"),
        ("cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1D",          "CHL"),
    ]
    date_candidates = [
        (now - datetime.timedelta(days=d)).strftime("%Y-%m-%dT00:00:00")
        for d in range(2, 6)
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
                        print(f"[CMEMS] ✓ Chl-a Ocean Colour: {date_str[:10]}")
                        break
            except Exception as e:
                print(f"[CMEMS] Bio {dataset_id}: {str(e)[:60]}")
                break
        if "chla" in result:
            break

    # ── Kembalikan hasil ──────────────────────────────────────────────────
    if result:
        # [FIX] Clip uo/vo dari ERDDAP juga — data ERDDAP kadang mengandung
        # outlier saat grid query mengenai area transisi atau data yang belum QC.
        if "uo" in result:
            result["uo"] = float(np.clip(result["uo"], UO_MIN, UO_MAX))
        if "vo" in result:
            result["vo"] = float(np.clip(result["vo"], VO_MIN, VO_MAX))

        if "ssta" not in result and "sst" in result:
            result["ssta"] = result["sst"] - 28.5
        result["source_cmems"] = True
        return {"ok": True, "data": result, "error": None}

    return {"ok": False, "data": None,
            "error": "CMEMS gagal (ERDDAP + fallback). Cek email/password di marine.copernicus.eu"}


# =============================================================================
# 2. NASA MODIS — DINONAKTIFKAN
# =============================================================================
def fetch_nasa_modis(user="", password=""):
    return {
        "ok":    False,
        "data":  None,
        "error": "NASA MODIS dinonaktifkan — Chl-a diambil dari CMEMS Ocean Colour.",
    }


# =============================================================================
# 3. ERA5 / Open-Meteo — Angin 10m
# =============================================================================
def fetch_era5(cds_uid="", cds_key=""):
    """
    Angin 10m (u10/v10) dari ERA5 atau Open-Meteo.
    Open-Meteo gratis (tanpa akun), ERA5 via CDS opsional.
    """
    print("[ERA5] Mencoba Open-Meteo...")
    result_om = _fetch_openmeteo_wind()
    if result_om["ok"]:
        return result_om

    uid_str = str(cds_uid).strip()
    if uid_str and uid_str not in ("", "0", "ISI_UID_KAMU"):
        print(f"[ERA5] Mencoba CDS API UID: {uid_str[:6]}...")
        result_cds = _fetch_era5_cds(uid_str, str(cds_key).strip())
        if result_cds["ok"]:
            return result_cds

    return {"ok": False, "data": None,
            "error": "ERA5 & Open-Meteo keduanya gagal. Cek koneksi internet."}


def _fetch_openmeteo_wind():
    all_u, all_v, all_wave, all_wave_dir = [], [], [], []
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
                    "current":  "wave_height,wind_wave_height,wave_direction",
                    "timezone": "UTC", "forecast_days": 1,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                cur = resp.json().get("current", {})
                wh  = cur.get("wave_height") or cur.get("wind_wave_height")
                wd  = cur.get("wave_direction")
                if wh is not None:
                    all_wave.append(float(wh))
                if wd is not None:
                    all_wave_dir.append(float(wd))
        except Exception:
            pass

    if all_u or all_v or all_wave:
        return {
            "ok": True,
            "data": {
                "angin_u":   float(np.mean(all_u))   if all_u   else -1.5,
                "angin_v":   float(np.mean(all_v))   if all_v   else -0.5,
                "gelombang": float(np.clip(np.mean(all_wave), 0.2, 2.5))
                "wave_direction": float(np.mean(all_wave_dir)) if all_wave_dir else None, if all_wave else 0.8,
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
    tmp_rc = os.path.join(tempfile.gettempdir(), ".cdsapirc_oceana")
    tmp_nc = os.path.join(tempfile.gettempdir(), "era5_oceana.nc")
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
            "year": target.strftime("%Y"), "month": target.strftime("%m"),
            "day":  target.strftime("%d"),
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],
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
# 4. BMKG — Tinggi Gelombang
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
                        print(f"[BMKG] ✓ {url.split('/')[-1]}: {wave:.2f} m")
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
        print(f"[BMKG] ✓ Open-Meteo Marine fallback: {wave:.2f} m")
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
# GABUNGAN — build_realtime_dataframe()
# =============================================================================
def build_realtime_dataframe(cmems_user, cmems_pass, cds_uid="", cds_key=""):
    """
    Panggil semua API aktif, gabungkan ke satu DataFrame kompatibel dengan app.py.

    Urutan pengisian:
      1. Semua kolom diisi nilai klimatologis musiman (prior/fallback)
      2. CMEMS  → timpa: uo, vo, sst, ssta, salinitas, chla
      3. ERA5/Open-Meteo → timpa: angin_u, angin_v, gelombang
      4. BMKG  → timpa: gelombang (override ERA5 jika berhasil)
      5. Derivasi: current_speed = sqrt(uo²+vo²)
                   ssta = sst - 28.5  (setelah SST final diketahui)
                   do   = derivasi empiris dari SST (tidak ada API gratis)
                   ph   = klimatologis (tidak ada API gratis real-time)

    Returns:
        {"data": pd.DataFrame | None, "status": dict, "errors": dict}
    """
    now  = datetime.datetime.utcnow()
    klim = _klimatologi_bulan(now.month)

    # Isi semua dengan klimatologi sebagai prior
    merged = {
        **klim,
        "time":  pd.Timestamp(now),
        "year":  int(now.year),
        "month": int(now.month),
    }

    status = {
        "CMEMS":           False,
        "ERA5/Open-Meteo": False,
        "BMKG":            False,
    }
    errors  = {}
    any_ok  = False

    # ── 1. CMEMS ──────────────────────────────────────────────────────────
    print("\n[OCEANA] CMEMS...")
    r = fetch_cmems(cmems_user, cmems_pass)
    if r["ok"] and r["data"]:
        for k in ("uo", "vo", "sst", "ssta", "salinitas", "chla"):
            if k in r["data"]:
                merged[k] = float(r["data"][k])
        status["CMEMS"] = True
        any_ok = True
        print("[OCEANA] ✓ CMEMS OK")
    else:
        errors["CMEMS"] = r.get("error", "unknown")
        print(f"[OCEANA] ✗ CMEMS: {errors['CMEMS'][:80]}")

    # ── 2. ERA5 / Open-Meteo ──────────────────────────────────────────────
    print("[OCEANA] ERA5/Open-Meteo...")
    r = fetch_era5(cds_uid, cds_key)
    if r["ok"] and r["data"]:
        for k in ("angin_u", "angin_v", "gelombang"):
            if k in r["data"]:
                merged[k] = float(r["data"][k])
        status["ERA5/Open-Meteo"] = True
        any_ok = True
        src = "ERA5" if r["data"].get("source_era5") else "Open-Meteo"
        print(f"[OCEANA] ✓ Angin dari {src}")
    else:
        errors["ERA5"] = r.get("error", "unknown")
        print(f"[OCEANA] ✗ ERA5: {errors.get('ERA5','')[:80]}")

    # ── 3. BMKG ───────────────────────────────────────────────────────────
    print("[OCEANA] BMKG...")
    r = fetch_bmkg()
    if r["ok"] and r["data"]:
        merged["gelombang"] = float(r["data"]["gelombang"])
        status["BMKG"] = True
        any_ok = True
        src = "BMKG" if r["data"].get("source_bmkg") else "Open-Meteo Marine"
        print(f"[OCEANA] ✓ Gelombang dari {src}")
    else:
        errors["BMKG"] = r.get("error", "unknown")
        print(f"[OCEANA] ✗ BMKG: {errors.get('BMKG','')[:80]}")

    # ── Derivasi & clip ───────────────────────────────────────────────────

    # [FIX] Clip uo/vo ke rentang fisik realistis Laut Arafura SEBELUM
    # menghitung current_speed. Ini mencegah nilai ekstrem dari OPeNDAP
    # fallback atau outlier ERDDAP menyebabkan current_speed tidak wajar.
    merged["uo"] = float(np.clip(merged.get("uo", -0.06), UO_MIN, UO_MAX))
    merged["vo"] = float(np.clip(merged.get("vo", -0.01), VO_MIN, VO_MAX))

    # current_speed: dihitung dari komponen arus yang sudah di-clip
    merged["current_speed"] = float(np.clip(
        np.sqrt(merged["uo"] ** 2 + merged["vo"] ** 2),
        0.0,
        CS_MAX,  # [FIX] clip eksplisit ke maksimum realistis Arafura (~0.45 m/s)
    ))

    # SSTA: SST dikurangi baseline klimatologi ~28.5°C (World Ocean Atlas Arafura)
    merged["ssta"] = float(merged.get("sst", 28.5)) - 28.5

    # DO: derivasi empiris. Tidak ada API publik real-time gratis untuk DO laut.
    sst_val = float(merged.get("sst", 28.5))
    merged["do"] = float(np.clip(6.5 - 0.15 * (sst_val - 28.0), 4.5, 7.5))

    # pH: tidak ada API publik real-time gratis.
    merged["ph"] = float(np.clip(8.12 - 0.005 * (sst_val - 28.0), 7.9, 8.4))

    # Clip semua ke rentang fisik yang valid
    merged["chla"]      = float(np.clip(merged.get("chla",      0.20), 0.05, 0.8))
    merged["salinitas"] = float(np.clip(merged.get("salinitas", 34.2), 32.0, 36.5))
    merged["gelombang"] = float(np.clip(merged.get("gelombang", 0.85), 0.2,  2.5))

    print(f"[OCEANA] uo={merged['uo']:.4f} m/s, vo={merged['vo']:.4f} m/s, "
          f"current_speed={merged['current_speed']:.4f} m/s")

    df_out = pd.DataFrame([merged])

    # ── Hitung indeks komposit (sama dengan app.py) ───────────────────────
    def _norm(s, vmin, vmax):
        s = np.asarray(s, dtype=float)
        if (vmax - vmin) == 0:
            return np.zeros_like(s)
        return np.clip((s - vmin) / (vmax - vmin), 0.0, 1.0)

    def _suit(s, lo, opt_lo, opt_hi, hi):
        s = np.asarray(s, dtype=float)
        naik  = np.clip((s - lo)  / max(opt_lo - lo,  1e-9), 0.0, 1.0)
        turun = np.clip((hi - s)  / max(hi - opt_hi,  1e-9), 0.0, 1.0)
        return np.minimum(naik, turun)

    # ── Ocean Health Index ────────────────────────────────────────────────
    df_out["Ocean_Health_Index"] = np.clip((
        0.30 * _norm(df_out["do"], 4.5, 7.5) +
        0.25 * _norm(df_out["ph"], 7.9, 8.4) +
        0.20 * _suit(df_out["chla"], 0.05, 0.10, 0.40, 0.80) +
        0.15 * _suit(df_out["salinitas"], 32.0, 33.5, 35.0, 36.5) +
        0.10 * _suit(df_out["sst"], 22.0, 26.0, 30.0, 32.0)
    ) * 100, 0, 100)

    # ── Fisheries Index ──────────────────────────────────────────────────
    df_out["Fisheries_Index"] = np.clip((
        0.35 * _suit(df_out["chla"], 0.05, 0.10, 0.50, 0.80) +
        0.25 * _suit(df_out["sst"], 24.0, 28.0, 30.0, 33.0) +
        0.20 * _norm(df_out["do"], 4.5, 7.5) +
        0.10 * _norm(df_out["current_speed"], 0.0, 0.25) +
        0.10 * (1 - _norm(df_out["gelombang"], 0.2, 2.5))
    ) * 100, 0, 100)

    n_ok = sum(status.values())
    print(f"\n[OCEANA] Selesai: {n_ok}/3 API berhasil")

    return {
        "data": df_out if any_ok else None,
        "status": status,
        "errors": errors,
    }
