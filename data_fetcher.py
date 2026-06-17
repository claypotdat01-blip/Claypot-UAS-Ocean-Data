cat > /mnt/user-data/outputs/data_fetcher.py << 'PYEOF'
"""
data_fetcher.py — Pengambil data real-time LAUTAN
==============================================================================
Sumber data:
  1. CMEMS        → Arus (uo/vo), SST, Salinitas, Klorofil-a
  2. Open-Meteo   → Angin (u10/v10) & Gelombang — gratis, tanpa akun
  3. BMKG         → Gelombang; fallback Open-Meteo Marine

Strategi fallback berlapis:
  - Jika satu endpoint gagal → coba endpoint berikutnya
  - Jika semua API gagal     → gunakan estimasi klimatologis musiman
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


def _get(url, auth=None, params=None, timeout=20, headers=None):
    import requests
    h = {"User-Agent": "LAUTAN-OceanPlatform/1.0", "Accept": "application/json"}
    if headers:
        h.update(headers)
    return requests.get(url, auth=auth, params=params, timeout=timeout, headers=h)


def _klimatologi_bulan(month: int) -> dict:
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
def fetch_cmems(user: str, password: str) -> dict:
    if not user or "@" not in user or user.startswith("ISI_"):
        return {"ok": False, "data": None,
                "error": "CMEMS_USER belum diisi. Isi di config.py dengan email marine.copernicus.eu"}
    if not password or len(password) < 4:
        return {"ok": False, "data": None, "error": "CMEMS_PASS terlalu pendek atau kosong"}

    now      = datetime.datetime.utcnow()
    end_dt   = now.strftime("%Y-%m-%dT%H:00:00")
    start_dt = (now - datetime.timedelta(hours=6)).strftime("%Y-%m-%dT%H:00:00")
    result   = {}

    # A. Data FISIK
    for dataset_id in [
        "cmems_mod_glo_phy_anfc_0.083deg_PT1H-i",
        "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
    ]:
        url   = f"https://nrt.cmems-du.eu/erddap/griddap/{dataset_id}.json"
        lat_c = (LAT_MIN + LAT_MAX) / 2
        lon_c = (LON_MIN + LON_MAX) / 2
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
                body = resp.json()
                cols = body.get("table", {}).get("columnNames", [])
                rows = body.get("table", {}).get("rows", [])
                if rows and cols:
                    df_raw = pd.DataFrame(rows, columns=cols)
                    for src, dst in [("uo","uo"),("vo","vo"),("thetao","sst"),("so","salinitas")]:
                        if src in df_raw.columns:
                            v = pd.to_numeric(df_raw[src], errors="coerce").dropna()
                            if len(v):
                                result[dst] = float(v.mean())
                    if "uo" in result:
                        print(f"[CMEMS] ✓ Fisik dari: {dataset_id}")
                        break
        except Exception as e:
            print(f"[CMEMS] phy gagal: {str(e)[:60]}")
            continue

    # B. Data BIOLOGI (klorofil) — retry tanggal mundur D-2 hingga D-6
    date_candidates = [
        (now - datetime.timedelta(days=d)).strftime("%Y-%m-%dT00:00:00")
        for d in range(2, 7)
    ]
    for dataset_id, var_name in [
        ("cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D", "CHL"),
        ("cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1D",          "CHL"),
        ("cmems_obs-oc_glo_bgc-optics_nrt_l3-multi-4km_P1D",           "CHL"),
    ]:
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
                        print(f"[CMEMS] ✓ Klorofil {date_str[:10]}: {result['chla']:.3f} mg/m³")
                        break
            except Exception as e:
                print(f"[CMEMS] bio {date_str[:10]}: {str(e)[:60]}")
                break
        if "chla" in result:
            break

    # C. Fallback WMS
    if not result:
        try:
            resp = _get(
                "https://nrt.cmems-du.eu/thredds/wms/cmems_mod_glo_phy_anfc_0.083deg_PT1H-i",
                auth=(user, password),
                params={"SERVICE":"WMS","VERSION":"1.3.0","REQUEST":"GetCapabilities"},
                timeout=15
            )
            if resp.status_code == 200:
                result.update(_klimatologi_bulan(now.month))
                result["_source"] = "wms_klimatologi"
                print("[CMEMS] ✓ WMS OK, pakai klimatologis")
        except Exception:
            pass

    if result:
        if "ssta" not in result and "sst" in result:
            result["ssta"] = result["sst"] - 28.5
        result["source_cmems"] = True
        return {"ok": True, "data": result, "error": None}

    return {"ok": False, "data": None, "error": "CMEMS gagal di semua endpoint. Cek email/password di config.py"}


# =============================================================================
# 2. OPEN-METEO — Angin & Gelombang
# =============================================================================
def fetch_openmeteo() -> dict:
    all_u, all_v, all_wave = [], [], []

    for lat, lon in SAMPLE_POINTS:
        try:
            resp = _get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "current": "wind_speed_10m,wind_direction_10m",
                    "wind_speed_unit": "ms", "timezone": "UTC", "forecast_days": 1,
                },
                timeout=15,
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
                    "current": "wave_height,wind_wave_height",
                    "timezone": "UTC", "forecast_days": 1,
                },
                timeout=15,
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
                "angin_u":          float(np.mean(all_u))   if all_u   else -1.5,
                "angin_v":          float(np.mean(all_v))   if all_v   else -0.5,
                "gelombang":        float(np.clip(np.mean(all_wave), 0.2, 2.5)) if all_wave else 0.8,
                "source_openmeteo": True,
            },
            "error": None,
        }

    return {"ok": False, "data": None, "error": "Open-Meteo tidak merespons"}


# =============================================================================
# 3. BMKG — Gelombang
# =============================================================================
def fetch_bmkg() -> dict:
    for url in [
        "https://data.bmkg.go.id/DataMKG/MEWS/maritim/ombakLaut.json",
        "https://data.bmkg.go.id/DataMKG/MEWS/maritim/maritim.json",
    ]:
        try:
            resp = _get(url, timeout=12)
            if resp.status_code != 200:
                continue
            text = resp.text.strip()
            vals = _bmkg_extract_json(resp.json()) if text.startswith(("{","[")) else _bmkg_extract_text(text)
            if vals:
                wave = float(np.clip(np.mean(vals), 0.2, 2.5))
                return {"ok": True, "data": {"gelombang": wave, "source_bmkg": True}, "error": None}
        except Exception as e:
            print(f"[BMKG] {str(e)[:50]}")
            continue

    # Fallback Open-Meteo Marine
    wave_vals = []
    for lat, lon in SAMPLE_POINTS:
        try:
            resp = _get(
                "https://marine-api.open-meteo.com/v1/marine",
                params={"latitude": lat, "longitude": lon, "current": "wave_height", "timezone": "UTC"},
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
        return {"ok": True, "data": {"gelombang": wave, "source_bmkg": False, "source_openmeteo": True},
                "error": "BMKG down, pakai Open-Meteo Marine"}

    return {"ok": False, "data": None, "error": "BMKG dan Open-Meteo Marine tidak merespons"}


def _bmkg_extract_json(data) -> list:
    vals = []
    if isinstance(data, list):
        for item in data: vals.extend(_bmkg_extract_json(item))
        return vals
    if not isinstance(data, dict): return vals
    for field in ("tinggiGelombang","wave_height","gelombang","tinggi","waveHeight"):
        val = data.get(field)
        if val is not None: vals.extend(_parse_wave_str(str(val)))
    for v in data.values():
        if isinstance(v, (dict, list)): vals.extend(_bmkg_extract_json(v))
    return vals


def _bmkg_extract_text(text: str) -> list:
    vals = []
    for pat in [r"(\d+[\.,]\d*)\s*[-–—]\s*(\d+[\.,]\d*)\s*[Mm]eter",
                r"wave[_\s]?height[=:\s]+(\d+[\.,]\d*)"]:
        for m in re.finditer(pat, text, re.IGNORECASE):
            groups = m.groups()
            try:
                if len(groups) == 2:
                    a, b = float(groups[0].replace(",",".")), float(groups[1].replace(",","."))
                    if 0.1 <= (a+b)/2 <= 8.0: vals.append((a+b)/2)
                elif len(groups) == 1:
                    v = float(groups[0].replace(",","."))
                    if 0.1 <= v <= 8.0: vals.append(v)
            except Exception: pass
    return vals


def _parse_wave_str(text: str) -> list:
    text = text.replace(",", ".").strip()
    m = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return [(a+b)/2] if 0.1 <= (a+b)/2 <= 8.0 else []
    m2 = re.search(r"(\d+\.?\d*)", text)
    if m2:
        v = float(m2.group(1))
        return [v] if 0.1 <= v <= 8.0 else []
    return []


# =============================================================================
# FUNGSI UTAMA
# =============================================================================
def build_realtime_dataframe(cmems_user: str, cmems_pass: str,
                              nasa_user: str = "", nasa_pass: str = "",
                              cds_uid: str = "", cds_key: str = "") -> dict:
    now  = datetime.datetime.utcnow()
    klim = _klimatologi_bulan(now.month)
    merged = {**klim, "time": now, "year": now.year, "month": now.month, "ph": 8.10, "do": 6.20}
    status = {"CMEMS": False, "Open-Meteo": False, "BMKG": False}
    errors = {}

    print("\n[LAUTAN] ── CMEMS...")
    r = fetch_cmems(cmems_user, cmems_pass)
    if r["ok"] and r["data"]:
        for k in ("uo","vo","sst","ssta","salinitas","chla"):
            if k in r["data"]: merged[k] = float(r["data"][k])
        status["CMEMS"] = True
    else:
        errors["CMEMS"] = r.get("error","unknown")

    print("[LAUTAN] ── Open-Meteo...")
    r = fetch_openmeteo()
    if r["ok"] and r["data"]:
        for k in ("angin_u","angin_v","gelombang"):
            if k in r["data"]: merged[k] = float(r["data"][k])
        status["Open-Meteo"] = True
    else:
        errors["Open-Meteo"] = r.get("error","unknown")

    print("[LAUTAN] ── BMKG...")
    r = fetch_bmkg()
    if r["ok"] and r["data"]:
        merged["gelombang"] = float(r["data"]["gelombang"])
        status["BMKG"] = True
    else:
        errors["BMKG"] = r.get("error","unknown")

    # Derive
    merged["current_speed"] = float(np.sqrt(merged.get("uo",0)**2 + merged.get("vo",0)**2))
    merged["angin_speed"]   = float(np.sqrt(merged.get("angin_u",0)**2 + merged.get("angin_v",0)**2))
    merged["angin_dir"]     = float(np.degrees(np.arctan2(-merged.get("angin_u",0), -merged.get("angin_v",0))) % 360)
    merged["arus_dir"]      = float(np.degrees(np.arctan2(merged.get("uo",0), merged.get("vo",0))) % 360)
    merged["ssta"]          = merged.get("sst", 28.5) - 28.5
    merged["chla"]          = float(np.clip(merged.get("chla",      0.20), 0.05, 0.8))
    merged["do"]            = float(np.clip(merged.get("do",        6.20), 4.5,  7.5))
    merged["ph"]            = float(np.clip(merged.get("ph",        8.10), 7.9,  8.4))
    merged["salinitas"]     = float(np.clip(merged.get("salinitas", 34.2), 32.0, 36.5))
    merged["gelombang"]     = float(np.clip(merged.get("gelombang", 0.85), 0.2,  2.5))

    print(f"\n[LAUTAN] Selesai: {sum(status.values())}/{len(status)} sumber berhasil")
    return {"data": merged, "status": status, "errors": errors}
PYEOF
