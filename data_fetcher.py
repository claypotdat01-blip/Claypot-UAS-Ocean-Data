"""
data_fetcher.py — LAUTAN
Sumber data real-time:
  · CMEMS  : arus (uo/vo), SST, salinitas, klorofil-a (chla) ← chla pindah ke sini
  · ERA5   : angin (angin_u / angin_v)
  · BMKG   : gelombang
"""

import datetime
import numpy as np
import pandas as pd

from config import (
    CMEMS_USER, CMEMS_PASS,
    CDS_UID, CDS_KEY,
    BMKG_ADM4, BMKG_BASE_URL,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
)


# ─────────────────────────────────────────────────────────
# HELPER: rentang waktu 3 hari terakhir
# ─────────────────────────────────────────────────────────
def _date_range_recent(days=3):
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    return start.strftime("%Y-%m-%dT00:00:00"), end.strftime("%Y-%m-%dT23:59:59")


# ─────────────────────────────────────────────────────────
# 1. CMEMS — arus (uo/vo), SST, salinitas
# ─────────────────────────────────────────────────────────
def fetch_cmems(cmems_user: str, cmems_pass: str) -> dict:
    """
    Ambil uo, vo, SST, salinitas dari CMEMS Global Ocean Physics.
    Dataset: cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i  (arus)
             cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i (SST)
             cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i  (salinitas)
    Kembalikan dict dengan rata-rata scalar per variabel.
    """
    try:
        import copernicusmarine as cm

        start_dt, end_dt = _date_range_recent(3)

        # — arus —
        ds_cur = cm.open_dataset(
            dataset_id  = "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
            username    = cmems_user,
            password    = cmems_pass,
            minimum_latitude  = LAT_MIN,
            maximum_latitude  = LAT_MAX,
            minimum_longitude = LON_MIN,
            maximum_longitude = LON_MAX,
            start_datetime    = start_dt,
            end_datetime      = end_dt,
            variables         = ["uo", "vo"],
        )
        uo_val = float(ds_cur["uo"].mean())
        vo_val = float(ds_cur["vo"].mean())
        ds_cur.close()

        # — SST —
        ds_sst = cm.open_dataset(
            dataset_id  = "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i",
            username    = cmems_user,
            password    = cmems_pass,
            minimum_latitude  = LAT_MIN,
            maximum_latitude  = LAT_MAX,
            minimum_longitude = LON_MIN,
            maximum_longitude = LON_MAX,
            start_datetime    = start_dt,
            end_datetime      = end_dt,
            variables         = ["thetao"],
        )
        sst_val  = float(ds_sst["thetao"].mean())
        ssta_val = sst_val - 28.5          # anomali sederhana vs klimatologi
        ds_sst.close()

        # — salinitas —
        ds_sal = cm.open_dataset(
            dataset_id  = "cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i",
            username    = cmems_user,
            password    = cmems_pass,
            minimum_latitude  = LAT_MIN,
            maximum_latitude  = LAT_MAX,
            minimum_longitude = LON_MIN,
            maximum_longitude = LON_MAX,
            start_datetime    = start_dt,
            end_datetime      = end_dt,
            variables         = ["so"],
        )
        sal_val = float(ds_sal["so"].mean())
        ds_sal.close()

        return {
            "ok": True,
            "uo": uo_val, "vo": vo_val,
            "sst": sst_val, "ssta": ssta_val,
            "salinitas": sal_val,
        }

    except Exception as e:
        print(f"[CMEMS physics] ERROR: {e}")
        return {"ok": False, "uo": -0.05, "vo": -0.01,
                "sst": 28.5, "ssta": 0.0, "salinitas": 34.2}


# ─────────────────────────────────────────────────────────
# 2. CMEMS — Klorofil-a  ← BARU, ganti NASA MODIS
# ─────────────────────────────────────────────────────────
def fetch_cmems_chla(cmems_user: str, cmems_pass: str) -> dict:
    """
    Ambil klorofil-a dari CMEMS Ocean Colour (satellite L4 gap-free).
    Dataset: cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D
    Variabel: CHL  (mg/m³)

    Catatan: dataset 'my' (multi-year reprocessed) kadang lag ~2–4 minggu.
    Jika ingin NRT gunakan:
        cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D
    """
    try:
        import copernicusmarine as cm

        start_dt, end_dt = _date_range_recent(7)   # chla satelit bisa lag, ambil 7 hari

        ds = cm.open_dataset(
            dataset_id  = "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D",
            username    = cmems_user,
            password    = cmems_pass,
            minimum_latitude  = LAT_MIN,
            maximum_latitude  = LAT_MAX,
            minimum_longitude = LON_MIN,
            maximum_longitude = LON_MAX,
            start_datetime    = start_dt,
            end_datetime      = end_dt,
            variables         = ["CHL"],
        )

        chla_val = float(ds["CHL"].mean())
        chla_val = float(np.clip(chla_val, 0.05, 0.8))
        ds.close()

        return {"ok": True, "chla": chla_val}

    except Exception as e:
        print(f"[CMEMS chla] ERROR: {e}")
        return {"ok": False, "chla": 0.22}   # fallback klimatologis


# ─────────────────────────────────────────────────────────
# 3. ERA5 / CDS — angin (angin_u / angin_v)
# ─────────────────────────────────────────────────────────
def fetch_era5(cds_uid: str, cds_key: str) -> dict:
    """
    Ambil komponen angin permukaan 10 m dari ERA5 via CDS API.
    Variabel: u10 → angin_u,  v10 → angin_v
    """
    try:
        import cdsapi

        c = cdsapi.Client(
            url = "https://cds.climate.copernicus.eu/api/v2",
            key = f"{cds_uid}:{cds_key}",
            quiet = True,
        )

        today = datetime.date.today()
        # ERA5 biasanya tersedia dengan lag ~5 hari
        target = today - datetime.timedelta(days=5)

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tf:
            tmp_path = tf.name

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable"    : ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "year"  : str(target.year),
                "month" : target.strftime("%m"),
                "day"   : target.strftime("%d"),
                "time"  : ["00:00", "06:00", "12:00", "18:00"],
                "area"  : [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],   # N/W/S/E
                "format": "netcdf",
            },
            tmp_path,
        )

        import netCDF4 as nc
        ds = nc.Dataset(tmp_path)
        u_val = float(np.mean(ds.variables["u10"][:]))
        v_val = float(np.mean(ds.variables["v10"][:]))
        ds.close()
        os.unlink(tmp_path)

        return {"ok": True, "angin_u": u_val, "angin_v": v_val}

    except Exception as e:
        print(f"[ERA5] ERROR: {e}")
        return {"ok": False, "angin_u": -1.5, "angin_v": -0.5}


# ─────────────────────────────────────────────────────────
# 4. BMKG — gelombang
# ─────────────────────────────────────────────────────────
def fetch_bmkg() -> dict:
    """
    Ambil tinggi gelombang signifikan dari BMKG Open API.
    Endpoint: /api/v1/weather/significant-wave-height
    """
    try:
        import requests

        url = f"{BMKG_BASE_URL}/api/v1/weather/significant-wave-height"
        params = {"adm4": BMKG_ADM4}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Struktur respons BMKG: {"data": [{"wave_height": 1.2, ...}, ...]}
        records = data.get("data", [])
        if records:
            heights = [r.get("wave_height", 1.0) for r in records if r.get("wave_height")]
            wave_val = float(np.mean(heights)) if heights else 1.0
        else:
            wave_val = 1.0

        wave_val = float(np.clip(wave_val, 0.2, 2.5))
        return {"ok": True, "gelombang": wave_val}

    except Exception as e:
        print(f"[BMKG] ERROR: {e}")
        return {"ok": False, "gelombang": 1.0}


# ─────────────────────────────────────────────────────────
# 5. BUILD REALTIME DATAFRAME — fungsi utama dipanggil app
# ─────────────────────────────────────────────────────────
def build_realtime_dataframe(
    cmems_user: str, cmems_pass: str,
    nasa_user : str = "",   # tidak dipakai lagi, dipertahankan agar signature tidak patah
    nasa_pass : str = "",
    cds_uid   : str = "",
    cds_key   : str = "",
) -> dict:
    """
    Panggil semua sumber data, gabungkan jadi satu DataFrame baris tunggal,
    kembalikan {'data': df, 'status': {nama_api: bool}}.
    """

    # — panggil semua API —
    r_cmems = fetch_cmems(cmems_user, cmems_pass)
    r_chla  = fetch_cmems_chla(cmems_user, cmems_pass)   # ← CMEMS, bukan MODIS
    r_era5  = fetch_era5(cds_uid, cds_key)
    r_bmkg  = fetch_bmkg()

    # — status per API —
    status = {
        "CMEMS (Fisika)"  : r_cmems["ok"],
        "CMEMS (Klorofil)": r_chla["ok"],   # ← label baru
        "ERA5/ECMWF"      : r_era5["ok"],
        "BMKG"            : r_bmkg["ok"],
    }

    # — derivasi DO dan pH dari SST/salinitas (tidak ada API langsung) —
    sst_val = r_cmems["sst"]
    sal_val = r_cmems["salinitas"]
    do_val  = float(np.clip(6.2  - (sst_val - 28.5) * 0.1, 4.5, 7.5))
    ph_val  = float(np.clip(8.12 - (sal_val - 34.2) * 0.02, 7.9, 8.4))

    now = datetime.datetime.utcnow()

    row = {
        "time"     : now,
        "month"    : now.month,
        "year"     : now.year,
        "uo"       : r_cmems["uo"],
        "vo"       : r_cmems["vo"],
        "sst"      : sst_val,
        "ssta"     : r_cmems["ssta"],
        "salinitas": sal_val,
        "chla"     : r_chla["chla"],          # ← dari CMEMS sekarang
        "do"       : do_val,
        "ph"       : ph_val,
        "gelombang": r_bmkg["gelombang"],
        "angin_u"  : r_era5["angin_u"],
        "angin_v"  : r_era5["angin_v"],
    }

    df_rt = pd.DataFrame([row])
    df_rt["current_speed"] = np.sqrt(df_rt["uo"]**2 + df_rt["vo"]**2)

    return {"data": df_rt, "status": status}
