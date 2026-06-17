"""
data_fetcher.py — LAUTAN Platform Intelijen Oseanografi Papua
============================================================
Sumber data real-time:
  · CMEMS Fisika  : arus (uo/vo), SST, salinitas
  · CMEMS Klorofil: klorofil-a / chla  ← menggantikan NASA MODIS
  · ERA5 / CDS    : angin permukaan (angin_u / angin_v)
  · BMKG          : tinggi gelombang signifikan

NASA MODIS dihapus dari pipeline real-time karena API/akses sulit
dibuka (sering timeout / butuh autentikasi Earthdata terpisah).
Klorofil-a sekarang 100% diambil dari CMEMS Ocean Colour, yang
memakai kredensial yang sama dengan CMEMS Fisika (CMEMS_USER/PASS).
============================================================
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


# ─────────────────────────────────────────────────────────────
# HELPER: rentang waktu N hari terakhir dalam format ISO string
# ─────────────────────────────────────────────────────────────
def _date_range_recent(days: int = 3):
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    return (
        start.strftime("%Y-%m-%dT00:00:00"),
        end.strftime("%Y-%m-%dT23:59:59"),
    )


# ─────────────────────────────────────────────────────────────
# 1. CMEMS FISIKA — arus (uo/vo), SST, salinitas
# ─────────────────────────────────────────────────────────────
def fetch_cmems(cmems_user: str, cmems_pass: str) -> dict:
    """
    Ambil uo, vo, SST, dan salinitas dari CMEMS Global Ocean Physics
    Analysis and Forecast.

    Dataset yang digunakan:
      - Arus     : cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i   (var: uo, vo)
      - SST      : cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i (var: thetao)
      - Salinitas: cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i     (var: so)

    Mengembalikan dict scalar rata-rata area Laut Arafura.
    """
    try:
        import copernicusmarine as cm

        start_dt, end_dt = _date_range_recent(days=3)

        # ── Arus (uo, vo) ──────────────────────────────────
        ds_cur = cm.open_dataset(
            dataset_id        = "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
            username          = cmems_user,
            password          = cmems_pass,
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

        # ── SST (thetao) ───────────────────────────────────
        ds_sst = cm.open_dataset(
            dataset_id        = "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i",
            username          = cmems_user,
            password          = cmems_pass,
            minimum_latitude  = LAT_MIN,
            maximum_latitude  = LAT_MAX,
            minimum_longitude = LON_MIN,
            maximum_longitude = LON_MAX,
            start_datetime    = start_dt,
            end_datetime      = end_dt,
            variables         = ["thetao"],
        )
        sst_val  = float(ds_sst["thetao"].mean())
        ssta_val = round(sst_val - 28.5, 4)   # anomali sederhana vs klimatologi
        ds_sst.close()

        # ── Salinitas (so) ─────────────────────────────────
        ds_sal = cm.open_dataset(
            dataset_id        = "cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i",
            username          = cmems_user,
            password          = cmems_pass,
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

        print(f"[CMEMS Fisika] OK → uo={uo_val:.3f}, vo={vo_val:.3f}, "
              f"SST={sst_val:.2f}, SAL={sal_val:.2f}")

        return {
            "ok"       : True,
            "uo"       : uo_val,
            "vo"       : vo_val,
            "sst"      : sst_val,
            "ssta"     : ssta_val,
            "salinitas": sal_val,
        }

    except Exception as e:
        print(f"[CMEMS Fisika] ERROR: {e}")
        # Fallback nilai klimatologis rata-rata Laut Arafura
        return {
            "ok"       : False,
            "uo"       : -0.05,
            "vo"       : -0.01,
            "sst"      : 28.5,
            "ssta"     : 0.0,
            "salinitas": 34.2,
        }


# ─────────────────────────────────────────────────────────────
# 2. CMEMS KLOROFIL-A  — pengganti NASA MODIS
# ─────────────────────────────────────────────────────────────
def fetch_cmems_chla(cmems_user: str, cmems_pass: str) -> dict:
    """
    Ambil klorofil-a dari CMEMS Ocean Colour (produk Copernicus-GlobColour,
    L4 gap-free, multi-sensor 4km, resolusi harian).

    Dataset dicoba berurutan (yang pertama berhasil langsung dipakai):
      1. NRT (Near Real Time) — lag ~1-2 hari, dataset paling baru
         cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D
      2. MY  (Multi-Year/reprocessed) — lag lebih lama, lebih stabil
         cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D

    Variabel yang diambil: CHL (mg/m³)
    Rentang waktu: 7 hari ke belakang, untuk mengantisipasi lag produk
    satelit / hari tanpa data karena tutupan awan.

    Memakai kredensial CMEMS yang sama dengan fetch_cmems() —
    TIDAK butuh akun NASA Earthdata sama sekali.
    """
    DATASETS = [
        "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D",
        "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
    ]

    start_dt, end_dt = _date_range_recent(days=7)

    for dataset_id in DATASETS:
        try:
            import copernicusmarine as cm

            ds = cm.open_dataset(
                dataset_id        = dataset_id,
                username          = cmems_user,
                password          = cmems_pass,
                minimum_latitude  = LAT_MIN,
                maximum_latitude  = LAT_MAX,
                minimum_longitude = LON_MIN,
                maximum_longitude = LON_MAX,
                start_datetime    = start_dt,
                end_datetime      = end_dt,
                variables         = ["CHL"],
            )

            chla_raw = float(ds["CHL"].mean())
            ds.close()

            if np.isnan(chla_raw):
                raise ValueError("Nilai CHL kosong (NaN) untuk rentang waktu/wilayah ini")

            # Clip ke rentang realistis Laut Arafura
            chla_val = float(np.clip(chla_raw, 0.05, 0.8))

            print(f"[CMEMS Klorofil] OK via {dataset_id} → CHL={chla_val:.4f} mg/m³")
            return {"ok": True, "chla": chla_val, "source_dataset": dataset_id}

        except Exception as e:
            print(f"[CMEMS Klorofil] GAGAL dataset {dataset_id}: {e}")
            continue   # coba dataset berikutnya

    # Kedua dataset gagal → fallback klimatologis
    print("[CMEMS Klorofil] Semua dataset gagal, menggunakan fallback klimatologis.")
    return {"ok": False, "chla": 0.22, "source_dataset": "fallback"}


# ─────────────────────────────────────────────────────────────
# 3. ERA5 / CDS — angin permukaan (angin_u / angin_v)
# ─────────────────────────────────────────────────────────────
def fetch_era5(cds_uid: str, cds_key: str) -> dict:
    """
    Ambil komponen angin permukaan 10 m dari ERA5 via Copernicus CDS API.
      u10 → angin_u  (zonal,     positif ke timur)
      v10 → angin_v  (meridional, positif ke utara)

    ERA5 biasanya tersedia dengan lag ~5 hari dari hari ini.
    """
    try:
        import cdsapi, netCDF4 as nc, tempfile, os

        c = cdsapi.Client(
            url   = "https://cds.climate.copernicus.eu/api/v2",
            key   = f"{cds_uid}:{cds_key}",
            quiet = True,
        )

        today  = datetime.date.today()
        target = today - datetime.timedelta(days=5)   # ERA5 lag ~5 hari

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tf:
            tmp_path = tf.name

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable"    : [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                ],
                "year"  : str(target.year),
                "month" : target.strftime("%m"),
                "day"   : target.strftime("%d"),
                "time"  : ["00:00", "06:00", "12:00", "18:00"],
                "area"  : [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],  # N/W/S/E
                "format": "netcdf",
            },
            tmp_path,
        )

        ds    = nc.Dataset(tmp_path)
        u_val = float(np.mean(ds.variables["u10"][:]))
        v_val = float(np.mean(ds.variables["v10"][:]))
        ds.close()
        os.unlink(tmp_path)

        print(f"[ERA5] OK → u={u_val:.3f} m/s, v={v_val:.3f} m/s")
        return {"ok": True, "angin_u": u_val, "angin_v": v_val}

    except Exception as e:
        print(f"[ERA5] ERROR: {e}")
        return {"ok": False, "angin_u": -1.5, "angin_v": -0.5}


# ─────────────────────────────────────────────────────────────
# 4. BMKG — tinggi gelombang signifikan
# ─────────────────────────────────────────────────────────────
def fetch_bmkg() -> dict:
    """
    Ambil tinggi gelombang signifikan dari BMKG INA-OC Open API.
    Endpoint: /api/v1/weather/significant-wave-height
    Parameter: adm4 = kode wilayah administratif level-4
    """
    try:
        import requests

        url    = f"{BMKG_BASE_URL}/api/v1/weather/significant-wave-height"
        params = {"adm4": BMKG_ADM4}

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("data", [])
        if records:
            heights  = [
                r.get("wave_height", 1.0)
                for r in records
                if r.get("wave_height") is not None
            ]
            wave_val = float(np.mean(heights)) if heights else 1.0
        else:
            wave_val = 1.0

        wave_val = float(np.clip(wave_val, 0.2, 2.5))

        print(f"[BMKG] OK → gelombang={wave_val:.2f} m")
        return {"ok": True, "gelombang": wave_val}

    except Exception as e:
        print(f"[BMKG] ERROR: {e}")
        return {"ok": False, "gelombang": 1.0}


# ─────────────────────────────────────────────────────────────
# 5. BUILD REALTIME DATAFRAME — fungsi utama dipanggil app.py
# ─────────────────────────────────────────────────────────────
def build_realtime_dataframe(
    cmems_user: str,
    cmems_pass: str,
    nasa_user : str = "",   # sudah tidak dipakai — dipertahankan agar
    nasa_pass : str = "",   # signature lama (app.py) tidak patah
    cds_uid   : str = "",
    cds_key   : str = "",
) -> dict:
    """
    Panggil semua sumber data secara berurutan, gabungkan jadi satu
    DataFrame baris tunggal, kembalikan:
        {
          'data'  : pd.DataFrame  (1 baris, semua kolom siap pakai),
          'status': dict          (nama_api → bool),
        }

    Klorofil-a SEKARANG diambil dari CMEMS (fetch_cmems_chla), BUKAN
    dari NASA MODIS lagi. Jika suatu API gagal, nilai fallback
    klimatologis digunakan sehingga aplikasi tetap berjalan.
    """

    # ── panggil semua sumber ──────────────────────────────
    print("[build_realtime_dataframe] Menghubungi CMEMS Fisika...")
    r_cmems = fetch_cmems(cmems_user, cmems_pass)

    print("[build_realtime_dataframe] Menghubungi CMEMS Klorofil-a...")
    r_chla  = fetch_cmems_chla(cmems_user, cmems_pass)

    print("[build_realtime_dataframe] Menghubungi ERA5/CDS...")
    r_era5  = fetch_era5(cds_uid, cds_key)

    print("[build_realtime_dataframe] Menghubungi BMKG...")
    r_bmkg  = fetch_bmkg()

    # ── status koneksi per API (ditampilkan di sidebar) ───
    status = {
        "CMEMS (Fisika)"     : r_cmems["ok"],
        "CMEMS (Klorofil-a)" : r_chla["ok"],
        "ERA5 / ECMWF"       : r_era5["ok"],
        "BMKG"               : r_bmkg["ok"],
    }

    # ── derivasi DO dan pH (tidak ada API langsung) ───────
    # DO menurun saat SST naik; pH menurun saat salinitas naik
    sst_val = r_cmems["sst"]
    sal_val = r_cmems["salinitas"]
    do_val  = float(np.clip(6.2  - (sst_val - 28.5) * 0.1, 4.5, 7.5))
    ph_val  = float(np.clip(8.12 - (sal_val - 34.2) * 0.02, 7.9, 8.4))

    # ── rangkai satu baris DataFrame ─────────────────────
    now = datetime.datetime.utcnow()

    row = {
        "time"     : now,
        "month"    : now.month,
        "year"     : now.year,
        # Arus & fisika — CMEMS
        "uo"       : r_cmems["uo"],
        "vo"       : r_cmems["vo"],
        "sst"      : sst_val,
        "ssta"     : r_cmems["ssta"],
        "salinitas": sal_val,
        # Klorofil-a — CMEMS Ocean Colour (pengganti NASA MODIS)
        "chla"     : r_chla["chla"],
        # Derivasi
        "do"       : do_val,
        "ph"       : ph_val,
        # Gelombang — BMKG
        "gelombang": r_bmkg["gelombang"],
        # Angin — ERA5
        "angin_u"  : r_era5["angin_u"],
        "angin_v"  : r_era5["angin_v"],
    }

    df_rt = pd.DataFrame([row])
    df_rt["current_speed"] = np.sqrt(df_rt["uo"]**2 + df_rt["vo"]**2)

    print(f"[build_realtime_dataframe] Selesai. Status: {status}")
    return {"data": df_rt, "status": status}
