"""
config.py — Konfigurasi & kredensial dashboard Laut Arafura
===========================================================

KEAMANAN (penting):
  - JANGAN commit file ini ke GitHub dengan kredensial asli di dalamnya.
  - Default di bawah sengaja dikosongkan. Isi lewat environment variable
    atau file `.env`, jangan ditempel langsung di sini kalau repo-nya publik.
  - Kalau password/token pernah ter-ekspos (mis. ter-paste/ter-share),
    GANTI password CMEMS & NASA dan REGENERATE token CDS sekarang.

CARA ISI (pilih salah satu):
  1) Set environment variable sebelum menjalankan Streamlit:
       export CMEMS_USER="email_kamu@example.com"
       export CMEMS_PASS="password_cmems"
       export NASA_USER="username_earthdata"
       export NASA_PASS="password_earthdata"
       export CDS_KEY="personal-access-token-cds"
       export BMKG_ADM4="31.71.03.1001"   # opsional
       streamlit run app.py
  2) Atau pakai file `.env` (lihat README) — direkomendasikan untuk lokal.

DAFTAR AKUN:
  CMEMS : https://marine.copernicus.eu                  (gratis, approval 1-2 hari kerja)
  NASA  : https://urs.earthdata.nasa.gov                (aktif setelah verifikasi email)
  CDS   : https://cds.climate.copernicus.eu/profile     (WAJIB akun BARU pasca-migrasi 2025,
                                                          lalu copy "Personal Access Token")
  BMKG  : tidak perlu akun (open data publik)
"""

import os

# Muat file .env kalau ada (opsional; aman kalau python-dotenv belum terpasang)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ── CMEMS (Copernicus Marine Service) ───────────────────────────────────────
# Sekarang WAJIB pakai toolbox `copernicusmarine` (motuclient lama sudah pensiun).
CMEMS_USER = os.getenv("CMEMS_USER", "")
CMEMS_PASS = os.getenv("CMEMS_PASS", "")

# ── NASA Earthdata (klorofil-a MODIS Aqua) ──────────────────────────────────
NASA_USER = os.getenv("NASA_USER", "")
NASA_PASS = os.getenv("NASA_PASS", "")

# ── CDS / ERA5 (angin ECMWF) ────────────────────────────────────────────────
# Setelah migrasi 2025: endpoint baru + Personal Access Token (string tunggal,
# TANPA prefix "UID:" gaya lama). Token kamu yang berformat UUID sudah sesuai.
CDS_URL = os.getenv("CDS_URL", "https://cds.climate.copernicus.eu/api")
CDS_KEY = os.getenv("CDS_KEY", "")

# ── BMKG (open data) ────────────────────────────────────────────────────────
# API cuaca publik resmi (JSON):
#   https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4=<kode_kelurahan>
# Cari kode wilayah (adm4) di: https://data.bmkg.go.id/prakiraan-cuaca
# Untuk Laut Arafura, pilih kelurahan/desa pesisir (mis. di Maluku/Papua Selatan).
BMKG_API_BASE = os.getenv("BMKG_API_BASE", "https://api.bmkg.go.id/publik")
BMKG_ADM4 = os.getenv("BMKG_ADM4", "")  # contoh: "81.71.01.1001"

# ── Batas wilayah Laut Arafura ──────────────────────────────────────────────
LAT_MIN, LAT_MAX = -12.0, -4.0
LON_MIN, LON_MAX = 129.0, 144.0


# ── Jembatan ke environment variable bawaan tiap library ────────────────────
# copernicusmarine & earthaccess membaca kredensial dari env var bernama khusus.
# Di sini kita mapping dari variabel di atas ke nama yang mereka harapkan.
if CMEMS_USER:
    os.environ.setdefault("COPERNICUSMARINE_SERVICE_USERNAME", CMEMS_USER)
if CMEMS_PASS:
    os.environ.setdefault("COPERNICUSMARINE_SERVICE_PASSWORD", CMEMS_PASS)
if NASA_USER:
    os.environ.setdefault("EARTHDATA_USERNAME", NASA_USER)
if NASA_PASS:
    os.environ.setdefault("EARTHDATA_PASSWORD", NASA_PASS)


def missing_credentials() -> dict:
    """Kembalikan dict {nama_sumber: True/False} — True artinya kredensial belum diisi."""
    return {
        "CMEMS": not (CMEMS_USER and CMEMS_PASS),
        "NASA": not (NASA_USER and NASA_PASS),
        "CDS": not CDS_KEY,
        "BMKG": not BMKG_ADM4,  # BMKG tidak perlu login, hanya butuh kode wilayah
    }
