"""
config.py — Konfigurasi API LAUTAN
====================================
CARA ISI:
  1. Isi langsung variabel di bawah (untuk testing lokal), ATAU
  2. Set environment variable sebelum jalankan Streamlit:
       export CMEMS_USER="email_kamu@example.com"
       export CMEMS_PASS="password_cmems"
       ...
       streamlit run app.py

CARA DAFTAR TIAP API:
  CMEMS  : https://marine.copernicus.eu → Register
  NASA   : https://urs.earthdata.nasa.gov → Register
  CDS    : https://cds.climate.copernicus.eu → Register → My Account → API Key
  BMKG   : Tidak perlu akun (open data)
"""

import os

# ── CMEMS (Copernicus Marine Service) ──────────────────────
# Daftar: https://marine.copernicus.eu
# Gratis, approved dalam 1-2 hari kerja
CMEMS_USER = os.getenv("CMEMS_USER", "ISI_EMAIL_CMEMS_KAMU_DI_SINI")
CMEMS_PASS = os.getenv("CMEMS_PASS", "ISI_PASSWORD_CMEMS_KAMU_DI_SINI")

# ── NASA Earthdata (MODIS Klorofil-a) ──────────────────────
# Daftar: https://urs.earthdata.nasa.gov
# Langsung aktif setelah verifikasi email
NASA_USER = os.getenv("NASA_USER", "ISI_USERNAME_NASA_EARTHDATA_DI_SINI")
NASA_PASS = os.getenv("NASA_PASS", "ISI_PASSWORD_NASA_EARTHDATA_DI_SINI")

# ── CDS / ERA5 (ECMWF Angin) ───────────────────────────────
# Daftar: https://cds.climate.copernicus.eu
# Setelah daftar: My Account → API Key → copy UID dan Key
CDS_UID = os.getenv("CDS_UID", "ISI_UID_CDS_KAMU_DI_SINI")   # angka, misal: 123456
CDS_KEY = os.getenv("CDS_KEY", "ISI_API_KEY_CDS_KAMU_DI_SINI")  # hex string panjang

# ── BMKG ───────────────────────────────────────────────────
# Tidak perlu akun — open data publik
BMKG_BASE_URL = "https://inaoc.bmkg.go.id"

# ── Batas Wilayah Laut Arafura ─────────────────────────────
LAT_MIN, LAT_MAX = -12.0, -4.0
LON_MIN, LON_MAX = 129.0, 144.0
