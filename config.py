# config.py — LAUTAN Platform Intelijen Oseanografi Papua
# ============================================================
# Kredensial API dan konfigurasi wilayah Laut Arafura
# ============================================================

# ── CMEMS (Copernicus Marine Service) ───────────────────────
# Digunakan untuk: arus (uo/vo), SST, salinitas, klorofil-a
CMEMS_USER = "claypotdat01@gmail.com"
CMEMS_PASS = "limaJuni_2026"

# ── NASA MODIS ───────────────────────────────────────────────
# Tidak lagi digunakan untuk real-time (diganti CMEMS chla)
# Dipertahankan agar tidak memecah import di app.py lama
NASA_USER = ""
NASA_PASS = ""

# ── ERA5 / Copernicus Climate Data Store (CDS) ───────────────
# Digunakan untuk: angin permukaan (angin_u / angin_v)
CDS_UID = ""
CDS_KEY = "e6b04d44-1240-4812-b642-a49e762e49b5"

# ── BMKG ─────────────────────────────────────────────────────
# Digunakan untuk: tinggi gelombang signifikan
BMKG_ADM4     = "81.71.01.1001"
BMKG_BASE_URL = "https://inaoc.bmkg.go.id"

# ── Batas Wilayah Laut Arafura ───────────────────────────────
LAT_MIN = -12.0
LAT_MAX = -4.0
LON_MIN = 129.0
LON_MAX = 144.0
