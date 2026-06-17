cat > /home/claude/lautan/config.py << 'ENDOFFILE'
# ============================================================
# LAUTAN — Konfigurasi API
# ============================================================

# ── CMEMS (Copernicus Marine) ──────────────────────────────
# Digunakan untuk: Arus (uo, vo), SST, Salinitas, Klorofil-a
CMEMS_USER = "claypotdat01@gmail.com"
CMEMS_PASS = "limaJuni_2026"

# ── NASA Earthdata ─────────────────────────────────────────
# Tidak lagi digunakan untuk klorofil-a (diganti CMEMS)
# Tetap disimpan sebagai fallback / referensi
NASA_USER = "claypotdat1"
NASA_PASS = "limaJuni_2026"

# ── CDS / ERA5 (ECMWF) ────────────────────────────────────
# Digunakan untuk: Angin (angin_u, angin_v)
CDS_UID = ""
CDS_KEY = "e6b04d44-1240-4812-b642-a49e762e49b5"

# ── BMKG ──────────────────────────────────────────────────
# Digunakan untuk: Gelombang
BMKG_ADM4     = "81.71.01.1001"
BMKG_BASE_URL = "https://inaoc.bmkg.go.id"

# ── Area of Interest: Laut Arafura ────────────────────────
LAT_MIN = -12.0
LAT_MAX =  -4.0
LON_MIN = 129.0
LON_MAX = 144.0

# ── Dataset CMEMS untuk Klorofil-a ────────────────────────
# CMEMS dataset ID untuk klorofil-a (Ocean Colour)
# Dataset: OCEANCOLOUR_GLO_BGC_L4_MY_009_104
# Variabel: CHL (mg/m³)
CMEMS_CHLA_DATASET  = "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"
CMEMS_CHLA_VARIABLE = "CHL"
ENDOFFILE
echo "Done"
