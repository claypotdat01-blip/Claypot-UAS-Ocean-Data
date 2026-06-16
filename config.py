"""
config.py
"""

import os

# ==========================================================
# CMEMS
# ==========================================================
CMEMS_USER = os.getenv(
    "CMEMS_USER",
    "claypotdat01@gmail.com"
)

CMEMS_PASS = os.getenv(
    "CMEMS_PASS",
    "limaJuni_2026"
)

# ==========================================================
# NASA EARTHDATA
# ==========================================================
NASA_USER = os.getenv(
    "NASA_USER",
    "claypotdat01@gmail.com"
)

NASA_PASS = os.getenv(
    "NASA_PASS",
    "limaJuni_2026"
)

# ==========================================================
# CDS
# ==========================================================

# DITAMBAHKAN AGAR IMPORT TIDAK ERROR
CDS_UID = ""

CDS_KEY = os.getenv(
    "CDS_KEY",
    "e6b04d44-1240-4812-b642-a49e762e49b5"
)

# ==========================================================
# BMKG
# ==========================================================
BMKG_BASE_URL = "https://inaoc.bmkg.go.id"

# ==========================================================
# DOMAIN ARAFURA
# ==========================================================
LAT_MIN = -12.0
LAT_MAX = -4.0

LON_MIN = 129.0
LON_MAX = 144.0
