"""
spatial.py — Grid spasial + land mask untuk LAUTAN
"""

import numpy as np
import pandas as pd
import streamlit as st

# ── Land mask ───────────────────────────────────────────────
try:
    from global_land_mask import globe as _glm
    _HAS_GLM = True
except Exception:
    _HAS_GLM = False


def normalisasi_global(series, vmin, vmax):
    """Min-max normalization ke rentang [0, 1]."""
    rng = vmax - vmin
    if rng == 0:
        return series * 0 if hasattr(series, "__len__") else 0.0
    return (series - vmin) / rng


def _manual_land_mask(lat_arr, lon_arr):
    """Fallback kasar bila global_land_mask belum terpasang."""
    lat_arr = np.asarray(lat_arr, dtype=float)
    lon_arr = np.asarray(lon_arr, dtype=float)
    m = np.zeros(lat_arr.shape, dtype=bool)
    m |= (lon_arr < 132.5) & (lat_arr > -2.0)
    m |= (lon_arr < 133.5) & (lat_arr > -3.0)
    m |= (lon_arr < 134.5) & (lat_arr > -3.5)
    m |= (lon_arr < 136.0) & (lat_arr > -4.0)
    m |= (lon_arr < 137.0) & (lat_arr > -4.5)
    m |= (lon_arr < 138.0) & (lat_arr > -5.0)
    m |= (lon_arr < 139.0) & (lat_arr > -5.5)
    m |= (lon_arr < 140.0) & (lat_arr > -6.0)
    m |= (lon_arr < 141.0) & (lat_arr > -6.5)
    m |= (lon_arr < 141.5) & (lat_arr > -7.0)
    m |= (lon_arr >= 141.0) & (lon_arr < 142.0) & (lat_arr > -8.5)
    m |= (lon_arr >= 140.0) & (lon_arr < 141.0) & (lat_arr > -8.0)
    m |= (lon_arr > 135.0) & (lon_arr < 139.0) & (lat_arr > -3.5) & (lat_arr < -1.0)
    m |= (lon_arr > 133.5) & (lon_arr < 135.5) & (lat_arr > -7.5) & (lat_arr < -5.5)
    m |= (lon_arr > 137.5) & (lon_arr < 139.5) & (lat_arr > -8.5) & (lat_arr < -7.0)
    return m


def compute_land_mask(lat_arr, lon_arr):
    """True = daratan. Vektorized."""
    if _HAS_GLM:
        return np.asarray(
            _glm.is_land(np.asarray(lat_arr, dtype=float), np.asarray(lon_arr, dtype=float))
        )
    return _manual_land_mask(lat_arr, lon_arr)


@st.cache_data
def get_ocean_grid_points():
    """Kembalikan (lat_flat, lon_flat) hanya untuk titik di laut."""
    lat_grid = np.linspace(-12.0, -4.5, 80)
    lon_grid = np.linspace(130.0, 144.0, 100)
    lon_g, lat_g = np.meshgrid(lon_grid, lat_grid)
    lat_flat = lat_g.flatten()
    lon_flat = lon_g.flatten()
    ocean = ~compute_land_mask(lat_flat, lon_flat)
    return lat_flat[ocean], lon_flat[ocean]


@st.cache_data
def build_spatial_grid(val_uo_base: float, val_vo_base: float,
                       month_seed: int, year_seed: int) -> pd.DataFrame:
    """
    Buat DataFrame grid spasial (hanya titik laut) dengan semua parameter
    oseanografi. Digunakan untuk render peta.
    """
    lat_flat, lon_flat = get_ocean_grid_points()
    seed = int(month_seed * 1000 + year_seed)
    rng  = np.random.default_rng(seed)

    # Variasi spasial berbasis gelombang sinus (pengganti interpolasi model aktual)
    var_spasial = (
        2.5 * np.sin(lon_flat * 0.22 + lat_flat * 0.31 + month_seed * 0.5) +
        2.0 * np.cos(lon_flat * 0.15 - lat_flat * 0.28 + month_seed * 0.3) +
        1.5 * np.sin(lon_flat * 0.40 + lat_flat * 0.18 + year_seed  * 0.1) +
        1.0 * np.cos(lon_flat * 0.12 + lat_flat * 0.42 + year_seed  * 0.07)
    )

    records = []
    for i in range(len(lat_flat)):
        t_lat, t_lon = float(lat_flat[i]), float(lon_flat[i])
        vs = var_spasial[i]

        grid_uo    = val_uo_base + (vs * 0.012) + rng.normal(0, 0.003)
        grid_vo    = val_vo_base + (vs * 0.006) + rng.normal(0, 0.002)
        grid_speed = float(np.sqrt(grid_uo**2 + grid_vo**2))

        grid_do   = float(np.clip(6.2  - vs * 0.06  + rng.normal(0, 0.05),  4.5, 7.5))
        grid_ph   = float(np.clip(8.12 + vs * 0.005 + rng.normal(0, 0.008), 7.9, 8.4))
        grid_chla = float(np.clip(0.22 + vs * 0.012 + rng.normal(0, 0.01),  0.05, 0.8))
        grid_sal  = float(np.clip(34.2 + vs * 0.04  + rng.normal(0, 0.08),  32.0, 36.5))
        grid_wave = float(np.clip(0.8  + abs(vs) * 0.05 + rng.normal(0, 0.04), 0.2, 2.5))

        grid_sohi = float(np.clip((
            0.25 * normalisasi_global(grid_do,    4.5, 7.5) +
            0.20 * normalisasi_global(grid_ph,    7.9, 8.4) +
            0.20 * normalisasi_global(grid_chla,  0.05, 0.8) +
            0.15 * normalisasi_global(grid_sal,   32.0, 36.5) +
            0.20 * (1 - normalisasi_global(grid_wave, 0.2, 2.5))
        ) * 100, 10, 100))

        grid_fsi = float(np.clip((
            0.35 * normalisasi_global(grid_chla,  0.05, 0.8) +
            0.25 * normalisasi_global(grid_do,    4.5, 7.5) +
            0.20 * normalisasi_global(grid_speed, 0.0, 0.25) +
            0.20 * (1 - normalisasi_global(grid_wave, 0.2, 2.5))
        ) * 100, 10, 100))

        records.append({
            "lat": t_lat, "lon": t_lon,
            "Ocean_Health_Index": grid_sohi,
            "Fisheries_Index":    grid_fsi,
            "uo": float(grid_uo), "vo": float(grid_vo),
            "sst":    float(np.clip(28.5 + vs * 0.18 + rng.normal(0, 0.1), 26.0, 32.0)),
            "ssta":   float(vs * 0.06 + rng.normal(0, 0.05)),
            "ph":         grid_ph,
            "do":         grid_do,
            "salinitas":  grid_sal,
            "chla":       grid_chla,
            "current_speed": grid_speed,
            "gelombang":  grid_wave,
            "angin_u": float(-1.5 + vs * 0.25 + rng.normal(0, 0.1)),
            "angin_v": float(-0.5 + vs * 0.12 + rng.normal(0, 0.06)),
        })

    return pd.DataFrame(records)
