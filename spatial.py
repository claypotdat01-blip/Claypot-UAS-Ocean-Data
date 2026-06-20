"""
spatial.py — Grid spasial + land mask untuk OCEANA
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
    rng = vmax - vmin

    if rng == 0:
        return series * 0 if hasattr(series, "__len__") else 0.0

    return np.clip(
        (series - vmin) / rng,
        0.0,
        1.0
    )


def suitabilitas_optimal(x, lo, opt_lo, opt_hi, hi):
    """
    Skor kesesuaian 0..1 berbentuk trapesium (IDENTIK dengan yang dipakai di app.py):
      0  bila x <= lo atau x >= hi
      1  bila opt_lo <= x <= opt_hi   (zona optimal)
      naik/turun linear di antaranya.
    Dipakai untuk parameter ber-'rentang ideal' (chl-a trofik, jendela termal SST).
    """
    x = np.asarray(x, dtype=float)
    naik  = np.clip((x - lo) / max(opt_lo - lo, 1e-9), 0.0, 1.0)
    turun = np.clip((hi - x) / max(hi - opt_hi, 1e-9), 0.0, 1.0)
    return np.minimum(naik, turun)


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
def build_spatial_grid(uo: float = -0.05, vo: float = -0.01,
                       sst: float = 28.5, do: float = 6.2, ph: float = 8.12,
                       chla: float = 0.22, sal: float = 34.2, wave: float = 0.8,
                       angin_u: float = -1.5, angin_v: float = -0.5, ssta: float = 0.0,
                       month_seed: int = 1, year_seed: int = 2020) -> pd.DataFrame:
    """
    Grid spasial (titik laut) dengan semua parameter oseanografi, DIPUSATKAN pada
    nilai rata-rata periode (uo, sst, chla, ...) yang dikirim app.py.

    var_spasial = gelombang berfrekuensi rendah yang sudah DI-NOL-RATA-RATAKAN,
    jadi ia hanya menambahkan TEKSTUR spasial yang mulus di sekitar nilai tengah.
    Akibatnya:
      • rata-rata peta = nilai periode  -> peta = headline = status (satu skala),
      • peta berubah mengikuti musim (tidak statis tiap bulan),
      • tidak ada derau per-titik (peta tidak berbintik).
    Indeks FSI/OHI memakai FORMULA IDENTIK dengan app.py.
    """
    lat_flat, lon_flat = get_ocean_grid_points()

    vs = (
        2.5 * np.sin(lon_flat * 0.22 + lat_flat * 0.31 + month_seed * 0.5) +
        2.0 * np.cos(lon_flat * 0.15 - lat_flat * 0.28 + month_seed * 0.3) +
        1.5 * np.sin(lon_flat * 0.40 + lat_flat * 0.18 + year_seed  * 0.1) +
        1.0 * np.cos(lon_flat * 0.12 + lat_flat * 0.42 + year_seed  * 0.07)
    )
    vs = vs - vs.mean()   # nol-rata-rata -> rata-rata field = nilai periode

    grid_uo    = uo + vs * 0.012
    grid_vo    = vo + vs * 0.006
    grid_speed = np.sqrt(grid_uo**2 + grid_vo**2)

    grid_do   = np.clip(do   - vs * 0.06,  4.5,  7.5)
    grid_ph   = np.clip(ph   + vs * 0.005, 7.9,  8.4)
    grid_chla = np.clip(chla + vs * 0.012, 0.05, 0.8)
    grid_sal  = np.clip(sal  + vs * 0.04,  32.0, 36.5)
    grid_wave = np.clip(wave + vs * 0.05,  0.2,  2.5)

    # [FIX] Clip SST sesuai rentang PENUH yang dipakai suitabilitas_optimal OHI
    # (lo=22, hi=32) dan FSI (lo=24, hi=33). Sebelumnya clip mulai 26.0 sehingga
    # ramp naik OHI (22→26) tidak pernah terjadi dan skor SST selalu mulai dari 1,
    # menyebabkan OHI terkesan "stuck" dan tidak berubah wajar antar bulan.
    grid_sst  = np.clip(sst  + vs * 0.18, 22.0, 33.0)

    grid_ssta = ssta + vs * 0.06

    # ── Indeks — FORMULA IDENTIK dengan app.py (sumber kebenaran tunggal) ──
    grid_sohi = (
        0.30 * normalisasi_global(grid_do,   4.5,  7.5) +
        0.25 * normalisasi_global(grid_ph,   7.9,  8.4) +
        0.20 * suitabilitas_optimal(grid_chla, 0.05, 0.10, 0.40, 0.80) +
        0.15 * suitabilitas_optimal(grid_sal,  32.0, 33.5, 35.0, 36.5) +
        0.10 * suitabilitas_optimal(grid_sst,  22.0, 26.0, 30.0, 32.0)
    ) * 100
    grid_sohi = np.clip(grid_sohi, 0, 100)

    grid_fsi = (
        0.35 * normalisasi_global(grid_chla,  0.05, 0.8) +
        0.25 * suitabilitas_optimal(grid_sst, 24.0, 28.0, 30.0, 33.0) +
        0.20 * normalisasi_global(grid_do,    4.5,  7.5) +
        0.10 * normalisasi_global(grid_speed, 0.0,  0.25) +
        0.10 * (1 - normalisasi_global(grid_wave, 0.2, 2.5))
    ) * 100
    grid_fsi = np.clip(grid_fsi, 0, 100)

    grid_angin_u = angin_u + vs * 0.25
    grid_angin_v = angin_v + vs * 0.12

    return pd.DataFrame({
        "lat": lat_flat, "lon": lon_flat,
        "Ocean_Health_Index": grid_sohi,
        "Fisheries_Index":    grid_fsi,
        "uo": grid_uo, "vo": grid_vo,
        "sst": grid_sst, "ssta": grid_ssta,
        "ph": grid_ph, "do": grid_do,
        "salinitas": grid_sal, "chla": grid_chla,
        "current_speed": grid_speed,
        "gelombang": grid_wave,
        "angin_u": grid_angin_u, "angin_v": grid_angin_v,
    })
