# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.
# config.py

"""
CONFIG.PY — Parametry wejściowe dla symulatora
================================================
Tutaj definiujesz własne układy gwiazd.
Możesz kopiować i modyfikować gotowe przykłady.

JEDNOSTKI:
  Masy:      kilogramy  (M_SUN = 1.989e30 kg)
  Pozycje:   metry      (AU = 1.496e11 m)
  Prędkości: m/s
  Czas dt:   sekundy    (dzień = 86400 s)
"""

# Stałe pomocnicze
M_SUN = 1.989e30    # kg
AU    = 1.496e11    # m
DZIEN = 86400       # s
ROK   = 3.156e7     # s

# ── Parametry globalne symulacji ─────────────────────────────────────────────

FOLDER_DANYCH   = "data"      # gdzie zapisywać wyniki
FOLDER_WYNIKOW  = "results"   # gdzie zapisywać analizę

# ── Przykład własnej konfiguracji ────────────────────────────────────────────
#
# Skopiuj ten słownik do simulator.py lub zaimportuj go:
#   from config import MOJA_SYMULACJA
#   symuluj(MOJA_SYMULACJA)

MOJA_SYMULACJA = {
    "nazwa":         "moja_symulacja",

    # Masy trzech gwiazd [kg]
    "masy": [
        1.0 * M_SUN,    # Gwiazda 1
        1.0 * M_SUN,    # Gwiazda 2
        1.0 * M_SUN,    # Gwiazda 3
    ],

    # Pozycje startowe [x, y] w metrach
    "pozycje": [
        [ 3.0 * AU,  0.0      ],   # Gwiazda 1
        [-3.0 * AU,  0.0      ],   # Gwiazda 2
        [ 0.0,       3.0 * AU ],   # Gwiazda 3
    ],

    # Prędkości startowe [vx, vy] w m/s
    "predkosci": [
        [ 0.0,    5000.0],   # Gwiazda 1
        [ 0.0,   -5000.0],   # Gwiazda 2
        [ 8000.0,  0.0  ],   # Gwiazda 3
    ],

    # Krok czasowy — mniejszy = dokładniejszy, ale wolniejszy
    "dt": 5 * DZIEN,

    # Ile cykli zapisać do pliku
    "n_cykli": 1000,

    # Ile kroków RK4 między zapisami
    # (razem: n_cykli * kroki_na_cykl kroków łącznie)
    "kroki_na_cykl": 20,
}