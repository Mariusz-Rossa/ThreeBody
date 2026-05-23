# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.
# pysr_analyzer.py

"""
PYSR ANALYZER — REGRESJA SYMBOLICZNA
=====================================
Szuka wzoru matematycznego w danych z symulatora trzech ciał.

UŻYCIE:
  # Podaj seed — skrypt sam znajdzie plik w data/
  python pysr_analyzer.py --seed 55485

  # Lub podaj pełną ścieżkę do pliku
  python pysr_analyzer.py --plik data/zwiazany_sredni_s55485_20260513_123456.csv

  # Dodatkowe opcje
  python pysr_analyzer.py --seed 55485 --kolumna v1 --iter 60

JAK DZIAŁA:
  1. Wczytuje CSV i JSON z konfigiem (masy, pozycje startowe)
  2. Filtruje dane przez Przekrój Poincarégo (perycentra r12)
  3. Przekazuje do PySR: czas + masy jako zmienne wejściowe
  4. PySR szuka wzoru matematycznego opisującego dane
  5. Zapisuje wyniki do results/pysr_<seed>_<kolumna>.txt
"""

import pandas as pd
import numpy as np
from scipy import signal
from pysr import PySRRegressor
import os
import glob
import json
import argparse
from datetime import datetime

# ── Stałe ───────────────────────────────────────────────────────────────────

FOLDER_DANYCH  = "data"
FOLDER_WYNIKOW = "results"
M_SUN          = 1.989e30
AU             = 1.496e11


# ── Szukanie pliku po seedzie ────────────────────────────────────────────────

def znajdz_plik_po_seedzie(seed, folder=FOLDER_DANYCH):
    """
    Szuka pliku CSV w folderze data/ zawierającego seed w nazwie.
    Seed może być liczbą lub stringiem (np. 55485 lub 's55485').

    Nazwy plików wyglądają tak:
      zwiazany_sredni_s55485_20260513_123456.csv

    Zwraca (plik_csv, plik_json) lub (None, None) jeśli nie znaleziono.
    """
    # Normalizuj seed — usuń 's' jeśli podano
    seed_str = str(seed).lstrip('s')

    wzorzec = os.path.join(folder, f"*_s{seed_str}_*.csv")
    pliki = glob.glob(wzorzec)

    # Pomiń pliki podsumowań
    pliki = [p for p in pliki if "podsumowanie" not in p]

    if not pliki:
        print(f"  ✗ Nie znaleziono pliku z seedem {seed_str} w folderze '{folder}/'")
        print(f"  Szukałem wzorca: *_s{seed_str}_*.csv")
        # Pokaż dostępne pliki jako podpowiedź
        wszystkie = sorted(glob.glob(os.path.join(folder, "*.csv")))
        wszystkie = [p for p in wszystkie if "podsumowanie" not in p]
        if wszystkie:
            print(f"\n  Dostępne pliki w {folder}/:")
            for p in wszystkie[:10]:
                print(f"    {os.path.basename(p)}")
            if len(wszystkie) > 10:
                print(f"    ... i {len(wszystkie)-10} więcej")
        return None, None

    if len(pliki) > 1:
        print(f"  ⚠ Znaleziono {len(pliki)} plików z seedem {seed_str}, używam pierwszego:")
        for p in pliki:
            print(f"    {os.path.basename(p)}")

    plik_csv = pliki[0]

    # Szukaj odpowiedniego JSON z configiem
    plik_json = plik_csv.replace(".csv", "_config.json")
    if not os.path.exists(plik_json):
        # Spróbuj bez _config
        alt = plik_csv.replace(".csv", ".json")
        plik_json = alt if os.path.exists(alt) else None

    return plik_csv, plik_json


# ── Przekrój Poincarégo ──────────────────────────────────────────────────────

def wyznacz_przekroj_poincarego(df, kolumna_trigger='r12', distance=3):
    """
    Filtruje dane do momentów perycentrum (lokalnych minimów r12).
    Zwraca przefiltrowany DataFrame i indeksy zdarzeń.
    """
    wartosci = df[kolumna_trigger].values
    minima_idx, _ = signal.find_peaks(-wartosci, distance=distance)

    if len(minima_idx) == 0:
        return pd.DataFrame(), np.array([])

    return df.iloc[minima_idx].reset_index(drop=True), minima_idx


# ── Przygotowanie danych dla PySR ────────────────────────────────────────────

def przygotuj_dane(df_poincare, kolumna, cfg=None):
    """
    Buduje macierz X (zmienne wejściowe) i wektor y (cel).

    Zmienne wejściowe:
      - czas_lat     : czas fizyczny [lata]
      - masa1/2/3    : masy gwiazd [M_Sun] — ze stałego configu
      - n_zdarzenia  : numer kolejnego zdarzenia (0, 1, 2, ...)

    Im więcej zmiennych wejściowych, tym lepiej PySR może znaleźć
    wzór który łączy fizykę z obserwowanym wzorcem.
    """
    n = len(df_poincare)

    # Zawsze dostępne: czas i numer zdarzenia
    czas = df_poincare['czas_lat'].values
    n_zdarzen = np.arange(n, dtype=float)

    # Kolumny do X
    X_cols = {
        'czas_lat':   czas,
        'n':          n_zdarzen,
    }

    # Dodaj masy z configu jeśli dostępny
    if cfg and 'masy' in cfg:
        masy_msun = [m / M_SUN for m in cfg['masy']]
        X_cols['masa1'] = np.full(n, masy_msun[0])
        X_cols['masa2'] = np.full(n, masy_msun[1])
        X_cols['masa3'] = np.full(n, masy_msun[2])

    # Dodaj odległości startowe z configu
    if cfg and 'pozycje' in cfg:
        pos = np.array(cfg['pozycje'])
        r12_start = np.linalg.norm(pos[1] - pos[0]) / AU
        r13_start = np.linalg.norm(pos[2] - pos[0]) / AU
        r23_start = np.linalg.norm(pos[2] - pos[1]) / AU
        X_cols['r12_start'] = np.full(n, r12_start)
        X_cols['r13_start'] = np.full(n, r13_start)
        X_cols['r23_start'] = np.full(n, r23_start)

    X = np.column_stack(list(X_cols.values()))
    y = df_poincare[kolumna].values

    return X, y, list(X_cols.keys())


# ── Główna funkcja ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PySR Analyzer — szukanie wzorów matematycznych w danych trzech ciał.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady:
  python pysr_analyzer.py --seed 55485
  python pysr_analyzer.py --seed 62746 --kolumna v2 --iter 60
  python pysr_analyzer.py --plik data/zwiazany_sredni_s55485_20260513_123456.csv
        """
    )

    # Sposób podania pliku — seed LUB pełna ścieżka
    grupa = parser.add_mutually_exclusive_group(required=True)
    grupa.add_argument("--seed", type=str,
                       help="Seed układu (np. 55485) — skrypt sam znajdzie plik w data/")
    grupa.add_argument("--plik", type=str,
                       help="Pełna ścieżka do pliku CSV")

    parser.add_argument("--kolumna", type=str, default="r12",
                        choices=["r12", "r13", "r23", "v1", "v2", "v3"],
                        help="Kolumna do analizy (domyślnie: r12)")
    parser.add_argument("--iter", type=int, default=40,
                        help="Liczba iteracji PySR (domyślnie: 40, więcej = dokładniej ale wolniej)")
    parser.add_argument("--folder", type=str, default=FOLDER_DANYCH,
                        help=f"Folder z danymi (domyślnie: {FOLDER_DANYCH})")

    args = parser.parse_args()

    # ── Znajdź pliki ─────────────────────────────────────────────────────────
    if args.seed:
        plik_csv, plik_json = znajdz_plik_po_seedzie(args.seed, args.folder)
        if plik_csv is None:
            return
        identyfikator = f"s{args.seed.lstrip('s')}"
    else:
        plik_csv = args.plik
        plik_json = args.plik.replace(".csv", "_config.json")
        if not os.path.exists(plik_csv):
            print(f"  ✗ Nie znaleziono pliku: {plik_csv}")
            return
        identyfikator = os.path.basename(plik_csv).replace(".csv", "")

    print(f"\n{'='*60}")
    print(f"  PYSR ANALYZER")
    print(f"{'='*60}")
    print(f"  Plik:     {os.path.basename(plik_csv)}")
    print(f"  Kolumna:  {args.kolumna}")
    print(f"  Iteracje: {args.iter}")
    print(f"{'='*60}\n")

    # ── Wczytaj dane ─────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(plik_csv)
        print(f"  ✓ Wczytano CSV: {len(df)} cykli, "
              f"{df['czas_lat'].iloc[-1]:.1f} lat symulacji")
    except Exception as e:
        print(f"  ✗ Błąd wczytywania CSV: {e}")
        return

    # Wczytaj config (opcjonalnie)
    cfg = None
    if plik_json and os.path.exists(plik_json):
        try:
            with open(plik_json) as f:
                cfg = json.load(f)
            masy = [m/M_SUN for m in cfg['masy']]
            print(f"  ✓ Wczytano config: masy {masy[0]:.2f} {masy[1]:.2f} {masy[2]:.2f} M☉")
        except Exception as e:
            print(f"  ⚠ Nie udało się wczytać configu: {e}")
    else:
        print(f"  ⚠ Brak pliku config — PySR będzie miał tylko czas jako zmienną")

    # ── Przekrój Poincarégo ───────────────────────────────────────────────────
    df_poincare, idx = wyznacz_przekroj_poincarego(df, 'r12')

    if len(df_poincare) < 15:
        print(f"\n  ✗ Za mało zdarzeń Poincarégo ({len(df_poincare)}) — minimum 15")
        return

    print(f"  ✓ Zdarzeń Poincarégo (perycentra r12): {len(df_poincare)}")

    # ── Przygotuj dane dla PySR ───────────────────────────────────────────────
    X, y, nazwy_kolumn = przygotuj_dane(df_poincare, args.kolumna, cfg)

    print(f"\n  Zmienne wejściowe dla PySR:")
    for i, nazwa in enumerate(nazwy_kolumn):
        print(f"    X[{i}] = {nazwa}")
    print(f"  Zmienna wyjściowa: {args.kolumna}")
    print(f"  Punktów danych: {len(y)}")

    # Statystyki y
    print(f"\n  Statystyki {args.kolumna}:")
    print(f"    Min: {y.min():.3f}  Max: {y.max():.3f}  "
          f"Śr: {y.mean():.3f}  Std: {y.std():.3f}")

    # ── Uruchom PySR ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Uruchamiam PySR ({args.iter} iteracji)...")
    print(f"  Ctrl+C = przerwij i pokaż dotychczasowe wyniki")
    print(f"{'─'*60}\n")

    model = PySRRegressor(
        niterations=args.iter,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["sin", "cos", "sqrt", "square", "cube", "exp", "log"],
        extra_sympy_mappings={
            "square": lambda x: x**2,
            "cube":   lambda x: x**3,
        },
        variable_names=nazwy_kolumn,
        parsimony=0.01,
        populations=20,
        population_size=50,
        maxsize=30,
        random_state=42,
        verbosity=1,
    )

    przerwano = False
    try:
        model.fit(X, y)
    except KeyboardInterrupt:
        print("\n  Przerwano — pokazuję dotychczasowe wyniki...")
        przerwano = True

    # ── Wyniki ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  WYNIKI PySR — {args.kolumna} dla seed {identyfikator}")
    print(f"{'='*60}\n")

    if not hasattr(model, 'equations_') or model.equations_ is None:
        print("  ✗ Brak wyników — PySR nie zdążył znaleźć żadnego wzoru.")
        return

    # Pokaż wszystkie odkryte równania
    eq = model.equations_[['complexity', 'loss', 'equation']].copy()
    eq['loss'] = eq['loss'].apply(lambda x: f"{x:.6f}")
    print(eq.to_string(index=False))

    print(f"\n  {'─'*58}")
    print(f"  NAJLEPSZY WZÓR (optymalny balans dokładność/prostota):")
    try:
        wzor = model.sympy()
        print(f"\n  {args.kolumna}(t) = {wzor}\n")
    except Exception as e:
        print(f"  (nie udało się skonwertować do SymPy: {e})")

    # ── Zapis wyników ─────────────────────────────────────────────────────────
    os.makedirs(FOLDER_WYNIKOW, exist_ok=True)
    znacznik = datetime.now().strftime("%Y%m%d_%H%M%S")
    plik_wynikow = os.path.join(
        FOLDER_WYNIKOW,
        f"pysr_{identyfikator}_{args.kolumna}_{znacznik}.txt"
    )

    with open(plik_wynikow, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"PYSR ANALYZER — WYNIKI\n")
        f.write(f"Plik:       {os.path.basename(plik_csv)}\n")
        f.write(f"Kolumna:    {args.kolumna}\n")
        f.write(f"Iteracje:   {args.iter}\n")
        f.write(f"Przerwano:  {'tak' if przerwano else 'nie'}\n")
        f.write(f"Data:       {znacznik}\n")
        f.write("=" * 60 + "\n\n")

        if cfg:
            masy = [m/M_SUN for m in cfg['masy']]
            f.write(f"PARAMETRY UKŁADU:\n")
            f.write(f"  Masy: {masy[0]:.3f}  {masy[1]:.3f}  {masy[2]:.3f} M☉\n")
            f.write(f"  Zdarzeń Poincarégo: {len(df_poincare)}\n\n")

        f.write("WSZYSTKIE ODKRYTE RÓWNANIA:\n")
        f.write(eq.to_string(index=False))
        f.write("\n\n")

        f.write("NAJLEPSZY WZÓR:\n")
        try:
            f.write(f"  {args.kolumna}(t) = {model.sympy()}\n")
        except Exception as e:
            f.write(f"  (błąd konwersji: {e})\n")

    print(f"  ✓ Zapisano wyniki: {plik_wynikow}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()