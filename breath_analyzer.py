# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.
# breath_analyzer.py

"""
BREATH ANALYZER — SZUKANIE STAŁEJ FEIGENBAUMA-LIKE
====================================================
Hipoteza (kierunek C): w układach które "oddychają" (r13, r23 oscylują
sinusoidalnie w długiej skali czasowej), stosunek amplitud kolejnych
oddechów R_n = A_{n+1}/A_n jest stały i uniwersalny.

Analogia do stałej Feigenbauma: nie szukamy stałej w wartościach,
tylko w PROPORCJACH między kolejnymi cyklami oscylacji.

ALGORYTM:
  1. Wczytaj CSV (50k cykli)
  2. Wyznacz zdarzenia Poincarégo (lokalne minima r12)
  3. Dla r13 w zdarzeniach:
     a. Dopasuj A·sin(ωt+φ)+C → jeśli R²>R2_MIN, układ "oddycha"
     b. Wyznacz węzły sinusoidy (co pół okresu T/2)
     c. W każdym segmencie [węzeł_n, węzeł_{n+1}] oblicz amplitudę
        jako (max-min)/2
     d. Policz stosunki R_n = A_{n+1}/A_n
     e. Sprawdź czy R_n jest stałe (CV < próg)
  4. Zbierz wyniki ze wszystkich układów
  5. Sprawdź czy mediana R_n jest podobna między układami

WYNIKI:
  results/breath_tabela.csv     — dla każdego układu: ω, A, R², mediana R_n, CV
  results/breath_stosunki.csv   — wszystkie stosunki R_n ze wszystkich układów
  results/breath_raport.txt     — interpretacja
  results/breath_wykresy.png    — wykresy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal, optimize, stats
import os, glob, warnings
warnings.filterwarnings('ignore')

# ── Konfiguracja ──────────────────────────────────────────────────────────────

FOLDER_DANYCH  = "data"
FOLDER_WYNIKOW = "results"

R2_MIN        = 0.25   # minimalny R² żeby uznać układ za "oddychający"
MIN_SEGMENTOW = 4      # minimalna liczba pełnych segmentów (co pół okresu)
CV_PROG       = 0.4    # maksymalny CV stosunków żeby uznać za "stabilne"
KOLUMNY       = ["r13", "r23", "v3"]

# ── Przekrój Poincarégo ───────────────────────────────────────────────────────

def poincare(df, trigger='r12', distance=3):
    idx, _ = signal.find_peaks(-df[trigger].values, distance=distance)
    if len(idx) < 20:
        return None, idx
    return df.iloc[idx].reset_index(drop=True), idx

# ── Dopasowanie sinusoidy ─────────────────────────────────────────────────────

def model_sin(t, A, omega, phi, C):
    return A * np.sin(omega * t + phi) + C

def r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res / (ss_tot + 1e-30)

def dopasuj_sinusoide(t, y):
    """
    Próbuje dopasować A·sin(ωt+φ)+C do danych.
    Zwraca (popt, R²) lub None jeśli się nie uda.
    """
    C0 = np.mean(y)
    A0 = (np.max(y) - np.min(y)) / 2.0 + 1e-10
    T_total = t[-1] - t[0]
    if T_total <= 0:
        return None, -1.0

    najlepszy = None
    najlepszy_R2 = -np.inf

    # Próbuj kilka początkowych wartości okresu
    for n_okresy in [1, 2, 3, 5, 8, 12, 20]:
        omega_guess = 2 * np.pi * n_okresy / T_total
        try:
            popt, _ = optimize.curve_fit(
                model_sin, t, y,
                p0=[A0, omega_guess, 0.0, C0],
                bounds=([0, 0, -np.pi, -np.inf],
                        [np.inf, np.inf, np.pi, np.inf]),
                maxfev=8000, ftol=1e-8
            )
            y_fit = model_sin(t, *popt)
            R2 = r_squared(y, y_fit)
            if R2 > najlepszy_R2:
                najlepszy_R2 = R2
                najlepszy = popt
        except Exception:
            pass

    return najlepszy, najlepszy_R2

# ── Amplitudy segmentów ───────────────────────────────────────────────────────

def oblicz_amplitudy_segmentow(t, y, popt):
    """
    Dzieli szereg czasowy na segmenty długości T/2 (pół okresu sinusoidy).
    W każdym segmencie oblicza amplitudę jako (max-min)/2.
    Zwraca listę amplitud.
    """
    A_fit, omega, phi, C = popt
    if omega <= 0:
        return []

    T = 2 * np.pi / omega
    T_half = T / 2.0

    t_start = t[0]
    t_end   = t[-1]

    amplitudy = []
    t_seg = t_start
    while t_seg + T_half <= t_end:
        mask = (t >= t_seg) & (t < t_seg + T_half)
        y_seg = y[mask]
        if len(y_seg) >= 3:
            amp = (np.max(y_seg) - np.min(y_seg)) / 2.0
            amplitudy.append(amp)
        t_seg += T_half

    return amplitudy

# ── Stosunki amplitud ─────────────────────────────────────────────────────────

def oblicz_stosunki(amplitudy, min_amp=1e-6):
    """
    Oblicza R_n = A_{n+1}/A_n dla kolejnych segmentów.
    Pomija skrajne wartości (outlier).
    """
    if len(amplitudy) < MIN_SEGMENTOW:
        return []

    stosunki = []
    for i in range(len(amplitudy) - 1):
        if amplitudy[i] > min_amp and amplitudy[i+1] > min_amp:
            r = amplitudy[i+1] / amplitudy[i]
            stosunki.append(r)

    if not stosunki:
        return []

    # Usuń skrajne outliers (poza 5×IQR)
    arr = np.array(stosunki)
    q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q75 - q25
    mask = (arr > q25 - 5*iqr) & (arr < q75 + 5*iqr)
    return arr[mask].tolist()

# ── Analiza jednego układu ────────────────────────────────────────────────────

def analizuj_uklad(plik_csv, verbose=False):
    seed = os.path.basename(plik_csv).split('_s')[1].split('_')[0]

    try:
        df = pd.read_csv(plik_csv)
    except Exception:
        return None

    df_p, _ = poincare(df, 'r12')
    if df_p is None or len(df_p) < 20:
        return None

    t = df_p['czas_lat'].values.astype(float)
    wyniki_ukladu = {'seed': seed, 'n_zdarzen': len(df_p)}

    for kol in KOLUMNY:
        if kol not in df_p.columns:
            continue

        y = df_p[kol].values.astype(float)

        # Dopasuj sinusoidę
        popt, R2 = dopasuj_sinusoide(t, y)
        if popt is None or R2 < R2_MIN:
            wyniki_ukladu[f'{kol}_R2']     = R2 if popt is not None else -1
            wyniki_ukladu[f'{kol}_oddech'] = False
            continue

        A, omega, phi, C = popt
        T = 2 * np.pi / omega

        # Amplitudy segmentów
        amplitudy = oblicz_amplitudy_segmentow(t, y, popt)
        stosunki  = oblicz_stosunki(amplitudy)

        if len(stosunki) < MIN_SEGMENTOW - 1:
            wyniki_ukladu[f'{kol}_R2']     = R2
            wyniki_ukladu[f'{kol}_oddech'] = False
            wyniki_ukladu[f'{kol}_omega']  = omega
            wyniki_ukladu[f'{kol}_T_lat']  = T
            wyniki_ukladu[f'{kol}_A']      = A
            wyniki_ukladu[f'{kol}_n_seg']  = len(amplitudy)
            continue

        stosunki_arr = np.array(stosunki)
        med_R   = np.median(stosunki_arr)
        mean_R  = np.mean(stosunki_arr)
        std_R   = np.std(stosunki_arr)
        cv_R    = std_R / (abs(mean_R) + 1e-10)
        stabilny = cv_R < CV_PROG

        wyniki_ukladu[f'{kol}_R2']       = round(R2, 4)
        wyniki_ukladu[f'{kol}_oddech']   = True
        wyniki_ukladu[f'{kol}_omega']    = round(omega, 8)
        wyniki_ukladu[f'{kol}_T_lat']    = round(T, 1)
        wyniki_ukladu[f'{kol}_A']        = round(A, 4)
        wyniki_ukladu[f'{kol}_C']        = round(C, 4)
        wyniki_ukladu[f'{kol}_n_seg']    = len(amplitudy)
        wyniki_ukladu[f'{kol}_n_R']      = len(stosunki)
        wyniki_ukladu[f'{kol}_med_R']    = round(med_R, 6)
        wyniki_ukladu[f'{kol}_mean_R']   = round(mean_R, 6)
        wyniki_ukladu[f'{kol}_std_R']    = round(std_R, 6)
        wyniki_ukladu[f'{kol}_cv_R']     = round(cv_R, 4)
        wyniki_ukladu[f'{kol}_stabilny'] = stabilny
        wyniki_ukladu['stosunki_' + kol] = stosunki  # lista — do osobnej tabeli

        if verbose:
            ocena = "🟢 STABILNE" if stabilny else "🟡 zmienne"
            print(f"    [{kol}] R²={R2:.3f}  T={T:.0f}lat  "
                  f"n_seg={len(amplitudy)}  "
                  f"med_R={med_R:.4f}  CV={cv_R:.3f}  {ocena}")

    return wyniki_ukladu

# ── Główna analiza ────────────────────────────────────────────────────────────

def main():
    os.makedirs(FOLDER_WYNIKOW, exist_ok=True)

    pliki = sorted(glob.glob(os.path.join(FOLDER_DANYCH, "zwiazany_sredni_*.csv")))
    pliki = [p for p in pliki if "podsumowanie" not in p]

    print(f"\n{'='*60}")
    print(f"  BREATH ANALYZER — {len(pliki)} układów")
    print(f"  Próg R² > {R2_MIN}  |  Min segmentów: {MIN_SEGMENTOW}")
    print(f"{'='*60}\n")

    wiersze     = []
    wszystkie_R = {kol: [] for kol in KOLUMNY}  # wszystkie stosunki R_n

    for i, plik in enumerate(pliki):
        seed = os.path.basename(plik).split('_s')[1].split('_')[0]
        w = analizuj_uklad(plik, verbose=False)
        if w is None:
            continue

        # Zbierz stosunki do globalnej listy
        for kol in KOLUMNY:
            key = f'stosunki_{kol}'
            if key in w:
                wszystkie_R[kol].extend(w.pop(key))

        wiersze.append(w)

        # Postęp
        if (i+1) % 20 == 0 or (i+1) == len(pliki):
            n_odd = sum(1 for x in wiersze if any(x.get(f'{k}_oddech', False) for k in KOLUMNY))
            print(f"  [{i+1:3d}/{len(pliki)}] przetworzonych | "
                  f"oddychających: {n_odd}")

    df_tab = pd.DataFrame(wiersze)

    # ── Podsumowanie oddychających ────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  WYNIKI")
    print(f"{'='*60}\n")

    for kol in KOLUMNY:
        col_odd = f'{kol}_oddech'
        if col_odd not in df_tab.columns:
            continue
        n_odd  = df_tab[col_odd].sum()
        n_stab = df_tab.get(f'{kol}_stabilny', pd.Series(dtype=bool)).sum()
        print(f"  {kol}: oddychających={n_odd}/{len(df_tab)}  "
              f"ze stabilnym R_n={n_stab}")

        if n_odd > 0:
            sub = df_tab[df_tab[col_odd] == True]
            print(f"       R²:   med={sub[f'{kol}_R2'].median():.3f}  "
                  f"min={sub[f'{kol}_R2'].min():.3f}  "
                  f"max={sub[f'{kol}_R2'].max():.3f}")
            print(f"       T[lat]: med={sub[f'{kol}_T_lat'].median():.0f}  "
                  f"min={sub[f'{kol}_T_lat'].min():.0f}  "
                  f"max={sub[f'{kol}_T_lat'].max():.0f}")

            if f'{kol}_med_R' in sub.columns:
                R_vals = sub[f'{kol}_med_R'].dropna()
                if len(R_vals) > 0:
                    print(f"       med_R: med={R_vals.median():.5f}  "
                          f"std={R_vals.std():.5f}  "
                          f"CV={R_vals.std()/(abs(R_vals.mean())+1e-10)*100:.1f}%")

        # Globalna dystrybucja R_n
        R_all = np.array(wszystkie_R[kol])
        if len(R_all) > 10:
            R_clean = R_all[(R_all > 0.01) & (R_all < 100)]
            print(f"       GLOBALNIE R_n: n={len(R_clean)}  "
                  f"med={np.median(R_clean):.5f}  "
                  f"std={np.std(R_clean):.5f}  "
                  f"CV={np.std(R_clean)/(abs(np.mean(R_clean))+1e-10)*100:.1f}%")
        print()

    # ── Zapis ─────────────────────────────────────────────────────────────────

    plik_tab = os.path.join(FOLDER_WYNIKOW, 'breath_tabela.csv')
    df_tab.to_csv(plik_tab, index=False)
    print(f"  → Tabela: {plik_tab}")

    # Osobna tabela wszystkich stosunków
    for kol in KOLUMNY:
        R_all = np.array(wszystkie_R[kol])
        if len(R_all) > 0:
            plik_R = os.path.join(FOLDER_WYNIKOW, f'breath_stosunki_{kol}.csv')
            pd.DataFrame({'R_n': R_all}).to_csv(plik_R, index=False)
            print(f"  → Stosunki {kol}: {plik_R} ({len(R_all)} wartości)")

    # ── Wykresy ───────────────────────────────────────────────────────────────

    print("\n  Rysuję wykresy...")
    rysuj(df_tab, wszystkie_R)

    # ── Raport ────────────────────────────────────────────────────────────────

    zapisz_raport(df_tab, wszystkie_R)
    print(f"\n{'='*60}")
    print(f"  GOTOWE — wyniki w {FOLDER_WYNIKOW}/")
    print(f"{'='*60}\n")


# ── Wykresy ───────────────────────────────────────────────────────────────────

def rysuj(df_tab, wszystkie_R):
    os.makedirs(FOLDER_WYNIKOW, exist_ok=True)
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0f0f13')
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    KOLOR_KOL = {'r13': '#4affb0', 'r23': '#4a9eff', 'v3': '#ff6b4a'}

    def styl(ax, title, xlabel, ylabel):
        ax.set_facecolor('#17171f')
        ax.set_title(title, color='#e8e8f0', fontsize=9, pad=5)
        ax.set_xlabel(xlabel, color='#6b6b80', fontsize=8)
        ax.set_ylabel(ylabel, color='#6b6b80', fontsize=8)
        ax.tick_params(colors='#6b6b80', labelsize=7)
        for sp in ax.spines.values():
            sp.set_color('#333340')

    for ci, kol in enumerate(KOLUMNY):
        kolor = KOLOR_KOL.get(kol, '#ffffff')
        R_all = np.array(wszystkie_R[kol])

        # Panel górny: histogram R_n
        ax = fig.add_subplot(gs[0, ci])
        if len(R_all) > 5:
            R_clean = R_all[(R_all > 0.05) & (R_all < 20)]
            if len(R_clean) > 5:
                ax.hist(R_clean, bins=60, color=kolor, alpha=0.7, edgecolor='none')
                med = np.median(R_clean)
                ax.axvline(med, color='#ffffff', lw=1.5, ls='--',
                           label=f'mediana={med:.4f}')
                ax.axvline(1.0, color='#555555', lw=0.8, ls=':',
                           label='R=1 (stała amplituda)')
                ax.legend(fontsize=7, facecolor='#17171f',
                          labelcolor='#e8e8f0', framealpha=0.5)
                n_total = len(R_clean)
                cv = np.std(R_clean) / (abs(np.mean(R_clean)) + 1e-10)
                styl(ax, f'{kol}: rozkład R_n  (n={n_total}, CV={cv:.2f})',
                     'R_n = A_{n+1}/A_n', 'liczba')
            else:
                styl(ax, f'{kol}: brak danych', '', '')
        else:
            styl(ax, f'{kol}: brak danych', '', '')

        # Panel dolny: R² vs okres T dla oddychających
        ax2 = fig.add_subplot(gs[1, ci])
        col_odd = f'{kol}_oddech'
        col_R2  = f'{kol}_R2'
        col_T   = f'{kol}_T_lat'
        col_med = f'{kol}_med_R'

        if col_odd in df_tab.columns:
            sub = df_tab[df_tab[col_odd] == True]
            if len(sub) > 0 and col_T in sub.columns and col_R2 in sub.columns:
                sc = ax2.scatter(sub[col_T], sub[col_R2],
                                 c=sub[col_med] if col_med in sub.columns else kolor,
                                 cmap='RdYlGn', vmin=0.5, vmax=1.5,
                                 s=30, alpha=0.8, zorder=3)
                try:
                    plt.colorbar(sc, ax=ax2, label='mediana R_n',
                                 fraction=0.046, pad=0.04)
                except Exception:
                    pass
                styl(ax2, f'{kol}: R² vs okres oddechu  (n={len(sub)})',
                     'okres T [lata]', 'R² dopasowania')
            else:
                styl(ax2, f'{kol}: brak oddychających', '', '')
        else:
            styl(ax2, f'{kol}: brak danych', '', '')

    fig.suptitle('Breath Analyzer — stosunki amplitud R_n = A_{n+1}/A_n',
                 color='#e8e8f0', fontsize=12, y=1.01)

    plik = os.path.join(FOLDER_WYNIKOW, 'breath_wykresy.png')
    plt.savefig(plik, dpi=120, bbox_inches='tight', facecolor='#0f0f13')
    plt.close()
    print(f"  → Wykresy: {plik}")


# ── Raport ────────────────────────────────────────────────────────────────────

def zapisz_raport(df_tab, wszystkie_R):
    os.makedirs(FOLDER_WYNIKOW, exist_ok=True)
    plik = os.path.join(FOLDER_WYNIKOW, 'breath_raport.txt')

    with open(plik, 'w', encoding='utf-8') as f:
        f.write('='*60 + '\n')
        f.write('BREATH ANALYZER — RAPORT\n')
        f.write(f'Układów: {len(df_tab)}\n')
        f.write(f'Próg R² > {R2_MIN}  |  Min segmentów: {MIN_SEGMENTOW}\n')
        f.write('='*60 + '\n\n')

        f.write('HIPOTEZA: R_n = A_{n+1}/A_n jest stałą (analogia Feigenbauma)\n\n')

        for kol in KOLUMNY:
            f.write(f'[{kol}]\n')
            col_odd = f'{kol}_oddech'
            if col_odd not in df_tab.columns:
                f.write('  brak danych\n\n')
                continue

            sub = df_tab[df_tab[col_odd] == True]
            f.write(f'  Oddychających: {len(sub)}/{len(df_tab)}\n')

            if len(sub) > 0:
                if f'{kol}_T_lat' in sub.columns:
                    T = sub[f'{kol}_T_lat']
                    f.write(f'  Okres T [lat]: med={T.median():.0f}  '
                            f'min={T.min():.0f}  max={T.max():.0f}\n')
                if f'{kol}_med_R' in sub.columns:
                    R_med = sub[f'{kol}_med_R'].dropna()
                    if len(R_med) > 0:
                        cv = R_med.std() / (abs(R_med.mean()) + 1e-10)
                        f.write(f'  med_R między układami: '
                                f'med={R_med.median():.5f}  '
                                f'std={R_med.std():.5f}  '
                                f'CV={cv*100:.1f}%\n')

            R_all = np.array(wszystkie_R[kol])
            if len(R_all) > 5:
                R_c = R_all[(R_all > 0.05) & (R_all < 20)]
                cv  = np.std(R_c) / (abs(np.mean(R_c)) + 1e-10)
                f.write(f'  GLOBALNE R_n (wszystkie stosunki łącznie):\n')
                f.write(f'    n={len(R_c)}  med={np.median(R_c):.5f}  '
                        f'mean={np.mean(R_c):.5f}  std={np.std(R_c):.5f}  '
                        f'CV={cv*100:.1f}%\n')

                # Percentyle
                p = np.percentile(R_c, [5, 25, 50, 75, 95])
                f.write(f'    Percentyle [5,25,50,75,95]: '
                        f'{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} '
                        f'{p[3]:.4f} {p[4]:.4f}\n')

                # Czy R_n ≈ 1? (stała amplituda)
                blisko_1 = np.sum(np.abs(R_c - 1.0) < 0.1) / len(R_c) * 100
                f.write(f'    Procent R_n w przedziale [0.9, 1.1]: {blisko_1:.1f}%\n')

            f.write('\n')

        f.write('WNIOSEK:\n')
        for kol in KOLUMNY:
            R_all = np.array(wszystkie_R[kol])
            if len(R_all) < 10:
                f.write(f'  {kol}: za mało danych\n')
                continue
            R_c = R_all[(R_all > 0.05) & (R_all < 20)]
            cv  = np.std(R_c) / (abs(np.mean(R_c)) + 1e-10)
            med = np.median(R_c)
            if cv < 0.15:
                ocena = f"POTENCJALNA STAŁA: R_n ≈ {med:.4f} (CV={cv*100:.1f}%)"
            elif cv < 0.30:
                ocena = f"SŁABY WZORZEC: R_n ≈ {med:.4f} (CV={cv*100:.1f}%)"
            else:
                ocena = f"BRAK STAŁEJ: R_n chaotyczne (CV={cv*100:.1f}%)"
            f.write(f'  {kol}: {ocena}\n')

    print(f"  → Raport: {plik}")


if __name__ == '__main__':
    main()