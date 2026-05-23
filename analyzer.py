# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.
# analyzer.py

"""
ANALYZER PROBLEMU TRZECH CIAŁ
================================
Program 2 — szuka ukrytych wzorców i stałych w danych z symulatora.

STRATEGIA (Zaktualizowana o Mapy Poincarégo):
  Badamy układ TYLKO w określonych fizycznie momentach (zdarzeniach),
  np. podczas najbliższych zbliżeń gwiazd (perycentrach).
  
  Szukamy stałej w stosunkach R(n) = δ(n+1) / δ(n), 
  gdzie δ(n) to zmiana wartości między kolejnymi perycentrami.

KROKI ANALIZY:
  1. Wczytaj CSV i przefiltruj dane przez Przekrój Poincarégo.
  2. Oblicz różnice δ(n) = wartość(n+1) - wartość(n) dla zdarzeń.
  3. Oblicz stosunki R(n) = δ(n+1) / δ(n) — czy to stała?
  4. Wykryj periodyczność przez FFT i autokorelację.
  5. Porównaj wzorce między różnymi układami.
  6. Zapisz wyniki i wykresy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# ── Konfiguracja ─────────────────────────────────────────────────────────────

FOLDER_DANYCH   = "data"
FOLDER_WYNIKOW  = "results"

KOLUMNY = ["r12", "r13", "r23", "v1", "v2", "v3"]
KOLUMNY_ETYKIETY = {
    "r12": "Odległość r₁₂ [AU]",
    "r13": "Odległość r₁₃ [AU]",
    "r23": "Odległość r₂₃ [AU]",
    "v1":  "Prędkość v₁ [m/s]",
    "v2":  "Prędkość v₂ [m/s]",
    "v3":  "Prędkość v₃ [m/s]",
}

KOLORY = ['#4a9eff', '#ff6b4a', '#4affb0', '#ff4af0', '#ffd04a', '#a04aff']

# ── Wczytywanie danych ───────────────────────────────────────────────────────

def wczytaj_dane(folder=FOLDER_DANYCH):
    pliki = sorted(glob.glob(os.path.join(folder, "*.csv")))
    pliki = [p for p in pliki if "podsumowanie" not in p]

    if not pliki:
        print(f"  ✗ Brak plików CSV w folderze '{folder}'")
        return []

    dane = []
    for plik in pliki:
        try:
            df = pd.read_csv(plik)
            nazwa = os.path.basename(plik).replace(".csv", "")
            dane.append({"nazwa": nazwa, "df": df, "plik": plik})
            print(f"  ✓ Wczytano: {nazwa} ({len(df)} cykli symulacji)")
        except Exception as e:
            print(f"  ✗ Błąd wczytywania {plik}: {e}")

    return dane

# ── Przekrój Poincarégo & Analiza ─────────────────────────────────────────────

def wyznacz_przekroj_poincarego(df, kolumna='r12'):
    """
    Filtruje dane do momentów "zdarzeń".
    Domyślnie: znajduje lokalne minima (perycentra) w odległości r12.
    """
    wartosci = df[kolumna].values
    # Szukamy dołków - znak minus odwraca wykres dla find_peaks
    # distance=3 zapobiega wyłapywaniu szumów blisko siebie
    minima_idx, _ = signal.find_peaks(-wartosci, distance=3)
    return df.iloc[minima_idx].reset_index(drop=True), minima_idx

def oblicz_roznice(df_poincare, kolumna):
    wartosci = df_poincare[kolumna].values
    return np.diff(wartosci)

def oblicz_stosunki(delta, min_abs=1e-10):
    wynik = []
    for i in range(len(delta) - 1):
        if abs(delta[i]) > min_abs:
            wynik.append(delta[i+1] / delta[i])
        else:
            wynik.append(np.nan)
    return np.array(wynik)

def statystyki_stosunkow(R, odrzuc_outliers=True):
    R_clean = R[~np.isnan(R)]
    if len(R_clean) < 5:
        return None

    if odrzuc_outliers:
        med = np.median(R_clean)
        mad = np.median(np.abs(R_clean - med))
        maska = np.abs(R_clean - med) < 10 * (mad + 1e-10)
        R_clean = R_clean[maska]

    if len(R_clean) < 5:
        return None

    return {
        "srednia":  np.mean(R_clean),
        "mediana":  np.median(R_clean),
        "std":      np.std(R_clean),
        "cv":       np.std(R_clean) / (abs(np.mean(R_clean)) + 1e-10),
        "min":      np.min(R_clean),
        "max":      np.max(R_clean),
        "n":        len(R_clean),
    }

def analiza_fft(szereg, dt_cykl=1.0):
    szereg_clean = szereg[~np.isnan(szereg)]
    n = len(szereg_clean)
    if n < 10:
        return [], []

    szereg_clean = szereg_clean - np.mean(szereg_clean)
    widmo = np.abs(fft(szereg_clean))[:n//2]
    czest = fftfreq(n, d=dt_cykl)[:n//2]

    if len(widmo) < 3:
        return [], []

    widmo_norm = widmo / (np.max(widmo) + 1e-10)
    piki_idx, _ = signal.find_peaks(widmo_norm, height=0.1, distance=3)

    if len(piki_idx) == 0:
        return [], []

    piki_sorted = sorted(piki_idx, key=lambda i: widmo_norm[i], reverse=True)[:5]
    dominujace_czest = czest[piki_sorted]
    dominujace_amp   = widmo_norm[piki_sorted]

    okresy = [1.0/f if f > 0 else np.inf for f in dominujace_czest]
    return okresy, dominujace_amp.tolist()

def analiza_autokorelacji(szereg, max_lag=200):
    szereg_clean = szereg[~np.isnan(szereg)]
    n = len(szereg_clean)
    if n < 20:
        return [], []

    szereg_norm = (szereg_clean - np.mean(szereg_clean)) / (np.std(szereg_clean) + 1e-10)
    max_lag = min(max_lag, n // 3)

    lagi = range(1, max_lag + 1)
    auto = [np.corrcoef(szereg_norm[:-lag], szereg_norm[lag:])[0,1]
            for lag in lagi]

    return list(lagi), auto

def funkcja_oscylacji(t, A, omega, phi, C):
    """Matematyczny model oddechu układu (fala sinusoidalna)."""
    return A * np.sin(omega * t + phi) + C

def badaj_ewolucje_wiekowa(df_poincare, kolumna='r12', verbose=True):
    """
    Szuka długofalowego 'oddechu' układu (np. oscylacji Kozai-Lidowa)
    dopasowując falę sinusoidalną do punktów z Mapy Poincarégo w dziedzinie czasu.
    """
    if len(df_poincare) < 15:
        return None

    # Zamiast numeru cyklu 'n', używamy prawdziwego czasu fizycznego 't' (w latach)
    t = df_poincare['czas_lat'].values
    y = df_poincare[kolumna].values

    # 1. Zgadywanie punktu startowego (Initial Guess) - kluczowe dla SciPy
    C_guess = np.mean(y)
    A_guess = (np.max(y) - np.min(y)) / 2.0
    
    # Zgadywanie okresu (zakładamy optymistycznie, że widzimy choć połowę fali)
    czas_calkowity = t[-1] - t[0]
    if czas_calkowity == 0: return None
    omega_guess = 2 * np.pi / (czas_calkowity / 1.5) 
    
    p0 = [A_guess, omega_guess, 0.0, C_guess]

    try:
        # 2. Właściwe dopasowanie ewolucyjne (Curve Fitting)
        popt, _ = curve_fit(funkcja_oscylacji, t, y, p0=p0, maxfev=15000)
        A_fit, omega_fit, phi_fit, C_fit = popt
        
        # 3. Ocena jak dobrze wzór opisuje rzeczywistość (R-kwadrat)
        y_fit = funkcja_oscylacji(t, *popt)
        ss_res = np.sum((y - y_fit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        
        okres_lat = abs(2 * np.pi / omega_fit)
        
        if verbose and r_squared > 0.3: # Pokazujemy tylko sensowne dopasowania
            print(f"    [WZÓR] {kolumna}(t) = {abs(A_fit):.3f} * sin(ωt + φ) + {C_fit:.3f}")
            print(f"           Okres oscylacji = {okres_lat:.1f} lat | Dokładność dopasowania (R²) = {r_squared*100:.1f}%")
            
        return {
            "A": abs(A_fit), "omega": omega_fit, "phi": phi_fit, "C": C_fit,
            "okres_lat": okres_lat,
            "R2": r_squared,
            "y_fit": y_fit,
            "t": t, "y": y
        }
    except Exception:
        # Zbyt duży chaos, SciPy się poddał - normalne w problemie trzech ciał
        return None

# ── Analiza jednego układu ───────────────────────────────────────────────────

def analizuj_uklad(dane_ukladu, verbose=True):
    nazwa = dane_ukladu["nazwa"]
    df_oryginalne = dane_ukladu["df"]

    # --- NOWE: Filtrujemy dane do Mapy Poincarégo ---
    df_poincare, idx_poincare = wyznacz_przekroj_poincarego(df_oryginalne, 'r12')

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  Analiza: {nazwa}")
        print(f"  Zdarzeń Poincarégo: {len(df_poincare)} (z {len(df_oryginalne)} całkowitych cykli)")
        print(f"  {'─'*50}")

    wyniki = {
        "nazwa": nazwa, 
        "kolumny": {}, 
        "idx_poincare": idx_poincare,
        "n_zdarzen": len(df_poincare)
    }

    if len(df_poincare) < 10:
        if verbose: print("  ⚠ Zbyt mało zdarzeń do analizy wzorców.")
        return wyniki

    for kol in KOLUMNY:
        if kol not in df_poincare.columns:
            continue

        # Analiza odbywa się TYLKO na odfiltrowanych zdarzeniach
        delta   = oblicz_roznice(df_poincare, kol)
        R       = oblicz_stosunki(delta)
        stats_R = statystyki_stosunkow(R)
        dopasowanie = badaj_ewolucje_wiekowa(df_poincare, kolumna=kol, verbose=verbose)

        if stats_R is None:
            continue

        okresy, amplitudy = analiza_fft(R)
        lagi, autokor     = analiza_autokorelacji(R)

        cv = stats_R["cv"]
        if cv < 0.1:   ocena = "🟢 BARDZO STABILNE"
        elif cv < 0.3: ocena = "🟡 UMIARKOWANIE STABILNE"
        elif cv < 1.0: ocena = "🟠 ZMIENNE"
        else:          ocena = "🔴 CHAOTYCZNE"

        wyniki["kolumny"][kol] = {
            "delta":      delta,
            "R":          R,
            "stats":      stats_R,
            "okresy":     okresy,
            "amplitudy":  amplitudy,
            "lagi":       lagi,
            "autokor":    autokor,
            "ocena":      ocena,
            "dopasowanie": dopasowanie,
        }

        if verbose:
            print(f"\n  [{kol}] {ocena}")
            print(f"    R — śr: {stats_R['srednia']:+.4f}  "
                  f"med: {stats_R['mediana']:+.4f}  "
                  f"std: {stats_R['std']:.4f}  CV: {cv:.3f}")
            if okresy:
                print(f"    Dominujący okres zdarzeń: {okresy[0]:.1f} "
                      f"(amplituda: {amplitudy[0]:.3f})")

    return wyniki

# ── Analiza porównawcza (uniwersalność) ─────────────────────────────────────

def analiza_uniwersalnosci(wyniki_wszystkich):
    print(f"\n{'='*57}")
    print(f"  ANALIZA UNIWERSALNOŚCI — porównanie między układami")
    print(f"{'='*57}\n")

    wyniki_porownania = {}

    for kol in KOLUMNY:
        wartosci_R = []
        nazwy = []

        for w in wyniki_wszystkich:
            if kol in w["kolumny"]:
                stats_R = w["kolumny"][kol]["stats"]
                if stats_R and stats_R["cv"] < 2.0:
                    wartosci_R.append(stats_R["mediana"])
                    nazwy.append(w["nazwa"])

        if len(wartosci_R) < 3:
            continue

        wartosci_R = np.array(wartosci_R)
        rozrzut = np.std(wartosci_R) / (abs(np.mean(wartosci_R)) + 1e-10)

        wyniki_porownania[kol] = {
            "wartosci":  wartosci_R,
            "nazwy":     nazwy,
            "srednia":   np.mean(wartosci_R),
            "std":       np.std(wartosci_R),
            "rozrzut":   rozrzut,
        }

        if rozrzut < 0.1:   ocena = "🟢 POTENCJALNIE UNIWERSALNA STAŁA!"
        elif rozrzut < 0.3: ocena = "🟡 SŁABY WZORZEC"
        else:               ocena = "🔴 BRAK WZORCA"

        print(f"  [{kol}] {ocena}")
        print(f"    Mediana R w układach: "
              f"śr={np.mean(wartosci_R):+.4f}  "
              f"std={np.std(wartosci_R):.4f}  rozrzut={rozrzut:.3f}")

    return wyniki_porownania

# ── Wykresy ──────────────────────────────────────────────────────────────────

def rysuj_uklad(dane_ukladu, wyniki, folder=FOLDER_WYNIKOW):
    os.makedirs(folder, exist_ok=True)
    nazwa = wyniki["nazwa"]
    df = dane_ukladu["df"]
    idx_poincare = wyniki.get("idx_poincare", [])

    kols_do_wykresu = [k for k in ["r12", "r13", "r23"] if k in wyniki["kolumny"]]
    if not kols_do_wykresu: return

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0f0f13')
    gs = gridspec.GridSpec(3, len(kols_do_wykresu), hspace=0.45, wspace=0.35)

    for ci, kol in enumerate(kols_do_wykresu):
        w = wyniki["kolumny"][kol]
        stats_R = w["stats"]
        kolor = KOLORY[ci]

        # ── Wiersz 1: oryginalne dane + ZDARZENIA ──────────────────
        ax1 = fig.add_subplot(gs[0, ci])
        ax1.set_facecolor('#17171f')
        ax1.plot(df["cykl"], df[kol], color=kolor, lw=0.8, alpha=0.9, label='ciągła ewolucja')
        
        # Zaznacz punkty Poincarégo na trajektorii
        if len(idx_poincare) > 0:
            ax1.scatter(df["cykl"].iloc[idx_poincare], df[kol].iloc[idx_poincare], 
                        color='red', s=15, zorder=3, label='Przekrój (Zdarzenia)')

        ax1.set_title(f"{kol} — wartości w czasie", color='#e8e8f0', fontsize=9, pad=4)
        ax1.tick_params(colors='#6b6b80', labelsize=7)
        for spine in ax1.spines.values(): spine.set_color('#333340')
        ax1.set_xlabel("Cykl symulacji", color='#6b6b80', fontsize=7)
        ax1.legend(fontsize=7, facecolor='#17171f', labelcolor='#e8e8f0', framealpha=0.5)

        # ── Wiersz 2: stosunki R(n) ────────────────────────────────
        ax2 = fig.add_subplot(gs[1, ci])
        ax2.set_facecolor('#17171f')
        R = w["R"]
        R_clean = R[~np.isnan(R)]

        if len(R_clean) > 0:
            p5, p95 = np.percentile(R_clean, [5, 95])
            margin = (p95 - p5) * 0.3 + 0.1
            ax2.set_ylim(p5 - margin, p95 + margin)

        ax2.plot(R, color=kolor, marker='o', markersize=3, lw=0.6, alpha=0.7)
        ax2.axhline(stats_R["mediana"], color='#ffffff', lw=1.5, ls='--', alpha=0.7,
                    label=f"mediana={stats_R['mediana']:+.3f}")
        ax2.axhline(0, color='#555555', lw=0.5)
        ax2.set_title(f"R(n) = δ(n+1)/δ(n) dla ZDARZEŃ  CV={stats_R['cv']:.3f}",
                      color='#e8e8f0', fontsize=9, pad=4)
        ax2.legend(fontsize=7, facecolor='#17171f', labelcolor='#e8e8f0', framealpha=0.5)
        ax2.tick_params(colors='#6b6b80', labelsize=7)
        for spine in ax2.spines.values(): spine.set_color('#333340')
        ax2.set_xlabel("Numer Zdarzenia (n)", color='#6b6b80', fontsize=7)

        # ── Wiersz 3: autokorelacja R ──────────────────────────────
        ax3 = fig.add_subplot(gs[2, ci])
        ax3.set_facecolor('#17171f')
        lagi, autokor = w["lagi"], w["autokor"]
        
        if lagi and autokor:
            ax3.plot(lagi, autokor, color=kolor, lw=0.8)
            ax3.axhline(0, color='#555555', lw=0.5)
            n_pts = len(R_clean)
            prog = 1.96 / np.sqrt(n_pts) if n_pts > 0 else 0.1
            ax3.axhline( prog, color='#ffffff', lw=0.5, ls=':', alpha=0.4)
            ax3.axhline(-prog, color='#ffffff', lw=0.5, ls=':', alpha=0.4)

        ax3.set_title(f"Autokorelacja R(n)", color='#e8e8f0', fontsize=9, pad=4)
        ax3.tick_params(colors='#6b6b80', labelsize=7)
        for spine in ax3.spines.values(): spine.set_color('#333340')
        ax3.set_xlabel("Lag [zdarzenia]", color='#6b6b80', fontsize=7)
        ax3.set_ylim(-1.1, 1.1)

    fig.suptitle(f"Analiza Przekroju Poincarégo — {nazwa}", color='#e8e8f0', fontsize=12, y=0.98)
    plik = os.path.join(folder, f"analiza_{nazwa}.png")
    plt.savefig(plik, dpi=120, bbox_inches='tight', facecolor='#0f0f13')
    plt.close()
    print(f"  → Wykres: {plik}")

def rysuj_porownanie(wyniki_porownania, wyniki_wszystkich, folder=FOLDER_WYNIKOW):
    os.makedirs(folder, exist_ok=True)
    kols = list(wyniki_porownania.keys())
    if not kols: return

    fig, axes = plt.subplots(1, len(kols), figsize=(5*len(kols), 5))
    fig.patch.set_facecolor('#0f0f13')
    if len(kols) == 1: axes = [axes]

    for ai, kol in enumerate(kols):
        ax = axes[ai]
        ax.set_facecolor('#17171f')

        wp = wyniki_porownania[kol]
        wartosci = wp["wartosci"]

        ax.boxplot(wartosci, patch_artist=True,
                   boxprops=dict(facecolor='#1e1e28', color='#4a9eff'),
                   medianprops=dict(color='#4affb0', lw=2),
                   whiskerprops=dict(color='#6b6b80'),
                   capprops=dict(color='#6b6b80'),
                   flierprops=dict(marker='o', color='#ff6b4a', ms=4))

        x_jitter = np.random.normal(1, 0.05, len(wartosci))
        ax.scatter(x_jitter, wartosci, color='#7c6aff', alpha=0.7, s=30, zorder=5)

        ax.axhline(wp["srednia"], color='#ff6b4a', lw=1.5, ls='--', alpha=0.8,
                   label=f"śr={wp['srednia']:+.3f}")
        ax.axhline(0, color='#555555', lw=0.5)

        rozrzut_pct = wp["rozrzut"] * 100
        ax.set_title(f"[{kol}]  rozrzut={rozrzut_pct:.1f}%", color='#e8e8f0', fontsize=10, pad=6)
        ax.set_ylabel("Mediana R(n)", color='#6b6b80', fontsize=8)
        ax.legend(fontsize=8, facecolor='#17171f', labelcolor='#e8e8f0', framealpha=0.5)
        ax.tick_params(colors='#6b6b80', labelsize=8)
        ax.set_xticklabels([])
        for spine in ax.spines.values(): spine.set_color('#333340')

    fig.suptitle("Uniwersalność stosunków R(n) dla zjawisk w układach", color='#e8e8f0', fontsize=12, y=1.02)
    plik = os.path.join(folder, "porownanie_ukladow.png")
    plt.savefig(plik, dpi=120, bbox_inches='tight', facecolor='#0f0f13')
    plt.close()
    print(f"  → Wykres porównawczy: {plik}")

def rysuj_orbity(dane_ukladu, wyniki, folder=FOLDER_WYNIKOW):
    os.makedirs(folder, exist_ok=True)
    df = dane_ukladu["df"]
    nazwa = dane_ukladu["nazwa"]
    idx_poincare = wyniki.get("idx_poincare", [])

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#0f0f13')
    ax.set_facecolor('#0f0f13')

    for i, (xk, yk, kol) in enumerate([('x1','y1',KOLORY[0]), ('x2','y2',KOLORY[1]), ('x3','y3',KOLORY[2])]):
        ax.plot(df[xk], df[yk], color=kol, lw=0.5, alpha=0.4)
        ax.scatter(df[xk].iloc[-1], df[yk].iloc[-1], color=kol, s=40, zorder=5, label=f"G{i+1} (koniec)")
        
        if len(idx_poincare) > 0:
            ax.scatter(df[xk].iloc[idx_poincare], df[yk].iloc[idx_poincare], 
                       color='white', s=5, alpha=0.8, zorder=6)

    # Legenda dla kropek zdarzeń
    if len(idx_poincare) > 0:
        ax.scatter([], [], color='white', s=20, label='Miejsca zdarzeń')

    ax.set_title(f"Orbity i Miejsca Zdarzeń — {nazwa}", color='#e8e8f0', fontsize=11)
    ax.set_xlabel("x [AU]", color='#6b6b80')
    ax.set_ylabel("y [AU]", color='#6b6b80')
    ax.tick_params(colors='#6b6b80')
    ax.legend(facecolor='#17171f', labelcolor='#e8e8f0', fontsize=9)
    for spine in ax.spines.values(): spine.set_color('#333340')
    ax.set_aspect('equal')

    plik = os.path.join(folder, f"orbity_{nazwa}.png")
    plt.savefig(plik, dpi=100, bbox_inches='tight', facecolor='#0f0f13')
    plt.close()

# ── Zapis raportu tekstowego ─────────────────────────────────────────────────

def zapisz_raport(wyniki_wszystkich, wyniki_porownania, folder=FOLDER_WYNIKOW):
    os.makedirs(folder, exist_ok=True)
    plik = os.path.join(folder, "raport.txt")

    with open(plik, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RAPORT ANALIZY POINCARÉGO — PROBLEM TRZECH CIAŁ\n")
        f.write("=" * 60 + "\n\n")

        f.write("HIPOTEZA: czy stosunek R(n) = δ(n+1)/δ(n) między kolejnymi ZDARZENIAMI (perycentrami) jest stały?\n\n")

        f.write("-" * 60 + "\n")
        f.write("WYNIKI DLA POSZCZEGÓLNYCH UKŁADÓW:\n\n")

        for w in wyniki_wszystkich:
            if w["n_zdarzen"] < 10: continue
            f.write(f"[{w['nazwa']}] (zdarzeń: {w['n_zdarzen']})\n")
            for kol, dane_kol in w["kolumny"].items():
                s = dane_kol["stats"]
                f.write(f"  {kol}: R_med={s['mediana']:+.4f}  "
                        f"std={s['std']:.4f}  CV={s['cv']:.3f}  {dane_kol['ocena']}\n")
                
                # Dodajemy info o wzorze do raportu, jeśli udało się dopasować falę
                dop = dane_kol.get("dopasowanie")
                if dop and dop["R2"] > 0.3:
                    f.write(f"    --> WYKRYTO ODDECH: f(t)={dop['A']:.2f}*sin(ωt+φ)+{dop['C']:.2f} (Okres: {dop['okres_lat']:.1f} lat, R²: {dop['R2']*100:.1f}%)\n")

            f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("ANALIZA UNIWERSALNOŚCI:\n\n")

        for kol, wp in wyniki_porownania.items():
            rozrzut_pct = wp["rozrzut"] * 100
            f.write(f"[{kol}]  śr R={wp['srednia']:+.4f}  std={wp['std']:.4f}  rozrzut={rozrzut_pct:.1f}%\n")
        f.write("\n")

    print(f"  → Raport: {plik}")
    return plik

# ── Główna funkcja ───────────────────────────────────────────────────────────

def main():
    os.makedirs(FOLDER_WYNIKOW, exist_ok=True)

    print("\n" + "=" * 57)
    print("  ANALYZER — SZUKANIE WZORCÓW (METODA POINCARÉGO)")
    print("=" * 57)

    # --- NOWE: Pytanie o generowanie indywidualnych plików PNG ---
    wybor = input("\n  Czy generować indywidualne wykresy PNG dla każdego układu? (T/N) [Domyślnie: N]: ").strip().upper()
    generuj_indywidualne_png = (wybor == 'T')

    print(f"\n  Wczytuję dane z '{FOLDER_DANYCH}/'...\n")
    wszystkie_dane = wczytaj_dane(FOLDER_DANYCH)

    if not wszystkie_dane: return

    print("=" * 57)
    print("  ANALIZA ZDARZEŃ W UKŁADACH")
    print("=" * 57)

    wyniki_wszystkich = []

    for dane in wszystkie_dane:
        wyniki = analizuj_uklad(dane, verbose=True)
        wyniki_wszystkich.append(wyniki)
        
        # --- Zmiana: Generujemy tylko na życzenie użytkownika ---
        if generuj_indywidualne_png:
            rysuj_uklad(dane, wyniki, FOLDER_WYNIKOW)
            rysuj_orbity(dane, wyniki, FOLDER_WYNIKOW)

    wyniki_porownania = analiza_uniwersalnosci(wyniki_wszystkich)

    # Generowanie podsumowań jest zawsze włączone (zgodnie z prośbą)
    if wyniki_porownania:
        print("\n  Rysuję wykres porównawczy...")
        rysuj_porownanie(wyniki_porownania, wyniki_wszystkich, FOLDER_WYNIKOW)

    print(f"\n{'='*57}")
    print(f"  ZAPIS WYNIKÓW")
    print(f"{'='*57}\n")
    zapisz_raport(wyniki_wszystkich, wyniki_porownania, FOLDER_WYNIKOW)

    print(f"\n{'='*57}")
    print(f"  GOTOWE — wyniki w folderze '{FOLDER_WYNIKOW}/'")
    print(f"{'='*57}\n")

if __name__ == "__main__":
    main()