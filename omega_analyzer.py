# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.
# omega_analyzer.py

"""
OMEGA ANALYZER — KIERUNKI A i B
=================================
Dla każdego układu z data/:
  1. Wczytuje CSV + JSON
  2. Wyznacza zdarzenia Poincarégo (lokalne minima r12)
  3. Dopasowuje trzy modele do r13(t) i r13(n):
       Model 1: A·sin²(ω·t) + C
       Model 2: A·cos(ω·log(t)) + C          (motyw z s67249)
       Model 3: A·sin(ω·n + φ) + C           (prosta sinusoida)
  4. Zapisuje tabelę wyników: omega, A, R², typ układu, masy
  5. Analizuje korelacje omega vs masy, omega vs typ

WYNIKI:
  results/omega_tabela.csv     — surowe dane
  results/omega_raport.txt     — interpretacja
  results/omega_wykresy.png    — 4 wykresy korelacji
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal, optimize
import os, json, glob, warnings
warnings.filterwarnings('ignore')

# ── Stałe ────────────────────────────────────────────────────────────────────

FOLDER_DANYCH   = "data"
FOLDER_WYNIKOW  = "results"
AU    = 1.496e11
M_SUN = 1.989e30
KOLORY_TYP = {"hierarchiczny": "#4a9eff", "sredni": "#ff6b4a"}

# ── Przekrój Poincarégo ───────────────────────────────────────────────────────

def poincare(df, kolumna='r12', distance=3):
    idx, _ = signal.find_peaks(-df[kolumna].values, distance=distance)
    if len(idx) < 15:
        return pd.DataFrame(), idx
    return df.iloc[idx].reset_index(drop=True), idx

# ── Modele do dopasowania ─────────────────────────────────────────────────────

def model_sin2(t, A, omega, C):
    """A·sin²(ω·t) + C"""
    return A * np.sin(omega * t) ** 2 + C

def model_coslog(t, A, omega, C):
    """A·cos(ω·log(t)) + C"""
    t_safe = np.where(t > 0, t, 1e-10)
    return A * np.cos(omega * np.log(t_safe)) + C

def model_sin(n, A, omega, phi, C):
    """A·sin(ω·n + φ) + C"""
    return A * np.sin(omega * n + phi) + C

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-30)

def dopasuj_modele(df_p, kolumna='r13'):
    """
    Próbuje dopasować wszystkie 3 modele.
    Zwraca słownik z wynikami najlepszego + wszystkich.
    """
    if kolumna not in df_p.columns or len(df_p) < 15:
        return None

    t = df_p['czas_lat'].values.astype(float)
    n = np.arange(len(df_p), dtype=float)
    y = df_p[kolumna].values.astype(float)

    C0  = np.mean(y)
    A0  = (np.max(y) - np.min(y)) / 2.0
    if A0 < 1e-6:
        return None

    wyniki = {}

    # --- Model 1: sin²(ω·t) ---
    for omega_guess in [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1]:
        try:
            popt, _ = optimize.curve_fit(
                model_sin2, t, y,
                p0=[A0, omega_guess, C0],
                bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=8000
            )
            y_fit = model_sin2(t, *popt)
            r2 = r_squared(y, y_fit)
            if 'sin2' not in wyniki or r2 > wyniki['sin2']['R2']:
                wyniki['sin2'] = {
                    'model': 'sin2',
                    'A': abs(popt[0]), 'omega': abs(popt[1]), 'phi': 0.0, 'C': popt[2],
                    'R2': r2, 'y_fit': y_fit
                }
        except Exception:
            pass

    # --- Model 2: cos(ω·log(t)) ---
    t_pos = t[t > 0]
    y_pos = y[t > 0]
    if len(t_pos) > 10:
        for omega_guess in [0.5, 1.0, 2.0, 2.09, 3.0, 5.0]:
            try:
                popt, _ = optimize.curve_fit(
                    model_coslog, t_pos, y_pos,
                    p0=[A0, omega_guess, C0],
                    maxfev=8000
                )
                y_fit_full = model_coslog(t, *popt)
                r2 = r_squared(y, y_fit_full)
                if 'coslog' not in wyniki or r2 > wyniki['coslog']['R2']:
                    wyniki['coslog'] = {
                        'model': 'coslog',
                        'A': abs(popt[0]), 'omega': abs(popt[1]), 'phi': 0.0, 'C': popt[2],
                        'R2': r2, 'y_fit': y_fit_full
                    }
            except Exception:
                pass

    # --- Model 3: A·sin(ω·n + φ) + C ---
    for omega_guess in [0.01, 0.05, 0.1, 0.5, 1.0, 5.69, 2*np.pi/10]:
        try:
            popt, _ = optimize.curve_fit(
                model_sin, n, y,
                p0=[A0, omega_guess, 0.0, C0],
                maxfev=8000
            )
            y_fit = model_sin(n, *popt)
            r2 = r_squared(y, y_fit)
            if 'sin_n' not in wyniki or r2 > wyniki['sin_n']['R2']:
                wyniki['sin_n'] = {
                    'model': 'sin_n',
                    'A': abs(popt[0]), 'omega': abs(popt[1]), 'phi': popt[2], 'C': popt[3],
                    'R2': r2, 'y_fit': y_fit
                }
        except Exception:
            pass

    if not wyniki:
        return None

    # Znajdź najlepszy model wg R²
    najlepszy_klucz = max(wyniki, key=lambda k: wyniki[k]['R2'])
    wynik = wyniki[najlepszy_klucz].copy()
    wynik['wszystkie'] = wyniki
    return wynik

# ── Wczytywanie i przetwarzanie ───────────────────────────────────────────────

def przetworz_wszystkie():
    pliki_json = sorted(glob.glob(os.path.join(FOLDER_DANYCH, "*_config.json")))

    print(f"\n{'='*60}")
    print(f"  OMEGA ANALYZER — {len(pliki_json)} układów")
    print(f"{'='*60}\n")

    wiersze = []

    for pj in pliki_json:
        # Znajdź odpowiedni CSV
        baza = pj.replace("_config.json", "")
        plik_csv = baza + ".csv"
        if not os.path.exists(plik_csv):
            print(f"  ✗ Brak CSV dla {os.path.basename(pj)}")
            continue

        # Wczytaj JSON
        try:
            with open(pj) as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"  ✗ Błąd JSON {pj}: {e}")
            continue

        # Wczytaj CSV
        try:
            df = pd.read_csv(plik_csv)
        except Exception as e:
            print(f"  ✗ Błąd CSV {plik_csv}: {e}")
            continue

        # Wyciągnij parametry
        nazwa = os.path.basename(baza)
        typ   = "hierarchiczny" if "hierarchiczny" in nazwa else "sredni"
        seed  = nazwa.split('_s')[1].split('_')[0] if '_s' in nazwa else "?"

        masy  = [m / M_SUN for m in cfg.get('masy', [1, 1, 1])]
        m1, m2, m3 = masy[0], masy[1], masy[2]
        m_total = m1 + m2 + m3
        m_max   = max(masy)
        m_min   = min(masy)
        # Masa gwiazdy "trzeciej" — tej która ma r13 (odległość od g1 do g3)
        # g3 = masy[2]

        # Przekrój Poincarégo
        df_p, _ = poincare(df, 'r12')
        n_zd = len(df_p)

        if n_zd < 15:
            print(f"  ⚠ {seed} ({typ}): za mało zdarzeń ({n_zd})")
            continue

        # Dopasowanie modeli
        wynik = dopasuj_modele(df_p, 'r13')

        if wynik is None:
            print(f"  ✗ {seed} ({typ}): dopasowanie nie powiodło się")
            continue

        model_best = wynik['model']
        omega      = wynik['omega']
        A          = wynik['A']
        C          = wynik['C']
        R2         = wynik['R2']

        # R² dla każdego modelu osobno
        r2_sin2   = wynik['wszystkie'].get('sin2',   {}).get('R2', np.nan)
        r2_coslog = wynik['wszystkie'].get('coslog', {}).get('R2', np.nan)
        r2_sin_n  = wynik['wszystkie'].get('sin_n',  {}).get('R2', np.nan)

        # Omega dla każdego modelu osobno
        o_sin2   = wynik['wszystkie'].get('sin2',   {}).get('omega', np.nan)
        o_coslog = wynik['wszystkie'].get('coslog', {}).get('omega', np.nan)
        o_sin_n  = wynik['wszystkie'].get('sin_n',  {}).get('omega', np.nan)

        # Statystyki r13 na przekroju
        r13_mean = df_p['r13'].mean()
        r13_std  = df_p['r13'].std()

        ikona = "🟢" if R2 > 0.5 else ("🟡" if R2 > 0.2 else "🔴")
        print(f"  {ikona} {seed:6s} ({typ:14s}) | n={n_zd:5d} | "
              f"best={model_best:8s} R²={R2:.3f} | ω={omega:.5f} | "
              f"m3={m3:.2f}M☉")

        wiersze.append({
            'seed': seed, 'typ': typ, 'nazwa': nazwa,
            'm1': round(m1, 3), 'm2': round(m2, 3), 'm3': round(m3, 3),
            'm_total': round(m_total, 3),
            'm_max': round(m_max, 3), 'm_min': round(m_min, 3),
            'n_zdarzen': n_zd,
            'model_best': model_best,
            'omega_best': omega, 'A_best': A, 'C_best': C, 'R2_best': R2,
            'omega_sin2': o_sin2,   'R2_sin2': r2_sin2,
            'omega_coslog': o_coslog, 'R2_coslog': r2_coslog,
            'omega_sin_n': o_sin_n,  'R2_sin_n': r2_sin_n,
            'r13_mean': round(r13_mean, 3), 'r13_std': round(r13_std, 3),
        })

    return pd.DataFrame(wiersze)

# ── Analiza korelacji ─────────────────────────────────────────────────────────

def analizuj_korelacje(df_tab):
    print(f"\n{'='*60}")
    print(f"  KORELACJE OMEGA vs PARAMETRY")
    print(f"{'='*60}\n")

    zmienne = ['m1', 'm2', 'm3', 'm_total', 'm_max', 'm_min']

    # Użyj omega z modelu sin2 (najczęstszy motyw)
    df_ok = df_tab[df_tab['R2_sin2'] > 0.05].copy()
    print(f"  Układów z R²(sin²) > 0.05: {len(df_ok)}/{len(df_tab)}\n")

    wyniki_kor = []
    for zm in zmienne:
        if df_ok[zm].nunique() < 3:
            continue
        r = np.corrcoef(df_ok[zm], df_ok['omega_sin2'])[0, 1]
        wyniki_kor.append({'zmienna': zm, 'korelacja_r': round(r, 4)})
        gwiazdki = "***" if abs(r) > 0.5 else ("**" if abs(r) > 0.3 else ("*" if abs(r) > 0.15 else ""))
        print(f"  ω(sin²) vs {zm:10s}: r = {r:+.4f}  {gwiazdki}")

    # Porównanie typów
    print(f"\n  Średnia ω(sin²) wg typu:")
    for typ in ['hierarchiczny', 'sredni']:
        sub = df_ok[df_ok['typ'] == typ]['omega_sin2']
        if len(sub) > 0:
            print(f"    {typ:15s}: śr={sub.mean():.5f}  med={sub.median():.5f}  "
                  f"std={sub.std():.5f}  n={len(sub)}")

    return pd.DataFrame(wyniki_kor)

# ── Wykresy ───────────────────────────────────────────────────────────────────

def rysuj(df_tab):
    os.makedirs(FOLDER_WYNIKOW, exist_ok=True)

    df_sin2 = df_tab[df_tab['R2_sin2'] > 0.05].copy()
    if len(df_sin2) < 3:
        print("  ⚠ Za mało danych do wykresów (R²>0.05)")
        return

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0f0f13')
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

    def styl_ax(ax, tytul, xlabel, ylabel):
        ax.set_facecolor('#17171f')
        ax.set_title(tytul, color='#e8e8f0', fontsize=10, pad=5)
        ax.set_xlabel(xlabel, color='#6b6b80', fontsize=9)
        ax.set_ylabel(ylabel, color='#6b6b80', fontsize=9)
        ax.tick_params(colors='#6b6b80', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333340')

    kolory = [KOLORY_TYP[t] for t in df_sin2['typ']]

    # 1. ω vs m3
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(df_sin2['m3'], df_sin2['omega_sin2'], c=kolory, s=60, alpha=0.8, zorder=3)
    r = np.corrcoef(df_sin2['m3'], df_sin2['omega_sin2'])[0,1]
    styl_ax(ax1, f'ω(sin²) vs masa gwiazdy 3  [r={r:+.3f}]', 'm3 [M☉]', 'ω')
    # Linia trendu
    try:
        z = np.polyfit(df_sin2['m3'], df_sin2['omega_sin2'], 1)
        xr = np.linspace(df_sin2['m3'].min(), df_sin2['m3'].max(), 50)
        ax1.plot(xr, np.polyval(z, xr), color='#7c6aff', lw=1.5, alpha=0.7)
    except Exception:
        pass

    # 2. ω vs m_total
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(df_sin2['m_total'], df_sin2['omega_sin2'], c=kolory, s=60, alpha=0.8, zorder=3)
    r2 = np.corrcoef(df_sin2['m_total'], df_sin2['omega_sin2'])[0,1]
    styl_ax(ax2, f'ω(sin²) vs masa całkowita  [r={r2:+.3f}]', 'm_total [M☉]', 'ω')
    try:
        z = np.polyfit(df_sin2['m_total'], df_sin2['omega_sin2'], 1)
        xr = np.linspace(df_sin2['m_total'].min(), df_sin2['m_total'].max(), 50)
        ax2.plot(xr, np.polyval(z, xr), color='#7c6aff', lw=1.5, alpha=0.7)
    except Exception:
        pass

    # 3. ω vs m_max/m_min (stosunek mas)
    ax3 = fig.add_subplot(gs[0, 2])
    stosunek = df_sin2['m_max'] / df_sin2['m_min']
    ax3.scatter(stosunek, df_sin2['omega_sin2'], c=kolory, s=60, alpha=0.8, zorder=3)
    r3 = np.corrcoef(stosunek, df_sin2['omega_sin2'])[0,1]
    styl_ax(ax3, f'ω(sin²) vs m_max/m_min  [r={r3:+.3f}]', 'm_max/m_min', 'ω')
    try:
        z = np.polyfit(stosunek, df_sin2['omega_sin2'], 1)
        xr = np.linspace(stosunek.min(), stosunek.max(), 50)
        ax3.plot(xr, np.polyval(z, xr), color='#7c6aff', lw=1.5, alpha=0.7)
    except Exception:
        pass

    # 4. Rozkład ω wg typu
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor('#17171f')
    typy = ['hierarchiczny', 'sredni']
    dane_box = [df_sin2[df_sin2['typ'] == t]['omega_sin2'].dropna().values for t in typy]
    bp = ax4.boxplot(dane_box, patch_artist=True,
                     boxprops=dict(facecolor='#1e1e28'),
                     medianprops=dict(color='#4affb0', lw=2),
                     whiskerprops=dict(color='#6b6b80'),
                     capprops=dict(color='#6b6b80'),
                     flierprops=dict(marker='o', ms=4))
    for patch, t in zip(bp['boxes'], typy):
        patch.set_facecolor(KOLORY_TYP[t])
        patch.set_alpha(0.4)
    for i, (t, d) in enumerate(zip(typy, dane_box)):
        jitter = np.random.normal(i+1, 0.05, len(d))
        ax4.scatter(jitter, d, color=KOLORY_TYP[t], s=30, alpha=0.8, zorder=5)
    ax4.set_xticks([1, 2])
    ax4.set_xticklabels(['hierarchiczny', 'średni'], color='#6b6b80', fontsize=8)
    styl_ax(ax4, 'Rozkład ω wg typu układu', 'typ', 'ω(sin²)')

    # 5. R² rozkład wg modelu
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor('#17171f')
    modele = ['R2_sin2', 'R2_coslog', 'R2_sin_n']
    etykiety = ['sin²(ωt)', 'cos(ω·log)', 'sin(ωn+φ)']
    kol_m = ['#4a9eff', '#4affb0', '#ff6b4a']
    for i, (m, et, kol) in enumerate(zip(modele, etykiety, kol_m)):
        dane = df_tab[m].dropna()
        ax5.hist(dane, bins=15, alpha=0.6, color=kol, label=et,
                 range=(-0.5, 1.0), histtype='step', linewidth=2)
    ax5.legend(fontsize=8, facecolor='#17171f', labelcolor='#e8e8f0', framealpha=0.5)
    styl_ax(ax5, 'Rozkład R² dla trzech modeli', 'R²', 'liczba układów')
    ax5.axvline(0.5, color='#ffffff', lw=0.7, ls='--', alpha=0.4)
    ax5.axvline(0.0, color='#555555', lw=0.5)

    # 6. ω vs n_zdarzen (kontrola jakości — czy omega zależy od długości serii?)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(df_sin2['n_zdarzen'], df_sin2['omega_sin2'], c=kolory, s=60, alpha=0.8, zorder=3)
    r6 = np.corrcoef(df_sin2['n_zdarzen'], df_sin2['omega_sin2'])[0,1]
    styl_ax(ax6, f'ω(sin²) vs długość serii  [r={r6:+.3f}]', 'liczba zdarzeń', 'ω')

    # Legenda kolorów (typ)
    from matplotlib.patches import Patch
    legend_el = [Patch(facecolor=KOLORY_TYP[t], label=t) for t in ['hierarchiczny', 'sredni']]
    fig.legend(handles=legend_el, loc='upper center', ncol=2,
               facecolor='#17171f', labelcolor='#e8e8f0', fontsize=9,
               bbox_to_anchor=(0.5, 0.98))

    fig.suptitle('Omega Analyzer — r13: korelacje ω z parametrami układu',
                 color='#e8e8f0', fontsize=13, y=1.01)

    plik = os.path.join(FOLDER_WYNIKOW, 'omega_wykresy.png')
    plt.savefig(plik, dpi=120, bbox_inches='tight', facecolor='#0f0f13')
    plt.close()
    print(f"\n  → Wykresy: {plik}")

# ── Raport tekstowy ───────────────────────────────────────────────────────────

def zapisz_raport(df_tab, df_kor):
    os.makedirs(FOLDER_WYNIKOW, exist_ok=True)
    plik = os.path.join(FOLDER_WYNIKOW, 'omega_raport.txt')

    with open(plik, 'w', encoding='utf-8') as f:
        f.write('='*60 + '\n')
        f.write('OMEGA ANALYZER — RAPORT\n')
        f.write('Kierunek A: typ układu vs forma wzoru\n')
        f.write('Kierunek B: korelacje ω vs masy\n')
        f.write('='*60 + '\n\n')

        f.write('TABELA WYNIKÓW:\n')
        cols = ['seed', 'typ', 'm1', 'm2', 'm3', 'n_zdarzen',
                'model_best', 'omega_best', 'R2_best',
                'omega_sin2', 'R2_sin2', 'R2_coslog', 'R2_sin_n']
        f.write(df_tab[cols].to_string(index=False))
        f.write('\n\n')

        f.write('KORELACJE ω(sin²) vs masy:\n')
        if not df_kor.empty:
            f.write(df_kor.to_string(index=False))
        f.write('\n\n')

        # Kierunek A — forma wg typu
        f.write('KIERUNEK A — model_best wg typu układu:\n')
        for typ in ['hierarchiczny', 'sredni']:
            sub = df_tab[df_tab['typ'] == typ]['model_best'].value_counts()
            f.write(f'  {typ}: {dict(sub)}\n')
        f.write('\n')

        # Kierunek B — omega wg masy
        df_ok = df_tab[df_tab['R2_sin2'] > 0.05]
        f.write('KIERUNEK B — ω(sin²) wg masy gwiazdy 3:\n')
        if len(df_ok) > 3:
            r = np.corrcoef(df_ok['m3'], df_ok['omega_sin2'])[0,1]
            f.write(f'  Korelacja r(ω, m3) = {r:.4f}\n')
            try:
                z = np.polyfit(df_ok['m3'], df_ok['omega_sin2'], 1)
                f.write(f'  Przybliżenie liniowe: ω ≈ {z[0]:.5f}·m3 + {z[1]:.5f}\n')
            except Exception:
                pass

    print(f"  → Raport: {plik}")
    return plik

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FOLDER_WYNIKOW, exist_ok=True)

    df_tab = przetworz_wszystkie()

    if df_tab.empty:
        print('\n  ✗ Brak wyników — sprawdź folder data/')
        return

    # Zapis tabeli CSV
    plik_csv = os.path.join(FOLDER_WYNIKOW, 'omega_tabela.csv')
    df_tab.to_csv(plik_csv, index=False)
    print(f"\n  → Tabela: {plik_csv}")

    # Analiza korelacji
    df_kor = analizuj_korelacje(df_tab)

    # Wykresy
    print("\n  Rysuję wykresy...")
    rysuj(df_tab)

    # Raport
    print("\n  Zapisuję raport...")
    zapisz_raport(df_tab, df_kor)

    # Podsumowanie w konsoli
    print(f"\n{'='*60}")
    print(f"  PODSUMOWANIE")
    print(f"{'='*60}")
    print(f"  Układów przetworzonych: {len(df_tab)}")
    print(f"  Modele best:")
    print(df_tab['model_best'].value_counts().to_string())
    print(f"\n  R² > 0.5 (dobre dopasowanie): {(df_tab['R2_best'] > 0.5).sum()}/{len(df_tab)}")
    print(f"  R² > 0.2 (akceptowalne):       {(df_tab['R2_best'] > 0.2).sum()}/{len(df_tab)}")

    df_ok = df_tab[df_tab['R2_sin2'] > 0.05]
    if len(df_ok) > 3:
        r_m3 = np.corrcoef(df_ok['m3'], df_ok['omega_sin2'])[0,1]
        print(f"\n  Korelacja ω(sin²) vs m3: r = {r_m3:+.4f}")
        for typ in ['hierarchiczny', 'sredni']:
            sub = df_ok[df_ok['typ'] == typ]['omega_sin2']
            if len(sub) > 0:
                print(f"  ω śr [{typ}]: {sub.mean():.5f} ± {sub.std():.5f}")

    print(f"\n  Wyniki w: {FOLDER_WYNIKOW}/")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()