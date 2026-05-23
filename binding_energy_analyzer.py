# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.
# binding_energy_analyzer.py

"""
BINDING ENERGY ANALYZER
========================
Hipoteza: CV(r13) koreluje ze stosunkiem energii wiązania pary (g1,g2)
do energii wiązania g3 z parą.

Dla każdego układu z data/ wczytuje JSON i CSV, oblicza:

  E_para    = -G·m1·m2 / r12_start       [energia wiązania pary g1-g2]
  E_g3      = -G·(m1+m2)·m3 / r13_start  [energia wiązania g3 z CM pary]
  R_bind    = |E_para| / |E_g3|           [stosunek: im większy, tym para
                                            mocniej związana względem g3]
  lambda    = r13_start / r12_start       [stosunek odległości]
  mu        = m3 / (m1+m2)               [stosunek mas]

Następnie sprawdza korelacje tych wielkości z:
  r13_cv    — zmienność r13 (std/mean) w trakcie symulacji
  r13_mean  — średnia r13 w trakcie symulacji

Zapisuje:
  results/binding_tabela.csv
  results/binding_raport.txt
  results/binding_wykresy.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats, signal
import os, json, glob
import warnings
warnings.filterwarnings('ignore')

# ── Stałe ────────────────────────────────────────────────────────────────────

G     = 6.674e-11
AU    = 1.496e11
M_SUN = 1.989e30

FOLDER_DANYCH  = "data"
FOLDER_WYNIKOW = "results"
KOLORY_TYP = {"hierarchiczny": "#4a9eff", "sredni": "#ff6b4a"}

# ── Przekrój Poincarégo ───────────────────────────────────────────────────────

def poincare_stats(df, kolumna_trigger='r12', kolumna_cel='r13', distance=3):
    idx, _ = signal.find_peaks(-df[kolumna_trigger].values, distance=distance)
    if len(idx) < 5:
        return None
    y = df[kolumna_cel].values[idx]
    return {
        'mean': np.mean(y),
        'std':  np.std(y),
        'cv':   np.std(y) / np.mean(y) if np.mean(y) > 0 else np.nan,
        'n':    len(idx),
        'min':  np.min(y),
        'max':  np.max(y),
    }

# ── Wczytywanie ───────────────────────────────────────────────────────────────

def wczytaj_wszystkie(folder=FOLDER_DANYCH):
    pliki_json = sorted(glob.glob(os.path.join(folder, "*_config.json")))
    wiersze = []

    print(f"\n{'='*60}")
    print(f"  BINDING ENERGY ANALYZER — {len(pliki_json)} układów")
    print(f"{'='*60}\n")

    for pj in pliki_json:
        baza    = pj.replace("_config.json", "")
        plik_csv = baza + ".csv"
        if not os.path.exists(plik_csv):
            continue

        try:
            with open(pj) as f:
                cfg = json.load(f)
        except Exception:
            continue

        try:
            df = pd.read_csv(plik_csv)
        except Exception:
            continue

        nazwa = os.path.basename(baza)
        typ   = "hierarchiczny" if "hierarchiczny" in nazwa else "sredni"
        seed  = nazwa.split('_s')[1].split('_')[0] if '_s' in nazwa else "?"

        # Masy
        masy  = [m / M_SUN for m in cfg.get('masy', [1, 1, 1])]
        m1, m2, m3 = masy
        m12 = m1 + m2

        # Pozycje startowe — z JSON
        pozycje = cfg.get('pozycje', None)
        if pozycje is None:
            continue
        pos = np.array(pozycje)  # w metrach

        r12_start = np.linalg.norm(pos[1] - pos[0]) / AU
        r13_start = np.linalg.norm(pos[2] - pos[0]) / AU
        r23_start = np.linalg.norm(pos[2] - pos[1]) / AU

        if r12_start < 1e-6 or r13_start < 1e-6:
            continue

        # Energie wiązania [J] — wartości ujemne, bierzemy moduł
        E_para = G * (m1 * M_SUN) * (m2 * M_SUN) / (r12_start * AU)
        E_g3   = G * (m12 * M_SUN) * (m3 * M_SUN) / (r13_start * AU)
        E_g3_12 = G * (m1 * M_SUN) * (m3 * M_SUN) / (r13_start * AU) + \
                  G * (m2 * M_SUN) * (m3 * M_SUN) / (r23_start * AU)

        # Parametry bezwymiarowe
        R_bind  = E_para / E_g3        # im > 1, tym para mocniej związana niż g3
        lambda_ = r13_start / r12_start # stosunek odległości
        mu      = m3 / m12              # stosunek mas

        # Statystyki r13 z symulacji (przekrój Poincarégo)
        st = poincare_stats(df, 'r12', 'r13')
        if st is None:
            continue

        # Statystyki r12 (jak bardzo para oscyluje)
        st12 = poincare_stats(df, 'r12', 'r12')
        r12_cv = st12['cv'] if st12 else np.nan

        print(f"  ✓ {seed:6s} ({typ:14s}) | "
              f"R_bind={R_bind:.3f}  λ={lambda_:.2f}  μ={mu:.2f}  "
              f"CV(r13)={st['cv']:.3f}")

        wiersze.append({
            'seed': seed, 'typ': typ, 'nazwa': nazwa,
            'm1': round(m1, 3), 'm2': round(m2, 3), 'm3': round(m3, 3),
            'm12': round(m12, 3), 'm_total': round(m1+m2+m3, 3),
            'mu': round(mu, 4),          # m3 / (m1+m2)
            'r12_start': round(r12_start, 3),
            'r13_start': round(r13_start, 3),
            'r23_start': round(r23_start, 3),
            'lambda': round(lambda_, 4),  # r13/r12
            'E_para': round(E_para, 6),
            'E_g3':   round(E_g3, 6),
            'E_g3_12': round(E_g3_12, 6),
            'R_bind': round(R_bind, 6),   # |E_para|/|E_g3|
            'log_R_bind': round(np.log(R_bind), 4) if R_bind > 0 else np.nan,
            'log_lambda': round(np.log(lambda_), 4),
            'r13_mean': round(st['mean'], 3),
            'r13_std':  round(st['std'], 3),
            'r13_cv':   round(st['cv'], 4),
            'r13_min':  round(st['min'], 3),
            'r13_max':  round(st['max'], 3),
            'r13_n':    st['n'],
            'r12_cv':   round(r12_cv, 4) if not np.isnan(r12_cv) else np.nan,
        })

    return pd.DataFrame(wiersze)

# ── Analiza korelacji ─────────────────────────────────────────────────────────

def analizuj(df):
    print(f"\n{'='*60}")
    print(f"  KORELACJE Z r13_cv")
    print(f"{'='*60}\n")

    cele = ['r13_cv', 'r13_mean']
    predyktory = ['R_bind', 'log_R_bind', 'lambda', 'log_lambda',
                  'mu', 'm1', 'm2', 'm3', 'm_total',
                  'r12_start', 'r13_start', 'r12_cv']

    wyniki = []
    for cel in cele:
        print(f"  Zmienna zależna: {cel}\n")
        for pred in predyktory:
            col = df[pred].dropna()
            cel_col = df.loc[col.index, cel].dropna()
            idx = col.index.intersection(cel_col.index)
            if len(idx) < 5:
                continue
            x, y = df.loc[idx, pred].values, df.loc[idx, cel].values
            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)
            flag = " ★★★" if abs(r_s) > 0.5 else (" ★★" if abs(r_s) > 0.35 else (" ★" if abs(r_s) > 0.2 else ""))
            print(f"    {pred:15s}  Pearson={r_p:+.3f}(p={p_p:.3f})  "
                  f"Spearman={r_s:+.3f}(p={p_s:.3f}){flag}")
            wyniki.append({
                'cel': cel, 'predyktor': pred,
                'r_pearson': round(r_p, 4), 'p_pearson': round(p_p, 4),
                'r_spearman': round(r_s, 4), 'p_spearman': round(p_s, 4),
            })
        print()

    return pd.DataFrame(wyniki)

# ── Wykresy ───────────────────────────────────────────────────────────────────

def rysuj(df):
    os.makedirs(FOLDER_WYNIKOW, exist_ok=True)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0f0f13')
    gs = gridspec.GridSpec(2, 4, hspace=0.45, wspace=0.38)

    def styl(ax, title, xlabel, ylabel):
        ax.set_facecolor('#17171f')
        ax.set_title(title, color='#e8e8f0', fontsize=9, pad=5)
        ax.set_xlabel(xlabel, color='#6b6b80', fontsize=8)
        ax.set_ylabel(ylabel, color='#6b6b80', fontsize=8)
        ax.tick_params(colors='#6b6b80', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#333340')

    def scatter_z_trendem(ax, x_col, y_col, xlabel, ylabel, title, log_x=False, log_y=False):
        kolory = [KOLORY_TYP[t] for t in df['typ']]
        x = df[x_col].values
        y = df[y_col].values
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y, kol = x[mask], y[mask], np.array(kolory)[mask]

        if log_x: x_plot = np.log10(x + 1e-10)
        else:      x_plot = x
        if log_y: y_plot = np.log10(y + 1e-10)
        else:      y_plot = y

        ax.scatter(x_plot, y_plot, c=kol, s=40, alpha=0.8, zorder=3)

        # Linia trendu
        if len(x_plot) > 3:
            try:
                z = np.polyfit(x_plot, y_plot, 1)
                xr = np.linspace(x_plot.min(), x_plot.max(), 50)
                ax.plot(xr, np.polyval(z, xr), color='#7c6aff', lw=1.5, alpha=0.8)
                r_s, _ = stats.spearmanr(x_plot, y_plot)
                ax.text(0.05, 0.93, f'r_S={r_s:+.3f}',
                        transform=ax.transAxes, color='#7c6aff', fontsize=8)
            except Exception:
                pass

        styl(ax, title, xlabel, ylabel)

    # Rząd 1: CV(r13) vs różne predyktory
    scatter_z_trendem(fig.add_subplot(gs[0, 0]),
        'R_bind', 'r13_cv', 'R_bind = |E_para|/|E_g3|', 'CV(r13)',
        'CV(r13) vs R_bind')

    scatter_z_trendem(fig.add_subplot(gs[0, 1]),
        'log_R_bind', 'r13_cv', 'log(R_bind)', 'CV(r13)',
        'CV(r13) vs log(R_bind)')

    scatter_z_trendem(fig.add_subplot(gs[0, 2]),
        'lambda', 'r13_cv', 'λ = r13_start / r12_start', 'CV(r13)',
        'CV(r13) vs λ (stosunek odległości)')

    scatter_z_trendem(fig.add_subplot(gs[0, 3]),
        'mu', 'r13_cv', 'μ = m3 / (m1+m2)', 'CV(r13)',
        'CV(r13) vs μ (stosunek mas)')

    # Rząd 2: r13_mean vs predyktory
    scatter_z_trendem(fig.add_subplot(gs[1, 0]),
        'R_bind', 'r13_mean', 'R_bind', 'r13_mean [AU]',
        'r13_mean vs R_bind')

    scatter_z_trendem(fig.add_subplot(gs[1, 1]),
        'lambda', 'r13_mean', 'λ = r13_start / r12_start', 'r13_mean [AU]',
        'r13_mean vs λ')

    scatter_z_trendem(fig.add_subplot(gs[1, 2]),
        'r13_start', 'r13_mean', 'r13_start [AU]', 'r13_mean [AU]',
        'r13_mean vs r13_start ← kluczowy')

    scatter_z_trendem(fig.add_subplot(gs[1, 3]),
        'r12_start', 'r13_mean', 'r12_start [AU]', 'r13_mean [AU]',
        'r13_mean vs r12_start')

    # Legenda
    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(facecolor=KOLORY_TYP[t], label=t)
                 for t in ['hierarchiczny', 'sredni']],
        loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.99),
        facecolor='#17171f', labelcolor='#e8e8f0', fontsize=9
    )
    fig.suptitle('Binding Energy Analyzer — r13_cv i r13_mean vs parametry fizyczne',
                 color='#e8e8f0', fontsize=12, y=1.02)

    plik = os.path.join(FOLDER_WYNIKOW, 'binding_wykresy.png')
    plt.savefig(plik, dpi=120, bbox_inches='tight', facecolor='#0f0f13')
    plt.close()
    print(f"  → Wykresy: {plik}")

# ── Raport ────────────────────────────────────────────────────────────────────

def zapisz_raport(df, df_kor):
    os.makedirs(FOLDER_WYNIKOW, exist_ok=True)

    plik = os.path.join(FOLDER_WYNIKOW, 'binding_raport.txt')
    with open(plik, 'w', encoding='utf-8') as f:
        f.write('='*60 + '\n')
        f.write('BINDING ENERGY ANALYZER — RAPORT\n')
        f.write('='*60 + '\n\n')

        f.write('HIPOTEZA: CV(r13) koreluje z R_bind = |E_para|/|E_g3|\n')
        f.write('gdzie E_para = energia wiązania pary (g1,g2)\n')
        f.write('      E_g3   = energia wiązania g3 z CM pary\n\n')

        f.write('PARAMETRY FIZYCZNE UKŁADÓW:\n')
        cols = ['seed','typ','m1','m2','m3','r12_start','r13_start',
                'lambda','mu','R_bind','r13_cv','r13_mean']
        f.write(df[cols].to_string(index=False))
        f.write('\n\n')

        f.write('NAJSILNIEJSZE KORELACJE Z r13_cv:\n')
        sub = df_kor[df_kor['cel']=='r13_cv'].sort_values('r_spearman', key=abs, ascending=False)
        f.write(sub[['predyktor','r_pearson','p_pearson','r_spearman','p_spearman']].to_string(index=False))
        f.write('\n\n')

        f.write('NAJSILNIEJSZE KORELACJE Z r13_mean:\n')
        sub2 = df_kor[df_kor['cel']=='r13_mean'].sort_values('r_spearman', key=abs, ascending=False)
        f.write(sub2[['predyktor','r_pearson','p_pearson','r_spearman','p_spearman']].to_string(index=False))
        f.write('\n\n')

        # Wniosek
        top_cv = sub.iloc[0]
        f.write('WNIOSEK:\n')
        f.write(f"  Najsilniejsza korelacja z r13_cv: {top_cv['predyktor']} "
                f"(r_S={top_cv['r_spearman']:+.3f}, p={top_cv['p_spearman']:.3f})\n")

        # Regresja liniowa r13_mean ~ r13_start
        mask = df['r13_start'].notna() & df['r13_mean'].notna()
        sl, ic, r, p, _ = stats.linregress(df.loc[mask,'r13_start'], df.loc[mask,'r13_mean'])
        f.write(f"\n  r13_mean ≈ {sl:.3f} · r13_start + {ic:.2f}  "
                f"(R²={r**2:.3f}, p={p:.4f})\n")

    print(f"  → Raport: {plik}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FOLDER_WYNIKOW, exist_ok=True)

    df = wczytaj_wszystkie()

    if df.empty:
        print('  ✗ Brak danych — sprawdź folder data/')
        return

    plik_csv = os.path.join(FOLDER_WYNIKOW, 'binding_tabela.csv')
    df.to_csv(plik_csv, index=False)
    print(f"\n  → Tabela: {plik_csv}")

    df_kor = analizuj(df)

    print("  Rysuję wykresy...")
    rysuj(df)

    print("  Zapisuję raport...")
    zapisz_raport(df, df_kor)

    # Podsumowanie w konsoli
    print(f"\n{'='*60}")
    print(f"  PODSUMOWANIE")
    print(f"{'='*60}")

    print(f"\n  Układów przetworzonych: {len(df)}")
    print(f"  Hierarchicznych: {(df['typ']=='hierarchiczny').sum()}")
    print(f"  Średnich:        {(df['typ']=='sredni').sum()}")

    print(f"\n  Top korelacje z r13_cv (Spearman):")
    df_cv = df_kor[df_kor['cel']=='r13_cv'].sort_values('r_spearman', key=abs, ascending=False)
    for _, row in df_cv.head(5).iterrows():
        stars = "★★★" if abs(row['r_spearman']) > 0.5 else ("★★" if abs(row['r_spearman']) > 0.35 else "★")
        print(f"    {stars} {row['predyktor']:15s}: r_S={row['r_spearman']:+.3f}  p={row['p_spearman']:.3f}")

    print(f"\n  Top korelacje z r13_mean (Spearman):")
    df_m = df_kor[df_kor['cel']=='r13_mean'].sort_values('r_spearman', key=abs, ascending=False)
    for _, row in df_m.head(5).iterrows():
        stars = "★★★" if abs(row['r_spearman']) > 0.5 else ("★★" if abs(row['r_spearman']) > 0.35 else "★")
        print(f"    {stars} {row['predyktor']:15s}: r_S={row['r_spearman']:+.3f}  p={row['p_spearman']:.3f}")

    # Regresja r13_mean ~ r13_start
    mask = df['r13_start'].notna() & df['r13_mean'].notna()
    if mask.sum() > 3:
        sl, ic, r, p, _ = stats.linregress(
            df.loc[mask,'r13_start'], df.loc[mask,'r13_mean'])
        print(f"\n  Regresja r13_mean ~ r13_start:")
        print(f"    r13_mean = {sl:.3f} · r13_start + {ic:.2f}")
        print(f"    R² = {r**2:.3f}  p = {p:.4f}")

    print(f"\n  Wyniki w: {FOLDER_WYNIKOW}/")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()