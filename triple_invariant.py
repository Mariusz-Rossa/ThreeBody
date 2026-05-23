# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.
# triple_invariant.py
 
"""
TRIPLE INVARIANT SCAN
=====================
Bada trzy kandydatów na niezmienniki jednocześnie na siatce parametrów.

Kandydaci:
  1. CV(r13)           — rozrzut odległości g3, kandydat ≈ 0.23
  2. r13_mean/r13_start — współczynnik skurczenia, kandydat ≈ 0.78
  3. reżim dynamiczny  — pct_pos > 50% (reżim +) vs < 50% (reżim -)

Siatka:
  m_para = 1, 5, 10, 20, 50 M☉  (m1=m2, symetryczna para)
  r12    = 3, 6, 10, 20 AU       (różne separacje pary)
  r13    = 200 AU (stałe)
  m3     = 1 M☉  (stałe)

Łącznie: 5 × 4 = 20 układów

Pytania:
  - Czy CV(r13) ≈ 0.23 trzyma się dla wszystkich mas pary?
  - Czy r13_mean/r13_start ≈ 0.78 trzyma się dla wszystkich r12?
  - Czy separacja krytyczna (sep*) zależy od m_para?
  - Które niezmienniki psują się razem → wspólna przyczyna?

Wyniki → results/triple_invariant_raport.txt
         results/triple_invariant_wykres.png
         results/triple_invariant_tabela.csv

Uruchomienie:
    cd ~/Documents/three_body
    source venv/bin/activate
    python triple_invariant.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator import symuluj, oblicz_energie, G, AU, YEAR, DAY, M_SUN

# ── Siatka parametrów ────────────────────────────────────────────────────────

M_PARA_LISTA = [1.0, 5.0, 10.0, 20.0, 50.0]   # M☉ — masa każdej gwiazdy pary
R12_LISTA    = [3.0, 6.0, 10.0, 20.0]           # AU
R13          = 200.0                             # AU — stałe
M3           = 1.0                              # M☉ — stałe
N_CYKLI      = 35_000

# Separacja krytyczna znaleziona dla m_para=50: sep* ≈ 23×
# Dla innych mas może być inna — to sprawdzamy

# ── Budowanie konfiguracji ───────────────────────────────────────────────────

def zbuduj_config(m_para, r12_au, nazwa):
    m1 = m_para * M_SUN
    m2 = m_para * M_SUN
    m3 = M3 * M_SUN
    M_12  = m1 + m2
    M_tot = m1 + m2 + m3

    r12_m = r12_au * AU
    r13_m = R13 * AU

    # Para na osi X (symetrycznie), g3 na osi Y
    pos = np.array([
        [-r12_m / 2,  0.0],
        [ r12_m / 2,  0.0],
        [ 0.0,        r13_m],
    ])
    r_cm = (m1*pos[0] + m2*pos[1] + m3*pos[2]) / M_tot
    pos -= r_cm

    # Prędkości kepleriańskie
    v_para = np.sqrt(G * M_12 / r12_m)
    v3_mag = np.sqrt(G * M_12 / r13_m) * 0.80

    vel = np.array([
        [0.0,    v_para / 2],
        [0.0,   -v_para / 2],
        [v3_mag, 0.0],
    ])
    p_cm = m1*vel[0] + m2*vel[1] + m3*vel[2]
    vel -= p_cm / M_tot

    # Sprawdź związanie
    E = oblicz_energie(pos, vel, np.array([m1, m2, m3]))
    if E >= 0:
        vel[2] *= 0.65
        p_cm = m1*vel[0] + m2*vel[1] + m3*vel[2]
        vel -= p_cm / M_tot

    # dt: 200 kroków na okres pary
    T_para_dni = 2 * np.pi * np.sqrt(r12_m**3 / (G * M_12)) / DAY
    dt_auto    = min(0.5, T_para_dni / 200)
    dt_auto    = max(dt_auto, 0.001)

    # kpc: ~1/200 okresu g3
    T_g3_dni = 2 * np.pi * np.sqrt(r13_m**3 / (G * M_12)) / DAY
    kpc = max(10, int(T_g3_dni / dt_auto / 200))
    kpc = min(kpc, 500)

    return {
        "nazwa":            nazwa,
        "masy":             [m1, m2, m3],
        "pozycje":          pos.tolist(),
        "predkosci":        vel.tolist(),
        "dt":               DAY * dt_auto,
        "n_cykli":          N_CYKLI,
        "kroki_na_cykl":    kpc,
        "max_blad_energii": 1e-2,
        "max_r_au":         10_000,
    }, dt_auto, kpc


# ── Analiza ──────────────────────────────────────────────────────────────────

def zdarzenia_poincare(df):
    r12 = df['r12'].values
    zdarz = []
    for i in range(1, len(r12) - 1):
        if r12[i] < r12[i-1] and r12[i] < r12[i+1]:
            zdarz.append(i)
    return zdarz


def analiza_R_med(vals):
    delty = np.diff(vals)
    mask  = np.abs(delty[:-1]) > 1e-10
    if np.sum(mask) < 10:
        return None
    R_n = delty[1:][mask] / delty[:-1][mask]
    R_n = R_n[np.abs(R_n) < 100]
    if len(R_n) < 5:
        return None
    return {
        "med":     np.median(R_n),
        "std":     np.std(R_n),
        "pct_pos": float(np.mean(R_n > 0) * 100),
        "n":       len(R_n),
    }


def przeanalizuj(df, r13_start):
    zdarz   = zdarzenia_poincare(df)
    n_zdarz = len(zdarz)

    if n_zdarz < 20:
        return None

    r13_z = df['r13'].values[zdarz]
    r23_z = df['r23'].values[zdarz]
    v3_z  = df['v3'].values[zdarz]

    res13 = analiza_R_med(r13_z)
    res23 = analiza_R_med(r23_z)
    resv3 = analiza_R_med(v3_z)

    # ── Niezmiennik 1: CV(r13) ──
    cv_r13 = np.std(r13_z) / np.mean(r13_z) if np.mean(r13_z) > 0 else np.nan

    # ── Niezmiennik 2: r13_mean / r13_start ──
    r13_mean  = df['r13'].mean()
    ratio_r13 = r13_mean / r13_start

    # ── Niezmiennik 3: reżim dynamiczny (pct_pos) ──
    pct_pos   = res13["pct_pos"] if res13 else np.nan
    rezim     = "+" if pct_pos >= 50 else "-"

    # ── Dodatkowe ──
    R_med_r13 = res13["med"] if res13 else np.nan
    R_med_r23 = res23["med"] if res23 else np.nan
    R_med_v3  = resv3["med"] if resv3 else np.nan
    cv_r12    = np.std(df['r12'].values[zdarz]) / \
                np.mean(df['r12'].values[zdarz])

    return {
        # Trzy kandydaci na niezmienniki
        "cv_r13":       cv_r13,
        "ratio_r13":    ratio_r13,
        "pct_pos":      pct_pos,
        "rezim":        rezim,
        # Pomocnicze
        "R_med_r13":    R_med_r13,
        "R_med_r23":    R_med_r23,
        "R_med_v3":     R_med_v3,
        "r13_mean":     r13_mean,
        "r13_start":    r13_start,
        "cv_r12":       cv_r12,
        "n_zdarzen":    n_zdarz,
    }


# ── Główna pętla ─────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("data",    exist_ok=True)

    wyniki = []
    total  = len(M_PARA_LISTA) * len(R12_LISTA)
    done   = 0

    print("\n" + "="*65)
    print("  TRIPLE INVARIANT SCAN")
    print(f"  m3={M3} M☉, r13={R13} AU (stałe)")
    print(f"  m_para: {M_PARA_LISTA} M☉")
    print(f"  r12:    {R12_LISTA} AU")
    print(f"  Łącznie: {total} układów × {N_CYKLI} cykli")
    print("="*65)

    for m_para in M_PARA_LISTA:
        mu = M3 / (2 * m_para)
        print(f"\n{'━'*65}")
        print(f"  m_para={m_para} M☉  (mu=m3/M_para={mu:.4f})")
        print(f"{'━'*65}")

        for r12 in R12_LISTA:
            done += 1
            sep   = R13 / r12
            nazwa = f"tri_mp{m_para:.0f}_r12_{r12:.0f}AU"

            print(f"\n  [{done:2d}/{total}] r12={r12:5.1f} AU  "
                  f"sep={sep:.1f}×  ", end="", flush=True)

            cfg, dt_auto, kpc = zbuduj_config(m_para, r12, nazwa)
            print(f"dt={dt_auto:.4f}d kpc={kpc}  ", end="", flush=True)

            df, powod = symuluj(cfg, folder="data", cicho=True)

            if powod is not None or df.empty:
                print(f"✗ {powod}")
                wyniki.append({
                    "m_para": m_para, "r12": r12, "sep": sep,
                    "mu": mu, "status": powod or "blad",
                    "cv_r13": np.nan, "ratio_r13": np.nan,
                    "pct_pos": np.nan, "rezim": "?",
                    "R_med_r13": np.nan, "n_zdarzen": 0,
                })
                continue

            wynik = przeanalizuj(df, R13)

            if wynik is None:
                print(f"✗ za mało zdarzeń")
                wyniki.append({
                    "m_para": m_para, "r12": r12, "sep": sep,
                    "mu": mu, "status": "malo_zdarzen",
                    "cv_r13": np.nan, "ratio_r13": np.nan,
                    "pct_pos": np.nan, "rezim": "?",
                    "R_med_r13": np.nan, "n_zdarzen": 0,
                })
                continue

            print(f"✓  CV={wynik['cv_r13']:.4f}  "
                  f"ratio={wynik['ratio_r13']:.4f}  "
                  f"pct_pos={wynik['pct_pos']:.1f}%  "
                  f"reżim={wynik['rezim']}  "
                  f"zdarz={wynik['n_zdarzen']}")

            wynik.update({
                "m_para": m_para, "r12": r12, "sep": sep,
                "mu": mu, "status": "ok",
                "blad_E": df['blad_energii'].iloc[-1],
                "n_cykli": len(df),
            })
            wyniki.append(wynik)

    # ── Zapisz ───────────────────────────────────────────────────────────────
    tabela = pd.DataFrame(wyniki)
    tabela.to_csv("results/triple_invariant_tabela.csv", index=False)

    raport(tabela)
    rysuj(tabela)

    print("\n  Tabela  → results/triple_invariant_tabela.csv")
    print("  Raport  → results/triple_invariant_raport.txt")
    print("  Wykres  → results/triple_invariant_wykres.png")
    print("\nGOTOWE.\n")


# ── Raport ───────────────────────────────────────────────────────────────────

def raport(tabela):
    ok = tabela[tabela["status"] == "ok"].copy()

    linie = []
    linie.append("=" * 75)
    linie.append("TRIPLE INVARIANT SCAN — RAPORT")
    linie.append(f"m3={M3} M☉, r13={R13} AU (stałe)")
    linie.append("Kandydaci: CV(r13), r13_mean/r13_start, reżim (pct_pos>50%)")
    linie.append("=" * 75)

    # ── Tabela główna ──
    linie.append("")
    linie.append(f"  {'m_para':>7} {'r12':>6} {'sep':>6} "
                 f"{'CV(r13)':>9} {'ratio_r13':>10} {'pct_pos':>8} "
                 f"{'reżim':>6} {'R_med_r13':>10} {'n_zdarz':>8}")
    linie.append("  " + "-"*75)

    for _, row in tabela.sort_values(["m_para","r12"]).iterrows():
        if row["status"] != "ok":
            linie.append(f"  {row['m_para']:>7.1f} {row['r12']:>6.1f} "
                         f"{row['sep']:>6.1f}   ✗ {row['status']}")
            continue
        linie.append(
            f"  {row['m_para']:>7.1f} {row['r12']:>6.1f} {row['sep']:>6.1f} "
            f"  {row['cv_r13']:>8.4f}  {row['ratio_r13']:>9.4f} "
            f"  {row['pct_pos']:>7.1f}%  {row['rezim']:>5}  "
            f"  {row['R_med_r13']:>+9.4f}  {int(row['n_zdarzen']):>8}"
        )

    # ── Analiza niezmiennika 1: CV(r13) ──
    linie.append("\n" + "─"*75)
    linie.append("NIEZMIENNIK 1: CV(r13)")
    linie.append("─"*75)
    for m_para in M_PARA_LISTA:
        sub = ok[ok["m_para"] == m_para]["cv_r13"].dropna()
        if len(sub) < 2:
            continue
        linie.append(f"  m_para={m_para:5.1f} M☉:  "
                     f"mean={sub.mean():.4f}  std={sub.std():.4f}  "
                     f"min={sub.min():.4f}  max={sub.max():.4f}  "
                     f"CV_of_CV={sub.std()/sub.mean():.3f}")
    linie.append("")
    all_cv = ok["cv_r13"].dropna()
    if len(all_cv) >= 4:
        linie.append(f"  GLOBALNIE: mean={all_cv.mean():.4f}  "
                     f"std={all_cv.std():.4f}  "
                     f"rozstęp={all_cv.max()-all_cv.min():.4f}")
        if all_cv.std() < 0.03:
            linie.append("  ★★★ CV(r13) jest NIEZMIENNIKIEM globalnym (std<0.03)")
        elif all_cv.std() < 0.06:
            linie.append("  ★★  CV(r13) jest CZĘŚCIOWYM niezmiennikiem (std<0.06)")
        else:
            linie.append("  ✗   CV(r13) NIE jest niezmiennikiem (std≥0.06)")

    # ── Analiza niezmiennika 2: ratio_r13 ──
    linie.append("\n" + "─"*75)
    linie.append("NIEZMIENNIK 2: r13_mean / r13_start")
    linie.append("─"*75)
    for m_para in M_PARA_LISTA:
        sub = ok[ok["m_para"] == m_para]["ratio_r13"].dropna()
        if len(sub) < 2:
            continue
        linie.append(f"  m_para={m_para:5.1f} M☉:  "
                     f"mean={sub.mean():.4f}  std={sub.std():.4f}  "
                     f"min={sub.min():.4f}  max={sub.max():.4f}")
    linie.append("")
    all_ratio = ok["ratio_r13"].dropna()
    if len(all_ratio) >= 4:
        linie.append(f"  GLOBALNIE: mean={all_ratio.mean():.4f}  "
                     f"std={all_ratio.std():.4f}  "
                     f"rozstęp={all_ratio.max()-all_ratio.min():.4f}")
        if all_ratio.std() < 0.02:
            linie.append("  ★★★ ratio_r13 jest NIEZMIENNIKIEM globalnym (std<0.02)")
        elif all_ratio.std() < 0.05:
            linie.append("  ★★  ratio_r13 jest CZĘŚCIOWYM niezmiennikiem (std<0.05)")
        else:
            linie.append("  ✗   ratio_r13 NIE jest niezmiennikiem (std≥0.05)")

    # ── Analiza niezmiennika 3: reżim ──
    linie.append("\n" + "─"*75)
    linie.append("NIEZMIENNIK 3: reżim dynamiczny (pct_pos vs 50%)")
    linie.append("─"*75)
    linie.append("  Mapa reżimów (+ = pct_pos>50%, - = pct_pos<50%):")
    linie.append(f"  {'m_para↓ r12→':>12} " +
                 "  ".join(f"{r12:>6.0f}AU" for r12 in R12_LISTA))
    linie.append("  " + "-"*55)
    for m_para in M_PARA_LISTA:
        row_str = f"  {m_para:>8.1f} M☉  "
        for r12 in R12_LISTA:
            sub = ok[(ok["m_para"]==m_para) & (ok["r12"]==r12)]
            if sub.empty:
                row_str += f"  {'?':>6}"
            else:
                r = sub.iloc[0]
                row_str += f"  {r['rezim']:>3}({r['pct_pos']:.0f}%)"
        linie.append(row_str)

    # Czy separacja krytyczna zależy od m_para?
    linie.append("")
    linie.append("  Separacja krytyczna sep* (gdzie reżim się zmienia):")
    for m_para in M_PARA_LISTA:
        sub = ok[ok["m_para"]==m_para].sort_values("r12")
        rezim_list = sub["rezim"].values
        r12_list   = sub["r12"].values
        pct_list   = sub["pct_pos"].values
        # Znajdź gdzie pct_pos spada poniżej 50
        crossings  = []
        for i in range(len(pct_list)-1):
            if not (np.isnan(pct_list[i]) or np.isnan(pct_list[i+1])):
                if (pct_list[i] >= 50) and (pct_list[i+1] < 50):
                    sep_star = R13 / r12_list[i+1]
                    crossings.append(f"r12∈({r12_list[i]},{r12_list[i+1]}) "
                                     f"→ sep*≈{sep_star:.0f}–"
                                     f"{R13/r12_list[i]:.0f}×")
        if crossings:
            linie.append(f"  m_para={m_para:5.1f}: {'; '.join(crossings)}")
        else:
            linie.append(f"  m_para={m_para:5.1f}: brak wyraźnego przejścia "
                         f"w przeskanowanym zakresie")

    # ── Korelacje między niezmiennikmi ──
    linie.append("\n" + "─"*75)
    linie.append("KORELACJE MIĘDZY KANDYDATAMI")
    linie.append("─"*75)
    if len(ok) >= 6:
        from scipy.stats import spearmanr
        pairs = [
            ("cv_r13", "ratio_r13", "CV(r13) vs ratio_r13"),
            ("cv_r13", "pct_pos",   "CV(r13) vs pct_pos"),
            ("ratio_r13", "pct_pos", "ratio_r13 vs pct_pos"),
        ]
        for c1, c2, label in pairs:
            d1 = ok[c1].dropna()
            d2 = ok[c2].dropna()
            idx = d1.index.intersection(d2.index)
            if len(idx) >= 4:
                r, p = spearmanr(d1[idx], d2[idx])
                linie.append(f"  {label}: r_S={r:+.3f}  p={p:.4f}")

    linie.append("\n" + "="*75)

    tekst = "\n".join(linie)
    print("\n" + tekst)
    with open("results/triple_invariant_raport.txt", "w") as f:
        f.write(tekst)


# ── Wykresy ──────────────────────────────────────────────────────────────────

def rysuj(tabela):
    ok = tabela[tabela["status"] == "ok"].copy()
    if ok.empty:
        return

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0d0d1a')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    def styl(ax, xlabel, ylabel, title):
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444466')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)

    kolory = {
        1.0:  '#ff4444',
        5.0:  '#ff9944',
        10.0: '#ffff44',
        20.0: '#44ff88',
        50.0: '#44aaff',
    }
    markery = {3.0: 'o', 6.0: 's', 10.0: '^', 20.0: 'D'}

    # ── Rząd 1: CV(r13) ──────────────────────────────────────────────────────

    # 1a: CV(r13) vs r12, kolory = m_para
    ax = fig.add_subplot(gs[0, 0])
    for m_para in M_PARA_LISTA:
        sub = ok[ok["m_para"]==m_para].sort_values("r12")
        if sub.empty: continue
        ax.plot(sub["r12"], sub["cv_r13"], 'o-',
                color=kolory[m_para], markersize=7, linewidth=1.8,
                label=f"{m_para}M☉")
    ax.axhline(0.23, color='white', linestyle='--', linewidth=1,
               alpha=0.6, label='CV=0.23 (kandydat)')
    ax.axhspan(0.20, 0.26, alpha=0.08, color='white')
    ax.legend(framealpha=0.2, labelcolor='white', fontsize=7)
    styl(ax, "r12 [AU]", "CV(r13)", "Niezmiennik 1: CV(r13) vs r12\n(linie = różne m_para)")

    # 1b: CV(r13) vs m_para, kolory = r12
    ax = fig.add_subplot(gs[0, 1])
    for r12 in R12_LISTA:
        sub = ok[ok["r12"]==r12].sort_values("m_para")
        if sub.empty: continue
        ax.plot(sub["m_para"], sub["cv_r13"],
                marker=markery[r12], linestyle='-',
                color=plt.cm.plasma(R12_LISTA.index(r12)/len(R12_LISTA)),
                markersize=7, linewidth=1.8, label=f"r12={r12}AU")
    ax.axhline(0.23, color='white', linestyle='--', linewidth=1, alpha=0.6)
    ax.axhspan(0.20, 0.26, alpha=0.08, color='white')
    ax.set_xscale('log')
    ax.legend(framealpha=0.2, labelcolor='white', fontsize=7)
    styl(ax, "m_para [M☉] (log)", "CV(r13)",
         "Niezmiennik 1: CV(r13) vs m_para\n(linie = różne r12)")

    # 1c: histogram CV(r13) globalny
    ax = fig.add_subplot(gs[0, 2])
    cv_all = ok["cv_r13"].dropna()
    ax.hist(cv_all, bins=12, color='#44aaff', alpha=0.8, edgecolor='#0d0d1a')
    ax.axvline(cv_all.mean(), color='white', linewidth=2,
               label=f"mean={cv_all.mean():.4f}")
    ax.axvline(0.23, color='#ff9944', linewidth=1.5,
               linestyle='--', label="kandydat 0.23")
    ax.legend(framealpha=0.2, labelcolor='white', fontsize=7)
    styl(ax, "CV(r13)", "Liczba układów",
         f"Rozkład CV(r13)\nstd={cv_all.std():.4f}")

    # ── Rząd 2: ratio_r13 ────────────────────────────────────────────────────

    # 2a: ratio_r13 vs r12
    ax = fig.add_subplot(gs[1, 0])
    for m_para in M_PARA_LISTA:
        sub = ok[ok["m_para"]==m_para].sort_values("r12")
        if sub.empty: continue
        ax.plot(sub["r12"], sub["ratio_r13"], 'o-',
                color=kolory[m_para], markersize=7, linewidth=1.8,
                label=f"{m_para}M☉")
    ax.axhline(0.78, color='white', linestyle='--', linewidth=1,
               alpha=0.6, label='ratio=0.78 (kandydat)')
    ax.axhspan(0.75, 0.81, alpha=0.08, color='white')
    ax.legend(framealpha=0.2, labelcolor='white', fontsize=7)
    styl(ax, "r12 [AU]", "r13_mean / r13_start",
         "Niezmiennik 2: ratio_r13 vs r12")

    # 2b: ratio_r13 vs m_para
    ax = fig.add_subplot(gs[1, 1])
    for r12 in R12_LISTA:
        sub = ok[ok["r12"]==r12].sort_values("m_para")
        if sub.empty: continue
        ax.plot(sub["m_para"], sub["ratio_r13"],
                marker=markery[r12], linestyle='-',
                color=plt.cm.plasma(R12_LISTA.index(r12)/len(R12_LISTA)),
                markersize=7, linewidth=1.8, label=f"r12={r12}AU")
    ax.axhline(0.78, color='white', linestyle='--', linewidth=1, alpha=0.6)
    ax.axhspan(0.75, 0.81, alpha=0.08, color='white')
    ax.set_xscale('log')
    ax.legend(framealpha=0.2, labelcolor='white', fontsize=7)
    styl(ax, "m_para [M☉] (log)", "r13_mean / r13_start",
         "Niezmiennik 2: ratio_r13 vs m_para")

    # 2c: histogram ratio_r13
    ax = fig.add_subplot(gs[1, 2])
    ratio_all = ok["ratio_r13"].dropna()
    ax.hist(ratio_all, bins=12, color='#ff9944', alpha=0.8, edgecolor='#0d0d1a')
    ax.axvline(ratio_all.mean(), color='white', linewidth=2,
               label=f"mean={ratio_all.mean():.4f}")
    ax.axvline(0.78, color='#44ff88', linewidth=1.5,
               linestyle='--', label="kandydat 0.78")
    ax.legend(framealpha=0.2, labelcolor='white', fontsize=7)
    styl(ax, "r13_mean / r13_start", "Liczba układów",
         f"Rozkład ratio_r13\nstd={ratio_all.std():.4f}")

    # ── Rząd 3: pct_pos i mapa reżimów ───────────────────────────────────────

    # 3a: pct_pos vs r12
    ax = fig.add_subplot(gs[2, 0])
    for m_para in M_PARA_LISTA:
        sub = ok[ok["m_para"]==m_para].sort_values("r12")
        if sub.empty: continue
        ax.plot(sub["r12"], sub["pct_pos"], 'o-',
                color=kolory[m_para], markersize=7, linewidth=1.8,
                label=f"{m_para}M☉")
    ax.axhline(50, color='white', linestyle='--', linewidth=1.5,
               alpha=0.8, label='50% = przejście')
    ax.fill_between([0, 25], 50, 100, alpha=0.05, color='#44ff88')
    ax.fill_between([0, 25], 0,   50, alpha=0.05, color='#ff4444')
    ax.set_ylim(0, 105)
    ax.legend(framealpha=0.2, labelcolor='white', fontsize=7)
    styl(ax, "r12 [AU]", "% R_n > 0",
         "Niezmiennik 3: pct_pos vs r12\n(>50% = reżim +)")

    # 3b: mapa reżimów jako heatmap
    ax = fig.add_subplot(gs[2, 1])
    m_para_arr = sorted(M_PARA_LISTA)
    r12_arr    = sorted(R12_LISTA)
    heatmap    = np.full((len(m_para_arr), len(r12_arr)), np.nan)
    for i, mp in enumerate(m_para_arr):
        for j, r12 in enumerate(r12_arr):
            sub = ok[(ok["m_para"]==mp) & (ok["r12"]==r12)]
            if not sub.empty:
                heatmap[i, j] = sub.iloc[0]["pct_pos"]

    im = ax.imshow(heatmap, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=100, origin='lower')
    ax.set_xticks(range(len(r12_arr)))
    ax.set_xticklabels([f"{r:.0f}" for r in r12_arr], color='white')
    ax.set_yticks(range(len(m_para_arr)))
    ax.set_yticklabels([f"{m:.0f}" for m in m_para_arr], color='white')
    for i in range(len(m_para_arr)):
        for j in range(len(r12_arr)):
            if not np.isnan(heatmap[i, j]):
                txt = f"{heatmap[i,j]:.0f}%"
                col = 'black' if 30 < heatmap[i,j] < 70 else 'white'
                ax.text(j, i, txt, ha='center', va='center',
                        color=col, fontsize=9, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('pct_pos [%]', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    styl(ax, "r12 [AU]", "m_para [M☉]",
         "Mapa reżimów\n(zielony=+, czerwony=-)")

    # 3c: CV(r13) vs ratio_r13 — scatter wszystkich układów
    ax = fig.add_subplot(gs[2, 2])
    sc = ax.scatter(ok["cv_r13"], ok["ratio_r13"],
                    c=ok["pct_pos"], cmap='RdYlGn',
                    vmin=0, vmax=100, s=80, zorder=5)
    cbar2 = plt.colorbar(sc, ax=ax)
    cbar2.set_label('pct_pos [%]', color='white')
    cbar2.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white')
    # Annotacje m_para
    for _, row in ok.iterrows():
        ax.annotate(f"{row['m_para']:.0f}M,r{row['r12']:.0f}",
                    (row["cv_r13"], row["ratio_r13"]),
                    textcoords="offset points", xytext=(3, 3),
                    fontsize=6, color='#aaaacc')
    ax.axvline(0.23, color='white', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(0.78, color='white', linestyle='--', linewidth=1, alpha=0.5)
    styl(ax, "CV(r13)", "ratio_r13",
         "Przestrzeń niezmienników\n(kolor = pct_pos)")

    fig.suptitle(
        "TRIPLE INVARIANT SCAN\n"
        f"m3={M3}M☉, r13={R13}AU | "
        f"m_para={M_PARA_LISTA}M☉ | r12={R12_LISTA}AU",
        color='white', fontsize=12, y=1.01
    )

    plt.savefig("results/triple_invariant_wykres.png",
                dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print("\n  Wykres zapisany.")


if __name__ == "__main__":
    main()