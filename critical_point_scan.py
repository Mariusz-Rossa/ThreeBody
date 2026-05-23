# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.
# critical_point_scan.py
 
"""
CRITICAL POINT SCAN
===================
Szuka dokładnego punktu krytycznego zmiany znaku R_med(r13)
między r12=1 AU (R_med≈+1.000) a r12=10 AU (R_med≈−0.558).

Geometria: m1=m2=50 M☉, m3=1 M☉, r13=200 AU
Skan: r12 = 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
            5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 12.0, 15.0

Pytanie: przy jakim r12 R_med przechodzi przez zero?
         Czy to przejście jest ostre (skok) czy ciągłe?

Wyniki → results/critical_point_raport.txt
         results/critical_point_wykres.png
         results/critical_point_tabela.csv

Uruchomienie:
    cd ~/Documents/three_body
    source venv/bin/activate
    python critical_point_scan.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator import symuluj, oblicz_energie, G, AU, YEAR, DAY, M_SUN

# ── Parametry ────────────────────────────────────────────────────────────────

M1      = 50.0    # M☉
M2      = 50.0    # M☉
M3      =  1.0    # M☉
R13     = 200.0   # AU — stałe
N_CYKLI = 40_000  # więcej cykli = stabilniejsza mediana

# Gęsty skan r12
R12_LISTA = [
    1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
    9.0, 10.0, 12.0, 15.0
]

# ── Funkcje (te same co w geometry_invariant) ────────────────────────────────

def zbuduj_config(r12_au, nazwa):
    m1 = M1 * M_SUN
    m2 = M2 * M_SUN
    m3 = M3 * M_SUN
    M_12  = m1 + m2
    M_tot = m1 + m2 + m3

    r12_m = r12_au * AU
    r13_m = R13 * AU

    # Para na osi X, g3 na osi Y
    x1 = -r12_m * m2 / M_12
    x2 =  r12_m * m1 / M_12
    pos = np.array([[x1, 0.0], [x2, 0.0], [0.0, r13_m]])

    r_cm = (m1*pos[0] + m2*pos[1] + m3*pos[2]) / M_tot
    pos -= r_cm

    # Prędkości kepleriańskie
    v_para = np.sqrt(G * M_12 / r12_m)
    v3_mag = np.sqrt(G * M_12 / r13_m) * 0.80

    vel = np.array([
        [0.0,  v_para * m2/M_12],
        [0.0, -v_para * m1/M_12],
        [v3_mag, 0.0],
    ])

    p_cm = m1*vel[0] + m2*vel[1] + m3*vel[2]
    vel -= p_cm / M_tot

    E = oblicz_energie(pos, vel, np.array([m1, m2, m3]))
    if E >= 0:
        vel[2] *= 0.6
        p_cm = m1*vel[0] + m2*vel[1] + m3*vel[2]
        vel -= p_cm / M_tot

    # dt: 200 kroków na okres pary
    T_para_dni = 2 * np.pi * np.sqrt(r12_m**3 / (G * M_12)) / DAY
    dt_auto = min(0.5, T_para_dni / 200)
    dt_auto = max(dt_auto, 0.001)

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
        "max_r_au":         5000,
    }, dt_auto, kpc


def zdarzenia_poincare(df):
    r12 = df['r12'].values
    zdarz = []
    for i in range(1, len(r12) - 1):
        if r12[i] < r12[i-1] and r12[i] < r12[i+1]:
            zdarz.append(i)
    return zdarz


def analiza_R_med(vals):
    """
    Pełna analiza R_med dla szeregu wartości w zdarzeniach.
    Zwraca słownik z medianą, std, CV, percentylami, % w [0.9,1.1].
    """
    delty = np.diff(vals)
    mask  = np.abs(delty[:-1]) > 1e-10
    if np.sum(mask) < 10:
        return None

    R_n = delty[1:][mask] / delty[:-1][mask]
    R_n = R_n[np.abs(R_n) < 100]  # usuń artefakty
    if len(R_n) < 5:
        return None

    return {
        "med":    np.median(R_n),
        "mean":   np.mean(R_n),
        "std":    np.std(R_n),
        "cv":     np.std(R_n) / (np.abs(np.median(R_n)) + 1e-10),
        "p25":    np.percentile(R_n, 25),
        "p75":    np.percentile(R_n, 75),
        "n":      len(R_n),
        "pct_pos": np.mean(R_n > 0) * 100,   # % dodatnich R_n
    }


# ── Główna pętla ─────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("data",    exist_ok=True)

    wyniki = []

    print("\n" + "="*65)
    print("  CRITICAL POINT SCAN")
    print(f"  m1=m2={M1} M☉, m3={M3} M☉, r13={R13} AU")
    print(f"  Skan r12: {R12_LISTA[0]}–{R12_LISTA[-1]} AU | {N_CYKLI} cykli")
    print("="*65)

    for r12 in R12_LISTA:
        sep   = R13 / r12
        nazwa = f"crit_r12_{r12:.1f}AU".replace('.', 'p')

        print(f"\n  r12={r12:5.1f} AU  (r13/r12={sep:6.1f}×)", end="  ")

        cfg, dt_auto, kpc = zbuduj_config(r12, nazwa)
        print(f"dt={dt_auto:.4f}d kpc={kpc}", end="  ")

        df, powod = symuluj(cfg, folder="data", cicho=True)

        if powod is not None or df.empty:
            print(f"✗ {powod}")
            wyniki.append({
                "r12": r12, "sep": sep,
                "status": powod or "blad",
                "R_med_r13": np.nan, "R_med_r23": np.nan,
                "pct_pos_r13": np.nan,
                "cv_r13": np.nan, "n_zdarzen": 0,
            })
            continue

        zdarz   = zdarzenia_poincare(df)
        n_zdarz = len(zdarz)

        if n_zdarz < 20:
            print(f"✗ za mało zdarzeń ({n_zdarz})")
            wyniki.append({
                "r12": r12, "sep": sep, "status": "malo_zdarzen",
                "R_med_r13": np.nan, "n_zdarzen": n_zdarz,
            })
            continue

        r13_z = df['r13'].values[zdarz]
        r23_z = df['r23'].values[zdarz]
        v3_z  = df['v3'].values[zdarz]

        res13 = analiza_R_med(r13_z)
        res23 = analiza_R_med(r23_z)
        resv3 = analiza_R_med(v3_z)

        cv_r13 = np.std(r13_z) / np.mean(r13_z) if np.mean(r13_z) > 0 else np.nan
        cv_r12 = np.std(df['r12'].values[zdarz]) / np.mean(df['r12'].values[zdarz])

        r13_med = res13["med"] if res13 else np.nan
        r23_med = res23["med"] if res23 else np.nan
        v3_med  = resv3["med"] if resv3 else np.nan
        pct_pos = res13["pct_pos"] if res13 else np.nan

        print(f"✓ zdarz={n_zdarz:5d} | "
              f"R_med(r13)={r13_med:+.4f} | "
              f"R_med(r23)={r23_med:+.4f} | "
              f"pct_pos={pct_pos:.1f}%")

        wyniki.append({
            "r12":         r12,
            "sep":         sep,
            "status":      "ok",
            "R_med_r13":   r13_med,
            "R_std_r13":   res13["std"] if res13 else np.nan,
            "R_cv_r13":    res13["cv"]  if res13 else np.nan,
            "R_p25_r13":   res13["p25"] if res13 else np.nan,
            "R_p75_r13":   res13["p75"] if res13 else np.nan,
            "pct_pos_r13": pct_pos,
            "R_med_r23":   r23_med,
            "R_med_v3":    v3_med,
            "cv_r13":      cv_r13,
            "cv_r12":      cv_r12,
            "r13_mean":    df['r13'].mean(),
            "n_zdarzen":   n_zdarz,
            "n_stosunkow": res13["n"] if res13 else 0,
            "blad_E":      df['blad_energii'].iloc[-1],
            "n_cykli":     len(df),
            "dt_dni":      cfg["dt"] / DAY,
            "kpc":         kpc,
        })

    # ── Tabela ───────────────────────────────────────────────────────────────
    tabela = pd.DataFrame(wyniki)
    tabela.to_csv("results/critical_point_tabela.csv", index=False)

    # ── Raport ───────────────────────────────────────────────────────────────
    raport(tabela)

    # ── Wykresy ──────────────────────────────────────────────────────────────
    rysuj(tabela)

    print("\n  Tabela  → results/critical_point_tabela.csv")
    print("  Raport  → results/critical_point_raport.txt")
    print("  Wykres  → results/critical_point_wykres.png")
    print("\nGOTOWE.\n")


# ── Raport ───────────────────────────────────────────────────────────────────

def raport(tabela):
    ok = tabela[tabela["status"] == "ok"].copy()

    linie = []
    linie.append("=" * 70)
    linie.append("CRITICAL POINT SCAN — RAPORT")
    linie.append(f"m1=m2={M1} M☉, m3={M3} M☉, r13={R13} AU")
    linie.append(f"Szukamy r12* gdzie R_med(r13) zmienia znak")
    linie.append("=" * 70)
    linie.append("")
    linie.append(f"  {'r12':>6} {'sep':>8} {'R_med(r13)':>12} "
                 f"{'R_med(r23)':>12} {'pct_pos':>8} {'CV(r13)':>8} {'n_zdarz':>8}")
    linie.append("  " + "-"*68)

    for _, row in tabela.iterrows():
        if row["status"] != "ok":
            linie.append(f"  {row['r12']:>6.1f} {row['sep']:>8.1f}  "
                         f"{'— '+str(row['status']):>40}")
            continue
        znak13 = "+" if row["R_med_r13"] > 0 else "-"
        linie.append(
            f"  {row['r12']:>6.1f} {row['sep']:>8.1f} "
            f"  {row['R_med_r13']:>+10.4f}   "
            f"  {row['R_med_r23']:>+10.4f}   "
            f"{row['pct_pos_r13']:>7.1f}%  "
            f"{row['cv_r13']:>7.4f}  "
            f"{int(row['n_zdarzen']):>8}"
        )

    linie.append("")

    # Znajdź punkt zerowy
    ok_sorted = ok.sort_values("r12")
    r_med_vals = ok_sorted["R_med_r13"].values
    r12_vals   = ok_sorted["r12"].values

    zero_crossings = []
    for i in range(len(r_med_vals) - 1):
        if not (np.isnan(r_med_vals[i]) or np.isnan(r_med_vals[i+1])):
            if r_med_vals[i] * r_med_vals[i+1] < 0:
                # Interpolacja liniowa
                r12_star = r12_vals[i] + (r12_vals[i+1] - r12_vals[i]) * \
                           abs(r_med_vals[i]) / (abs(r_med_vals[i]) + abs(r_med_vals[i+1]))
                zero_crossings.append((r12_vals[i], r12_vals[i+1], r12_star))
                linie.append(f"ZMIANA ZNAKU: między r12={r12_vals[i]} a r12={r12_vals[i+1]} AU")
                linie.append(f"  Interpolowany punkt krytyczny: r12* ≈ {r12_star:.2f} AU")
                linie.append(f"  Separacja krytyczna: r13/r12* ≈ {R13/r12_star:.1f}×")

    if not zero_crossings:
        linie.append("BRAK zmiany znaku w przeskanowanym zakresie.")

    # Dodaj analizę pct_pos
    linie.append("")
    linie.append("ANALIZA pct_pos (% stosunków R_n > 0):")
    linie.append("  pct_pos=50% odpowiada R_med≈0 (przejście)")
    for _, row in ok_sorted.iterrows():
        bar = "█" * int(row["pct_pos_r13"] / 5)
        linie.append(f"  r12={row['r12']:5.1f} AU: {row['pct_pos_r13']:5.1f}% {bar}")

    linie.append("")
    linie.append("=" * 70)

    tekst = "\n".join(linie)
    print("\n" + tekst)
    with open("results/critical_point_raport.txt", "w") as f:
        f.write(tekst)


# ── Wykresy ──────────────────────────────────────────────────────────────────

def rysuj(tabela):
    ok = tabela[tabela["status"] == "ok"].copy().sort_values("r12")
    if ok.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0d0d1a')

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
        ax.set_title(title)

    x = ok["r12"].values

    # 1. R_med(r13) i R_med(r23) vs r12
    ax = axes[0, 0]
    ax.axhline(0, color='white', linewidth=1.5, alpha=0.8, linestyle='-')
    ax.axhline(+1, color='#44ff88', linewidth=1, alpha=0.5, linestyle='--')
    ax.axhline(-1, color='#ff4444', linewidth=1, alpha=0.5, linestyle='--')
    ax.fill_between(x, ok["R_p25_r13"], ok["R_p75_r13"],
                    alpha=0.15, color='#ff9944', label='IQR r13')
    ax.plot(x, ok["R_med_r13"], 'o-', color='#ff9944',
            markersize=8, linewidth=2, label='R_med(r13)', zorder=5)
    ax.plot(x, ok["R_med_r23"], 's--', color='#44aaff',
            markersize=6, linewidth=1.5, label='R_med(r23)', zorder=4)

    # Annotacje wartości
    for xi, yi in zip(x, ok["R_med_r13"].values):
        if not np.isnan(yi):
            ax.annotate(f"{yi:+.3f}", (xi, yi),
                        textcoords="offset points", xytext=(3, 8),
                        fontsize=7, color='#ffcc88')

    # Zaznacz punkt zerowy jeśli jest
    r_med_v = ok["R_med_r13"].values
    r12_v   = ok["r12"].values
    for i in range(len(r_med_v)-1):
        if not (np.isnan(r_med_v[i]) or np.isnan(r_med_v[i+1])):
            if r_med_v[i] * r_med_v[i+1] < 0:
                r12_star = r12_v[i] + (r12_v[i+1]-r12_v[i]) * \
                           abs(r_med_v[i])/(abs(r_med_v[i])+abs(r_med_v[i+1]))
                ax.axvline(r12_star, color='#ff44ff', linewidth=2,
                           linestyle=':', alpha=0.8,
                           label=f'r12*≈{r12_star:.1f} AU')

    ax.legend(framealpha=0.3, labelcolor='white', fontsize=8)
    styl(ax, "r12 [AU]", "R_med", "R_med(r13) i R_med(r23) vs r12\n← punkt krytyczny →")

    # 2. pct_pos vs r12
    ax = axes[0, 1]
    ax.axhline(50, color='white', linewidth=1.5, linestyle='--',
               alpha=0.8, label='50% = punkt przejścia')
    ax.fill_between(x, 50, ok["pct_pos_r13"],
                    where=ok["pct_pos_r13"] >= 50,
                    alpha=0.2, color='#44ff88')
    ax.fill_between(x, 50, ok["pct_pos_r13"],
                    where=ok["pct_pos_r13"] < 50,
                    alpha=0.2, color='#ff4444')
    ax.plot(x, ok["pct_pos_r13"], 'o-', color='#88ffaa',
            markersize=8, linewidth=2, zorder=5)
    for xi, yi in zip(x, ok["pct_pos_r13"].values):
        ax.annotate(f"{yi:.0f}%", (xi, yi),
                    textcoords="offset points", xytext=(3, 6),
                    fontsize=7, color='#aaffcc')
    ax.set_ylim(0, 100)
    ax.legend(framealpha=0.3, labelcolor='white', fontsize=8)
    styl(ax, "r12 [AU]", "% stosunków R_n > 0",
         "Odsetek R_n dodatnich vs r12\n(50% = przejście)")

    # 3. CV(r13) vs r12
    ax = axes[1, 0]
    ax.plot(x, ok["cv_r13"], 'o-', color='#44aaff', markersize=8, linewidth=2)
    ax.axhline(0.25, color='#ff9944', linewidth=1, linestyle='--',
               alpha=0.7, label='CV=0.25 (poprzednia seria)')
    for xi, yi in zip(x, ok["cv_r13"].values):
        ax.annotate(f"{yi:.3f}", (xi, yi),
                    textcoords="offset points", xytext=(3, 6),
                    fontsize=7, color='#88ccff')
    ax.legend(framealpha=0.3, labelcolor='white', fontsize=8)
    styl(ax, "r12 [AU]", "CV(r13)", "CV(r13) vs r12")

    # 4. n_zdarzeń vs r12 — kontrola jakości
    ax = axes[1, 1]
    kolory_bars = ['#44ff88' if s == 'ok' else '#ff4444'
                   for s in tabela["status"]]
    n_vals = tabela["n_zdarzen"].fillna(0).values
    r12_all = tabela["r12"].values
    ax.bar(r12_all, n_vals, color=kolory_bars, alpha=0.8, width=0.4)
    ax.set_yscale('log')
    styl(ax, "r12 [AU]", "Liczba zdarzeń Poincarégo (log)",
         "Liczba zdarzeń vs r12\n(kontrola jakości)")

    fig.suptitle(
        f"CRITICAL POINT SCAN\n"
        f"m1=m2={M1}M☉, m3={M3}M☉, r13={R13}AU — gdzie R_med zmienia znak?",
        color='white', fontsize=13
    )
    plt.tight_layout()
    plt.savefig("results/critical_point_wykres.png",
                dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()


if __name__ == "__main__":
    main()