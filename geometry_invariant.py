# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.
# geometry_invariant.py

"""
GEOMETRY INVARIANT TEST
=======================
Testuje czy R_med ≈ 1.000 jest prawdziwym niezmiennikiem
czy artefaktem konkretnej geometrii.

Hipoteza z mu_transition: R_med = 1.0000 niezależnie od mu.
Pytanie: czy to działa też przy INNYCH r12?

Eksperyment:
  Stałe:  m1=m2=50 M☉, m3=1 M☉ (mu=0.01), r13=200 AU
  Zmienne: r12 = 1, 5, 10, 20, 30, 50, 80 AU

  + Kontrola geometrii r13:
  Stałe:  m1=m2=50 M☉, m3=1 M☉, r12=10 AU
  Zmienne: r13 = 50, 100, 150, 200, 300, 500 AU

  + Kontrola symetrii mas:
  Stałe:  r12=10 AU, r13=200 AU, m3=1 M☉
  Zmienne: (m1,m2) = (50,50), (40,60), (30,70), (20,80), (10,90) M☉

Wyniki → results/geometry_invariant_raport.txt
         results/geometry_invariant_wykres.png
         results/geometry_invariant_tabela.csv

Uruchomienie:
    cd ~/Documents/three_body
    source venv/bin/activate
    python geometry_invariant.py
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

# ── Parametry wspólne ────────────────────────────────────────────────────────

N_CYKLI  = 30_000   # wystarczy do stabilnej mediany R_med
DT_DNI   = 0.5      # bezpieczny krok dla szerszego zakresu geometrii

# ── Serie eksperymentów ──────────────────────────────────────────────────────

# Seria A: różne r12, stałe r13=200 AU, m1=m2=50, m3=1
SERIA_A = {
    "nazwa":  "A_r12_scan",
    "opis":   "Skan r12 przy stałym r13=200 AU",
    "m1":     50.0, "m2": 50.0, "m3": 1.0,
    "r13":    200.0,
    "r12_lista": [1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0],
    "zmienna": "r12",
}

# Seria B: różne r13, stałe r12=10 AU, m1=m2=50, m3=1
SERIA_B = {
    "nazwa":  "B_r13_scan",
    "opis":   "Skan r13 przy stałym r12=10 AU",
    "m1":     50.0, "m2": 50.0, "m3": 1.0,
    "r12":    10.0,
    "r13_lista": [50.0, 100.0, 150.0, 200.0, 300.0, 500.0],
    "zmienna": "r13",
}

# Seria C: różna asymetria mas pary, stała geometria
SERIA_C = {
    "nazwa":  "C_asymetria_mas",
    "opis":   "Asymetria mas pary przy stałej geometrii",
    "m3":     1.0,
    "r12":    10.0, "r13": 200.0,
    "pary_mas": [
        (50.0, 50.0),   # symetryczny
        (40.0, 60.0),
        (30.0, 70.0),
        (20.0, 80.0),
        (10.0, 90.0),   # bardzo asymetryczny
    ],
    "zmienna": "asymetria",
}

# ── Budowanie konfiguracji ───────────────────────────────────────────────────

def zbuduj_config(m1_msun, m2_msun, m3_msun, r12_au, r13_au,
                  nazwa, n_cykli=N_CYKLI, dt_dni=DT_DNI):
    """
    Układ: para (m1,m2) na r12, g3 (m3) na r13.
    g3 umieszczona prostopadle do osi pary (kąt 90°) — unikamy
    osobliwości przy małym r12 i dużym r13 w jednej linii.
    Prędkości kepleriańskie + korekta środka masy.
    """
    m1 = m1_msun * M_SUN
    m2 = m2_msun * M_SUN
    m3 = m3_msun * M_SUN
    M_12 = m1 + m2

    r12_m = r12_au * AU
    r13_m = r13_au * AU

    # Pozycje: para na osi X, g3 na osi Y (90° — dalej od pary)
    # Środek masy pary w początku układu
    x1 = -r12_m * m2 / M_12
    x2 =  r12_m * m1 / M_12

    pos = np.array([
        [x1,    0.0],
        [x2,    0.0],
        [0.0,   r13_m],
    ])

    # Korekta: środek masy całości w (0,0)
    M_tot = m1 + m2 + m3
    r_cm = (m1 * pos[0] + m2 * pos[1] + m3 * pos[2]) / M_tot
    pos -= r_cm

    # Prędkości kepleriańskie
    # Para: prędkość orbitalna wokół wspólnego CM pary
    v_para = np.sqrt(G * M_12 / r12_m)
    vx1 = 0.0;  vy1 =  v_para * m2 / M_12
    vx2 = 0.0;  vy2 = -v_para * m1 / M_12

    # g3: prędkość orbitalna wokół CM (m1+m2), lekko sub-kepleriańska
    # Kierunek: prostopadły do r13 (wzdłuż osi X gdy g3 na osi Y)
    v3_mag = np.sqrt(G * M_12 / r13_m) * 0.80
    vx3 = v3_mag;  vy3 = 0.0

    vel = np.array([
        [vx1, vy1],
        [vx2, vy2],
        [vx3, vy3],
    ])

    # Wyzeruj pęd środka masy
    p_cm = m1*vel[0] + m2*vel[1] + m3*vel[2]
    vel -= p_cm / M_tot

    # Sprawdź związanie, popraw jeśli trzeba
    E = oblicz_energie(pos, vel, np.array([m1, m2, m3]))
    if E >= 0:
        vel[2] *= 0.6
        p_cm = m1*vel[0] + m2*vel[1] + m3*vel[2]
        vel -= p_cm / M_tot
        E = oblicz_energie(pos, vel, np.array([m1, m2, m3]))

    # Dobierz dt dynamicznie: musi być << okres pary
    # T_para ≈ 2π√(r12³/GM12)
    T_para = 2 * np.pi * np.sqrt(r12_m**3 / (G * M_12))
    T_para_dni = T_para / DAY
    dt_auto = min(dt_dni, T_para_dni / 200)  # co najmniej 200 kroków na okres pary
    dt_auto = max(dt_auto, 0.001)             # minimum 0.001 dnia

    # kroki_na_cykl: jeden "cykl" = ~1/100 okresu g3
    T_g3 = 2 * np.pi * np.sqrt(r13_m**3 / (G * M_12))
    T_g3_dni = T_g3 / DAY
    kpc = max(10, int(T_g3_dni / dt_auto / 200))
    kpc = min(kpc, 500)  # cap żeby nie było za wolno

    return {
        "nazwa":            nazwa,
        "masy":             [m1, m2, m3],
        "pozycje":          pos.tolist(),
        "predkosci":        vel.tolist(),
        "dt":               DAY * dt_auto,
        "n_cykli":          n_cykli,
        "kroki_na_cykl":    kpc,
        "max_blad_energii": 1e-2,
        "max_r_au":         max(r13_au * 20, 5000),
    }, dt_auto, kpc


# ── Funkcje analizy (identyczne jak w mu_transition) ────────────────────────

def zdarzenia_poincare(df):
    r12 = df['r12'].values
    zdarz = []
    for i in range(1, len(r12) - 1):
        if r12[i] < r12[i-1] and r12[i] < r12[i+1]:
            zdarz.append(i)
    return zdarz


def analiza_R_med(df, zdarz, kolumna='r13'):
    """
    Mediana R(n) = δ(n+1)/δ(n) dla zadanej kolumny w zdarzeniach.
    Zwraca (R_med, R_std, CV, n_stosunkow).
    """
    if len(zdarz) < 20:
        return np.nan, np.nan, np.nan, 0

    vals = df[kolumna].values[zdarz]
    delty = np.diff(vals)
    mask = np.abs(delty[:-1]) > 1e-10
    n_ok = np.sum(mask)

    if n_ok < 10:
        return np.nan, np.nan, np.nan, n_ok

    R_n = delty[1:][mask] / delty[:-1][mask]

    # Usuń ekstrema (|R_n| > 100) — artefakty przy δ≈0
    R_n = R_n[np.abs(R_n) < 100]
    if len(R_n) < 5:
        return np.nan, np.nan, np.nan, len(R_n)

    med = np.median(R_n)
    std = np.std(R_n)
    cv  = std / (np.abs(med) + 1e-10)
    return med, std, cv, len(R_n)


def analiza_cv(df, zdarz, kolumna='r13'):
    if len(zdarz) < 10:
        return np.nan
    vals = df[kolumna].values[zdarz]
    return np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else np.nan


def przeanalizuj(df, nazwa_ukladu):
    """Pełna analiza jednego układu. Zwraca słownik wyników."""
    zdarz = zdarzenia_poincare(df)
    n = len(zdarz)

    r13_med, r13_std, r13_cv_R, n_R = analiza_R_med(df, zdarz, 'r13')
    r23_med, r23_std, r23_cv_R, _   = analiza_R_med(df, zdarz, 'r23')
    v3_med,  v3_std,  v3_cv_R,  _   = analiza_R_med(df, zdarz, 'v3')

    cv_r13 = analiza_cv(df, zdarz, 'r13')
    cv_r12 = analiza_cv(df, zdarz, 'r12')

    print(f"    Zdarzeń: {n} | R_med(r13)={r13_med:.6f} | "
          f"CV(r13)={cv_r13:.4f} | n_stosunków={n_R}")

    return {
        "nazwa":      nazwa_ukladu,
        "n_zdarzen":  n,
        "n_stosunkow": n_R,
        "R_med_r13":  r13_med,
        "R_std_r13":  r13_std,
        "R_cv_r13":   r13_cv_R,
        "R_med_r23":  r23_med,
        "R_med_v3":   v3_med,
        "cv_r13":     cv_r13,
        "cv_r12":     cv_r12,
        "r13_mean":   df['r13'].mean(),
        "r12_mean":   df['r12'].mean(),
        "blad_E":     df['blad_energii'].iloc[-1],
        "n_cykli":    len(df),
    }


# ── Główna pętla ─────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("data",    exist_ok=True)

    wszystkie_wyniki = []

    # ════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  SERIA A — skan r12")
    print(f"  m1=m2={SERIA_A['m1']} M☉, m3={SERIA_A['m3']} M☉, r13={SERIA_A['r13']} AU")
    print("="*65)

    for r12 in SERIA_A["r12_lista"]:
        sep = r13 / r12 if (r13 := SERIA_A['r13']) else 0
        nazwa = f"A_r12_{r12:.0f}AU"
        print(f"\n  r12={r12} AU  (r13/r12={sep:.1f}×)")

        cfg, dt_auto, kpc = zbuduj_config(
            SERIA_A["m1"], SERIA_A["m2"], SERIA_A["m3"],
            r12, SERIA_A["r13"], nazwa
        )
        print(f"    dt={dt_auto:.3f} dni | kpc={kpc}")

        df, powod = symuluj(cfg, folder="data", cicho=True)

        if powod is not None or df.empty:
            print(f"    ✗ Nieudana: {powod}")
            wszystkie_wyniki.append({
                "seria": "A", "zmienna": r12, "zmienna_nazwa": "r12_au",
                "status": powod or "blad", "R_med_r13": np.nan,
                "cv_r13": np.nan, "n_zdarzen": 0,
            })
            continue

        print(f"    ✓ OK — {len(df)} cykli | błąd E={df['blad_energii'].iloc[-1]:.2e}")
        wynik = przeanalizuj(df, nazwa)
        wynik.update({"seria": "A", "zmienna": r12, "zmienna_nazwa": "r12_au",
                      "status": "ok"})
        wszystkie_wyniki.append(wynik)

    # ════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  SERIA B — skan r13")
    print(f"  m1=m2={SERIA_B['m1']} M☉, m3={SERIA_B['m3']} M☉, r12={SERIA_B['r12']} AU")
    print("="*65)

    for r13 in SERIA_B["r13_lista"]:
        sep = r13 / SERIA_B['r12']
        nazwa = f"B_r13_{r13:.0f}AU"
        print(f"\n  r13={r13} AU  (r13/r12={sep:.1f}×)")

        cfg, dt_auto, kpc = zbuduj_config(
            SERIA_B["m1"], SERIA_B["m2"], SERIA_B["m3"],
            SERIA_B["r12"], r13, nazwa
        )
        print(f"    dt={dt_auto:.3f} dni | kpc={kpc}")

        df, powod = symuluj(cfg, folder="data", cicho=True)

        if powod is not None or df.empty:
            print(f"    ✗ Nieudana: {powod}")
            wszystkie_wyniki.append({
                "seria": "B", "zmienna": r13, "zmienna_nazwa": "r13_au",
                "status": powod or "blad", "R_med_r13": np.nan,
                "cv_r13": np.nan, "n_zdarzen": 0,
            })
            continue

        print(f"    ✓ OK — {len(df)} cykli | błąd E={df['blad_energii'].iloc[-1]:.2e}")
        wynik = przeanalizuj(df, nazwa)
        wynik.update({"seria": "B", "zmienna": r13, "zmienna_nazwa": "r13_au",
                      "status": "ok"})
        wszystkie_wyniki.append(wynik)

    # ════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  SERIA C — asymetria mas pary")
    print(f"  m3={SERIA_C['m3']} M☉, r12={SERIA_C['r12']} AU, r13={SERIA_C['r13']} AU")
    print("="*65)

    for m1, m2 in SERIA_C["pary_mas"]:
        asym = abs(m1 - m2) / (m1 + m2)
        nazwa = f"C_m1_{m1:.0f}_m2_{m2:.0f}"
        print(f"\n  m1={m1}, m2={m2} M☉  (asymetria={asym:.2f})")

        cfg, dt_auto, kpc = zbuduj_config(
            m1, m2, SERIA_C["m3"],
            SERIA_C["r12"], SERIA_C["r13"], nazwa
        )
        print(f"    dt={dt_auto:.3f} dni | kpc={kpc}")

        df, powod = symuluj(cfg, folder="data", cicho=True)

        if powod is not None or df.empty:
            print(f"    ✗ Nieudana: {powod}")
            wszystkie_wyniki.append({
                "seria": "C", "zmienna": asym, "zmienna_nazwa": "asymetria",
                "m1": m1, "m2": m2,
                "status": powod or "blad", "R_med_r13": np.nan,
                "cv_r13": np.nan, "n_zdarzen": 0,
            })
            continue

        print(f"    ✓ OK — {len(df)} cykli | błąd E={df['blad_energii'].iloc[-1]:.2e}")
        wynik = przeanalizuj(df, nazwa)
        wynik.update({
            "seria": "C", "zmienna": asym, "zmienna_nazwa": "asymetria",
            "m1": m1, "m2": m2, "status": "ok"
        })
        wszystkie_wyniki.append(wynik)

    # ── Zapisz tabelę ────────────────────────────────────────────────────────
    tabela = pd.DataFrame(wszystkie_wyniki)
    tabela.to_csv("results/geometry_invariant_tabela.csv", index=False)

    # ── Raport ───────────────────────────────────────────────────────────────
    raport(tabela)

    # ── Wykresy ──────────────────────────────────────────────────────────────
    rysuj(tabela)

    print("\nGOTOWE.")
    print("  Tabela  → results/geometry_invariant_tabela.csv")
    print("  Raport  → results/geometry_invariant_raport.txt")
    print("  Wykres  → results/geometry_invariant_wykres.png\n")


# ── Raport tekstowy ──────────────────────────────────────────────────────────

def raport(tabela):
    linie = []
    linie.append("=" * 70)
    linie.append("GEOMETRY INVARIANT TEST — RAPORT")
    linie.append("Hipoteza: R_med(r13) ≈ 1.000 niezależnie od geometrii")
    linie.append("=" * 70)

    for seria, tytul, zmienna_kol in [
        ("A", "SERIA A — skan r12 (r13=200 AU, m1=m2=50, m3=1 M☉)", "r12_au"),
        ("B", "SERIA B — skan r13 (r12=10 AU, m1=m2=50, m3=1 M☉)",  "r13_au"),
        ("C", "SERIA C — asymetria mas pary (r12=10, r13=200 AU)",    "asymetria"),
    ]:
        linie.append(f"\n{tytul}")
        linie.append("-" * 70)
        df_s = tabela[tabela["seria"] == seria]

        linie.append(f"  {'param':>12}  {'R_med(r13)':>12}  {'R_std':>8}  "
                     f"{'CV(r13)':>9}  {'n_zdarz':>8}  status")
        linie.append("  " + "-" * 60)

        for _, row in df_s.iterrows():
            if row["status"] != "ok":
                linie.append(f"  {row['zmienna']:>12.2f}  {'—':>12}  {'—':>8}  "
                             f"{'—':>9}  {0:>8}  ✗ {row['status']}")
                continue
            rmed = row.get("R_med_r13", np.nan)
            rstd = row.get("R_std_r13", np.nan)
            cv   = row.get("cv_r13", np.nan)
            n    = int(row.get("n_zdarzen", 0))
            rmed_s = f"{rmed:.6f}" if not np.isnan(rmed) else "   nan   "
            rstd_s = f"{rstd:.4f}" if not np.isnan(rstd) else "  nan "
            cv_s   = f"{cv:.4f}"   if not np.isnan(cv)   else "  nan "
            linie.append(f"  {row['zmienna']:>12.2f}  {rmed_s:>12}  {rstd_s:>8}  "
                         f"{cv_s:>9}  {n:>8}  ✓")

        # Podsumowanie serii
        ok = df_s[df_s["status"] == "ok"]["R_med_r13"].dropna()
        if len(ok) >= 2:
            linie.append(f"\n  → R_med: min={ok.min():.6f}  max={ok.max():.6f}  "
                         f"rozstęp={ok.max()-ok.min():.6f}  std={ok.std():.6f}")
            if ok.std() < 0.01:
                linie.append(f"  ★ NIEZMIENNIK POTWIERDZONY w serii {seria} "
                             f"(std={ok.std():.6f} < 0.01)")
            else:
                linie.append(f"  ✗ Brak niezmiennika w serii {seria} "
                             f"(std={ok.std():.6f} ≥ 0.01)")

    # Wniosek globalny
    ok_all = tabela[tabela["status"] == "ok"]["R_med_r13"].dropna()
    linie.append("\n" + "=" * 70)
    linie.append("WNIOSEK GLOBALNY")
    linie.append(f"  Układów z wynikiem: {len(ok_all)}")
    if len(ok_all) >= 3:
        linie.append(f"  R_med globalnie: "
                     f"mean={ok_all.mean():.6f}  "
                     f"std={ok_all.std():.6f}  "
                     f"min={ok_all.min():.6f}  max={ok_all.max():.6f}")
        if ok_all.std() < 0.01:
            linie.append("  ★★★ R_med ≈ 1.000 JEST NIEZMIENNIKIEM GEOMETRYCZNYM")
        elif ok_all.std() < 0.05:
            linie.append("  ★★  R_med ≈ 1.000 CZĘŚCIOWY (zależy od geometrii)")
        else:
            linie.append("  ✗   R_med NIE jest niezmiennikiem — zależy od geometrii")
    linie.append("=" * 70)

    tekst = "\n".join(linie)
    print("\n" + tekst)
    with open("results/geometry_invariant_raport.txt", "w") as f:
        f.write(tekst)


# ── Wykresy ──────────────────────────────────────────────────────────────────

def rysuj(tabela):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
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

    kolory = {"A": "#ff9944", "B": "#44aaff", "C": "#88ff44"}

    for i, (seria, label_x, title_r, title_cv) in enumerate([
        ("A", "r12 [AU]",    "Seria A: R_med(r13) vs r12",    "Seria A: CV(r13) vs r12"),
        ("B", "r13 [AU]",    "Seria B: R_med(r13) vs r13",    "Seria B: CV(r13) vs r13"),
        ("C", "asymetria",   "Seria C: R_med(r13) vs asym",   "Seria C: CV(r13) vs asym"),
    ]):
        df_s = tabela[(tabela["seria"] == seria) & (tabela["status"] == "ok")].copy()
        kolor = kolory[seria]

        ax_r = axes[0, i]
        ax_c = axes[1, i]

        if not df_s.empty:
            x = df_s["zmienna"].values
            r_med = df_s["R_med_r13"].values
            r_std = df_s["R_std_r13"].values
            cv    = df_s["cv_r13"].values

            ax_r.errorbar(x, r_med, yerr=r_std, fmt='o', color=kolor,
                          ecolor='#888888', capsize=4, markersize=8, zorder=5)
            ax_r.plot(x, r_med, color=kolor, alpha=0.4, linewidth=1.5)
            ax_r.axhline(1.0, color='white', linestyle='--', linewidth=1,
                         alpha=0.6, label='R_med=1.000')
            ax_r.axhspan(0.99, 1.01, alpha=0.08, color='white',
                         label='±1% od 1.000')

            # Annotacje wartości
            for xi, ri in zip(x, r_med):
                if not np.isnan(ri):
                    ax_r.annotate(f"{ri:.4f}", (xi, ri),
                                  textcoords="offset points", xytext=(4, 6),
                                  fontsize=7, color='#ccccee')

            ax_c.scatter(x, cv, color=kolor, s=80, zorder=5)
            ax_c.plot(x, cv, color=kolor, alpha=0.4, linewidth=1.5)
            ax_c.axhline(0.1, color='#ff4444', linestyle='--', linewidth=1,
                         alpha=0.8, label='CV=0.1')

        styl(ax_r, label_x, "R_med(r13)", title_r)
        styl(ax_c, label_x, "CV(r13)",    title_cv)
        ax_r.legend(framealpha=0.2, labelcolor='white', fontsize=8)
        ax_c.legend(framealpha=0.2, labelcolor='white', fontsize=8)

    # Zbiorczy panel: wszystkie serie razem
    fig.suptitle(
        "GEOMETRY INVARIANT TEST\n"
        "Czy R_med(r13) ≈ 1.000 jest niezmiennikiem geometrii?",
        color='white', fontsize=13
    )

    plt.tight_layout()
    plt.savefig("results/geometry_invariant_wykres.png",
                dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()


if __name__ == "__main__":
    main()