# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.
# mu_transition.py

"""
MU TRANSITION ANALYZER
======================
Bada przejście od chaosu do ruchu perturbacyjnego w układzie trzech ciał.

Układ: m1=m2=50 M☉ (para), m3 malejące → mu = m3/(m1+m2) maleje
Geometria: r12=1 AU (para blisko), r13=r23=200 AU (g3 daleko)
Hipoteza: sin²(ωt) coraz lepiej opisuje r13 gdy mu → 0

Uruchomienie:
    cd ~/Documents/three_body
    source venv/bin/activate
    python mu_transition.py

Wyniki → results/mu_transition_raport.txt
         results/mu_transition_wykres.png
         results/mu_transition_tabela.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import os, sys, warnings
warnings.filterwarnings('ignore')

# ── Importuj symulator ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator import symuluj, oblicz_energie, G, AU, YEAR, DAY, M_SUN

# ── Parametry eksperymentu ───────────────────────────────────────────────────

M_PARA    = 50.0          # masa każdej gwiazdy w parze [M☉]
R12       = 1.0           # odległość pary [AU]
R13       = 200.0         # odległość g3 [AU]
N_CYKLI   = 50_000        # liczba cykli symulacji
DT_DNI    = 0.3           # krok czasowy [dni] — sprawdzony dla BH

# Seria mas gwiazdy 3 — od gwiazdowej do ekstremalnie małej
M3_SERIE = [
    50.0,    # mu ≈ 0.500  — równe masy (pełny chaos)
    20.0,    # mu ≈ 0.167
    10.0,    # mu ≈ 0.091
     5.0,    # mu ≈ 0.048
     2.0,    # mu ≈ 0.020
     1.0,    # mu ≈ 0.010
     0.5,    # mu ≈ 0.005
     0.1,    # mu ≈ 0.001  — jak BH (już zbadane, R²≈0.988)
     0.05,   # mu ≈ 0.0005
     0.01,   # mu ≈ 0.0001
]

# ── Funkcje analizy ──────────────────────────────────────────────────────────

def sin2_model(t, omega, A, C):
    """r13(t) = A * sin²(ω·t) + C"""
    return A * np.sin(omega * t)**2 + C


def dopasuj_sin2(t_arr, r13_arr):
    """
    Dopasowuje sin²(ωt) do danych r13.
    Zwraca (omega, A, C, R²) lub None przy błędzie.
    """
    # Oszacowanie startowe omega z FFT
    if len(t_arr) < 20:
        return None

    # Normalizacja czasu do jednostek [lat]
    t = t_arr / YEAR

    # Startowe omega: szukamy dominującej częstotliwości
    n = len(t)
    dt_mean = np.mean(np.diff(t)) if n > 1 else 1.0
    freqs = np.fft.rfftfreq(n, d=dt_mean)
    fft_amp = np.abs(np.fft.rfft(r13_arr - np.mean(r13_arr)))
    if len(freqs) > 1:
        omega0 = 2 * np.pi * freqs[np.argmax(fft_amp[1:]) + 1]
    else:
        omega0 = 1e-4

    A0 = (np.max(r13_arr) - np.min(r13_arr)) / 2
    C0 = np.min(r13_arr)

    try:
        popt, _ = curve_fit(
            sin2_model, t, r13_arr,
            p0=[omega0, A0, C0],
            maxfev=10000,
            bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf])
        )
        r13_pred = sin2_model(t, *popt)
        ss_res = np.sum((r13_arr - r13_pred)**2)
        ss_tot = np.sum((r13_arr - np.mean(r13_arr))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return popt[0], popt[1], popt[2], r2
    except Exception:
        return None


def zdarzenia_poincare(df):
    """
    Wyznacza lokalne minima r12 (zdarzenia Poincarégo).
    Zwraca indeksy zdarzeń.
    """
    r12 = df['r12'].values
    zdarz = []
    for i in range(1, len(r12) - 1):
        if r12[i] < r12[i-1] and r12[i] < r12[i+1]:
            zdarz.append(i)
    return zdarz


def analiza_r13_cv(df, zdarz):
    """CV(r13) w zdarzeniach Poincarégo."""
    if len(zdarz) < 10:
        return np.nan
    r13_z = df['r13'].values[zdarz]
    return np.std(r13_z) / np.mean(r13_z) if np.mean(r13_z) > 0 else np.nan


def analiza_R_med(df, zdarz):
    """
    Mediana R(n) = δ(n+1)/δ(n) dla r13 w zdarzeniach.
    δ(n) = r13[n+1] - r13[n]
    """
    if len(zdarz) < 10:
        return np.nan, np.nan
    r13_z = df['r13'].values[zdarz]
    delty = np.diff(r13_z)
    # Unikaj dzielenia przez zero
    mask = np.abs(delty[:-1]) > 1e-10
    if np.sum(mask) < 5:
        return np.nan, np.nan
    R_n = delty[1:][mask] / delty[:-1][mask]
    return np.median(R_n), np.std(R_n) / (np.abs(np.median(R_n)) + 1e-10)


# ── Budowanie konfiguracji układu ────────────────────────────────────────────

def zbuduj_config(m3_msun, nazwa, n_cykli=N_CYKLI, dt_dni=DT_DNI):
    """
    Tworzy config dla układu: para (50+50 M☉) na r12=1 AU,
    g3 (m3_msun M☉) na r13=200 AU.
    Prędkości orbitalne kepleriańskie.
    """
    m1 = M_PARA * M_SUN
    m2 = M_PARA * M_SUN
    m3 = m3_msun * M_SUN

    r12_m = R12 * AU
    r13_m = R13 * AU

    # Para na osi X, symetrycznie
    pos = np.array([
        [-r12_m / 2,  0.0],
        [ r12_m / 2,  0.0],
        [ r13_m,      0.0],
    ])

    # Prędkości kepleriańskie
    # Para obraca się wokół wspólnego środka masy
    v_para = np.sqrt(G * (m1 + m2) / r12_m) * 0.5  # prędkość każdej gwiazdy pary
    # g3 obiega środek masy całości
    M_tot = m1 + m2 + m3
    v3 = np.sqrt(G * (m1 + m2) / r13_m) * 0.85     # lekko sub-kepleriańska → związana

    vel = np.array([
        [0.0,  v_para],
        [0.0, -v_para],
        [0.0,  v3],
    ])

    # Wyzeruj pęd środka masy
    p_cm = m1 * vel[0] + m2 * vel[1] + m3 * vel[2]
    M_tot_kg = m1 + m2 + m3
    vel -= p_cm / M_tot_kg

    # Sprawdź czy układ jest związany
    from simulator import oblicz_energie
    E = oblicz_energie(pos, vel, np.array([m1, m2, m3]))
    if E >= 0:
        # Zmniejsz prędkość g3
        vel[2] *= 0.7
        p_cm = m1 * vel[0] + m2 * vel[1] + m3 * vel[2]
        vel -= p_cm / M_tot_kg

    return {
        "nazwa":            nazwa,
        "masy":             [m1, m2, m3],
        "pozycje":          pos.tolist(),
        "predkosci":        vel.tolist(),
        "dt":               DAY * dt_dni,
        "n_cykli":          n_cykli,
        "kroki_na_cykl":    60,      # ~18 min per cykl pary (okres ~18 dni)
        "max_blad_energii": 1e-2,
        "max_r_au":         50_000,  # g3 może daleko odlecieć — nie przerywaj
    }


# ── Główna pętla eksperymentu ────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("data",    exist_ok=True)

    wyniki = []

    print("\n" + "="*65)
    print("  MU TRANSITION EXPERIMENT")
    print(f"  Para: {M_PARA}+{M_PARA} M☉, r12={R12} AU | g3 na r13={R13} AU")
    print(f"  Cykle: {N_CYKLI} | dt={DT_DNI} dni")
    print("="*65 + "\n")

    for m3 in M3_SERIE:
        mu = m3 / (2 * M_PARA)
        nazwa = f"mu_trans_m3_{m3:.3f}".replace('.', 'p')

        print(f"\n{'─'*55}")
        print(f"  m3={m3:6.3f} M☉  →  mu={mu:.5f}  ({nazwa})")
        print(f"{'─'*55}")

        cfg = zbuduj_config(m3, nazwa, n_cykli=N_CYKLI, dt_dni=DT_DNI)

        df, powod = symuluj(cfg, folder="data", cicho=False)

        if powod is not None or df.empty:
            print(f"  ✗ Symulacja nieudana: {powod}")
            wyniki.append({
                "m3": m3, "mu": mu,
                "R2_sin2": np.nan, "omega": np.nan,
                "r13_cv": np.nan, "R_med": np.nan, "R_cv": np.nan,
                "n_zdarzen": 0, "n_cykli_ukon": 0,
                "status": powod or "blad"
            })
            continue

        print(f"  ✓ Symulacja OK — {len(df)} cykli")

        # --- Dopasowanie sin² do r13 ciągłego ---
        t_arr   = df['czas_lat'].values * YEAR  # sekundy
        r13_arr = df['r13'].values

        fit = dopasuj_sin2(t_arr, r13_arr)
        if fit:
            omega, A, C, r2_sin2 = fit
            print(f"  sin²: R²={r2_sin2:.4f}  ω={omega:.6f} [1/lat]  A={A:.1f} AU")
        else:
            omega, A, C, r2_sin2 = np.nan, np.nan, np.nan, np.nan
            print(f"  sin²: dopasowanie nieudane")

        # --- Analiza Poincarégo ---
        zdarz = zdarzenia_poincare(df)
        n_zdarz = len(zdarz)
        r13_cv = analiza_r13_cv(df, zdarz)
        R_med, R_cv = analiza_R_med(df, zdarz)

        print(f"  Poincaré: {n_zdarz} zdarzeń | CV(r13)={r13_cv:.4f} | R_med={R_med:.4f} | R_cv={R_cv:.4f}")

        wyniki.append({
            "m3":         m3,
            "mu":         mu,
            "R2_sin2":    r2_sin2,
            "omega":      omega,
            "A_sin2":     A,
            "C_sin2":     C,
            "r13_cv":     r13_cv,
            "r13_mean":   df['r13'].mean(),
            "r13_std":    df['r13'].std(),
            "R_med":      R_med,
            "R_cv":       R_cv,
            "n_zdarzen":  n_zdarz,
            "n_cykli_ukon": len(df),
            "blad_E_fin": df['blad_energii'].iloc[-1],
            "status":     "ok"
        })

    # ── Zapisz tabelę ────────────────────────────────────────────────────────
    tabela = pd.DataFrame(wyniki)
    tabela.to_csv("results/mu_transition_tabela.csv", index=False)
    print(f"\n  Tabela → results/mu_transition_tabela.csv")

    # ── Raport tekstowy ──────────────────────────────────────────────────────
    linie = []
    linie.append("=" * 65)
    linie.append("MU TRANSITION — RAPORT")
    linie.append(f"Para: {M_PARA}+{M_PARA} M☉, r12={R12} AU | g3 na r13={R13} AU")
    linie.append(f"Cykle: {N_CYKLI} | dt={DT_DNI} dni")
    linie.append("=" * 65)
    linie.append("")
    linie.append(f"{'m3':>8} {'mu':>8} {'R²(sin²)':>10} {'CV(r13)':>9} {'R_med':>8} {'n_zdarz':>8} {'status'}")
    linie.append("-" * 65)
    for r in wyniki:
        r2   = f"{r['R2_sin2']:.4f}" if not np.isnan(r['R2_sin2']) else "  nan "
        cv   = f"{r['r13_cv']:.4f}"  if not np.isnan(r['r13_cv'])  else "  nan "
        rmed = f"{r['R_med']:.4f}"   if not np.isnan(r['R_med'])   else "  nan "
        linie.append(f"{r['m3']:>8.3f} {r['mu']:>8.5f} {r2:>10} {cv:>9} {rmed:>8} {r['n_zdarzen']:>8} {r['status']}")
    linie.append("")

    # Próg przejścia
    ok = tabela[tabela['status'] == 'ok'].dropna(subset=['R2_sin2'])
    if len(ok) >= 3:
        prog = ok[ok['R2_sin2'] > 0.5]
        if len(prog) > 0:
            mu_prog = prog['mu'].max()
            linie.append(f"Przejście perturbacyjne (R²>0.5) poniżej mu ≈ {mu_prog:.5f}")
        else:
            linie.append("Brak wyraźnego przejścia perturbacyjnego (R²<0.5 dla wszystkich)")

    raport_txt = "\n".join(linie)
    with open("results/mu_transition_raport.txt", "w") as f:
        f.write(raport_txt)
    print(raport_txt)

    # ── Wykresy ──────────────────────────────────────────────────────────────
    rysuj_wykresy(tabela)
    print("\n  Wykres → results/mu_transition_wykres.png")
    print("\nGOTOWE.\n")


def rysuj_wykresy(tabela):
    ok = tabela[tabela['status'] == 'ok'].copy()
    if ok.empty:
        print("  Brak danych do wykresów.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0d0d1a')
    for ax in axes.flat:
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    mu_vals = ok['mu'].values

    # 1. R²(sin²) vs mu
    ax = axes[0, 0]
    r2_vals = ok['R2_sin2'].dropna()
    mu_r2   = ok.loc[ok['R2_sin2'].notna(), 'mu']
    ax.scatter(mu_r2, r2_vals, color='#ff9944', s=80, zorder=5)
    ax.plot(mu_r2, r2_vals, color='#ff9944', alpha=0.5, linewidth=1.5)
    ax.axhline(0.5,  color='#44ff88', linestyle='--', linewidth=1, label='R²=0.5 (próg)')
    ax.axhline(0.9,  color='#4488ff', linestyle='--', linewidth=1, label='R²=0.9 (dobry fit)')
    ax.set_xscale('log')
    ax.set_xlabel('μ = m3 / (m1+m2)')
    ax.set_ylabel('R² dopasowania sin²(ωt)')
    ax.set_title('R²(sin²) vs μ — przejście perturbacyjne')
    ax.legend(framealpha=0.3, labelcolor='white')
    ax.set_ylim(-0.1, 1.05)

    # 2. CV(r13) vs mu
    ax = axes[0, 1]
    cv_vals = ok['r13_cv'].dropna()
    mu_cv   = ok.loc[ok['r13_cv'].notna(), 'mu']
    ax.scatter(mu_cv, cv_vals, color='#44aaff', s=80, zorder=5)
    ax.plot(mu_cv, cv_vals, color='#44aaff', alpha=0.5, linewidth=1.5)
    ax.axhline(0.1, color='#ff4444', linestyle='--', linewidth=1, label='CV=0.1 (stabilny)')
    ax.set_xscale('log')
    ax.set_xlabel('μ = m3 / (m1+m2)')
    ax.set_ylabel('CV(r13) w zdarzeniach Poincarégo')
    ax.set_title('CV(r13) vs μ — stabilność orbity g3')
    ax.legend(framealpha=0.3, labelcolor='white')

    # 3. R²(sin²) vs CV(r13) — mapa przejścia
    ax = axes[1, 0]
    ok2 = ok.dropna(subset=['R2_sin2', 'r13_cv'])
    sc = ax.scatter(ok2['r13_cv'], ok2['R2_sin2'],
                    c=np.log10(ok2['mu'] + 1e-6),
                    cmap='plasma', s=100, zorder=5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('log₁₀(μ)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    ax.set_xlabel('CV(r13)')
    ax.set_ylabel('R²(sin²)')
    ax.set_title('Mapa przejścia: CV(r13) vs R²(sin²)')

    # Opisz punkty wartością mu
    for _, row in ok2.iterrows():
        ax.annotate(f"μ={row['mu']:.3f}",
                    (row['r13_cv'], row['R2_sin2']),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color='#aaaacc')

    # 4. R_med vs mu
    ax = axes[1, 1]
    ok3 = ok.dropna(subset=['R_med'])
    if not ok3.empty:
        ax.scatter(ok3['mu'], ok3['R_med'], color='#ff44aa', s=80, zorder=5)
        ax.plot(ok3['mu'], ok3['R_med'], color='#ff44aa', alpha=0.5, linewidth=1.5)
        ax.axhline(1.0, color='#ffffff', linestyle='--', linewidth=1, alpha=0.5, label='R_med=1 (stała amplituda)')
        ax.set_xscale('log')
        ax.set_xlabel('μ = m3 / (m1+m2)')
        ax.set_ylabel('Mediana R(n) w zdarzeniach Poincarégo')
        ax.set_title('R_med vs μ — mediana stosunku kolejnych zdarzeń')
        ax.legend(framealpha=0.3, labelcolor='white')

    fig.suptitle(
        f'Przejście perturbacyjne: m1=m2={M_PARA}M☉, r12={R12}AU, r13={R13}AU\n'
        f'μ = m3/(m1+m2) od {M3_SERIE[-1]/(2*M_PARA):.4f} do {M3_SERIE[0]/(2*M_PARA):.3f}',
        color='white', fontsize=13, y=1.01
    )

    plt.tight_layout()
    plt.savefig("results/mu_transition_wykres.png",
                dpi=150, bbox_inches='tight',
                facecolor='#0d0d1a')
    plt.close()


if __name__ == "__main__":
    main()