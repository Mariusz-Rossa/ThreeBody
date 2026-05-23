# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.
# custom_system.py

"""
CUSTOM SYSTEM — generator układów z zadanymi masami i odległościami
=====================================================================
Podajesz masy i odległości między gwiazdami, skrypt:
  1. Oblicza pozycje i prędkości orbitalne (układ kołowy)
  2. Weryfikuje czy układ jest grawitacyjnie związany
  3. Uruchamia symulację RK4
  4. Dopasowuje cos(k·log(t)) do r13 z przekroju Poincarégo
  5. Wyświetla wynik i zapisuje CSV

UŻYCIE:
  python custom_system.py                        # tryb interaktywny
  python custom_system.py --preset test          # układ jak s67249
  python custom_system.py --preset kompaktowy    # mały r13

PARAMETRY (tryb interaktywny):
  Podajesz kolejno:
    m1, m2, m3  [M_Sun]
    r12         [AU]  — odległość g1–g2
    r13         [AU]  — odległość g1–g3
    r23         [AU]  — odległość g2–g3 (opcjonalnie, obliczana geometrycznie)
    n_cykli     [int]

GEOMETRIA:
  g1 w środku układu współrzędnych.
  g2 umieszczona wzdłuż osi X w odległości r12.
  g3 umieszczona tak żeby r13 i r23 zgadzały się z podanymi.
  Prędkości: każda gwiazda dostaje prędkość orbitalną odpowiadającą
  przyciąganiu pozostałych dwóch, w kierunku prostopadłym do promienia.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, optimize
import os, json, argparse
from datetime import datetime
from numba import njit

# ── Stałe ────────────────────────────────────────────────────────────────────

G     = 6.674e-11
AU    = 1.496e11
YEAR  = 3.156e7
DAY   = 86400
M_SUN = 1.989e30

FOLDER = "data"
RESULTS = "results"

# ── Fizyka RK4 ────────────────────────────────────────────────────────────────

@njit
def przyspieszenia(pos, masy):
    n = len(masy)
    acc = np.zeros((n, 2))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_vec = pos[j] - pos[i]
            r_mag = max(np.linalg.norm(r_vec), 1e8)
            acc[i] += G * masy[j] / r_mag**2 * (r_vec / r_mag)
    return acc

@njit
def rk4(pos, vel, masy, dt):
    def deriv(p, v):
        return v, przyspieszenia(p, masy)
    v1, a1 = deriv(pos,              vel)
    v2, a2 = deriv(pos + 0.5*dt*v1, vel + 0.5*dt*a1)
    v3, a3 = deriv(pos + 0.5*dt*v2, vel + 0.5*dt*a2)
    v4, a4 = deriv(pos + dt*v3,      vel + dt*a3)
    return (pos + (dt/6)*(v1+2*v2+2*v3+v4),
            vel + (dt/6)*(a1+2*a2+2*a3+a4))

def energia(pos, vel, masy):
    Ek = sum(0.5*masy[i]*np.dot(vel[i],vel[i]) for i in range(3))
    Ep = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            r = max(np.linalg.norm(pos[j]-pos[i]), 1e8)
            Ep -= G*masy[i]*masy[j]/r
    return Ek + Ep

# ── Geometria i prędkości ─────────────────────────────────────────────────────

def buduj_uklad(m1, m2, m3, r12_au, r13_au, r23_au=None):
    """
    Umieszcza trzy gwiazdy w przestrzeni i nadaje prędkości orbitalne.

    Jeśli r23_au nie podano, g3 umieszczana pod kątem 60° od osi g1-g2.
    Prędkości dobierane automatycznie (binarnie) tak żeby Ek = 0.5·|Ep|
    — układ stabilnie związany z marginesem energetycznym.
    """
    r12 = r12_au * AU
    r13 = r13_au * AU

    if r23_au is not None:
        r23 = r23_au * AU
        if r13 + r23 < r12 or r12 + r23 < r13 or r12 + r13 < r23:
            print("  ⚠ Podane odległości nie tworzą trójkąta — używam kąta 60°")
            r23_au = None

    if r23_au is None:
        kat = np.pi / 3
        x3 = r13 * np.cos(kat)
        y3 = r13 * np.sin(kat)
    else:
        cos_kat = (r12**2 + r13**2 - r23**2) / (2 * r12 * r13)
        cos_kat = np.clip(cos_kat, -1, 1)
        kat = np.arccos(cos_kat)
        x3 = r13 * np.cos(kat)
        y3 = r13 * np.sin(kat)

    masy = np.array([m1, m2, m3])
    pos_raw = np.array([
        [0.0, 0.0],
        [r12, 0.0],
        [x3,  y3 ],
    ])

    r_cm = np.sum(masy[:, None] * pos_raw, axis=0) / np.sum(masy)
    pos = pos_raw - r_cm

    # Energia potencjalna
    Ep = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            r = np.linalg.norm(pos[j] - pos[i])
            Ep -= G * masy[i] * masy[j] / r

    M_total = np.sum(masy)

    def oblicz_vel(f):
        vel = np.zeros((3, 2))
        for i in range(3):
            r_i = np.linalg.norm(pos[i])
            if r_i < 1e9:
                continue
            # Prędkość od całkowitej masy układu — stabilniejsza niż od m_inne
            v_orb = np.sqrt(G * M_total / r_i) * f
            kier = pos[i] / r_i
            perp = np.array([-kier[1], kier[0]])
            vel[i] = v_orb * perp
        v_cm = np.sum(masy[:, None] * vel, axis=0) / np.sum(masy)
        vel -= v_cm
        return vel

    def ek(vel):
        return sum(0.5 * masy[i] * np.dot(vel[i], vel[i]) for i in range(3))

    # Szukaj f binarnie: cel Ek = 0.45 * |Ep| (bezpiecznie związany)
    cel_ek = 0.45 * abs(Ep)
    f_lo, f_hi = 0.01, 0.99
    for _ in range(60):
        f_mid = (f_lo + f_hi) / 2
        if ek(oblicz_vel(f_mid)) < cel_ek:
            f_lo = f_mid
        else:
            f_hi = f_mid

    vel = oblicz_vel(f_lo * 0.98)  # minimalny margines bezpieczeństwa
    E_final = ek(vel) + Ep
    print(f"  Czynnik prędkości: f={f_lo:.4f}  E={E_final:+.3e} J  "
          f"Ek/|Ep|={ek(vel)/abs(Ep):.3f}")

    return pos, vel, masy

# ── Symulacja ─────────────────────────────────────────────────────────────────

def symuluj(pos, vel, masy, n_cykli=1000, kroki_na_cykl=30,
            dt_dni=2, nazwa="custom", cicho=False):

    dt = dt_dni * DAY
    E0 = energia(pos, vel, masy)

    if E0 >= 0:
        print(f"  ✗ Układ NIEZWIĄZANY (E={E0:.2e} > 0)")
        print(f"    Spróbuj zmniejszyć prędkości lub zwiększyć odległości.")
        return pd.DataFrame(), 'niezwiazany'

    if not cicho:
        print(f"\n  Energia startowa: {E0:.3e} J  ✓ układ związany")
        print(f"  Symulacja: {n_cykli} cykli × {kroki_na_cykl} kroków × dt={dt_dni} dni")
        print(f"  Czas całkowity: {n_cykli*kroki_na_cykl*dt/YEAR:.1f} lat\n")

    wyniki = []
    pos = pos.copy()
    vel = vel.copy()

    for cykl in range(n_cykli):
        for _ in range(kroki_na_cykl):
            pos, vel = rk4(pos, vel, masy, dt)

        czas = (cykl + 1) * kroki_na_cykl * dt
        E = energia(pos, vel, masy)
        dE = abs((E - E0) / E0) if E0 != 0 else 0

        r12 = np.linalg.norm(pos[1]-pos[0]) / AU
        r13 = np.linalg.norm(pos[2]-pos[0]) / AU
        r23 = np.linalg.norm(pos[2]-pos[1]) / AU

        wyniki.append({
            'cykl': cykl+1,
            'czas_lat': czas/YEAR,
            'x1': pos[0,0]/AU, 'y1': pos[0,1]/AU,
            'x2': pos[1,0]/AU, 'y2': pos[1,1]/AU,
            'x3': pos[2,0]/AU, 'y3': pos[2,1]/AU,
            'r12': r12, 'r13': r13, 'r23': r23,
            'v1': np.linalg.norm(vel[0]),
            'v2': np.linalg.norm(vel[1]),
            'v3': np.linalg.norm(vel[2]),
            'energia': E,
            'blad_energii': dE,
        })

        if dE > 0.01:
            if not cicho:
                print(f"  ⚠ Błąd energii za duży w cyklu {cykl+1}: {dE:.2e} — stop")
            break

        if not cicho and (cykl+1) % 200 == 0:
            print(f"  Cykl {cykl+1:4d}/{n_cykli} | r12={r12:.2f} r13={r13:.2f} r23={r23:.2f} AU | błąd E={dE:.2e}")

    return pd.DataFrame(wyniki), None

# ── Analiza cos(k·log(t)) ─────────────────────────────────────────────────────

def model_coslog(t, A, k, C):
    t_safe = np.where(t > 0, t, 1e-10)
    return A * np.cos(k * np.log(t_safe)) + C

def model_sin2(t, A, omega, C):
    return A * np.sin(omega * t)**2 + C

def model_sin_n(n, A, omega, phi, C):
    return A * np.sin(omega * n + phi) + C

def r2(y, y_fit):
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / (ss_tot + 1e-30)

def analizuj(df, kolumna='r13', verbose=True):
    # Przekrój Poincarégo
    idx, _ = signal.find_peaks(-df['r12'].values, distance=3)
    if len(idx) < 15:
        print(f"  ⚠ Za mało zdarzeń Poincarégo ({len(idx)})")
        return None

    df_p = df.iloc[idx].reset_index(drop=True)
    t = df_p['czas_lat'].values.astype(float)
    n = np.arange(len(df_p), dtype=float)
    y = df_p[kolumna].values.astype(float)

    C0 = np.mean(y)
    A0 = (np.max(y) - np.min(y)) / 2.0 + 1e-6

    wyniki = {}

    # cos(k·log(t))
    for k_guess in [0.3, 0.5, 1.0, 2.0, 2.08, 2.094, 3.0, 5.0]:
        try:
            popt, _ = optimize.curve_fit(
                model_coslog, t[t>0], y[t>0],
                p0=[A0, k_guess, C0], maxfev=10000
            )
            yf = model_coslog(t, *popt)
            r2v = r2(y, yf)
            if 'coslog' not in wyniki or r2v > wyniki['coslog']['R2']:
                wyniki['coslog'] = {'A': abs(popt[0]), 'k': abs(popt[1]),
                                    'C': popt[2], 'R2': r2v, 'y_fit': yf,
                                    'wzor': f"{abs(popt[0]):.3f}·cos({abs(popt[1]):.5f}·log(t)) + {popt[2]:.3f}"}
        except Exception:
            pass

    # sin²(ω·t)
    for omega_guess in [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1]:
        try:
            popt, _ = optimize.curve_fit(
                model_sin2, t, y,
                p0=[A0, omega_guess, C0],
                bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=8000
            )
            yf = model_sin2(t, *popt)
            r2v = r2(y, yf)
            if 'sin2' not in wyniki or r2v > wyniki['sin2']['R2']:
                wyniki['sin2'] = {'A': abs(popt[0]), 'omega': abs(popt[1]),
                                  'C': popt[2], 'R2': r2v, 'y_fit': yf,
                                  'wzor': f"{abs(popt[0]):.3f}·sin²({abs(popt[1]):.5f}·t) + {popt[2]:.3f}"}
        except Exception:
            pass

    # A·sin(ω·n + φ) + C
    for omega_guess in [0.01, 0.05, 0.1, 0.5, 5.69]:
        try:
            popt, _ = optimize.curve_fit(
                model_sin_n, n, y,
                p0=[A0, omega_guess, 0.0, C0], maxfev=8000
            )
            yf = model_sin_n(n, *popt)
            r2v = r2(y, yf)
            if 'sin_n' not in wyniki or r2v > wyniki['sin_n']['R2']:
                wyniki['sin_n'] = {'A': abs(popt[0]), 'omega': abs(popt[1]),
                                   'phi': popt[2], 'C': popt[3], 'R2': r2v, 'y_fit': yf,
                                   'wzor': f"{abs(popt[0]):.3f}·sin({abs(popt[1]):.5f}·n + {popt[2]:.3f}) + {popt[3]:.3f}"}
        except Exception:
            pass

    if not wyniki:
        return None

    najlepszy = max(wyniki, key=lambda k: wyniki[k]['R2'])

    if verbose:
        print(f"\n  {'─'*55}")
        print(f"  Analiza {kolumna} — {len(df_p)} zdarzeń Poincarégo")
        print(f"  r13: mean={y.mean():.2f} AU  std={y.std():.3f} AU  CV={y.std()/y.mean():.3f}")
        print(f"  {'─'*55}")
        for nazwa_m, w in sorted(wyniki.items(), key=lambda x: -x[1]['R2']):
            gwiazdki = "★★★" if w['R2'] > 0.5 else ("★★" if w['R2'] > 0.2 else ("★" if w['R2'] > 0.05 else "  "))
            print(f"  {gwiazdki} {nazwa_m:8s} R²={w['R2']:.4f}  →  {w['wzor']}")

        best = wyniki[najlepszy]
        print(f"\n  NAJLEPSZY: {najlepszy}  R²={best['R2']:.4f}")
        if najlepszy == 'coslog':
            k_val = best['k']
            print(f"  k = {k_val:.6f}")
            print(f"  k / (2π/3) = {k_val / (2*np.pi/3):.6f}  (s67249 miał: 0.9929)")
            print(f"  k / π      = {k_val / np.pi:.6f}")
            print(f"  k / 2      = {k_val / 2:.6f}")

    wyniki['_najlepszy'] = najlepszy
    wyniki['_df_p'] = df_p
    wyniki['_t'] = t
    wyniki['_y'] = y
    return wyniki

# ── Wykres ────────────────────────────────────────────────────────────────────

def rysuj_wynik(df, wyniki_r13, nazwa, masy_msun, r_start):
    os.makedirs(RESULTS, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0f0f13')

    m1, m2, m3 = masy_msun
    tytul = (f"Custom: m1={m1:.2f} m2={m2:.2f} m3={m3:.2f} M☉  |  "
             f"r12={r_start[0]:.1f} r13={r_start[1]:.1f} r23={r_start[2]:.1f} AU")
    fig.suptitle(tytul, color='#e8e8f0', fontsize=11)

    KOLORY = ['#4a9eff', '#ff6b4a', '#4affb0']

    # Panel 1: Orbity
    ax = axes[0]
    ax.set_facecolor('#0f0f13')
    for i, (xk, yk, kol) in enumerate([('x1','y1',KOLORY[0]),('x2','y2',KOLORY[1]),('x3','y3',KOLORY[2])]):
        ax.plot(df[xk], df[yk], color=kol, lw=0.4, alpha=0.5)
        ax.scatter(df[xk].iloc[-1], df[yk].iloc[-1], color=kol, s=30, zorder=5, label=f'G{i+1}')
    ax.set_title('Orbity [AU]', color='#e8e8f0', fontsize=9)
    ax.legend(fontsize=8, facecolor='#17171f', labelcolor='#e8e8f0')
    ax.tick_params(colors='#6b6b80')
    ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_color('#333340')

    # Panel 2: r13 w czasie
    ax = axes[1]
    ax.set_facecolor('#17171f')
    ax.plot(df['czas_lat'], df['r13'], color='#4affb0', lw=0.6, alpha=0.8, label='r13 ciągłe')
    if wyniki_r13:
        df_p = wyniki_r13['_df_p']
        ax.scatter(df_p['czas_lat'], df_p['r13'], color='red', s=8, zorder=5, label='Zdarzenia')
        najl = wyniki_r13['_najlepszy']
        w = wyniki_r13[najl]
        ax.plot(df_p['czas_lat'], w['y_fit'], color='#ffd04a', lw=1.5,
                label=f"{najl} R²={w['R2']:.3f}", zorder=6)
    ax.set_title('r13 + dopasowanie', color='#e8e8f0', fontsize=9)
    ax.set_xlabel('czas [lat]', color='#6b6b80', fontsize=8)
    ax.legend(fontsize=8, facecolor='#17171f', labelcolor='#e8e8f0', framealpha=0.5)
    ax.tick_params(colors='#6b6b80')
    for sp in ax.spines.values(): sp.set_color('#333340')

    # Panel 3: r12, r13, r23
    ax = axes[2]
    ax.set_facecolor('#17171f')
    for kol_name, kol in [('r12',KOLORY[0]),('r13',KOLORY[2]),('r23',KOLORY[1])]:
        ax.plot(df['czas_lat'], df[kol_name], color=kol, lw=0.6, alpha=0.7, label=kol_name)
    ax.set_title('Odległości r12 r13 r23 [AU]', color='#e8e8f0', fontsize=9)
    ax.set_xlabel('czas [lat]', color='#6b6b80', fontsize=8)
    ax.legend(fontsize=8, facecolor='#17171f', labelcolor='#e8e8f0', framealpha=0.5)
    ax.tick_params(colors='#6b6b80')
    for sp in ax.spines.values(): sp.set_color('#333340')

    znacznik = datetime.now().strftime("%Y%m%d_%H%M%S")
    plik = os.path.join(RESULTS, f"custom_{nazwa}_{znacznik}.png")
    plt.savefig(plik, dpi=110, bbox_inches='tight', facecolor='#0f0f13')
    plt.close()
    print(f"\n  → Wykres: {plik}")
    return plik

# ── Zapis CSV ─────────────────────────────────────────────────────────────────

def zapisz(df, cfg, nazwa):
    os.makedirs(FOLDER, exist_ok=True)
    znacznik = datetime.now().strftime("%Y%m%d_%H%M%S")
    plik_csv  = os.path.join(FOLDER, f"custom_{nazwa}_{znacznik}.csv")
    plik_json = os.path.join(FOLDER, f"custom_{nazwa}_{znacznik}_config.json")
    df.to_csv(plik_csv, index=False)
    with open(plik_json, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"  → CSV:  {plik_csv}")
    return plik_csv

# ── Presets ───────────────────────────────────────────────────────────────────

PRESETS = {
    'test': {
        # Odtwarza warunki podobne do s67249 (małe r13, duże m3)
        'opis':    'Podobny do s67249 — małe r13, duże m3',
        'm1': 1.0, 'm2': 1.5, 'm3': 2.0,
        'r12': 8.0, 'r13': 6.0, 'r23': None,
        'n_cykli': 1000, 'dt_dni': 1, 'kroki_na_cykl': 40,
    },
    'kompaktowy': {
        'opis':    'Bardzo bliski układ — r13 < 5 AU',
        'm1': 1.2, 'm2': 1.8, 'm3': 1.9,
        'r12': 4.0, 'r13': 4.5, 'r23': None,
        'n_cykli': 1000, 'dt_dni': 1, 'kroki_na_cykl': 40,
    },
    'rowny': {
        'opis':    'Równe masy, małe r',
        'm1': 1.0, 'm2': 1.0, 'm3': 1.0,
        'r12': 5.0, 'r13': 5.0, 'r23': 5.0,
        'n_cykli': 1000, 'dt_dni': 1, 'kroki_na_cykl': 40,
    },
}

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Custom System — symulacja trzech ciał z zadanymi masami i odległościami',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady:
  python custom_system.py                          # tryb interaktywny
  python custom_system.py --preset test            # układ podobny do s67249
  python custom_system.py --preset kompaktowy      # mały r13

  # Pełne parametry z linii komend:
  python custom_system.py --m1 1.0 --m2 1.5 --m3 2.0 --r12 8 --r13 6 --n 1000
        '''
        """
    )
    parser.add_argument('--preset', choices=list(PRESETS.keys()), help='Gotowy preset')
    parser.add_argument('--m1',  type=float, help='Masa g1 [M_Sun]')
    parser.add_argument('--m2',  type=float, help='Masa g2 [M_Sun]')
    parser.add_argument('--m3',  type=float, help='Masa g3 [M_Sun]')
    parser.add_argument('--r12', type=float, help='Odległość g1-g2 [AU]')
    parser.add_argument('--r13', type=float, help='Odległość g1-g3 [AU]')
    parser.add_argument('--r23', type=float, default=None, help='Odległość g2-g3 [AU] (opcja)')
    parser.add_argument('--n',   type=int,   default=1000, help='Liczba cykli (domyślnie 1000)')
    parser.add_argument('--dt',  type=float, default=2.0,  help='Krok dt [dni] (domyślnie 2)')
    parser.add_argument('--kpc', type=int,   default=30,   help='Kroków na cykl (domyślnie 30)')
    parser.add_argument('--nazwa', type=str, default='uk', help='Nazwa układu')
    args = parser.parse_args()

    print("\n" + "="*57)
    print("  CUSTOM SYSTEM — symulator z zadanymi parametrami")
    print("="*57)

    # Wybór źródła parametrów
    if args.preset:
        p = PRESETS[args.preset]
        print(f"\n  Preset: {args.preset} — {p['opis']}")
        m1, m2, m3 = p['m1'], p['m2'], p['m3']
        r12, r13   = p['r12'], p['r13']
        r23        = p.get('r23', None)
        n_cykli    = p.get('n_cykli', args.n)
        dt_dni     = p.get('dt_dni', args.dt)
        kpc        = p.get('kroki_na_cykl', args.kpc)
        nazwa      = args.preset

    elif args.m1 and args.m2 and args.m3 and args.r12 and args.r13:
        m1, m2, m3 = args.m1, args.m2, args.m3
        r12, r13   = args.r12, args.r13
        r23        = args.r23
        n_cykli    = args.n
        dt_dni     = args.dt
        kpc        = args.kpc
        nazwa      = args.nazwa

    else:
        # Tryb interaktywny
        print("\n  Podaj parametry układu:\n")
        try:
            m1  = float(input("  Masa g1 [M_Sun]: "))
            m2  = float(input("  Masa g2 [M_Sun]: "))
            m3  = float(input("  Masa g3 [M_Sun]: "))
            r12 = float(input("  Odległość r12 [AU]: "))
            r13 = float(input("  Odległość r13 [AU]: "))
            r23_inp = input("  Odległość r23 [AU] (Enter = oblicz automatycznie): ").strip()
            r23 = float(r23_inp) if r23_inp else None
            n_cykli = int(input(f"  Liczba cykli [Enter = 1000]: ").strip() or "1000")
            dt_dni  = float(input(f"  Krok dt [dni, Enter = 2]: ").strip() or "2")
            kpc     = int(input(f"  Kroków na cykl [Enter = 30]: ").strip() or "30")
            nazwa   = input("  Nazwa układu [Enter = 'uk']: ").strip() or "uk"
        except (ValueError, EOFError) as e:
            print(f"\n  ✗ Błąd: {e}")
            return

    # Podsumowanie parametrów
    print(f"\n  {'─'*55}")
    print(f"  m1={m1:.3f}  m2={m2:.3f}  m3={m3:.3f}  M☉")
    print(f"  r12={r12:.2f}  r13={r13:.2f}  r23={'auto' if r23 is None else f'{r23:.2f}'}  AU")
    print(f"  n_cykli={n_cykli}  dt={dt_dni} dni  kpc={kpc}")
    print(f"  {'─'*55}")

    # Zbuduj układ
    masy_si = np.array([m1, m2, m3]) * M_SUN
    pos, vel, masy = buduj_uklad(m1*M_SUN, m2*M_SUN, m3*M_SUN, r12, r13, r23)

    # Rzeczywiste odległości startowe
    r12_real = np.linalg.norm(pos[1]-pos[0]) / AU
    r13_real = np.linalg.norm(pos[2]-pos[0]) / AU
    r23_real = np.linalg.norm(pos[2]-pos[1]) / AU
    print(f"\n  Odległości startowe po przesunięciu do środka masy:")
    print(f"  r12={r12_real:.3f}  r13={r13_real:.3f}  r23={r23_real:.3f}  AU")

    r_start = (r12_real, r13_real, r23_real)

    # Symulacja
    df, powod = symuluj(pos, vel, masy, n_cykli=n_cykli,
                        kroki_na_cykl=kpc, dt_dni=dt_dni, nazwa=nazwa)

    if df.empty or powod == 'niezwiazany':
        print("\n  ✗ Symulacja nieudana.")
        return

    print(f"\n  ✓ Symulacja zakończona: {len(df)} cykli, "
          f"{df['czas_lat'].iloc[-1]:.1f} lat")
    print(f"  Błąd energii (końcowy): {df['blad_energii'].iloc[-1]:.2e}")

    # Analiza
    print(f"\n  Analizuję r13...")
    wyniki = analizuj(df, 'r13', verbose=True)

    # Zapis
    cfg = {
        'masy': (masy_si).tolist(),
        'r12_au': r12, 'r13_au': r13, 'r23_au': r23,
        'n_cykli': n_cykli, 'dt': dt_dni*DAY, 'kroki_na_cykl': kpc,
    }
    zapisz(df, cfg, nazwa)
    rysuj_wynik(df, wyniki, nazwa, (m1, m2, m3), r_start)

    print(f"\n{'='*57}")
    print(f"  GOTOWE")
    print(f"{'='*57}\n")

if __name__ == '__main__':
    main()