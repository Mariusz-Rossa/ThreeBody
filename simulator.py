# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.
# simulator.py

"""
SYMULATOR PROBLEMU TRZECH CIAŁ
================================
Program 1 — generuje dane o ruchu trzech gwiazd pod wpływem grawitacji.

Używa metody integracji Runge-Kutta 4 rzędu (RK4) — dokładnej i stabilnej.
Zapisuje wyniki co określoną liczbę kroków do pliku CSV.

KLUCZOWE ULEPSZENIA:
  - Sprawdzanie czy układ jest grawitacyjnie związany (energia < 0)
    przed rozpoczęciem symulacji
  - Monitoring rozpadu w trakcie: jeśli para gwiazd ucieka > max_r AU
    i energia par jest dodatnia → układ się rozlatuje → stop
  - Tryb serii szuka aż znajdzie N ZWIĄZANYCH układów (nie odpada po N próbach)
  - Możliwość dynamicznego określenia liczby cykli podczas uruchamiania

TRYBY:
  1 — Trójkąt równoboczny (test symulatora)
  2 — Alpha Centauri (układ hierarchiczny)
  3 — Losowy układ związany (średni: 10-50 AU)
  4 — Losowy układ związany (bliski: 3-15 AU)
  5 — Losowy układ związany (hierarchiczny)
  6 — Seria 10 układów związanych (średni) ← GŁÓWNY TRYB
  7 — Seria 20 układów związanych (mieszane zakresy)
"""

import numpy as np
import pandas as pd
import os
import json
from numba import njit
from datetime import datetime


# ── Stałe fizyczne ──────────────────────────────────────────────────────────

G     = 6.674e-11
AU    = 1.496e11
YEAR  = 3.156e7
DAY   = 86400
M_SUN = 1.989e30

N_CYKLI = 1000

# ── Fizyka — rdzeń symulatora ────────────────────────────────────────────────
@njit
def oblicz_przyspieszenia(pos, masy):
    n = len(masy)
    acc = np.zeros((n, 2))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_vec = pos[j] - pos[i]
            r_mag = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
            if r_mag < 1e8:
                r_mag = 1e8
            acc[i] += G * masy[j] / r_mag**2 * (r_vec / r_mag)
    return acc

@njit
def rk4_krok(pos, vel, masy, dt):
    v1 = vel
    a1 = oblicz_przyspieszenia(pos,              masy)
    v2 = vel + 0.5*dt*a1
    a2 = oblicz_przyspieszenia(pos + 0.5*dt*v1,  masy)
    v3 = vel + 0.5*dt*a2
    a3 = oblicz_przyspieszenia(pos + 0.5*dt*v2,  masy)
    v4 = vel + dt*a3
    a4 = oblicz_przyspieszenia(pos + dt*v3,       masy)
    nowe_pos = pos + (dt/6.0) * (v1 + 2.0*v2 + 2.0*v3 + v4)
    nowe_vel = vel + (dt/6.0) * (a1 + 2.0*a2 + 2.0*a3 + a4)
    return nowe_pos, nowe_vel


def oblicz_energie(pos, vel, masy):
    """Energia całkowita układu. Jeśli < 0 — układ związany grawitacyjnie."""
    Ek = sum(0.5 * masy[i] * np.dot(vel[i], vel[i]) for i in range(len(masy)))
    Ep = 0.0
    for i in range(len(masy)):
        for j in range(i+1, len(masy)):
            r = max(np.linalg.norm(pos[j] - pos[i]), 1e8)
            Ep -= G * masy[i] * masy[j] / r
    return Ek + Ep


def oblicz_energie_par(pos, vel, masy):
    """
    Energia każdej pary gwiazd osobno.
    Jeśli energia pary > 0 i odległość rośnie → ta para się rozlatuje.
    Zwraca (E12, E13, E23).
    """
    pary = [(0,1), (0,2), (1,2)]
    energie = []
    for i, j in pary:
        r = max(np.linalg.norm(pos[j] - pos[i]), 1e8)
        v_rel = vel[j] - vel[i]
        Ek_red = 0.5 * (masy[i]*masy[j]/(masy[i]+masy[j])) * np.dot(v_rel, v_rel)
        Ep_ij  = -G * masy[i] * masy[j] / r
        energie.append(Ek_red + Ep_ij)
    return tuple(energie)


def oblicz_moment_pedu(pos, vel, masy):
    return sum(masy[i] * (pos[i][0]*vel[i][1] - pos[i][1]*vel[i][0])
               for i in range(len(masy)))


def czy_zwiazany(pos, vel, masy):
    """
    Sprawdza czy układ jest grawitacyjnie związany.
    Warunek: energia całkowita < 0.
    Zwraca (True/False, energia).
    """
    E = oblicz_energie(pos, vel, masy)
    return E < 0, E


def czy_rozlatuje_sie(pos, vel, masy, max_r_au=500):
    """
    Sprawdza czy układ zaczyna się rozlatywać w trakcie symulacji.
    Warunek: dowolna para ma odległość > max_r_au AU
             ORAZ energia tej pary > 0 (uciekają od siebie).
    Zwraca (True/False, opis).
    """
    pary = [(0,1,'r12'), (0,2,'r13'), (1,2,'r23')]
    E_par = oblicz_energie_par(pos, vel, masy)

    for idx, (i, j, nazwa) in enumerate(pary):
        r = np.linalg.norm(pos[j] - pos[i]) / AU
        if r > max_r_au and E_par[idx] > 0:
            return True, f"{nazwa}={r:.0f} AU, E_para={E_par[idx]:.2e}"

    return False, ""


# ── Główna pętla symulacji ───────────────────────────────────────────────────

def symuluj(config, folder="data", cicho=False):
    """
    Przeprowadza symulację i zapisuje CSV + JSON.
    """
    os.makedirs(folder, exist_ok=True)

    masy     = np.array(config["masy"])
    pos      = np.array(config["pozycje"])
    vel      = np.array(config["predkosci"])
    dt       = config["dt"]
    n_cykli  = config["n_cykli"]
    kpc      = config["kroki_na_cykl"]
    nazwa    = config.get("nazwa", "symulacja")
    max_blad = config.get("max_blad_energii", 1e-2)
    max_r    = config.get("max_r_au", 500)

    zwiazany, E0 = czy_zwiazany(pos, vel, masy)
    if not zwiazany:
        if not cicho:
            print(f"  ✗ Układ NIEZWIĄZANY (E={E0:.2e} > 0) — pomijam")
        return pd.DataFrame(), 'niezwiazany'

    if not cicho:
        print(f"\n{'='*55}")
        print(f"  SYMULACJA: {nazwa}")
        print(f"{'='*55}")
        print(f"  Energia startowa: {E0:.3e} J  (związany ✓)")
        print(f"  Cykle:            {n_cykli}")
        print(f"  Krok dt:          {dt/DAY:.2f} dni")
        print(f"  Czas całkowity:   {n_cykli*kpc*dt/YEAR:.1f} lat")
        print(f"{'='*55}\n")

    wyniki    = []
    powod_stopu = None

    for cykl in range(n_cykli):
        for _ in range(kpc):
            pos, vel = rk4_krok(pos, vel, masy, dt)

        czas = (cykl + 1) * kpc * dt
        E    = oblicz_energie(pos, vel, masy)
        L    = oblicz_moment_pedu(pos, vel, masy)
        dE   = abs((E - E0) / E0) if E0 != 0 else 0

        r12 = np.linalg.norm(pos[1] - pos[0]) / AU
        r13 = np.linalg.norm(pos[2] - pos[0]) / AU
        r23 = np.linalg.norm(pos[2] - pos[1]) / AU

        wyniki.append({
            "cykl":        cykl + 1,
            "czas_lat":    czas / YEAR,
            "x1": pos[0][0]/AU, "y1": pos[0][1]/AU,
            "x2": pos[1][0]/AU, "y2": pos[1][1]/AU,
            "x3": pos[2][0]/AU, "y3": pos[2][1]/AU,
            "r12": r12, "r13": r13, "r23": r23,
            "v1": np.linalg.norm(vel[0]),
            "v2": np.linalg.norm(vel[1]),
            "v3": np.linalg.norm(vel[2]),
            "energia":      E,
            "moment_pedu":  L,
            "blad_energii": dE,
        })

        if dE > max_blad:
            powod_stopu = 'blad_energii'
            if not cicho:
                print(f"  ⚠ Błąd energii za duży w cyklu {cykl+1}: {dE:.2e}")
            break

        rozlatuje, opis = czy_rozlatuje_sie(pos, vel, masy, max_r)
        if rozlatuje:
            powod_stopu = 'rozlot'
            if not cicho:
                print(f"  ⚠ Układ rozlatuje się w cyklu {cykl+1}: {opis}")
            break

        if not cicho and (cykl + 1) % 100 == 0:
            print(f"  Cykl {cykl+1:4d}/{n_cykli} | "
                  f"t={czas/YEAR:7.1f} lat | "
                  f"r12={r12:.1f} AU | błąd E={dE:.2e}")

    df = pd.DataFrame(wyniki)

    if powod_stopu is None:
        znacznik  = datetime.now().strftime("%Y%m%d_%H%M%S")
        plik_csv  = os.path.join(folder, f"{nazwa}_{znacznik}.csv")
        plik_json = os.path.join(folder, f"{nazwa}_{znacznik}_config.json")
        df.to_csv(plik_csv, index=False)
        cfg_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in config.items()}
        with open(plik_json, "w") as f:
            json.dump(cfg_json, f, indent=2)
        if not cicho:
            print(f"\n  ✓ Zapisano: {plik_csv}")
            print(f"  ✓ Błąd energii (końcowy): {df['blad_energii'].iloc[-1]:.2e}\n")

    return df, powod_stopu


# ── Generator losowych układów ───────────────────────────────────────────────

def losowy_uklad(seed=None, zakres="sredni", n_cykli=N_CYKLI):
    """
    Generuje fizycznie sensowny losowy układ trzech gwiazd.
    """
    rng  = np.random.default_rng(seed)
    masy = rng.uniform(0.5, 2.0, 3) * M_SUN

    if zakres == "hierarchiczny":
        r_para    = rng.uniform(5,   25)  * AU
        r_trzecia = rng.uniform(200, 800) * AU
        kat1 = rng.uniform(0, 2*np.pi)
        kat3 = rng.uniform(0, 2*np.pi)

        pos = np.array([
            [ r_para/2 * np.cos(kat1),  r_para/2 * np.sin(kat1)],
            [-r_para/2 * np.cos(kat1), -r_para/2 * np.sin(kat1)],
            [ r_trzecia * np.cos(kat3),  r_trzecia * np.sin(kat3)],
        ])

        m12 = masy[0] + masy[1]
        v_para    = np.sqrt(G * m12 / r_para)    * rng.uniform(0.6, 0.95)
        v_trzecia = np.sqrt(G * sum(masy) / r_trzecia) * rng.uniform(0.4, 0.85)

        perp1 = np.array([-np.sin(kat1), np.cos(kat1)])
        perp3 = np.array([-np.sin(kat3), np.cos(kat3)])

        vel = np.array([
            perp1 * v_para * masy[1]/m12,
            perp1 * -v_para * masy[0]/m12,
            perp3 * v_trzecia,
        ])
        dt_dni, kpc, max_r = 5, 30, 2000

    else:
        if zakres == "bliski":
            r_min, r_max, dt_dni, kpc, max_r = 3,  15, 1, 30, 200
        else:  # sredni
            r_min, r_max, dt_dni, kpc, max_r = 10, 50, 2, 30, 2000

        katy = np.sort(rng.uniform(0, 2*np.pi, 3))
        dyst = rng.uniform(r_min, r_max, 3) * AU

        pos = np.array([
            [dyst[i] * np.cos(katy[i]), dyst[i] * np.sin(katy[i])]
            for i in range(3)
        ])

        r_cm = np.sum(masy[:, None] * pos, axis=0) / np.sum(masy)
        pos -= r_cm

        vel = np.zeros((3, 2))
        for i in range(3):
            r_i = np.linalg.norm(pos[i])
            if r_i < 1e9:
                continue
            m_inne = sum(masy[j] for j in range(3) if j != i)
            v_orb  = np.sqrt(G * m_inne / r_i) * rng.uniform(0.5, 0.85)
            kier   = pos[i] / r_i
            perp   = np.array([-kier[1], kier[0]])
            vel[i] = rng.choice([-1, 1]) * v_orb * perp

    v_cm = np.sum(masy[:, None] * vel, axis=0) / np.sum(masy)
    vel -= v_cm

    return {
        "masy":             masy.tolist(),
        "pozycje":          pos.tolist(),
        "predkosci":        vel.tolist(),
        "dt":               DAY * dt_dni,
        "n_cykli":          n_cykli,
        "kroki_na_cykl":    kpc,
        "max_blad_energii": 1e-2,
        "max_r_au":         max_r,
    }


def generuj_serie(n_docelowych=100, zakres="sredni", folder="data", n_cykli=N_CYKLI):
    """
    Generuje serię układów AŻ DO uzyskania n_docelowych ZWIĄZANYCH układów.
    """
    print(f"\n{'='*57}")
    print(f"  SZUKAM {n_docelowych} ZWIĄZANYCH układów | zakres: {zakres} | cykle: {n_cykli}")
    print(f"{'='*57}\n")

    znalezione = 0
    prob       = 0
    max_prob   = n_docelowych * 5
    wyniki_serii = []
    seed_base  = int(datetime.now().timestamp()) % 100000

    while znalezione < n_docelowych and prob < max_prob:
        seed = prob * 137 + seed_base
        cfg  = losowy_uklad(seed=seed, zakres=zakres, n_cykli=n_cykli)
        cfg["nazwa"] = f"zwiazany_{zakres}_s{seed:04d}"

        m1, m2, m3 = [cfg["masy"][j]/M_SUN for j in range(3)]
        prob += 1

        print(f"  [próba {prob:2d}] seed={seed:4d} | "
              f"masy: {m1:.2f} {m2:.2f} {m3:.2f} M☉ ", end="", flush=True)

        df, powod = symuluj(cfg, folder=folder, cicho=True)

        if powod == 'niezwiazany':
            print("→ ✗ niezwiązany (E>0)")
        elif powod == 'rozlot':
            cykli = len(df)
            print(f"→ ✗ rozlatuje się (cykl {cykli})")
        elif powod == 'blad_energii':
            print(f"→ ✗ błąd numeryczny (cykl {len(df)})")
        else:
            znalezione += 1
            blad = df['blad_energii'].iloc[-1]
            r_max = max(df['r12'].max(), df['r13'].max(), df['r23'].max())
            print(f"→ ✓ ZWIĄZANY [{znalezione}/{n_docelowych}] | "
                  f"błąd E={blad:.2e} | max r={r_max:.1f} AU")
            wyniki_serii.append({
                "seed":    seed,
                "nazwa":   cfg["nazwa"],
                "cykli":   len(df),
                "blad_E":  blad,
                "max_r":   r_max,
                "masa1":   m1, "masa2": m2, "masa3": m3,
            })

    print(f"\n{'─'*57}")
    if znalezione == n_docelowych:
        print(f"  ✓ Znaleziono {znalezione} układów w {prob} próbach")
    else:
        print(f"  ⚠ Znaleziono tylko {znalezione}/{n_docelowych} w {prob} próbach")
    print(f"  Dane zapisane w: {folder}/")
    print(f"{'─'*57}\n")

    if wyniki_serii:
        plik = os.path.join(folder, f"seria_{zakres}_podsumowanie.csv")
        pd.DataFrame(wyniki_serii).to_csv(plik, index=False)
        print(f"  Podsumowanie → {plik}\n")

    return wyniki_serii


# ── Gotowe konfiguracje referencyjne ────────────────────────────────────────

def config_trojkat_rownoboczny(n_cykli=N_CYKLI):
    m = 1.0 * M_SUN
    a = 5.0 * AU
    p1 = np.array([ a/np.sqrt(3), 0.0])
    p2 = np.array([-a/(2*np.sqrt(3)),  a/2])
    p3 = np.array([-a/(2*np.sqrt(3)), -a/2])
    omega = np.sqrt(G * m / a**3)
    v = omega * a / np.sqrt(3)
    return {
        "nazwa": "trojkat_rownoboczny",
        "masy":      [m, m, m],
        "pozycje":   [p1.tolist(), p2.tolist(), p3.tolist()],
        "predkosci": [[0.0, v], [-v*np.sqrt(3)/2, -v/2], [v*np.sqrt(3)/2, -v/2]],
        "dt": DAY*2, "n_cykli": n_cykli, "kroki_na_cykl": 50,
        "max_blad_energii": 1e-2, "max_r_au": 200,
    }


def config_alpha_centauri(n_cykli=N_CYKLI):
    mA, mB, mC = 1.1*M_SUN, 0.9*M_SUN, 0.12*M_SUN
    v_AB = np.sqrt(G*(mA+mB)/(23*AU))
    return {
        "nazwa": "alpha_centauri",
        "masy":      [mA, mB, mC],
        "pozycje":   [[11.5*AU, 0.0], [-11.5*AU, 0.0], [0.0, 13000*AU]],
        "predkosci": [
            [0.0,  v_AB * mB/(mA+mB) * 2],
            [0.0, -v_AB * mA/(mA+mB) * 2],
            [np.sqrt(G*(mA+mB)/(13000*AU)) * 0.3, 0.0],
        ],
        "dt": DAY*30, "n_cykli": n_cykli, "kroki_na_cykl": 12,
        "max_blad_energii": 1e-3, "max_r_au": 20000,
    }


# ── Menu główne ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🌟 SYMULATOR PROBLEMU TRZECH CIAŁ")
    print("─" * 45)
    print("  1 — Trójkąt równoboczny  (test symulatora)")
    print("  2 — Alpha Centauri       (układ realistyczny)")
    print("  3 — Losowy układ związany (średni: 10-50 AU)")
    print("  4 — Losowy układ związany (bliski: 3-15 AU)")
    print("  5 — Losowy układ związany (hierarchiczny)")
    print("  6 — Seria 100 związanych  (średni) ← GŁÓWNY TRYB")
    print("  7 — Seria 20 związanych  (mieszane zakresy)")
    print("─" * 45)

    wybor = input("\n  Twój wybór [1-7]: ").strip()

    # Zabezpieczenie przed niewłaściwym wyborem przed pytaniem o cykle
    if wybor in [str(i) for i in range(1, 8)]:
        try:
            wejscie_cykli = input(f"  Podaj liczbę cykli (Enter = domyślnie {N_CYKLI}): ").strip()
            wybrane_cykle = int(wejscie_cykli) if wejscie_cykli else N_CYKLI
        except ValueError:
            print(f"  ⚠ Błędna wartość, ustawiono domyślnie: {N_CYKLI}")
            wybrane_cykle = N_CYKLI
    else:
        wybrane_cykle = N_CYKLI # Zabezpieczenie (choć kod i tak trafi do bloku 'else' poniżej)

    if wybor == "1":
        symuluj(config_trojkat_rownoboczny(n_cykli=wybrane_cykle))
    elif wybor == "2":
        symuluj(config_alpha_centauri(n_cykli=wybrane_cykle))
    elif wybor in ("3", "4", "5"):
        zakres = {"3": "sredni", "4": "bliski", "5": "hierarchiczny"}[wybor]
        # Szukaj aż znajdzie związany
        for proba in range(20):
            seed = int(datetime.now().timestamp()) + proba
            cfg  = losowy_uklad(seed=seed, zakres=zakres, n_cykli=wybrane_cykle)
            cfg["nazwa"] = f"losowy_{zakres}"
            df, powod = symuluj(cfg)
            if powod is None:
                break
            print(f"  Próba {proba+1}: {powod} — szukam dalej...")
    elif wybor == "6":
        generuj_serie(n_docelowych=100, zakres="sredni", n_cykli=wybrane_cykle)
    elif wybor == "7":
        generuj_serie(n_docelowych=10, zakres="sredni", n_cykli=wybrane_cykle)
        generuj_serie(n_docelowych=5,  zakres="bliski", n_cykli=wybrane_cykle)
        generuj_serie(n_docelowych=5,  zakres="hierarchiczny", n_cykli=wybrane_cykle)
    else:
        print("  Nieprawidłowy wybór.")