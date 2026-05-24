#  ThreeBody - Statistical Invariants in Hierarchical Three-Body Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20356331.svg)](https://doi.org/10.5281/zenodo.20356331)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

A Python framework for numerical simulation and statistical analysis of hierarchical three-body gravitational systems. The project systematically searches for hidden invariants — quantities that remain statistically stable despite the chaotic nature of the three-body problem.

---

## 🔭 Scientific Background

The three-body problem has no general closed-form solution and exhibits chaotic behaviour. However, in *hierarchical* configurations — where a close binary pair is orbited by a distant third body — certain statistical properties of the motion may remain surprisingly stable over long timescales.

This framework investigates whether such statistical invariants exist, and if so, what physical meaning they carry.

**Key findings so far:**
- `r12_cv ↔ r13_cv` correlation (Spearman r_S = +0.672, p < 0.0001) across 128 random systems — chaos transfers from the inner pair to the outer body
- `CV(r13) ≈ 0.25` is stable across geometry and mass variations (for mass ratio μ < 0.1, separation > 10×)
- `ratio_r13 = r13_mean / r13_start → 0.781` as a function of pair mass in the perturbative limit
- A critical separation `sep* ≈ 23×` marks a regime transition in the sign of the dynamical regime
- Candidate universal invariant: `R_med(v3) ≈ 1.000` (velocity of the third body at Poincaré events) — under systematic investigation

---

## 📦 Installation

```bash
git clone https://github.com/Mariusz-Rossa/ThreeBody.git
cd ThreeBody
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, numpy, pandas, scipy, matplotlib, numba

---

## 🚀 Quick Start

### Simulate your own system

```bash
python custom_system.py
```

You will be prompted to enter:
- Masses of three stars (in solar masses)
- Separations r12, r13 (in AU)
- Number of simulation cycles

The script computes orbital velocities automatically, verifies that the system is gravitationally bound, runs the RK4 integrator, and outputs a Poincaré section analysis.

### Preset examples

```bash
python custom_system.py --preset test        # reference hierarchical system
python custom_system.py --preset kompaktowy  # compact configuration
```

### Run a batch simulation and full analysis

```bash
# 1. Generate simulation data (mode 6 = 10 random bound systems)
python simulator.py

# 2. Poincaré section analysis + R(n) statistics
python analyzer.py

# 3. Binding energy correlations (CV analysis)
python binding_energy_analyzer.py

# 4. Oscillation amplitude analysis
python breath_analyzer.py
```

---

## 🗂️ Project Structure

```
ThreeBody/
├── simulator.py              # RK4 integrator, 7 simulation modes
├── analyzer.py               # Poincaré sections, R(n) statistics, stability
├── custom_system.py          # Interactive: run your own star system
├── config.py                 # Physical constants, example configurations
├── binding_energy_analyzer.py # CV correlations: pair chaos ↔ outer body
├── breath_analyzer.py        # Amplitude oscillations between Poincaré events
├── omega_analyzer.py         # sin²/coslog curve fitting to r13
├── pysr_analyzer.py          # Symbolic regression on Poincaré events
├── mu_transition.py          # CV(r13) and R²(sin²) vs mass ratio μ
├── geometry_invariant.py     # Invariant testing on r12/r13/mass grid
├── critical_point_scan.py    # Dense r12 scan for regime transition point
├── triple_invariant.py       # CV, ratio_r13, pct_pos on m_para × r12 grid
├── data/                     # Simulation output (CSV + JSON)
├── results/                  # Analysis output
└── requirements.txt
```

---

## 🔬 Research Phases

| Phase | What | Key result |
|-------|------|------------|
| 1 | Poincaré sections, 33 systems, 500k cycles | r13/r23 stable, r12 chaotic |
| 2 | PySR symbolic regression | cos(k·log(t)) most frequent pattern |
| 3 | Omega analyzer, 33 systems | r13_mean ≈ 0.695·r13_start + 47 AU |
| 4 | Binding energy, 33 systems | r12_cv ↔ r13_cv r_S = +0.34 |
| 6 | Breath analyzer, 196 systems | median R_n ≈ 1.000 |
| 7 | Poincaré 150k cycles, 128 systems | r_S = +0.672 (strongest correlation) |
| 7b | Black hole mass ratio (500:1) | sin²(ωt) R² = 0.988 — perturbative limit |
| 8 | μ transition scan | CV(r13) decreases monotonically with μ |
| 9 | Geometry invariant grid | sep* ≈ 23× regime transition identified |
| 10 | Critical point scan (dense r12) | sep* ≈ 23.6×, orbital resonance zone 4–8 AU |
| 11 | Triple invariant scan | CV ≈ 0.25 stable; ratio → 0.781; R_med(v3) ≈ 1.000 candidate |

---

## 📊 Current Invariant Candidates

| Candidate | Description | Validity range | Evidence |
|-----------|-------------|----------------|----------|
| R_med(v3) ≈ 1.000 | v3 ratio at Poincaré events | all tested systems | ★★★ (under investigation) |
| r12_cv ↔ r13_cv | chaos transfer correlation | 128 random systems | ★★★ r_S = +0.672 |
| CV(r13) ≈ 0.25 | scatter of outer body distance | μ < 0.1, sep > 10× | ★★ |
| ratio_r13 = f(m) | r13_mean/r13_start → 0.781 | m_para ≥ 5 M☉ | ★★ |
| R ≈ 0.93 | median ratio of consecutive events | 160+ random systems | ★★ |
| sep* ≈ 23× | critical separation for regime change | m_para = 50 M☉ | ★ |

---

## 📄 Citation

If you use this software, please cite:

```bibtex
@software{threebodyproblem,
  author  = {Rossa, Mariusz},
  title   = {ThreeBody: Statistical Invariants in Hierarchical Three-Body Systems},
  year    = {2026},
  doi     = {10.5281/zenodo.20356331},
  url     = {https://github.com/Mariusz-Rossa/ThreeBody}
}
```

---

## 🔗 Related Projects

- [WheelPhysics](https://github.com/Mariusz-Rossa/WheelPhysics) — Wheel Algebra applied to singularities in theoretical physics (DOI: [10.5281/zenodo.20305458](https://doi.org/10.5281/zenodo.20305458))
- [CollatzWheel](https://github.com/Mariusz-Rossa/CollatzWheel) — Collatz conjecture through Wheel Algebra (mod 6) (DOI: [10.5281/zenodo.20355730](https://doi.org/10.5281/zenodo.20355730))

---

## 👤 Author

**Mariusz Rossa** — independent researcher  
ORCID: [0009-0006-1060-2883](https://orcid.org/0009-0006-1060-2883)

---

## 📜 License

MIT License — see [`LICENSE`](LICENSE). for details.

*Independent research project*