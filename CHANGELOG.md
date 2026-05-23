# Changelog

All notable changes to this project will be documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- Systematic verification of R_med(v3) ≈ 1.000 on full parameter grid (Phase 12)
- Critical separation sep* scan for varying m_para (Phase 13)
- Orbital resonance analysis in r12 = 4–8 AU zone (Phase 14)
- Small MLP network for R_med(v3) validation (post Phase 15)

---

## [1.0.0] — 2026-05-23

### Added
- `simulator.py` — RK4 integrator with 7 simulation modes, energy conservation monitoring, automatic bound-system detection
- `analyzer.py` — Poincaré section analysis, R(n) statistics, stability classification
- `custom_system.py` — interactive framework: user provides masses and separations, script computes orbits and runs full analysis
- `config.py` — physical constants and example configuration
- `binding_energy_analyzer.py` — CV(r13) ↔ CV(r12) correlation analysis across batches of systems
- `breath_analyzer.py` — amplitude oscillation analysis between Poincaré events
- `omega_analyzer.py` — sin²/coslog/sin curve fitting to r13 Poincaré time series
- `pysr_analyzer.py` — symbolic regression on Poincaré event data
- `mu_transition.py` — CV(r13) and R²(sin²) as a function of mass ratio μ = m3/(m1+m2)
- `geometry_invariant.py` — invariant testing on r12 / r13 / mass asymmetry grid
- `critical_point_scan.py` — dense r12 scan for regime transition point sep*
- `triple_invariant.py` — simultaneous CV, ratio_r13, pct_pos analysis on m_para × r12 grid

### Key scientific results in v1.0.0
- r12_cv ↔ r13_cv Spearman correlation r_S = +0.672 (p < 0.0001), 128 systems
- CV(r13) ≈ 0.25 stable for μ < 0.1, sep > 10× across geometry variations
- ratio_r13 = r13_mean / r13_start → 0.781 as m_para → ∞
- Critical separation sep* ≈ 23.6× marks dynamic regime transition
- Candidate universal invariant: R_med(v3) ≈ 1.000 (under investigation)