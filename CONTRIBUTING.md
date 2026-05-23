# Contributing to ThreeBody

Thank you for your interest in contributing. This project is developed by an independent researcher and welcomes contributions of all kinds.

---

## Ways to Contribute

### Bug reports
If you find incorrect physics, numerical instabilities, or unexpected outputs — please open a GitHub Issue with:
- Python version and OS
- Exact command used (`python custom_system.py --preset ...` etc.)
- Expected vs actual output
- If possible, the simulation parameters that trigger the bug

### New simulation modes or analysis scripts
Open an Issue first to discuss the idea before submitting a Pull Request.  
New scripts should follow the existing structure: physical constants at the top, clear docstring explaining inputs/outputs, results saved to `data/` or `results/`.

### New invariant candidates
If you find a quantity that appears statistically stable across multiple hierarchical three-body configurations, open an Issue describing:
- What the quantity is and how it is computed
- Which configurations were tested
- Numerical evidence (mean, std, sample size)

### Documentation and README improvements
Pull requests for typos, clarity, or additional usage examples are always welcome.

---

## Development Setup

```bash
git clone https://github.com/Mariusz-Rossa/ThreeBody.git
cd ThreeBody
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run a quick smoke-test:
```bash
python custom_system.py --preset test
```

---

## Code Style

- Python 3.10+
- No external dependencies beyond those in `requirements.txt`
- Physical units always in SI internally (meters, kg, seconds); AU and M☉ as display conversions only
- All new scripts should include a module-level docstring explaining purpose, inputs, and outputs
- Commit messages in English, one topic per commit

---

## Questions

Open a GitHub Issue with the `question` label.

---

*This project follows the [MIT License](LICENSE). By contributing, you agree that your contributions will be licensed under the same terms.*