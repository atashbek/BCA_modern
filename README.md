# BCA_modern

**An open-source universal binary collision approximation code for low-energy ion scattering from crystalline surfaces**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20547246-blue)](https://doi.org/10.5281/zenodo.20547246)

## Overview

BCA_modern is a Python code for simulating low-energy ion scattering (LEIS) from crystalline surfaces using the binary collision approximation. It combines several features not previously available in a single open-source tool:

- **Pair-specific NLH interatomic potentials** for 70 ion–atom combinations (He⁺, Ne⁺, Ar⁺, Kr⁺, Xe⁺ on 14 target elements), based on DFT calculations by Nordlund, Lehtola & Hobler (Phys. Rev. A 111, 2025)
- **Position-dependent electronic stopping** Se(v, ρ) with channel reduction
- **Debye thermal vibrations** with element-specific amplitudes
- **Hagstrum ion neutralisation** model
- **35 built-in crystal structures**: zincblende (GaP, GaAs, InP, CdTe, ZnSe, ...), wurtzite (GaN, AlN, InN), rocksalt (MgO), corundum (α-Al₂O₃), cristobalite (β-SiO₂), FCC metals (Au, Cu, Ag, Al, Ni, Pt), BCC metals (W, Fe, Mo), diamond cubic (Si, Ge, C, Sn)
- **CIF file input** for user-defined structures
- **Ternary alloy support** via `make_fcc_ternary()`
- **Azimuthal scanning** via `azimuthal_scan()`
- **Automatic calibration** — all physical parameters derived from just three inputs

## Installation

```bash
pip install numpy scipy matplotlib
```

No compilation required. Python 3.8+ with NumPy ≥ 1.20 and SciPy ≥ 1.7.

## Quick Start

```python
from BCA_modern import *

# Run a complete LEIS simulation in one line
setup = auto_setup(ion_Z=10, crystal_name='GaP', E0=2000)
results = setup['sim'].run_simulation()

# Analyse results
back = [r for r in results if r.backscattered]
print(f'Backscattered: {len(back)}/{len(results)}')
for r in back[:5]:
    print(f'  E = {r.final_energy:.0f} eV, angle = {r.polar_angle:.1f}°')
```

## Optional Parameters

```python
setup = auto_setup(
    ion_Z=10,              # Projectile atomic number (Ne)
    crystal_name='GaP',    # Crystal from database (35 available)
    E0=2000,               # Beam energy (eV)
    psi_deg=3.0,           # Glancing angle (degrees)
    xi_deg=0.0,            # Azimuthal angle (degrees)
    temperature_K=300,     # Lattice temperature (K)
    n_trajectories=5000,   # Number of trajectories
)
```

## Output Fields

Each trajectory result contains:
- `final_energy` — energy at exit (eV)
- `polar_angle` — polar exit angle (degrees)
- `azimuthal_angle` — azimuthal exit angle (degrees)
- `n_collisions` — number of binary collisions
- `backscattered` — True/False
- `max_depth` — maximum penetration depth (Å)
- `ion_survival_prob` — Hagstrum neutralisation probability

## Normal Incidence Example

```python
# He+ → Au at 3 keV, normal incidence, detector at 145°
setup = auto_setup(ion_Z=2, crystal_name='Au', E0=3000, psi_deg=90.0)
results = setup['sim'].run_simulation()

det = Detector(theta_deg=145.0)
back = [r for r in results if r.backscattered]
detected = [r for r in back
    if abs(det.scattering_angle_for_ion(90.0, r.polar_angle, r.azimuthal_angle) - 145.0) < 5.0]

print(f'Detected at 145°±5°: {len(detected)}')
```

## Built-in Crystal Structures (35)

| Family | Materials | Atoms/cell |
|--------|-----------|-----------|
| Zincblende | GaP, GaAs, InP, InAs, CdTe, ZnSe, ZnS, ZnTe, HgTe, CdS, GaSb, InSb, AlAs, AlP, 3C-SiC | 8 |
| Wurtzite | GaN, AlN, InN | 4 |
| Rocksalt | MgO | 8 |
| Corundum | α-Al₂O₃ | 30 |
| Cristobalite | β-SiO₂ | 24 |
| FCC metals | Au, Cu, Ag, Al, Ni, Pt | 4 |
| BCC metals | W, Fe, Mo | 2 |
| Diamond cubic | Si, Ge, C, Sn | 8 |

## Performance

| System | Atoms/cell | Speed (traj/s) | Time for 10⁴ traj |
|--------|-----------|----------------|-------------------|
| GaP | 8 | ~530 | ~19 s |
| CdTe | 8 | ~400 | ~25 s |
| β-SiO₂ | 24 | ~250 | ~40 s |
| α-Al₂O₃ | 30 | ~200 | ~50 s |

Single CPU core, Intel i5 equivalent.

## Example Scripts

- `example_basic.py` — Ne⁺ → GaP at 2 keV, basic spectrum
- `example_validation.py` — Kinematic factor verification + CaSiO₃ benchmark
- `example_normal.py` — He⁺ → Au/Cu at normal incidence, comparison with experiment

## Citation

If you use BCA_modern in your work, please cite:

```
A.S. Ashirov, U.O. Kutliev, BCA_modern: an open-source universal binary
collision approximation code for low-energy ion scattering from crystalline
surfaces, Computer Physics Communications (2026).
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Repository

- GitHub: https://github.com/atashbek/BCA_modern
- Zenodo: https://doi.org/10.5281/zenodo.20547246
