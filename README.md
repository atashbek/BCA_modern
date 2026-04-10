# BCA_modern

**Universal binary collision approximation code for low-energy ion scattering simulation from crystalline surfaces**

A.S. Ashirov

Department of Physics, Urgench State University, Uzbekistan

---

## Overview

BCA_modern is a Python implementation of the binary collision approximation (BCA) for simulating low-energy ion scattering (LEIS) from single-crystal surfaces. The code is designed to handle arbitrary crystal structures — from simple zincblende semiconductors to complex oxides — with a single unified framework, eliminating the need to modify source code when changing the target material.

The main physical models incorporated in the code are:

- **Pair-specific NLH interatomic potentials** (Nordlund, Lehtola, Hobler, Phys. Rev. A 111, 2025) for 70 ion–atom combinations, replacing the universal ZBL potential
- **Position-dependent electronic stopping** Se(v, ρ) with reduced stopping in channel centres
- **Debye thermal vibrations** with configurable temperature
- **Hagstrum ion neutralisation** P⁺ = exp(−v₀/v⊥) for ESA-LEIS quantification
- **Adaptive Gauss–Kronrod quadrature** for the scattering integral (scipy.integrate.quad)
- **Universal crystal navigator** with numpy-accelerated atom search

## Requirements

- Python 3.8 or later
- NumPy
- SciPy
- Matplotlib (optional, for plotting)

No compilation is required. No proprietary libraries or specialised hardware are needed.

## Installation

```bash
pip install -r requirements.txt
```

Or install dependencies manually:

```bash
pip install numpy scipy matplotlib
```

Place `BCA_modern.py` in your working directory. The code is a single self-contained file with no external data dependencies — all 70 NLH potential coefficients and 21 crystal structure definitions are embedded in the source.

## Quick start

The simplest way to run a simulation is through the automatic calibration module, which derives all physical parameters from three inputs:

```python
from BCA_modern import *

setup = auto_setup(ion_Z=10, crystal_name='GaP', E0=2000)
results = setup['sim'].run_simulation()

backscattered = [r for r in results if r.backscattered]
print(f'Backscattered: {len(backscattered)} ions')
```

This computes the scattering of 2 keV Ne⁺ ions from GaP(100)⟨110⟩ with NLH potentials, position-dependent stopping, and thermal vibrations at 300 K — all configured automatically.

## File structure

```
BCA_modern/
├── BCA_modern.py          Main code (single file, ~1650 lines)
├── README.md                  This file
├── LICENSE                    MIT licence
├── requirements.txt           Python dependencies
├── examples/
│   ├── example_basic.py       Minimal working example
│   ├── example_validation.py  Kinematic factor verification
│   └── example_normal.py      Normal-incidence LEIS comparison
└── output/
    └── (generated at runtime)
```

## Built-in crystal structures (21 materials)

| Structure type | Materials | Atoms/cell |
|:--------------|:----------|:----------:|
| Zincblende (F‑43m) | GaP, GaAs, InP, InAs, CdTe, ZnSe, ZnS, ZnTe, HgTe, CdS, GaSb, InSb, AlAs, AlP, 3C‑SiC | 8 |
| Wurtzite (P6₃mc) | GaN, AlN, InN | 4 |
| Rocksalt (Fm‑3m) | MgO | 8 |
| Corundum (R‑3c) | α‑Al₂O₃ | 30 |
| Cristobalite (Fd‑3m) | β‑SiO₂ | 24 |

Additional structures can be loaded from CIF files using `load_cif('structure.cif')`.

## NLH potential coverage (70 pairs)

Five noble gas projectiles (He, Ne, Ar, Kr, Xe) × fourteen target elements (N, O, Mg, Al, Si, P, S, Zn, Ga, As, Se, Cd, In, Te). Coefficients from Zenodo dataset doi:10.5281/zenodo.14172632, including the November 2025 erratum.

## Available ions

| Symbol | Z | Mass (amu) |
|:------:|:-:|:----------:|
| He⁺ | 2 | 4.003 |
| Ne⁺ | 10 | 20.180 |
| Ar⁺ | 18 | 39.948 |
| Kr⁺ | 36 | 83.798 |
| Xe⁺ | 54 | 131.293 |

## Key parameters

| Parameter | Meaning | Typical values |
|:----------|:--------|:---------------|
| `ion_Z` | Projectile atomic number | 2, 10, 18, 36, 54 |
| `crystal_name` | Target crystal | 'GaP', 'SiO2', etc. |
| `E0` | Beam energy (eV) | 500–10000 |
| `psi_deg` | Glancing angle (degrees) | 1–90 (90 = normal incidence) |
| `xi_deg` | Azimuthal angle (degrees) | 0–90 |
| `n_trajectories` | Number of ion trajectories | 1000–100000 |

## Output

Each trajectory returns:

| Field | Description |
|:------|:-----------|
| `final_energy` | Ion energy after scattering (eV) |
| `polar_angle` | Exit polar angle (degrees from surface) |
| `azimuthal_angle` | Exit azimuthal angle (degrees) |
| `n_collisions` | Number of binary collisions |
| `total_inelastic_loss` | Electronic energy loss (eV) |
| `max_depth` | Maximum penetration depth (Å) |
| `ion_survival_prob` | Neutralisation survival probability P⁺ |
| `backscattered` | True if ion returned above surface |

A structured text file `bca_modern_output.dat` is also generated with per-trajectory data.

## Performance

| Crystal | Atoms/cell | Speed (traj/s) | Time for 10⁴ traj |
|:--------|:----------:|:--------------:|:-----------------:|
| GaP | 8 | ~530 | ~19 s |
| CdTe | 8 | ~400 | ~25 s |
| β‑SiO₂ | 24 | ~250 | ~40 s |
| α‑Al₂O₃ | 30 | ~200 | ~50 s |

Measured on a single core of an Intel i5-class processor. Performance scales linearly with trajectory count.

## Advanced usage

### Custom incidence geometry

```python
setup = auto_setup(
    ion_Z=2,               # He+
    crystal_name='SiO2',
    E0=3000,               # 3 keV
    psi_deg=90.0,          # normal incidence
    n_trajectories=10000,
)
```

### Detector filtering

```python
setup = auto_setup(ion_Z=10, crystal_name='GaP', E0=2000,
                   enable_detector=True, detector_angle=136.0)
```

### Neutralisation

```python
setup = auto_setup(ion_Z=10, crystal_name='GaP', E0=2000,
                   enable_neutralisation=True)
```

### Ternary alloys

```python
crystal = make_ternary_alloy(
    base_crystal='GaAs',
    substitute_type=2,
    substitute_name='In',
    fraction=0.3,           # In₀.₃Ga₀.₇As
    which_sublattice=0,
)
```

### User-defined crystal from CIF

```python
crystal = load_cif('my_structure.cif')
```

## Validation

The code has been validated through:

1. **Kinematic factors**: machine-precision agreement (< 10⁻¹⁵ relative error) for 54 ion–target pairs
2. **Peak positions**: 3–4% agreement with kinematic predictions under normal-incidence LEIS conditions for He⁺ → Au and Cu
3. **Statistical convergence**: 1% precision in mean backscattered energy at 5000 trajectories
4. **Spectral shape**: reproduces characteristic LEIS features (surface peak, multiple-scattering background, thermal dechannelling)

## Citation

If you use BCA_modern in your research, please cite:

A.S. Ashirov, BCA_modern: an open-source universal binary collision approximation code for low-energy ion scattering from crystalline surfaces, Comput. Phys. Commun. (2026).

## Licence

MIT — see LICENSE file.

## Support

For questions, bug reports, or feature requests, please use the GitHub Issues page.
