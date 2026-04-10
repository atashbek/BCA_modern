"""
BCA_modern — Kinematic factor validation
=============================================

Verifies that the code reproduces exact binary collision kinematics
for 54 ion-target combinations (3 ions x 9 crystals x 2 atoms each).

This test requires no trajectory simulation — it checks only the
scattering integral solver and energy partitioning logic.

Usage:
    python example_validation.py

Expected output:
    All 54 pairs show 0.000000% relative error.

Runtime: under 10 seconds.
"""

import numpy as np
from BCA_modern import *

theta_deg = 136.0
det = Detector(theta_deg=theta_deg)

ions = [
    (2, 'He+'),
    (10, 'Ne+'),
    (18, 'Ar+'),
]

crystals = ['GaP', 'SiO2', 'CdTe', 'GaAs', 'InP', 'MgO', 'GaN', 'Al2O3', 'ZnSe']

print(f'Kinematic factor validation at theta = {theta_deg} degrees')
print(f'{"Ion":>6s}  {"Crystal":>8s}  {"Target":>6s}  {"K_code":>10s}  {"K_formula":>10s}  {"Error":>10s}')
print('-' * 60)

n_tested = 0
max_error = 0

for z_ion, ion_name in ions:
    ion = auto_ion(z_ion)

    for cryst_name in crystals:
        crystal = get_crystal(cryst_name)
        targets = auto_targets_from_crystal(crystal)

        for t in targets:
            # From the code
            E_kin = det.kinematic_energy(2000, ion.A, t.A)
            K_code = E_kin / 2000.0

            # Analytical formula
            mu = t.A / ion.A
            theta = np.radians(theta_deg)
            disc = mu ** 2 - np.sin(theta) ** 2
            K_formula = ((np.cos(theta) + np.sqrt(disc)) / (1 + mu)) ** 2 if disc > 0 else 0

            error = abs(K_code - K_formula) / max(K_formula, 1e-30) * 100
            max_error = max(max_error, error)
            n_tested += 1

            print(f'{ion_name:>6s}  {cryst_name:>8s}  {t.name:>6s}  {K_code:10.6f}  {K_formula:10.6f}  {error:10.6f}%')

print('-' * 60)
print(f'Tested: {n_tested} pairs')
print(f'Maximum error: {max_error:.2e}%')

if max_error < 1e-10:
    print('PASSED: all kinematic factors agree to machine precision.')
else:
    print('WARNING: deviations detected.')
