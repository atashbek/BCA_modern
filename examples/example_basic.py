"""
BCA_modern — Basic example
==============================

Minimal working example demonstrating the core functionality.
This script reproduces the main result of the accompanying paper:
Ne+ scattering from GaP(100)<110> at 2 keV.

Usage:
    python example_basic.py

Expected output:
    - Energy spectrum of backscattered ions (PNG)
    - Summary statistics printed to console
    - Structured output file (bca_modern_output.dat)

Runtime: approximately 30 seconds on a modern CPU.
"""

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from BCA_modern import *

# ---------------------------------------------------------------
#  Configure and run
# ---------------------------------------------------------------

setup = auto_setup(
    ion_Z=10,  # Ne+
    crystal_name='GaP',  # gallium phosphide, zincblende
    E0=2000.0,  # beam energy, eV
    psi_deg=1.8,  # glancing angle, degrees
    xi_deg=0.0,  # azimuthal angle (along <110>)
    n_trajectories=2000,  # number of ion trajectories
)

results = setup['sim'].run_simulation(verbose=True)

# ---------------------------------------------------------------
#  Analyse results
# ---------------------------------------------------------------

backscattered = [r for r in results if r.backscattered]
total = len(results)
n_back = len(backscattered)

print(f'\n--- Summary ---')
print(f'Total trajectories:  {total}')
print(f'Backscattered:       {n_back} ({100 * n_back / total:.1f}%)')

if backscattered:
    energies = [r.final_energy for r in backscattered]
    angles = [r.polar_angle for r in backscattered]

    print(f'Mean energy:         {np.mean(energies):.0f} eV')
    print(f'Energy range:        {min(energies):.0f} – {max(energies):.0f} eV')
    print(f'Mean polar angle:    {np.mean(angles):.1f} degrees')
    print(f'Mean collisions:     {np.mean([r.n_collisions for r in backscattered]):.1f}')

# ---------------------------------------------------------------
#  Plot energy spectrum
# ---------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

if backscattered:
    # Left panel: energy spectrum
    ax1.hist(energies, bins=np.arange(0, 2050, 50),
             color='steelblue', alpha=0.8, edgecolor='navy')

    # Kinematic lines for theta = 136 degrees
    det = Detector(theta_deg=136.0)
    for t in setup['targets']:
        ek = det.kinematic_energy(2000, setup['ion'].A, t.A)
        if ek > 10:
            ax1.axvline(ek, color='red', linestyle='--', alpha=0.6,
                        label=f'{t.name}: K={ek / 2000:.3f} ({ek:.0f} eV)')

    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Counts')
    ax1.set_title(f'Ne+ -> GaP(100)<110>, E0=2 keV, psi=1.8 deg')
    ax1.legend(fontsize=9)

    # Right panel: angular distribution
    ax2.hist(angles, bins=np.arange(0, 90, 2),
             color='coral', alpha=0.8, edgecolor='darkred')
    ax2.set_xlabel('Polar exit angle (degrees)')
    ax2.set_ylabel('Counts')
    ax2.set_title('Angular distribution')

plt.tight_layout()
plt.savefig('example_output.png', dpi=150)
plt.close()

print(f'\nFigure saved: example_output.png')
print(f'Data file:    bca_modern_output.dat')
