"""
BCA_modern — Normal-incidence LEIS example
===============================================

Reproduces the experimental comparison presented in the paper:
He+ scattering from Au and Cu at 3 keV under normal incidence
with detector at theta = 145 degrees.

Conditions match Prusa et al., Appl. Surf. Sci. 657 (2024) 158793.

Usage:
    python example_normal.py

Expected output:
    - Au peak at E/E0 ~ 0.89 (kinematic K = 0.929, deviation ~4%)
    - Cu peak at E/E0 ~ 0.77 (kinematic K = 0.795, deviation ~3%)

Runtime: approximately 5-10 minutes.
"""

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from BCA_modern import *


def make_fcc(a, Z, A, name):
    crystal = CrystalStructure(
        a=a, b=a, c=a,
        sites=[
            AtomSite(0.0, 0.0, 0.0, 0, name),
            AtomSite(0.5, 0.5, 0.0, 0, name),
            AtomSite(0.5, 0.0, 0.5, 0, name),
            AtomSite(0.0, 0.5, 0.5, 0, name),
        ],
        name=f'{name} FCC',
        channel_spacing=a / np.sqrt(2),
    )
    return crystal, [TargetAtom(Z=Z, A=A, name=name)]


def run_and_filter(ion_Z, crystal, targets, E0, u_therm, nx, ny):
    """Run at normal incidence and filter by detector angle."""
    ion = auto_ion(ion_Z)
    stopping = auto_electronic_stopping(ion_Z, targets, crystal)
    params = SimulationParams(E0=E0, n_x=nx, n_y=ny, stopping=stopping)

    engine = BCAEngine(ion, targets, crystal, 90.0, 0.0, u_therm, params)
    pgr = BoundaryParameterCalculator(engine)
    pgr.compute(verbose=False)
    sim = ChannelingSimulation(engine, pgr)
    results = sim.run_simulation(verbose=True)

    back = [r for r in results if r.backscattered]

    # Filter: scattering angle = 145 +/- 5 degrees
    detected = []
    for r in back:
        pol = np.radians(r.polar_angle)
        cos_theta = -np.sin(pol)
        theta = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        if abs(theta - 145.0) < 5.0:
            detected.append(r)

    return back, detected


# ---------------------------------------------------------------
#  Run simulations
# ---------------------------------------------------------------

print('He+ -> Au, 3 keV, normal incidence')
crystal_au, targets_au = make_fcc(4.078, 79, 196.97, 'Au')
back_au, det_au = run_and_filter(2, crystal_au, targets_au, 3000, 0.05, 300, 300)
print(f'  Backscattered: {len(back_au)}, in detector: {len(det_au)}')

print('\nHe+ -> Cu, 3 keV, normal incidence')
crystal_cu, targets_cu = make_fcc(3.615, 29, 63.546, 'Cu')
back_cu, det_cu = run_and_filter(2, crystal_cu, targets_cu, 3000, 0.065, 300, 300)
print(f'  Backscattered: {len(back_cu)}, in detector: {len(det_cu)}')

# ---------------------------------------------------------------
#  Kinematic factors
# ---------------------------------------------------------------

det = Detector(theta_deg=145.0)
ion = auto_ion(2)
K_Au = det.kinematic_energy(3000, ion.A, 196.97) / 3000
K_Cu = det.kinematic_energy(3000, ion.A, 63.546) / 3000

# ---------------------------------------------------------------
#  Plot overlay
# ---------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, detected, K, name in [(ax1, det_au, K_Au, 'Au'), (ax2, det_cu, K_Cu, 'Cu')]:
    if detected:
        e_norm = [r.final_energy / 3000 for r in detected]
        bins = np.arange(0, 1.05, 0.02)
        counts, edges = np.histogram(e_norm, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        smooth = gaussian_filter1d(counts.astype(float), sigma=1.5)
        if max(smooth) > 0:
            smooth /= max(smooth)
        ax.plot(centers, smooth, 'b-', lw=2.5, label=f'BCA_modern (N={len(detected)})')
        ax.fill_between(centers, smooth, alpha=0.15, color='steelblue')

    x = np.linspace(0.3, 1.05, 200)
    sigma = 0.03
    y = np.exp(-(x - K) ** 2 / (2 * sigma ** 2))
    ax.plot(x, y, 'r--', lw=2.5, label=f'Kinematic (K={K:.3f})')
    ax.fill_between(x, y, alpha=0.1, color='red')

    ax.set_xlabel('E/E0')
    ax.set_ylabel('Normalized intensity')
    ax.set_title(f'He+ -> {name}, 3 keV, normal incidence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.3, 1.05)

plt.tight_layout()
plt.savefig('example_normal_output.png', dpi=150)
plt.close()

print(f'\nK(Au) = {K_Au:.4f}, K(Cu) = {K_Cu:.4f}')
print(f'Figure saved: example_normal_output.png')
