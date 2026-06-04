"""
BCA_modern — Normal-incidence LEIS example
===============================================

He+ scattering from Au and Cu at 3 keV under normal incidence
with detector at theta = 145 degrees.
Conditions match Prusa et al., Appl. Surf. Sci. 657 (2024) 158793.

Usage:
    python example_normal.py

Expected output:
    - Au peak at E/E0 ~ 0.930 (K = 0.929, deviation ~0.1%)
    - Cu peak at E/E0 ~ 0.790 (K = 0.795, deviation ~0.6%)
    - Figure saved: example_normal_output.png

Runtime: approximately 5-10 minutes.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')
from BCA_modern import *

E0 = 3000.0
N_TRAJ = 20000  # increase to 90000 for publication quality
det = Detector(theta_deg=145.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, cryst in [(ax1, 'Au'), (ax2, 'Cu')]:
    print(f'\nHe+ -> {cryst}, {E0:.0f} eV, normal incidence')

    setup = auto_setup(ion_Z=2, crystal_name=cryst, E0=E0,
                       psi_deg=90.0, n_trajectories=N_TRAJ)

    K = det.kinematic_energy(E0, setup['ion'].A, setup['targets'][0].A) / E0

    results = setup['sim'].run_simulation(verbose=True)
    back = [r for r in results if r.backscattered]

    # Filter by detector angle 145 +/- 5 degrees
    detected = [r for r in back
                if abs(det.scattering_angle_for_ion(
                    90.0, r.polar_angle, r.azimuthal_angle) - 145.0) < 5.0]

    print(f'  Backscattered: {len(back)}, in detector: {len(detected)}')
    print(f'  K = {K:.4f}')

    peak_pos, dev = 0, 0
    if detected:
        e_norm = [r.final_energy / E0 for r in detected]
        bins = np.arange(0, 1.05, 0.02)
        counts, edges = np.histogram(e_norm, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        smooth = gaussian_filter1d(counts.astype(float), sigma=1.5)
        if max(smooth) > 0:
            smooth /= max(smooth)
        ax.plot(centers, smooth, 'b-', lw=2.5,
                label=f'BCA_modern (N={len(detected)})')
        ax.fill_between(centers, smooth, alpha=0.15, color='steelblue')

        mask = centers > 0.3
        if np.any(smooth[mask] > 0):
            peak_pos = centers[mask][np.argmax(smooth[mask])]
            dev = abs(peak_pos - K) / K * 100
        print(f'  Peak: {peak_pos:.3f}, deviation: {dev:.1f}%')

    x = np.linspace(0.3, 1.05, 200)
    ax.plot(x, np.exp(-(x-K)**2/(2*0.03**2)), 'r--', lw=2.5,
            label=f'Kinematic (K={K:.3f})')
    ax.set_xlabel('E/E0')
    ax.set_ylabel('Normalized intensity')
    ax.set_title(f'He+ -> {cryst}, 3 keV, normal incidence\n'
                 f'Peak: {peak_pos:.3f} vs K={K:.3f} ({dev:.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.3, 1.05)

plt.suptitle('Normal incidence LEIS (Prusa et al., 2024)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('example_normal_output.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nFigure saved: example_normal_output.png')
