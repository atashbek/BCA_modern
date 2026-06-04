#!/usr/bin/env python3
"""
BCA_MODERN — Universal Binary Collision Approximation code
for low-energy ion scattering (LEIS) from ANY crystal surface

This code simulates ion trajectories in crystalline solids using:
- pair-specific NLH interatomic potentials
- adaptive Gauss–Kronrod integration of the scattering integral
- position-dependent electronic stopping Se(v, ρ)
- Debye thermal vibrations
- Hagstrum ion neutralisation model
- universal crystal structure module (21 built-in structures + CIF input)

The code supports arbitrary crystal geometries, disordered alloys,
and automated parameter setup for LEIS simulations.
"""

import numpy as np
from scipy import integrate
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import warnings
import time

try:
    from numba import njit, float64

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


    def njit(*args, **kwargs):
        def decorator(func): return func

        return decorator if not (len(args) == 1 and callable(args[0])) else args[0]


    float64 = float

# ===========================================================================
#  Physical constants
# ===========================================================================
E2 = 14.4  # e^2 in eV*Angstrom
RD = 57.29577951  # 180/pi
PI = np.pi
BOHR = 0.529177  # Bohr radius in Angstrom
AMU_TO_KG = 1.6605e-27
EV_TO_J = 1.602e-19

NLH_DATABASE = {
    # =================================================================
    # REAL NLH coefficients from Zenodo doi:10.5281/zenodo.14172632
    # Nordlund, Lehtola, Hobler, Phys.Rev.A 111, 032818 (2025)
    # With erratum corrections (Nov 2025)
    #
    # Format: V(r) = Z1*Z2*e²/r * Σ d_i * exp(b_i * r)
    # d = [a1, a2, a3], b = [-b1, -b2, -b3] (b negative in code)
    # r in Angstrom, b in 1/Angstrom
    # =================================================================

    # Ne+ (Z=10) interactions
    (10, 7): {'d': [0.19253, 0.80747, 0.00000], 'b': [-26.08549, -4.27909, 0.00000]},  # Ne-N  err=2.33%
    (10, 8): {'d': [0.17925, 0.82075, 0.00000], 'b': [-28.08091, -4.37536, 0.00000]},  # Ne-O  err=2.51%
    (10, 12): {'d': [0.48433, 0.51567, 0.00000], 'b': [-11.07994, -3.29040, 0.00000]},  # Ne-Mg err=5.31%
    (10, 13): {'d': [0.42144, 0.57856, 0.00000], 'b': [-13.18333, -3.52801, 0.00000]},  # Ne-Al err=5.30%
    (10, 14): {'d': [0.39751, 0.60249, 0.00000], 'b': [-14.24701, -3.65335, 0.00000]},  # Ne-Si err=4.93%
    (10, 15): {'d': [0.37387, 0.62613, 0.00000], 'b': [-16.32003, -3.75944, 0.00000]},  # Ne-P  err=4.04%
    (10, 16): {'d': [0.36465, 0.63535, 0.00000], 'b': [-16.07529, -3.82228, 0.00000]},  # Ne-S  err=3.20%
    (10, 30): {'d': [0.06744, 0.50356, 0.42900], 'b': [-52.30059, -10.30821, -3.64124]},  # Ne-Zn err=2.38%
    (10, 31): {'d': [0.05186, 0.49892, 0.44922], 'b': [-64.20790, -11.09637, -3.70066]},  # Ne-Ga err=2.46%
    (10, 33): {'d': [0.04556, 0.45511, 0.49933], 'b': [-74.23554, -12.92664, -3.84358]},  # Ne-As err=2.30%
    (10, 34): {'d': [0.03817, 0.45501, 0.50683], 'b': [-86.99566, -13.12334, -3.86522]},  # Ne-Se err=2.02%
    (10, 48): {'d': [0.08128, 0.52206, 0.39666], 'b': [-58.68307, -10.83210, -3.42695]},  # Ne-Cd err=1.85%
    (10, 49): {'d': [0.06429, 0.52426, 0.41145], 'b': [-71.74130, -11.48269, -3.47916]},  # Ne-In err=1.99%
    (10, 52): {'d': [0.03649, 0.50965, 0.45386], 'b': [-124.70253, -13.57206, -3.60043]},  # Ne-Te err=2.13%

    # Ar+ (Z=18) interactions
    (18, 7): {'d': [0.30896, 0.69104, 0.00000], 'b': [-17.70541, -3.93207, 0.00000]},  # Ar-N  err=4.82%
    (18, 8): {'d': [0.26373, 0.73627, 0.00000], 'b': [-21.46136, -4.17289, 0.00000]},  # Ar-O  err=5.35%
    (18, 12): {'d': [0.18960, 0.57027, 0.24013], 'b': [-20.65303, -6.56615, -2.48803]},  # Ar-Mg err=2.81%
    (18, 13): {'d': [0.10006, 0.59380, 0.30615], 'b': [-38.18078, -8.03267, -2.70079]},  # Ar-Al err=3.15%
    (18, 14): {'d': [0.63215, 0.36785, 0.00000], 'b': [-10.18293, -2.90260, 0.00000]},  # Ar-Si err=3.73%
    (18, 15): {'d': [0.59671, 0.40329, 0.00000], 'b': [-10.86878, -3.04242, 0.00000]},  # Ar-P  err=4.25%
    (18, 16): {'d': [0.56113, 0.43887, 0.00000], 'b': [-11.78108, -3.18702, 0.00000]},  # Ar-S  err=5.12%
    (18, 30): {'d': [0.29252, 0.58186, 0.12562], 'b': [-20.60459, -5.76542, -2.68063]},  # Ar-Zn err=2.05%
    (18, 31): {'d': [0.24576, 0.58755, 0.16669], 'b': [-23.35100, -6.39997, -2.78859]},  # Ar-Ga err=2.08%
    (18, 33): {'d': [0.13807, 0.58679, 0.27514], 'b': [-35.01628, -8.35074, -3.13137]},  # Ar-As err=2.65%
    (18, 34): {'d': [0.09996, 0.59049, 0.30955], 'b': [-44.34779, -9.24977, -3.22546]},  # Ar-Se err=2.99%
    (18, 48): {'d': [0.43752, 0.38723, 0.17525], 'b': [-16.44833, -5.64613, -2.83829]},  # Ar-Cd err=1.43%
    (18, 49): {'d': [0.37906, 0.42856, 0.19238], 'b': [-18.33621, -6.27472, -2.84387]},  # Ar-In err=1.54%
    (18, 52): {'d': [0.12954, 0.59362, 0.27683], 'b': [-40.34115, -9.53986, -3.04310]},  # Ar-Te err=1.95%

    # He+ (Z=2) interactions
    (2, 7): {'d': [0.01725, 0.98275, 0.00000], 'b': [-145.87519, -4.64803, 0.00000]},  # He-N  err=3.50%
    (2, 8): {'d': [0.00251, 0.99749, 0.00000], 'b': [-999.99998, -5.11935, 0.00000]},  # He-O  err=4.22%
    (2, 12): {'d': [0.46969, 0.53031, 0.00000], 'b': [-11.17304, -2.72230, 0.00000]},  # He-Mg err=3.47%
    (2, 13): {'d': [0.38187, 0.61813, 0.00000], 'b': [-13.74509, -3.03497, 0.00000]},  # He-Al err=3.29%
    (2, 14): {'d': [0.30709, 0.69291, 0.00000], 'b': [-17.33509, -3.34101, 0.00000]},  # He-Si err=3.64%
    (2, 15): {'d': [0.25952, 0.74048, 0.00000], 'b': [-20.45628, -3.58266, 0.00000]},  # He-P  err=3.68%
    (2, 16): {'d': [0.20868, 0.79132, 0.00000], 'b': [-25.61218, -3.87320, 0.00000]},  # He-S  err=4.34%
    (2, 30): {'d': [0.04983, 0.60757, 0.34259], 'b': [-79.82594, -8.49164, -3.13860]},  # He-Zn err=1.59%
    (2, 31): {'d': [0.58154, 0.41846, 0.00000], 'b': [-10.34381, -3.34814, 0.00000]},  # He-Ga err=1.93%
    (2, 33): {'d': [0.02070, 0.47938, 0.49991], 'b': [-182.95700, -11.83738, -3.62650]},  # He-As err=2.42%
    (2, 34): {'d': [0.44104, 0.55896, 0.00000], 'b': [-14.10635, -3.83431, 0.00000]},  # He-Se err=2.80%
    (2, 48): {'d': [0.25189, 0.44512, 0.30299], 'b': [-25.44388, -6.73311, -2.92235]},  # He-Cd err=0.99%
    (2, 49): {'d': [0.20309, 0.42767, 0.36925], 'b': [-29.19319, -8.05494, -3.09496]},  # He-In err=1.27%
    (2, 52): {'d': [0.06018, 0.42936, 0.51047], 'b': [-61.27580, -13.47385, -3.48961]},  # He-Te err=1.69%

    # Kr+ (Z=36) interactions
    (36, 7): {'d': [0.46283, 0.53717, 0.00000], 'b': [-14.66899, -3.82318, 0.00000]},  # Kr-N  err=2.88%
    (36, 8): {'d': [0.41892, 0.58108, 0.00000], 'b': [-16.64935, -4.04204, 0.00000]},  # Kr-O  err=3.34%
    (36, 12): {'d': [0.27536, 0.57788, 0.14676], 'b': [-22.24562, -5.99657, -2.42851]},  # Kr-Mg err=1.83%
    (36, 13): {'d': [0.22802, 0.58322, 0.18876], 'b': [-26.23149, -6.70012, -2.62278]},  # Kr-Al err=2.09%
    (36, 14): {'d': [0.16828, 0.58880, 0.24292], 'b': [-32.69452, -7.74002, -2.85737]},  # Kr-Si err=2.46%
    (36, 15): {'d': [0.13242, 0.59029, 0.27728], 'b': [-37.34169, -8.43484, -3.00254]},  # Kr-P  err=2.69%
    (36, 16): {'d': [0.09263, 0.58988, 0.31749], 'b': [-48.54990, -9.44753, -3.17010]},  # Kr-S  err=3.20%
    (36, 30): {'d': [0.43535, 0.50055, 0.06410], 'b': [-16.86022, -5.09158, -2.50615]},  # Kr-Zn err=1.56%
    (36, 31): {'d': [0.40917, 0.50339, 0.08744], 'b': [-17.70988, -5.42087, -2.62521]},  # Kr-Ga err=1.67%
    (36, 33): {'d': [0.33351, 0.50108, 0.16541], 'b': [-20.49968, -6.48690, -3.00742]},  # Kr-As err=2.02%
    (36, 34): {'d': [0.29848, 0.50744, 0.19407], 'b': [-22.04538, -7.01390, -3.11245]},  # Kr-Se err=2.21%
    (36, 48): {'d': [0.32210, 0.47624, 0.20166], 'b': [-22.35119, -7.42592, -3.11038]},  # Kr-Cd err=1.56%
    (36, 49): {'d': [0.34330, 0.47069, 0.18602], 'b': [-21.53253, -7.15312, -3.02680]},  # Kr-In err=1.46%
    (36, 52): {'d': [0.30031, 0.49753, 0.20217], 'b': [-24.08335, -7.77258, -3.04522]},  # Kr-Te err=1.42%

    # Xe+ (Z=54) interactions
    (54, 7): {'d': [0.03384, 0.49830, 0.46786], 'b': [-123.99095, -13.65132, -3.53715]},  # Xe-N  err=2.15%
    (54, 8): {'d': [0.48881, 0.51119, 0.00000], 'b': [-16.25880, -3.74260, 0.00000]},  # Xe-O  err=3.19%
    (54, 12): {'d': [0.50108, 0.44306, 0.05586], 'b': [-15.43924, -4.34483, -1.88249]},  # Xe-Mg err=1.96%
    (54, 13): {'d': [0.45727, 0.42311, 0.11962], 'b': [-16.68637, -5.05416, -2.33886]},  # Xe-Al err=2.01%
    (54, 14): {'d': [0.35982, 0.43109, 0.20908], 'b': [-19.42858, -6.62314, -2.73388]},  # Xe-Si err=2.10%
    (54, 15): {'d': [0.18258, 0.55038, 0.26705], 'b': [-31.39955, -8.72930, -2.95439]},  # Xe-P  err=2.42%
    (54, 16): {'d': [0.11585, 0.59290, 0.29125], 'b': [-45.34890, -9.85310, -3.06125]},  # Xe-S  err=2.43%
    (54, 30): {'d': [0.55691, 0.40112, 0.04197], 'b': [-15.33605, -4.42934, -2.30533]},  # Xe-Zn err=1.47%
    (54, 31): {'d': [0.52039, 0.36683, 0.11278], 'b': [-16.04604, -5.16184, -2.78474]},  # Xe-Ga err=1.73%
    (54, 33): {'d': [0.35360, 0.44300, 0.20339], 'b': [-21.23216, -7.30883, -3.11095]},  # Xe-As err=1.86%
    (54, 34): {'d': [0.28901, 0.49211, 0.21887], 'b': [-24.61161, -8.05827, -3.15114]},  # Xe-Se err=1.78%
    (54, 48): {'d': [0.65835, 0.33069, 0.01096], 'b': [-14.23987, -3.71066, -1.80091]},  # Xe-Cd err=1.52%
    (54, 49): {'d': [0.64511, 0.31942, 0.03546], 'b': [-14.61006, -3.92526, -2.27955]},  # Xe-In err=1.58%
    (54, 52): {'d': [0.60938, 0.28166, 0.10896], 'b': [-15.70545, -4.51385, -2.72274]},  # Xe-Te err=1.94%
}

ZBL_C = np.array([0.1818, 0.5099, 0.2802, 0.02817])
ZBL_B = np.array([-3.200, -0.9423, -0.4029, -0.2016])


def get_potential_params(z1: int, z2: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Get NLH potential coefficients, fallback to ZBL."""
    for k in [(z1, z2), (z2, z1), (min(z1, z2), max(z1, z2))]:
        if k in NLH_DATABASE:
            p = NLH_DATABASE[k]
            return np.array(p['d']), np.array(p['b']), True
    a_scr = 0.4685 / (z1 ** 0.23 + z2 ** 0.23)
    warnings.warn(f"NLH not found for ({z1},{z2}), using ZBL")
    return ZBL_C.copy(), ZBL_B / a_scr, False


# ===========================================================================
#  UNIVERSAL CRYSTAL STRUCTURE (NEW in v2)
# ===========================================================================

@dataclass
class AtomSite:
    """One atom in the unit cell (fractional coordinates)."""
    x: float  # fractional coordinate 0..1
    y: float
    z: float
    atom_type: int  # index into targets[] list
    name: str = ""
    occupation: float = 1.0  # 1.0 = always present, 0.5 = 50% probability
    layer: int = 0  # surface layer index (0 = topmost)


@dataclass
class CrystalStructure:
    """
    Universal crystal structure definition.
    Works for ANY crystal: binary, ternary, quaternary, oxides, metals.

    The user provides:
      - Lattice parameters a, b, c (Angstrom) and angles
      - List of AtomSite with fractional coordinates and atom types
      - Surface orientation via miller indices or direct transformation

    """
    a: float  # lattice parameter a (Angstrom)
    b: float  # lattice parameter b (Angstrom)
    c: float  # lattice parameter c (Angstrom)
    alpha: float = 90.0  # angle b-c (degrees)
    beta: float = 90.0  # angle a-c (degrees)
    gamma: float = 90.0  # angle a-b (degrees)
    sites: List[AtomSite] = field(default_factory=list)
    name: str = ""

    # Surface properties
    surface_relaxation: Dict[int, float] = field(default_factory=dict)
    # key=layer_index, value=dz shift in Angstrom (positive = outward)

    # Inter-row distance along channel (used for reduced coordinates)
    channel_spacing: float = 0.0  # PD equivalent, auto-computed if 0

    def to_cartesian(self, fx: float, fy: float, fz: float) -> np.ndarray:
        """Convert fractional to Cartesian coordinates."""
        ar = np.radians(self.alpha)
        br = np.radians(self.beta)
        gr = np.radians(self.gamma)

        cos_a, cos_b, cos_g = np.cos(ar), np.cos(br), np.cos(gr)
        sin_g = np.sin(gr)

        # Transformation matrix
        v = self.a * self.b * self.c * np.sqrt(
            1 - cos_a ** 2 - cos_b ** 2 - cos_g ** 2 + 2 * cos_a * cos_b * cos_g)

        M = np.array([
            [self.a, self.b * cos_g, self.c * cos_b],
            [0, self.b * sin_g, self.c * (cos_a - cos_b * cos_g) / sin_g],
            [0, 0, v / (self.a * self.b * sin_g)]
        ])

        frac = np.array([fx, fy, fz])
        return M @ frac

    def generate_atom_positions(self, nx: int = 3, ny: int = 3, nz: int = 5,
                                rng: np.random.Generator = None) -> List[dict]:
        """
        Generate all atom positions in a search volume of nx*ny*nz unit cells.
        Applies occupation probability, surface relaxation, and vacancies.

        Returns list of dicts: {pos: [x,y,z], type: int, name: str}
        """
        if rng is None:
            rng = np.random.default_rng()

        from collections import defaultdict
        site_groups = defaultdict(list)
        EPS = 1e-6
        for site in self.sites:
            key = (round(site.x / EPS) * EPS,
                   round(site.y / EPS) * EPS,
                   round(site.z / EPS) * EPS)
            site_groups[key].append(site)

        for key, group in site_groups.items():
            sum_occ = sum(s.occupation for s in group)
            if sum_occ > 1.0 + 1e-6:
                warnings.warn(
                    f"Co-located AtomSites at {key} have sum_occupation = "
                    f"{sum_occ:.3f} > 1. Renormalising to sum = 1, ratios "
                    f"preserved. Check your CrystalStructure / "
                    f"make_ternary_alloy inputs.",
                    stacklevel=2)
                site_groups[key] = [
                    AtomSite(s.x, s.y, s.z, s.atom_type, s.name,
                             occupation=s.occupation / sum_occ,
                             layer=s.layer)
                    for s in group
                ]

        atoms = []
        for ix in range(-nx, nx + 1):
            for iy in range(-ny, ny + 1):
                for iz in range(0, nz):  # layers into the crystal
                    for group in site_groups.values():
                        sum_occ = sum(s.occupation for s in group)
                        if sum_occ <= 0.0:
                            continue
                        if len(group) == 1 and group[0].occupation >= 1.0:
                            chosen = group[0]
                        else:
                            r = rng.random()
                            cum = 0.0
                            chosen = None
                            for s in group:
                                cum += s.occupation
                                if r < cum:
                                    chosen = s
                                    break
                            if chosen is None:
                                continue

                        # Fractional position in supercell
                        fx = chosen.x + ix
                        fy = chosen.y + iy
                        fz = chosen.z + iz

                        # Convert to Cartesian
                        pos = self.to_cartesian(fx, fy, fz)

                        # Crystal is BELOW surface: z <= 0
                        pos[2] = -abs(pos[2])

                        # Surface relaxation (shift z for surface layers)
                        if iz in self.surface_relaxation:
                            pos[2] += self.surface_relaxation[iz]

                        atoms.append({
                            'pos': pos,
                            'type': chosen.atom_type,
                            'name': chosen.name,
                            'layer': iz,
                            'frac': (fx, fy, fz),
                        })

        return atoms

    def auto_channel_spacing(self) -> float:
        """Estimate inter-row distance for the current structure."""
        if self.channel_spacing > 0:
            return self.channel_spacing
        return min(self.a, self.b) / np.sqrt(2)


# ===========================================================================
#  Built-in Crystal Database
# ===========================================================================

def make_zincblende(a: float, name1: str, z1_type: int,
                    name2: str, z2_type: int, label: str) -> CrystalStructure:
    """Generate zincblende (F-43m) structure. 8 atoms per cell."""
    return CrystalStructure(
        a=a, b=a, c=a,
        sites=[
            AtomSite(0.00, 0.00, 0.00, z1_type, name1),
            AtomSite(0.50, 0.50, 0.00, z1_type, name1),
            AtomSite(0.50, 0.00, 0.50, z1_type, name1),
            AtomSite(0.00, 0.50, 0.50, z1_type, name1),
            AtomSite(0.25, 0.25, 0.25, z2_type, name2),
            AtomSite(0.75, 0.75, 0.25, z2_type, name2),
            AtomSite(0.75, 0.25, 0.75, z2_type, name2),
            AtomSite(0.25, 0.75, 0.75, z2_type, name2),
        ],
        name=label,
        channel_spacing=a / np.sqrt(2),
    )


def make_wurtzite(a: float, c: float, name1: str, z1_type: int,
                  name2: str, z2_type: int, label: str) -> CrystalStructure:
    """Generate wurtzite (P6_3mc) structure. 4 atoms per cell."""
    u = 3.0 / 8.0
    return CrystalStructure(
        a=a, b=a, c=c, gamma=120.0,
        sites=[
            AtomSite(1 / 3, 2 / 3, 0.0, z1_type, name1),
            AtomSite(2 / 3, 1 / 3, 0.5, z1_type, name1),
            AtomSite(1 / 3, 2 / 3, u, z2_type, name2),
            AtomSite(2 / 3, 1 / 3, 0.5 + u, z2_type, name2),
        ],
        name=label,
        channel_spacing=a,
    )


def make_rocksalt(a: float, name1: str, z1_type: int,
                  name2: str, z2_type: int, label: str) -> CrystalStructure:
    """Generate rocksalt (Fm-3m) structure. 8 atoms per cell."""
    return CrystalStructure(
        a=a, b=a, c=a,
        sites=[
            AtomSite(0.0, 0.0, 0.0, z1_type, name1),
            AtomSite(0.5, 0.5, 0.0, z1_type, name1),
            AtomSite(0.5, 0.0, 0.5, z1_type, name1),
            AtomSite(0.0, 0.5, 0.5, z1_type, name1),
            AtomSite(0.5, 0.0, 0.0, z2_type, name2),
            AtomSite(0.0, 0.5, 0.0, z2_type, name2),
            AtomSite(0.0, 0.0, 0.5, z2_type, name2),
            AtomSite(0.5, 0.5, 0.5, z2_type, name2),
        ],
        name=label,
        channel_spacing=a / np.sqrt(2),
    )


def _ensure_element_registered(name: str, where: str = "factory") -> None:
    """
    Raise ValueError if the element symbol `name` is not present
    in ATOM_DATA.

    Adding a new element to CRYSTAL_DB requires three coordinated edits:
      1. ATOM_DATA[Z] = (symbol, mass, u_300_K)
      2. (optional but recommended) MATERIAL_CLASS[crystal_name] for the
         neutralization model
      3. CRYSTAL_DB[name] = lambda: make_fcc(...) / make_bcc(...) / ...
    """
    SYM_TO_Z = {v[0]: k for k, v in ATOM_DATA.items()}
    if name not in SYM_TO_Z:
        raise ValueError(
            f"{where}: element '{name}' is not registered in ATOM_DATA. "
            f"Add an entry like  Z: ('{name}', mass_amu, u_300_K)  to "
            f"ATOM_DATA before using it as a target. Without this entry, "
            f"auto_targets_from_crystal would silently fall back to a "
            f"single-letter match (giving the wrong Z) or raise a "
            f"confusing ValueError later."
        )


def make_fcc(a: float, name: str, label: str) -> CrystalStructure:
    """
    FCC elemental metal (Fm-3m, #225).
    """
    _ensure_element_registered(name, where="make_fcc")
    return CrystalStructure(
        a=a, b=a, c=a,
        sites=[
            AtomSite(0.000, 0.000, 0.000, 0, name),
            AtomSite(0.500, 0.500, 0.000, 0, name),
            AtomSite(0.500, 0.000, 0.500, 0, name),
            AtomSite(0.000, 0.500, 0.500, 0, name),
        ],
        name=label,
        channel_spacing=a / np.sqrt(2),  # FCC <110> channel
    )


def make_bcc(a: float, name: str, label: str) -> CrystalStructure:
    """
    BCC elemental metal (Im-3m, #229).
    """
    _ensure_element_registered(name, where="make_bcc")
    return CrystalStructure(
        a=a, b=a, c=a,
        sites=[
            AtomSite(0.000, 0.000, 0.000, 0, name),
            AtomSite(0.500, 0.500, 0.500, 0, name),
        ],
        name=label,
        channel_spacing=a / np.sqrt(2),
    )


def make_diamond(a: float, name: str, label: str) -> CrystalStructure:
    """
    Diamond-cubic elemental crystal (Fd-3m, #227).
    """
    _ensure_element_registered(name, where="make_diamond")
    return CrystalStructure(
        a=a, b=a, c=a,
        sites=[
            # First FCC sub-lattice (origin)
            AtomSite(0.000, 0.000, 0.000, 0, name),
            AtomSite(0.500, 0.500, 0.000, 0, name),
            AtomSite(0.500, 0.000, 0.500, 0, name),
            AtomSite(0.000, 0.500, 0.500, 0, name),
            # Second FCC sub-lattice (offset by 1/4, 1/4, 1/4)
            AtomSite(0.250, 0.250, 0.250, 0, name),
            AtomSite(0.750, 0.750, 0.250, 0, name),
            AtomSite(0.750, 0.250, 0.750, 0, name),
            AtomSite(0.250, 0.750, 0.750, 0, name),
        ],
        name=label,
        channel_spacing=a / np.sqrt(2),  # <110> open channel
    )


# Pre-built structures
CRYSTAL_DB = {
    # ----------------------------------------------------------------------
    # FCC elemental metals (Fm-3m, #225) — experimental lattice parameters
    # ----------------------------------------------------------------------
    'Au': lambda: make_fcc(4.0782, 'Au', 'Au FCC (Fm-3m, a=4.0782 A)'),
    'Cu': lambda: make_fcc(3.6149, 'Cu', 'Cu FCC (Fm-3m, a=3.6149 A)'),
    'Ag': lambda: make_fcc(4.0862, 'Ag', 'Ag FCC (Fm-3m, a=4.0862 A)'),
    'Al': lambda: make_fcc(4.0494, 'Al', 'Al FCC (Fm-3m, a=4.0494 A)'),
    'Ni': lambda: make_fcc(3.5240, 'Ni', 'Ni FCC (Fm-3m, a=3.5240 A)'),
    'Pt': lambda: make_fcc(3.9242, 'Pt', 'Pt FCC (Fm-3m, a=3.9242 A)'),
    # ----------------------------------------------------------------------
    # BCC elemental metals (Im-3m, #229) — experimental lattice parameters
    # ----------------------------------------------------------------------
    'W':  lambda: make_bcc(3.1652, 'W',  'W BCC (Im-3m, a=3.1652 A)'),
    'Fe': lambda: make_bcc(2.8665, 'Fe', 'Fe BCC (Im-3m, a=2.8665 A)'),
    'Mo': lambda: make_bcc(3.1472, 'Mo', 'Mo BCC (Im-3m, a=3.1472 A)'),
    # ----------------------------------------------------------------------
    # Diamond-cubic elemental crystals (Fd-3m, #227)
    # 8 atoms per conventional cell. Classic substrates for ion-channelling
    # and LEIS validation experiments.
    # ----------------------------------------------------------------------
    'Si':  lambda: make_diamond(5.4307, 'Si', 'Si diamond-cubic (Fd-3m, a=5.4307 A)'),
    'Ge':  lambda: make_diamond(5.6580, 'Ge', 'Ge diamond-cubic (Fd-3m, a=5.6580 A)'),
    'C':   lambda: make_diamond(3.5667, 'C',  'C diamond (Fd-3m, a=3.5667 A)'),
    'Sn':  lambda: make_diamond(6.4892, 'Sn', 'alpha-Sn (Fd-3m, a=6.4892 A)'),
    # Zincblende III-V
    'GaP': lambda: make_zincblende(5.451, 'Ga', 0, 'P', 1, 'GaP zincblende'),
    'GaAs': lambda: make_zincblende(5.653, 'Ga', 0, 'As', 1, 'GaAs zincblende'),
    'InP': lambda: make_zincblende(5.869, 'In', 0, 'P', 1, 'InP zincblende'),
    'InAs': lambda: make_zincblende(6.058, 'In', 0, 'As', 1, 'InAs zincblende'),
    'GaSb': lambda: make_zincblende(6.096, 'Ga', 0, 'Sb', 1, 'GaSb zincblende'),
    'InSb': lambda: make_zincblende(6.479, 'In', 0, 'Sb', 1, 'InSb zincblende'),
    'AlAs': lambda: make_zincblende(5.661, 'Al', 0, 'As', 1, 'AlAs zincblende'),
    'AlP': lambda: make_zincblende(5.464, 'Al', 0, 'P', 1, 'AlP zincblende'),
    # Zincblende II-VI
    'CdTe': lambda: make_zincblende(6.482, 'Cd', 0, 'Te', 1, 'CdTe zincblende'),
    'ZnSe': lambda: make_zincblende(5.668, 'Zn', 0, 'Se', 1, 'ZnSe zincblende'),
    'ZnS': lambda: make_zincblende(5.409, 'Zn', 0, 'S', 1, 'ZnS zincblende'),
    'ZnTe': lambda: make_zincblende(6.101, 'Zn', 0, 'Te', 1, 'ZnTe zincblende'),
    'HgTe': lambda: make_zincblende(6.453, 'Hg', 0, 'Te', 1, 'HgTe zincblende'),
    'CdS': lambda: make_zincblende(5.832, 'Cd', 0, 'S', 1, 'CdS zincblende'),
    # IV-IV
    'SiC_3C': lambda: make_zincblende(4.360, 'Si', 0, 'C', 1, '3C-SiC zincblende'),
    # Wurtzite III-N
    'GaN': lambda: make_wurtzite(3.189, 5.185, 'Ga', 0, 'N', 1, 'GaN wurtzite'),
    'AlN': lambda: make_wurtzite(3.112, 4.982, 'Al', 0, 'N', 1, 'AlN wurtzite'),
    'InN': lambda: make_wurtzite(3.545, 5.703, 'In', 0, 'N', 1, 'InN wurtzite'),
    # Oxides — rocksalt
    'MgO': lambda: make_rocksalt(4.212, 'Mg', 0, 'O', 1, 'MgO rocksalt'),
    # Oxides — corundum (Al2O3)
    'Al2O3': lambda: CrystalStructure(
        a=4.759, b=4.759, c=12.991, alpha=90, beta=90, gamma=120,
        sites=[
            AtomSite(0.000, 0.000, 0.352, 0, 'Al'),
            AtomSite(0.000, 0.000, 0.648, 0, 'Al'),
            AtomSite(0.000, 0.000, 0.019, 0, 'Al'),
            AtomSite(0.000, 0.000, 0.981, 0, 'Al'),
            AtomSite(0.333, 0.667, 0.685, 0, 'Al'),
            AtomSite(0.333, 0.667, 0.315, 0, 'Al'),
            AtomSite(0.333, 0.667, 0.019, 0, 'Al'),
            AtomSite(0.333, 0.667, 0.981, 0, 'Al'),
            AtomSite(0.667, 0.333, 0.019, 0, 'Al'),
            AtomSite(0.667, 0.333, 0.981, 0, 'Al'),
            AtomSite(0.667, 0.333, 0.352, 0, 'Al'),
            AtomSite(0.667, 0.333, 0.648, 0, 'Al'),
            AtomSite(0.306, 0.000, 0.250, 1, 'O'),
            AtomSite(0.000, 0.306, 0.250, 1, 'O'),
            AtomSite(0.694, 0.694, 0.250, 1, 'O'),
            AtomSite(0.694, 0.000, 0.750, 1, 'O'),
            AtomSite(0.000, 0.694, 0.750, 1, 'O'),
            AtomSite(0.306, 0.306, 0.750, 1, 'O'),
            AtomSite(0.639, 0.667, 0.083, 1, 'O'),
            AtomSite(0.667, 0.973, 0.083, 1, 'O'),
            AtomSite(0.027, 0.361, 0.083, 1, 'O'),
            AtomSite(0.361, 0.667, 0.583, 1, 'O'),
            AtomSite(0.667, 0.027, 0.583, 1, 'O'),
            AtomSite(0.973, 0.639, 0.583, 1, 'O'),
            AtomSite(0.973, 0.333, 0.417, 1, 'O'),
            AtomSite(0.333, 0.306, 0.417, 1, 'O'),
            AtomSite(0.694, 0.027, 0.417, 1, 'O'),
            AtomSite(0.027, 0.333, 0.917, 1, 'O'),
            AtomSite(0.333, 0.360, 0.917, 1, 'O'),
            AtomSite(0.640, 0.973, 0.917, 1, 'O'),
        ],
        name='Al2O3 corundum (sapphire)',
        channel_spacing=4.759 / np.sqrt(3),
    ),
    # Beta-cristobalite SiO2
    'SiO2': lambda: CrystalStructure(
        a=7.16, b=7.16, c=7.16,
        sites=[
            AtomSite(0.000, 0.000, 0.000, 0, 'Si'),
            AtomSite(0.500, 0.500, 0.000, 0, 'Si'),
            AtomSite(0.250, 0.250, 0.250, 0, 'Si'),
            AtomSite(0.750, 0.750, 0.250, 0, 'Si'),
            AtomSite(0.500, 0.000, 0.500, 0, 'Si'),
            AtomSite(0.000, 0.500, 0.500, 0, 'Si'),
            AtomSite(0.250, 0.750, 0.750, 0, 'Si'),
            AtomSite(0.750, 0.250, 0.750, 0, 'Si'),
            AtomSite(0.125, 0.125, 0.125, 1, 'O'),
            AtomSite(0.875, 0.875, 0.125, 1, 'O'),
            AtomSite(0.375, 0.625, 0.125, 1, 'O'),
            AtomSite(0.625, 0.375, 0.125, 1, 'O'),
            AtomSite(0.125, 0.625, 0.375, 1, 'O'),
            AtomSite(0.875, 0.375, 0.375, 1, 'O'),
            AtomSite(0.375, 0.125, 0.375, 1, 'O'),
            AtomSite(0.625, 0.875, 0.375, 1, 'O'),
            AtomSite(0.125, 0.375, 0.625, 1, 'O'),
            AtomSite(0.875, 0.625, 0.625, 1, 'O'),
            AtomSite(0.375, 0.375, 0.625, 1, 'O'),
            AtomSite(0.625, 0.625, 0.625, 1, 'O'),
            AtomSite(0.125, 0.875, 0.875, 1, 'O'),
            AtomSite(0.875, 0.125, 0.875, 1, 'O'),
            AtomSite(0.375, 0.875, 0.875, 1, 'O'),
            AtomSite(0.625, 0.125, 0.875, 1, 'O'),
        ],
        name='beta-cristobalite SiO2 (Fd-3m)',
        channel_spacing=2.53,  # semi-channel width from your paper
    ),
    # Alpha-quartz SiO2
    'SiO2_alpha': lambda: CrystalStructure(
        a=4.9134, b=4.9134, c=5.4052, alpha=90, beta=90, gamma=120,
        sites=[
            # 3 Si atoms at Wyckoff 3a positions
            AtomSite(0.4699, 0.0000, 0.3333, 0, 'Si'),
            AtomSite(0.0000, 0.4699, 0.6667, 0, 'Si'),
            AtomSite(0.5301, 0.5301, 0.0000, 0, 'Si'),
            # 6 O atoms at Wyckoff 6c positions (symmetry-expanded)
            AtomSite(0.4145, 0.2662, 0.2858, 1, 'O'),
            AtomSite(0.7338, 0.1483, 0.6191, 1, 'O'),
            AtomSite(0.8517, 0.5855, 0.9525, 1, 'O'),
            AtomSite(0.2662, 0.4145, 0.7142, 1, 'O'),
            AtomSite(0.1483, 0.7338, 0.3809, 1, 'O'),
            AtomSite(0.5855, 0.8517, 0.0475, 1, 'O'),
        ],
        name='alpha-quartz SiO2 (P3_2 21)',
        channel_spacing=4.9134 / np.sqrt(3),
    ),
}


def get_crystal(name: str) -> CrystalStructure:
    """Get crystal structure from built-in database."""
    if name not in CRYSTAL_DB:
        available = ', '.join(sorted(CRYSTAL_DB.keys()))
        raise ValueError(f"Crystal '{name}' not in database. Available: {available}")
    return CRYSTAL_DB[name]()


def make_ternary_alloy(base_crystal: str, substitute_type: int,
                       substitute_name: str, fraction: float,
                       which_sublattice: int = 0) -> CrystalStructure:
    """
    Create ternary alloy by partial substitution.
    """
    import copy
    cs = copy.deepcopy(get_crystal(base_crystal))
    new_sites = []
    for site in cs.sites:
        if site.atom_type == which_sublattice:
            new_sites.append(AtomSite(
                site.x, site.y, site.z, site.atom_type,
                site.name, occupation=1.0 - fraction))
            new_sites.append(AtomSite(
                site.x, site.y, site.z, substitute_type,
                substitute_name, occupation=fraction))
        else:
            new_sites.append(site)
    cs.sites = new_sites
    cs.name = f"{substitute_name}{fraction:.1f}{cs.name}"
    return cs


# ========================================================================
#  TERNARY ALLOY EXTENSION (v2.1 ext — for CuAgAu noble-metal study)
# ========================================================================

def make_fcc_ternary(x_A: float, name_A: str, z_A: int,
                     x_B: float, name_B: str, z_B: int,
                     x_C: float, name_C: str, z_C: int,
                     a_lat: float = None) -> CrystalStructure:
    """
    Create a random-substitution fcc ternary alloy A_x B_y C_(1-x-y).

    Args:
        x_A, x_B, x_C : mole fractions; must sum to 1.0 (±1e-6).
        name_A/B/C    : element symbols (e.g. 'Cu', 'Ag', 'Au'). Each must
                        already be in ATOM_DATA.
        z_A/B/C       : atomic numbers — kept in the signature for
                        documentation only;
        a_lat         : lattice parameter in Å. If None, Vegard's law is
                        applied for noble metals (Cu, Ag, Au, Pt, Ni, Al).

    Returns:
        CrystalStructure with up to 12 AtomSite entries (3 species × 4 fcc
        positions).

    Example:
        >>> cs = make_fcc_ternary(0.33, 'Cu', 29,
        ...                       0.33, 'Ag', 47,
        ...                       0.34, 'Au', 79)
        >>> cs.a   # ≈ 3.928 Å (Vegard average)

    Notes:
        Co-located AtomSite entries with cumulative occupation = 1 are
        treated as mutually-exclusive choices in
        CrystalStructure.generate_atom_positions() (see fix #P6-2). Each
        physical site is occupied by exactly ONE of {A, B, C} per
        realisation, with probabilities (x_A, x_B, x_C).

    Raises:
        ValueError: if fractions don't sum to 1.0 (within 1e-6) or if
            any fraction is negative. Note: these checks were originally
            `assert` statements, which Python silently strips when run
            with the -O optimisation flag (or when bytecode is compiled
            to .pyo). With asserts, a typo like (0.4, 0.4, 0.4) → sum=1.2
            in production would silently produce a Vegard-averaged
            lattice parameter ~30% too large and a non-physical
            occupation of 1.2, with the corruption only manifesting
            much later as confusing renormalisation warnings during
            simulation. ValueError is enforced unconditionally.
    """
    s = x_A + x_B + x_C
    if abs(s - 1.0) >= 1e-6:
        raise ValueError(
            f"make_fcc_ternary: fractions must sum to 1.0, "
            f"got x_A={x_A}, x_B={x_B}, x_C={x_C} (sum={s:.6f}).")
    if min(x_A, x_B, x_C) < 0.0:
        raise ValueError(
            f"make_fcc_ternary: fractions must be non-negative, "
            f"got ({x_A}, {x_B}, {x_C}).")

    # Vegard interpolation
    VEGARD_A = {
        'Cu': 3.6149, 'Ag': 4.0862, 'Au': 4.0782,
        'Pt': 3.9242, 'Ni': 3.5240, 'Al': 4.0494,
        # 'Pd': 3.8901,  # uncomment after adding Pd to ATOM_DATA
    }
    if a_lat is None:
        unknown = [n for n in (name_A, name_B, name_C) if n not in VEGARD_A]
        if unknown:
            raise ValueError(
                f"make_fcc_ternary: no Vegard data for {unknown}. "
                f"Pass a_lat explicitly or extend VEGARD_A. "
                f"Known elements: {sorted(VEGARD_A.keys())}.")
        a_lat = (x_A * VEGARD_A[name_A] +
                 x_B * VEGARD_A[name_B] +
                 x_C * VEGARD_A[name_C])

    # Make sure the BCA engine can find elements
    for nm in (name_A, name_B, name_C):
        _ensure_element_registered(nm, where="make_fcc_ternary")

    fcc_pos = [(0.00, 0.00, 0.00), (0.50, 0.50, 0.00),
               (0.50, 0.00, 0.50), (0.00, 0.50, 0.50)]
    sites = []
    next_type = 0
    if x_A > 1e-6:
        for (fx, fy, fz) in fcc_pos:
            sites.append(AtomSite(fx, fy, fz, next_type, name_A,
                                   occupation=x_A))
        next_type += 1
    if x_B > 1e-6:
        for (fx, fy, fz) in fcc_pos:
            sites.append(AtomSite(fx, fy, fz, next_type, name_B,
                                   occupation=x_B))
        next_type += 1
    if x_C > 1e-6:
        for (fx, fy, fz) in fcc_pos:
            sites.append(AtomSite(fx, fy, fz, next_type, name_C,
                                   occupation=x_C))
        next_type += 1
    #The engine resolves Z via name.
    _ = (z_A, z_B, z_C)

    label = (f"{name_A}{x_A:.2f}{name_B}{x_B:.2f}{name_C}{x_C:.2f} "
             f"fcc ternary (a={a_lat:.4f} A, Vegard)")
    return CrystalStructure(
        a=a_lat, b=a_lat, c=a_lat,
        sites=sites,
        name=label,
        channel_spacing=a_lat / np.sqrt(2),  # FCC <110> channel
    )


def ternary_neutralization_v0(ion_sym: str,
                              x_A: float, name_A: str,
                              x_B: float, name_B: str,
                              x_C: float, name_C: str) -> float:
    """
    Effective Hagstrum v0 for a ternary alloy matrix.

    Args:
        ion_sym       : ion symbol ('He', 'Ne', 'Ar').
        x_A, x_B, x_C : mole fractions of the alloy components.
        name_A/B/C    : element symbols.

    Returns:
        Effective v0 in m/s.
    """
    def _lookup(elem: str) -> float:
        # 1. Element-specific entry, e.g. ('He','Cu')
        if (ion_sym, elem) in NeutralizationModel.V0_TABLE:
            return NeutralizationModel.V0_TABLE[(ion_sym, elem)]
        # 2. Fallback by material class (alloy components are metals)
        return NeutralizationModel.V0_TABLE.get(
            (ion_sym, 'metal'), 1.0e5)

    return (x_A * _lookup(name_A) +
            x_B * _lookup(name_B) +
            x_C * _lookup(name_C))


def load_cif(filename: str) -> CrystalStructure:
    """
    Load crystal structure from CIF file.
    Supports most standard CIF files from COD, ICSD, Materials Project.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    a = b = c = 1.0
    alpha = beta = gamma = 90.0
    sites = []
    reading_atoms = False
    col_map = {}
    has_symmetry_loop = False

    SYM_TO_Z = {v[0]: k for k, v in ATOM_DATA.items()}
    type_counter = {}

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue

        # Detect a symmetry loop — if present
        if line.startswith('_symmetry_equiv_pos') or \
           line.startswith('_space_group_symop'):
            has_symmetry_loop = True

        # Cell parameters
        if line.startswith('_cell_length_a'):
            a = float(line.split()[-1].split('(')[0])
        elif line.startswith('_cell_length_b'):
            b = float(line.split()[-1].split('(')[0])
        elif line.startswith('_cell_length_c'):
            c = float(line.split()[-1].split('(')[0])
        elif line.startswith('_cell_angle_alpha'):
            alpha = float(line.split()[-1].split('(')[0])
        elif line.startswith('_cell_angle_beta'):
            beta = float(line.split()[-1].split('(')[0])
        elif line.startswith('_cell_angle_gamma'):
            gamma = float(line.split()[-1].split('(')[0])

        elif line.startswith('_atom_site_aniso') or \
             line.startswith('_atom_site_attached'):
            if reading_atoms:
                reading_atoms = False
                col_map = {}
        elif line.startswith('_atom_site_'):
            key = line.split()[0]
            col_map[key] = len(col_map)
            reading_atoms = True
        elif reading_atoms and not line.startswith('_') and not line.startswith('loop_'):
            parts = line.split()
            if len(parts) >= len(col_map):
                try:
                    # Extract symbol
                    sym_key = '_atom_site_type_symbol'
                    label_key = '_atom_site_label'
                    sym = parts[col_map.get(sym_key, col_map.get(label_key, 0))]
                    # Clean symbol (remove charge, digits)
                    sym_clean = ''.join(c for c in sym if c.isalpha())[:2]
                    if sym_clean not in SYM_TO_Z:
                        warnings.warn(
                            f"load_cif: element '{sym_clean}' (from CIF "
                            f"label '{sym}') is not in ATOM_DATA — atom "
                            f"skipped. Add an entry to ATOM_DATA[Z] = "
                            f"('{sym_clean}', mass, u_300) to enable it.",
                            stacklevel=2)
                        continue

                    Z = SYM_TO_Z[sym_clean]

                    # Atom type index
                    if sym_clean not in type_counter:
                        type_counter[sym_clean] = len(type_counter)
                    atype = type_counter[sym_clean]

                    # Fractional coordinates
                    fx = float(parts[col_map.get('_atom_site_fract_x', 1)].split('(')[0])
                    fy = float(parts[col_map.get('_atom_site_fract_y', 2)].split('(')[0])
                    fz = float(parts[col_map.get('_atom_site_fract_z', 3)].split('(')[0])

                    # Occupancy
                    occ = 1.0
                    if '_atom_site_occupancy' in col_map:
                        occ = float(parts[col_map['_atom_site_occupancy']].split('(')[0])

                    sites.append(AtomSite(fx, fy, fz, atype, sym_clean, occ))
                except (ValueError, IndexError):
                    continue
        elif reading_atoms and (line.startswith('loop_') or line.startswith('_')):
            if not line.startswith('_atom_site_'):
                reading_atoms = False
                col_map = {}

    if has_symmetry_loop and len(sites) < 4:
        warnings.warn(
            f"CIF '{filename}' contains a symmetry loop but only "
            f"{len(sites)} atoms were parsed. Symmetry operations are "
            f"NOT applied by this loader. If this is the asymmetric "
            f"unit, expand to P1 before loading.",
            stacklevel=2)

    return CrystalStructure(
        a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
        sites=sites, name=f"CIF: {filename}",
        channel_spacing=min(a, b) / np.sqrt(2),
    )


# ===========================================================================
#  ION NEUTRALIZATION MODEL (NEW in v2)
# ===========================================================================

@dataclass
class NeutralizationModel:
    """
    Ion neutralization model.
    P+ = exp(-v0 / v_perp)   (Hagstrum)

    P+ = ion survival probability
    v0 = characteristic velocity
    v_perp = perpendicular component of ion velocity at the surface

    Set v0=0 to disable neutralisation (TOF-LEIS mode).
    """
    v0: float = 0.0  # Characteristic velocity (m/s), 0 = disabled
    enabled: bool = False  # Master switch

    # Known v0 values for common systems
    V0_TABLE = {
        ('Ne', 'metal'): 1.5e5,
        ('Ne', 'oxide'): 2.0e5,
        ('Ne', 'semi'):  1.6e5,
        ('Ar', 'metal'): 2.0e5,
        ('Ar', 'oxide'): 2.6e5,
        ('Ar', 'semi'):  2.2e5,
        ('He', 'metal'): 2.8e5,
        ('He', 'oxide'): 3.3e5,
        ('He', 'semi'):  3.0e5,
        ('He', 'Cu'):   4.05e5,
        ('He', 'Ag'):   3.35e5,
        ('He', 'Au'):   4.33e5,
    }

    def survival_probability(self, E_eV: float, ion_mass_amu: float,
                             polar_angle_deg: float) -> float:
        """
        Calculate ion survival probability P+.

        Args:
            E_eV: ion energy at exit (eV)
            ion_mass_amu: ion mass (amu)
            polar_angle_deg: exit angle from surface (degrees)
        Returns:
            P+ in range [0, 1]
        """
        if not self.enabled or self.v0 <= 0:
            return 1.0

        # Ion velocity
        v = np.sqrt(2.0 * E_eV * EV_TO_J / (ion_mass_amu * AMU_TO_KG))

        # Perpendicular component
        v_perp = v * np.sin(np.radians(abs(polar_angle_deg)))

        if v_perp < 1.0:
            return 0.0

        return np.exp(-self.v0 / v_perp)


# ===========================================================================
#  POSITION-DEPENDENT ELECTRONIC STOPPING (NEW in v2)
# ===========================================================================

@dataclass
class ElectronicStoppingModel:
    """
    Position-dependent electronic stopping Se(v, ρ)
    Based on: Nuñez et al., Commun. Mater. 6 (2025)
    Se = k_e * v * rho_e(r) / rho_avg
    Position dependence in channel:
    f(r) = 1 - alpha_ch * exp(-r^2 / (2*sigma_ch^2))
    where r is distance to nearest atomic row.
    alpha_ch controls channel reduction (typical 0.3–0.5).
    """
    # Continuous (non-local) stopping base coefficient
    se_base: float = 0.15  # eV^(1/2) / Angstrom

    # Channel reduction parameters
    alpha_channel: float = 0.3  # reduction factor in channel center (0=off)
    sigma_channel: float = 1.5  # channel width parameter (Angstrom)

    # Local stopping parameters
    local_model: str = 'srim'  # 'srim', 'oen_robinson', or 'off'

    def continuous_stopping(self, E: float, distance_A: float,
                            r_to_nearest_row: float) -> float:
        """
        Non-local electronic stopping between collisions.
        Reduced in channel center, full near atomic rows.
        """
        # Position-dependent factor
        if self.alpha_channel > 0 and self.sigma_channel > 0:
            f_pos = 1.0 - self.alpha_channel * np.exp(
                -r_to_nearest_row ** 2 / (2.0 * self.sigma_channel ** 2))
        else:
            f_pos = 1.0

        return self.se_base * np.sqrt(E) * distance_A * f_pos


# ===========================================================================
#  DEFECT MAP (NEW in v2)
# ===========================================================================

@dataclass
class DefectMap:
    """
    Pre-generated defect map for the crystal surface.
    The internal representation changes from
        vacancies: Set[(ix,iy,iz)]
    to
        _vacant_global_idx: Set[int]
    """
    vacancy_fraction: float = 0.0  # 0 = perfect crystal
    vacancies: Set[Tuple[int, ...]] = field(default_factory=set)

    def generate(self, nx: int = 20, ny: int = 20, nz: int = 5,
                 rng: np.random.Generator = None):
        self.vacancies = set()

    def is_vacant(self, ix: int, iy: int, iz: int) -> bool:
        return False


# ===========================================================================
#  Data classes (Ion, TargetAtom, Detector — same as v1 with minor additions)
# ===========================================================================

@dataclass
class Ion:
    Z: int
    A: float
    name: str = ""


@dataclass
class TargetAtom:
    Z: int
    A: float
    name: str = ""
    d_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(3))
    b_exponents: np.ndarray = field(default_factory=lambda: np.zeros(3))
    is_nlh: bool = False
    mu: float = 0.0
    cm: float = 0.0
    cei: float = 0.0
    cea: float = 0.0
    cv: float = 0.0
    cn1: float = 0.0
    cn2: float = 0.0
    g_limits: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Detector:
    theta_deg: float = 136.0
    acceptance_deg: float = 3.0
    azimuth_deg: float = 0.0
    azimuth_accept_deg: float = 180.0
    energy_resolution: float = 0.01

    def scattering_angle_for_ion(self, psi_deg, polar_deg, azimuth_deg, xi_deg=0.0):
        psi, xi = np.radians(psi_deg), np.radians(xi_deg)
        alpha, phi = np.radians(polar_deg), np.radians(azimuth_deg)
        # Incident direction: negative z
        ix = -np.cos(psi) * np.cos(xi)
        iy = -np.cos(psi) * np.sin(xi)
        iz = -np.sin(psi)
        # Outgoing direction: positive z
        ex = np.cos(alpha) * np.cos(phi)
        ey = np.cos(alpha) * np.sin(phi)
        ez = np.sin(alpha)
        ct = max(-1.0, min(1.0, ix * ex + iy * ey + iz * ez))
        return np.degrees(np.arccos(ct))

    def ion_in_detector(self, psi_deg, polar_deg, azimuth_deg, xi_deg=0.0):
        sc = self.scattering_angle_for_ion(psi_deg, polar_deg,
                                           azimuth_deg, xi_deg)
        if abs(sc - self.theta_deg) > self.acceptance_deg:
            return False
        if self.azimuth_accept_deg >= 180.0:
            return True
        d_az = (azimuth_deg - self.azimuth_deg + 540.0) % 360.0 - 180.0
        return abs(d_az) <= self.azimuth_accept_deg

    def kinematic_energy(self, E0, ion_A, target_A, target_Z=0):
        mu = target_A / ion_A
        th = np.radians(self.theta_deg)
        disc = mu ** 2 - np.sin(th) ** 2
        return E0 * ((np.cos(th) + np.sqrt(disc)) / (1 + mu)) ** 2 if disc > 0 else 0.0


@dataclass
class SimulationParams:
    E0: float = 2000.0
    n_x: int = 10
    n_y: int = 501
    x_step: int = 10
    y_step: int = 10
    max_collisions: int = 200
    min_energy: float = 100.0
    n_pgr_points: int = 12
    use_detector: bool = False
    detector: Detector = field(default_factory=Detector)
    neutralization: NeutralizationModel = field(default_factory=NeutralizationModel)
    stopping: ElectronicStoppingModel = field(default_factory=ElectronicStoppingModel)
    defects: DefectMap = field(default_factory=DefectMap)
    theta_boundary: float = 0.05


# ===========================================================================
#  BCA PHYSICS ENGINE (updated for v2)
# ===========================================================================

class BCAEngine:
    """Core physics: potential, scattering integral, stopping, neutralization."""
    def __init__(self, ion: Ion, targets: List[TargetAtom],
                 crystal: CrystalStructure, surface_psi: float, surface_xi: float,
                 u_therm, params: SimulationParams):
        self.ion = ion
        self.targets = targets
        self.crystal = crystal
        self.psi_deg = surface_psi
        self.xi_deg = surface_xi
        self.u_therm_full = u_therm
        if isinstance(u_therm, dict):
            self.u_therm = max(u_therm.values()) if u_therm else 0.0
        else:
            self.u_therm = float(u_therm)
        self.params = params
        self._init_targets()

    def _init_targets(self):
        for t in self.targets:
            t.d_coeffs, t.b_exponents, t.is_nlh = get_potential_params(self.ion.Z, t.Z)
            t.mu = t.A / self.ion.A
            t.cm = (1.0 + t.mu) / t.mu
            t.cei = 1.0 / (1.0 + t.mu) ** 2
            t.cea = t.mu * t.cei
            t.cv = E2 * self.ion.Z * t.Z * t.cm
            t.g_limits = np.array([
                -np.inf if abs(t.b_exponents[i]) < 1e-30 else 160.0 / t.b_exponents[i]
                for i in range(len(t.b_exponents))
            ])
            t.cn1, t.cn2 = self._calc_estop_coeffs(t)

    def _calc_estop_coeffs(self, target):
        z1, z2 = float(self.ion.Z), float(target.Z)
        a1 = self.ion.A
        zp = z1 ** (1 / 6)
        zs = (z1 ** (2 / 3) + z2 ** (2 / 3)) ** 1.5
        am = np.sqrt(a1 / (a1 + target.A))
        z_osc = 1.0
        if 2 <= z1 <= 36:
            z_osc = 1.0 + 0.13 * np.sin(PI * z1 / 10) * np.exp(-0.1 * z1 / 10)
        cn1 = 0.045 * z1 * z2 * zp * am * z_osc / zs
        cn2 = 0.55 / (0.4685 / (z1 ** 0.23 + z2 ** 0.23))
        return cn1, cn2

    def find_rmin(self, p, E, k, tol=1e-10, max_iter=200):
        t = self.targets[k]
        p2 = p * p
        r = max(p, 0.01)
        for _ in range(max_iter):
            phi, dphi = 0.0, 0.0
            for i in range(len(t.d_coeffs)):
                if r + t.g_limits[i] <= 0:
                    ei = t.d_coeffs[i] * np.exp(t.b_exponents[i] * r)
                    phi += ei
                    dphi += t.b_exponents[i] * ei
            vr = t.cv / (r * E)
            vre = vr * phi
            fn = 1.0 - p2 / (r * r) - vre
            if abs(fn) < tol: return r, vre
            dfn = 2 * p2 / (r ** 3) - vr * dphi + vre / r
            if abs(dfn) < 1e-30:
                r *= 1.1
                continue
            r = max(r - fn / dfn, 1e-4)
        return r, vre

    def scattering_angle_cm(self, p, E, k):
        rmin, vre = self.find_rmin(p, E, k)
        t = self.targets[k]

        def integrand(phi):
            if abs(phi - PI / 2) < 1e-12: return 0.0
            cp = np.cos(phi)
            if abs(cp) < 1e-15: return 0.0
            r = rmin / cp
            scr = 0.0
            for i in range(len(t.d_coeffs)):
                if r + t.g_limits[i] <= 0:
                    scr += t.d_coeffs[i] * np.exp(t.b_exponents[i] * r)
            arg = max(1.0 - (p / r) ** 2 - t.cv * scr / (r * E), 1e-30)
            return np.sin(phi) / np.sqrt(arg)

        ys, _ = integrate.quad(integrand, 0, PI / 2, limit=100, epsabs=1e-10, epsrel=1e-8)
        return abs(PI - 2 * p * ys / rmin), rmin, vre

    def electronic_loss_local(self, E, rmin, vre, k):
        t = self.targets[k]
        return max(t.cn1 * np.sqrt(E) * (1 - 0.7 * vre) * np.exp(-t.cn2 * rmin), 0.0)

    def single_collision(self, p, E, k):
        t = self.targets[k]
        ts, rmin, vre = self.scattering_angle_cm(p, E, k)
        en = self.electronic_loss_local(E, rmin, vre, k)
        f = np.sqrt(max(1 - t.cm * en / E, 0.01))
        fm = t.mu * f
        rti = np.arctan2(np.sin(ts), 1 / fm + np.cos(ts))
        if rti < 0: rti += PI
        ks = 1
        if fm < 1:
            tsi = np.arctan2(np.sqrt(max(1 - fm ** 2, 0)), -fm)
            if tsi < 0: tsi += PI
            if ts > tsi: ks = -1
        ei = t.cei * E * (np.cos(rti) + ks * np.sqrt(max(fm ** 2 - np.sin(rti) ** 2, 0))) ** 2
        rta = np.arctan2(np.sin(ts), 1 / f - np.cos(ts))
        if rta < 0: rta += PI
        ka = 1
        ea = t.cea * E * (np.cos(rta) + ka * np.sqrt(max(f ** 2 - np.sin(rta) ** 2, 0))) ** 2
        ea_balance = max(0.0, E - ei - en)
        if ts < np.radians(1.0) and abs(ea - ea_balance) > 1e-3:
            ea = ea_balance
        return {'theta_i_rad': rti, 'theta_i_deg': rti * RD, 'E_i': ei,
                'theta_a_rad': rta, 'theta_a_deg': rta * RD, 'E_a': ea,
                'en': en, 'r_min': rmin, 'theta_cm': ts, 'ks': ks, 'ka': ka, 'f': f, 'fm': fm}


# ===========================================================================
#  UNIVERSAL LATTICE NAVIGATOR (NEW in v2)
# ===========================================================================

class UniversalNavigator:
    """
    Finds the nearest atomic row to the ion trajectory
    for ANY crystal structure.
    """

    def __init__(self, crystal: CrystalStructure, u_therm, # may be float OR dict
                 defects: DefectMap, rng: np.random.Generator,
                 psi_deg: float = 90.0, E0: float = 2000.0,
                 supercell_cap: int = 20):
        self.crystal = crystal
        self.defects = defects
        self.rng = rng
        self.PD = crystal.auto_channel_spacing()

        if isinstance(u_therm, dict):
            self.u_therm_dict = dict(u_therm)
            # representative scalar (max) for legacy code paths and exit
            # threshold (z_exit_threshold = max(1.0, 3*u))
            self.u_therm = max(u_therm.values()) if u_therm else 0.0
        else:
            self.u_therm_dict = None
            self.u_therm = float(u_therm)

        psi_rad = max(np.radians(psi_deg), np.radians(0.5))
        a_typical = min(crystal.a, crystal.b)
        c_typical = crystal.c if crystal.c > 0 else a_typical
        max_depth_A = 5.0 + 25.0 * np.sin(psi_rad)

        # Lateral travel = max_depth / tan(psi)
        lat_travel_A = max_depth_A / np.tan(psi_rad)

        nx_needed = max(3, int(np.ceil(lat_travel_A / a_typical)) + 2)
        nx = min(nx_needed, supercell_cap)
        ny = min(nx_needed, supercell_cap)
        nz_needed = max(3, int(np.ceil(max_depth_A / c_typical)) + 2)
        nz = min(nz_needed, 25)

        # Store supercell extents for trajectory-bounds checks downstream
        self.nx_cells = nx
        self.ny_cells = ny
        self.nz_cells = nz
        self.x_extent = nx * crystal.a
        self.y_extent = ny * crystal.b if crystal.b > 0 else nx * crystal.a
        self.z_extent_below = nz * c_typical  # depth available below z=0
        self.max_depth_A = max_depth_A

        self.atoms = crystal.generate_atom_positions(nx=nx, ny=ny, nz=nz, rng=rng)

        if defects.vacancy_fraction > 0:
            keep_mask = rng.random(len(self.atoms)) >= defects.vacancy_fraction
            self.atoms = [a for a, keep in zip(self.atoms, keep_mask) if keep]

        # Pre-compute numpy arrays for fast distance calculation
        self._pos = np.array([a['pos'] for a in self.atoms])
        self._types = np.array([a['type'] for a in self.atoms])
        self._names = [a['name'] for a in self.atoms]

        if self.u_therm_dict is not None:
            self._u_per_atom = np.array(
                [self.u_therm_dict.get(int(t), 0.0) for t in self._types]
            )
        else:
            self._u_per_atom = np.full(len(self.atoms), self.u_therm)

        if np.any(self._u_per_atom > 0):
            # broadcast u (per-atom) over xyz: scale[:,None] * standard_normal
            self._thermal = (self.rng.standard_normal(self._pos.shape)
                             * self._u_per_atom[:, None])
            self._pos = self._pos + self._thermal
        else:
            self._thermal = np.zeros_like(self._pos)

    def refresh_thermal(self):
        """Generate a new frozen thermal snapshot. Call once per trajectory."""
        if not np.any(self._u_per_atom > 0):
            return
        # Restore equilibrium positions, then re-roll
        self._pos = self._pos - self._thermal
        self._thermal = (self.rng.standard_normal(self._pos.shape)
                         * self._u_per_atom[:, None])
        self._pos = self._pos + self._thermal

    def find_nearest(self, x, y, z, L, M, N, E, pgr_func, exclude_idx=-1):
        """
        Find the NEXT atom along the ion trajectory.
        """
        max_range = 15.0  # Angstrom

        bb_mask = (
            (self._pos[:, 0] > x - max_range) & (self._pos[:, 0] < x + max_range) &
            (self._pos[:, 1] > y - max_range) & (self._pos[:, 1] < y + max_range) &
            (self._pos[:, 2] > z - max_range) & (self._pos[:, 2] < z + max_range)
        )
        bb_idx = np.where(bb_mask)[0]
        if len(bb_idx) == 0:
            return False, {}

        # Only compute distances for atoms passing the bounding box
        local_pos = self._pos[bb_idx]
        dx = local_pos[:, 0] - x
        dy = local_pos[:, 1] - y
        dz = local_pos[:, 2] - z

        # Longitudinal projection along trajectory direction
        stp_arr = L * dx + M * dy + N * dz

        mask = (stp_arr > 0.01) & (stp_arr < max_range)
        if exclude_idx >= 0:
            local_excl = np.where(bb_idx == exclude_idx)[0]
            if len(local_excl):
                mask[local_excl[0]] = False
        local_candidates = np.where(mask)[0]
        if len(local_candidates) == 0:
            return False, {}
        # Map back to global indices
        candidates = bb_idx[local_candidates]

        ve = L * L + M * M + N * N
        sv = 1.0 / np.sqrt(ve) if ve > 0 else 1.0

        # Vectorized impact parameter for all candidates
        fx_arr = dx[local_candidates]
        fy_arr = dy[local_candidates]
        fz_arr = dz[local_candidates]
        cx = N * fy_arr - M * fz_arr
        cy = L * fz_arr - N * fx_arr
        cz = M * fx_arr - L * fy_arr
        di_arr = np.sqrt(cx * cx + cy * cy + cz * cz) * sv

        # Per-atom PGR threshold (depends on atom type)
        types_arr = self._types[candidates]
        unique_types = np.unique(types_arr)
        pg_per_type = {int(tt): pgr_func(E, int(tt)) for tt in unique_types}
        pg_arr = np.array([pg_per_type[int(tt)] for tt in types_arr])

        # Atoms within collision range
        hit_mask = di_arr < pg_arr
        stp_local = stp_arr[local_candidates]

        # Pack a result-dict given an index into local_candidates (= di_arr length)
        def pack(local_idx):
            # candidates[local_idx] is global index into self._pos / self.atoms
            global_idx = int(candidates[local_idx])
            atom = self.atoms[global_idx]
            ax, ay, az = self._pos[global_idx]
            return {
                'pos': (ax, ay, az),
                'type': atom['type'],
                'name': atom['name'],
                'global_idx': global_idx,    # FIX (audit-pass 7, pt 5)
                'di': float(di_arr[local_idx]),
                'fx': float(fx_arr[local_idx]),
                'fy': float(fy_arr[local_idx]),
                'fz': float(fz_arr[local_idx]),
                'stp': float(stp_local[local_idx]),
                'dist': float(np.sqrt(fx_arr[local_idx]**2 +
                                      fy_arr[local_idx]**2 +
                                      fz_arr[local_idx]**2)),
                've': ve,
                'sv': sv,
            }

        # Stage 1: among collision candidates, pick smallest stp
        if np.any(hit_mask):
            hit_indices = np.where(hit_mask)[0]
            best_local = hit_indices[np.argmin(stp_local[hit_indices])]
            return True, pack(best_local)

        best_local = int(np.argmin(stp_local))
        return False, pack(best_local)


# ===========================================================================
#  PGR Calculator (same as v1, uses new engine)
# ===========================================================================

class BoundaryParameterCalculator:
    def __init__(self, engine: BCAEngine):
        self.engine = engine
        self.results = {}

    def compute(self, verbose=True):
        params = self.engine.params
        for k, t in enumerate(self.engine.targets):
            if verbose:
                print(f"  PGR: {self.engine.ion.name} -> {t.name} (Z={t.Z})")
            energies, p_bounds = [], []
            e0 = params.E0
            step = max(e0 / (params.n_pgr_points + 1), 100)
            for ii in range(params.n_pgr_points):
                e1 = e0 + (params.n_pgr_points // 2 - ii) * step
                if e1 < 200: continue
                p_gr = self._find_boundary_p(e1, k, params.theta_boundary)
                if p_gr > 0:
                    energies.append(e1)
                    p_bounds.append(p_gr)
                    if verbose: print(f"    E={e1:7.0f} eV  p_gr={p_gr:.4f} A")
            if len(energies) < 3:
                self.results[k] = (1.0, 0.2)
                continue
            le = np.log10(energies)
            lp = np.log10(p_bounds)
            n = len(le)
            # FIX (BUG #5): protect against degenerate input (all energies equal
            # would make denominator zero -> NaN -> silently corrupted PGR).
            denom = np.sum(le) ** 2 - n * np.sum(le ** 2)
            if abs(denom) < 1e-12:
                warnings.warn(f"PGR regression denominator near zero for target {k}; "
                              f"using fallback (AA=1.0, CT=0.2).")
                self.results[k] = (1.0, 0.2)
                continue
            ct = (n * np.sum(le * lp) - np.sum(le) * np.sum(lp)) / denom
            aa = 10 ** ((np.sum(lp) + ct * np.sum(le)) / n)
            self.results[k] = (aa, ct)
            if verbose: print(f"    => AA={aa:.5f}, CT={ct:.5f}")
        return self.results

    def _find_boundary_p(self, E, k, theta_t, p_min=0.001, p_max=6.0):
        r0 = self.engine.single_collision(p_min, E, k)
        if r0['theta_i_rad'] < theta_t: return -1.0
        lo, hi = p_min, p_max
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            r = self.engine.single_collision(mid, E, k)
            if r['theta_i_rad'] > theta_t:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-6: break
        return 0.5 * (lo + hi)

    def get_pgr(self, E, k):
        if k not in self.results:
            warnings.warn(f"PGR not computed for target index {k}; "
                          f"using fallback 0.5 A. Did compute() run?",
                          stacklevel=2)
            return 0.5
        aa, ct = self.results[k]
        return aa / E ** ct


# ===========================================================================
#  TRAJECTORY RESULT
# ===========================================================================

@dataclass
class TrajectoryResult:
    final_energy: float = 0.0
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    total_inelastic_loss: float = 0.0
    n_collisions: int = 0
    max_depth: float = 0.0
    backscattered: bool = False
    in_detector: bool = False  # FIX audit-pass 6, bug #4: track detector hits
    ion_survival_prob: float = 1.0
    sputtered_recoils: list = field(default_factory=list)
    trajectory: list = field(default_factory=list)
    last_collision_atom: str = ""
    collision_record: list = field(default_factory=list)


# ===========================================================================
#  CHANNELING SIMULATION v2 (universal navigator)
# ===========================================================================

class ChannelingSimulation:
    """Full trajectory simulation using universal crystal navigation."""

    def __init__(self, engine: BCAEngine, pgr: BoundaryParameterCalculator):
        self.engine = engine
        self.pgr = pgr
        self.params = engine.params
        self.rng = np.random.default_rng()

        psi = np.radians(engine.psi_deg)
        xi = np.radians(engine.xi_deg)
        self.L1 = -np.cos(psi) * np.cos(xi)
        self.M1 = -np.cos(psi) * np.sin(xi)
        self.N1 = -np.sin(psi)
        self.PD = engine.crystal.auto_channel_spacing()

        # Initialize navigator
        self.params.defects.generate(rng=self.rng)
        self.navigator = UniversalNavigator(
            engine.crystal, engine.u_therm_full, self.params.defects, self.rng,
            psi_deg=engine.psi_deg, E0=self.params.E0)

    def run_single_trajectory(self, x0_A, y0_A, output_file=None):
        result = TrajectoryResult()
        self.navigator.refresh_thermal()
        E = self.params.E0
        L, M, N = self.L1, self.M1, self.N1
        x, y, z = x0_A, y0_A, 0.0  # Angstrom
        sne = 0.0
        kc = 0
        z_max = 0.0
        ko = 0
        kc0 = 0
        stopping = self.params.stopping

        z_exit_threshold = max(1.0, 3.0 * self.engine.u_therm)
        last_idx = -1  # FIX (audit-pass 7, pt 5): exclude last partner

        while True:
            kc0 += 1
            if kc0 >= self.params.max_collisions or E <= self.params.min_energy:
                break

            # Find nearest atom using universal navigator
            hit, atom = self.navigator.find_nearest(
                x, y, z, L, M, N, E,
                lambda e, k: self.pgr.get_pgr(e, k),
                exclude_idx=last_idx)

            if not atom:
                lateral_extent = max(self.navigator.x_extent,
                                     self.navigator.y_extent)
                if N > 1e-6 and E > self.params.min_energy:
                    # ion moving outward
                    dz_needed = z_exit_threshold - z + 0.001
                    if dz_needed > 0:
                        t_exit = dz_needed / N
                        lateral_disp = t_exit * np.sqrt(max(L * L + M * M, 0.0))
                        if lateral_disp <= lateral_extent:
                            x += L * t_exit
                            y += M * t_exit
                            z += N * t_exit
                break

            r_to_row = atom['di']  # impact parameter

            stp = atom['stp']
            ve = atom['ve']
            path_len = stp / np.sqrt(ve) if ve > 0 else 0.0

            de_cont = stopping.continuous_stopping(E, path_len, r_to_row)
            E -= de_cont
            sne += de_cont
            if E <= self.params.min_energy:
                break

            if not hit:
                if ve > 0:
                    x += L * stp / ve
                    y += M * stp / ve
                    z += N * stp / ve
                last_idx = -1
                continue

            # ===========================================================
            # COLLISION
            # ===========================================================
            k = atom['type']
            p_phys = atom['di']

            col = self.engine.single_collision(p_phys, E, k)

            if z < z_max:
                z_max = z

            if col['E_a'] > 3.0:
                ko += 1
                result.sputtered_recoils.append({
                    'energy': col['E_a'], 'atom': atom['name']})

            E = col['E_i']
            sne += col['en']

            if ve > 0:
                x += L * stp / ve
                y += M * stp / ve
                z += N * stp / ve

            theta_lab = col['theta_i_rad']
            if theta_lab > 1e-6:
                fx, fy, fz = atom['fx'], atom['fy'], atom['fz']
                nx = M * fz - N * fy
                ny = N * fx - L * fz
                nz = L * fy - M * fx
                nn = np.sqrt(nx * nx + ny * ny + nz * nz)
                if nn > 1e-12:
                    ux, uy, uz = nx / nn, ny / nn, nz / nn
                    ct, st = np.cos(theta_lab), np.sin(theta_lab)
                    d = L * ux + M * uy + N * uz
                    L_old, M_old, N_old = L, M, N
                    L = L_old * ct + (M_old * uz - N_old * uy) * st + ux * d * (1 - ct)
                    M = M_old * ct + (N_old * ux - L_old * uz) * st + uy * d * (1 - ct)
                    N = N_old * ct + (L_old * uy - M_old * ux) * st + uz * d * (1 - ct)
                else:
                    L, M, N = -L, -M, -N

                norm = np.sqrt(L * L + M * M + N * N)
                if norm > 1e-10:
                    L, M, N = L / norm, M / norm, N / norm

            kc += 1
            kc0 = 0
            result.last_collision_atom = atom['name']
            if kc <= 3:
                result.collision_record.append((x, y, z, atom['name']))
            last_idx = atom.get('global_idx', -1)

            # Exit check (z above threshold means ion has escaped through surface)
            if z > z_exit_threshold:
                break


        # RESULTS
        result.final_energy = E
        result.n_collisions = kc
        result.total_inelastic_loss = sne
        result.max_depth = abs(z_max)

        ve_norm = L * L + M * M + N * N
        if ve_norm > 1e-30:
            nn = N / np.sqrt(ve_norm)
            nn = max(-1.0, min(1.0, nn))
            result.polar_angle = RD * np.arcsin(nn)
        else:
            result.polar_angle = 0.0

        if (L * L + M * M) > 1e-20:
            result.azimuthal_angle = RD * np.arctan2(M, L)
        else:
            result.azimuthal_angle = 0.0

        result.backscattered = (z > z_exit_threshold and E > self.params.min_energy)

        # Neutralization
        neut = self.params.neutralization
        result.ion_survival_prob = neut.survival_probability(
            E, self.engine.ion.A, result.polar_angle)

        if result.backscattered and self.params.use_detector:
            det = self.params.detector
            result.in_detector = det.ion_in_detector(
                self.engine.psi_deg,
                result.polar_angle,
                result.azimuthal_angle,
                self.engine.xi_deg)
        else:
            result.in_detector = result.backscattered

        write_this = result.backscattered and (
            (not self.params.use_detector) or result.in_detector
        )
        if output_file and write_this:
            output_file.write(
                f"{0:5d}{kc:5d}{E:7.0f}{sne:6.0f}"
                f"{result.polar_angle:7.2f}{result.azimuthal_angle:7.2f}"
                f"{x:8.2f}{y:8.2f}{z:6.2f}{z_max:6.2f}{ko:4d}"
                f"  P+={result.ion_survival_prob:.3f}\n")

        return result

    def run_simulation(self, verbose=True, output_filename='bca_modern_output.dat'):
        """Run full simulation over entry point grid."""
        results = []
        n_total = 0
        n_back = 0
        n_detected = 0

        PD = self.PD
        # Grid: distribute n_x * n_y points over channel cross-section
        nx = max(2, self.params.n_x)
        ny = max(2, self.params.n_y)
        x_vals = np.linspace(0.01 * PD, 0.99 * PD, nx)
        y_vals = np.linspace(0.01 * PD, 0.99 * PD, ny)
        total = len(x_vals) * len(y_vals)

        if verbose:
            print(f"\n=== Channeling Simulation v2 ===")
            print(f"Crystal: {self.engine.crystal.name}")
            print(f"Ion: {self.engine.ion.name}, E0={self.params.E0:.0f} eV")
            print(f"Angles: psi={self.engine.psi_deg:.1f} deg, xi={self.engine.xi_deg:.1f} deg")
            u_full = self.engine.u_therm_full
            if isinstance(u_full, dict) and u_full:
                # Build "Si:0.065, O:0.085" string from sites
                type_to_name = {}
                for s in self.engine.crystal.sites:
                    if s.atom_type not in type_to_name:
                        type_to_name[s.atom_type] = s.name
                parts = [
                    f"{type_to_name.get(t, '?')}:{float(u):.3f}"
                    for t, u in sorted(u_full.items())
                ]
                print(f"Thermal: {{ {', '.join(parts)} }} A (per element)")
            else:
                print(f"Thermal: u={float(u_full):.3f} A")
            print(
                f"Neutralization: {'ON v0=' + str(self.params.neutralization.v0) if self.params.neutralization.enabled else 'OFF'}")
            print(f"Stopping: alpha_ch={self.params.stopping.alpha_channel}")
            print(f"Defects: {self.params.defects.vacancy_fraction * 100:.1f}% vacancies")
            print(f"Grid: {total} entry points")

        t0 = time.time()
        fout = open(output_filename, 'w')
        fout.write(f"# BCA_MODERN — {self.engine.crystal.name}\n")
        fout.write(f"# E0={self.params.E0:.0f} psi={self.engine.psi_deg:.1f} xi={self.engine.xi_deg:.1f}\n")

        for x0 in x_vals:
            for y0 in y_vals:
                n_total += 1
                res = self.run_single_trajectory(x0, y0, fout)
                results.append(res)  # store ALL trajectories
                if res.backscattered:
                    n_back += 1
                    if res.in_detector and self.params.use_detector:
                        n_detected += 1
                if verbose and n_total % 200 == 0:
                    dt = time.time() - t0
                    print(f"  {n_total}/{total} traj, {n_back} back, {n_total / dt:.0f} traj/s")

        ie = np.zeros(210, dtype=int)
        for r in results:
            if not r.backscattered:
                continue
            if self.params.use_detector and not r.in_detector:
                continue
            ien = int(r.final_energy / 25)
            if 0 <= ien < 210:
                ie[ien] += 1
        fout.write("# Energy histogram (25 eV bins, backscattered only)\n")
        for i in range(0, 210, 15):
            fout.write(' '.join(f"{ie[j]:5d}" for j in range(i, min(i + 15, 210))) + '\n')

        kr = sum(len(r.sputtered_recoils) for r in results)
        n_spectrum = sum(1 for r in results
                         if r.backscattered and
                         (not self.params.use_detector or r.in_detector))
        fout.write(f"# KSI={n_spectrum:6d} KR={kr:6d}\n")
        fout.close()

        if verbose:
            dt = time.time() - t0
            print(f"\nDone: {n_total} traj in {dt:.1f}s, {n_back} backscattered")
            if self.params.use_detector:
                print(f"  → {n_detected} ions in detector "
                      f"(theta={self.params.detector.theta_deg} deg, "
                      f"acceptance=+/-{self.params.detector.acceptance_deg} deg)")
            print(f"Output: {output_filename}")

        return results


# ===========================================================================
#  AZIMUTHAL SCANNING (NEW in v2)
# ===========================================================================

def azimuthal_scan(ion, targets, crystal, params, u_therm,
                   psi_deg, xi_range=(0, 91, 2), verbose=True):
    """
    Scan azimuthal angle to map all semi-channels.
    """
    import os as _os  # FIX (BUG #11 ext): cross-platform null device.
    xi_start, xi_end, xi_step = xi_range
    result_map = {}

    if verbose:
        print(f"\n=== Azimuthal Scan: xi = {xi_start}..{xi_end - 1} deg ===")

    for xi in np.arange(xi_start, xi_end, xi_step):
        engine = BCAEngine(ion, targets, crystal, psi_deg, xi, u_therm, params)
        pgr = BoundaryParameterCalculator(engine)
        pgr.compute(verbose=False)
        sim = ChannelingSimulation(engine, pgr)
        results = sim.run_simulation(verbose=False, output_filename=_os.devnull)
        n_back = len([r for r in results if r.backscattered])
        result_map[xi] = n_back
        if verbose:
            bar = '#' * max(1, n_back // 2)
            print(f"  xi={xi:5.1f} deg: N_back={n_back:4d}  {bar}")

    return result_map


# ===========================================================================
#  MAIN — Example: Ne+ → β-cristobalite SiO₂
# ===========================================================================


# ===========================================================================
#  AUTO-CALIBRATION MODULE
# ===========================================================================

# Atomic data: Z -> (symbol, A_amu, u_therm_300K in Angstrom)
ATOM_DATA = {
    1: ('H', 1.008, 0.10), 2: ('He', 4.003, 0.0),
    5: ('B', 10.81, 0.06), 6: ('C', 12.011, 0.05),
    7: ('N', 14.007, 0.07), 8: ('O', 15.999, 0.085),
    10: ('Ne', 20.183, 0.0), 12: ('Mg', 24.305, 0.07),
    13: ('Al', 26.982, 0.065), 14: ('Si', 28.086, 0.065),
    15: ('P', 30.974, 0.070), 16: ('S', 32.06, 0.075),
    18: ('Ar', 39.948, 0.0),
    22: ('Ti', 47.867, 0.06),
    26: ('Fe', 55.845, 0.055),
    28: ('Ni', 58.6934, 0.060),
    29: ('Cu', 63.546, 0.065),
    30: ('Zn', 65.38, 0.070), 31: ('Ga', 69.723, 0.070),
    32: ('Ge', 72.63, 0.065), 33: ('As', 74.922, 0.070),
    34: ('Se', 78.96, 0.075), 36: ('Kr', 83.798, 0.0),
    38: ('Sr', 87.62, 0.08), 40: ('Zr', 91.224, 0.06),
    42: ('Mo', 95.95, 0.050),
    47: ('Ag', 107.868, 0.07), 48: ('Cd', 112.414, 0.075),
    49: ('In', 114.818, 0.075), 50: ('Sn', 118.71, 0.070),
    51: ('Sb', 121.76, 0.070), 52: ('Te', 127.60, 0.075),
    54: ('Xe', 131.293, 0.0), 56: ('Ba', 137.327, 0.08),
    72: ('Hf', 178.49, 0.05), 74: ('W', 183.84, 0.04),
    78: ('Pt', 195.08, 0.045), 79: ('Au', 196.97, 0.05),
    82: ('Pb', 207.2, 0.08),
}

# Material classification for neutralization
MATERIAL_CLASS = {
    'SiO2': 'oxide', 'SiO2_alpha': 'oxide',
    'Al2O3': 'oxide', 'MgO': 'oxide',
    'GaP': 'semi', 'GaAs': 'semi', 'InP': 'semi', 'InAs': 'semi',
    'CdTe': 'semi', 'ZnSe': 'semi', 'ZnS': 'semi', 'ZnTe': 'semi',
    'GaN': 'semi', 'AlN': 'semi', 'SiC_3C': 'semi',
    'HgTe': 'semi', 'CdS': 'semi', 'GaSb': 'semi', 'InSb': 'semi',
    'AlAs': 'semi', 'AlP': 'semi', 'InN': 'semi',
    # Diamond-cubic elementals: Si/Ge/C are semiconductors,
    # alpha-Sn is a (semi)metal.
    'Si': 'semi', 'Ge': 'semi', 'C': 'semi', 'Sn': 'metal',
    'Cu': 'metal', 'Au': 'metal', 'Ag': 'metal', 'Pt': 'metal',
    'Fe': 'metal', 'W': 'metal',
    'Al': 'metal', 'Ni': 'metal', 'Mo': 'metal',
}


def get_atom_data(Z: int) -> Tuple[str, float, float]:
    """Returns (symbol, mass_amu, u_therm_300K) for given Z."""
    if Z in ATOM_DATA:
        return ATOM_DATA[Z]
    return (f"Z{Z}", 2.0 * Z, 0.07)


def auto_ion(Z: int) -> 'Ion':
    """Create Ion automatically from atomic number."""
    sym, mass, _ = get_atom_data(Z)
    return Ion(Z=Z, A=mass, name=f"{sym}+")


def auto_targets_from_crystal(crystal: CrystalStructure) -> List['TargetAtom']:
    """
    Automatically create TargetAtom list from crystal structure.
    Reads atom types from sites and looks up Z, A from ATOM_DATA.
    """
    # Collect unique atom types
    type_map = {}  # atom_type -> (name, Z)
    for site in crystal.sites:
        if site.atom_type not in type_map:
            type_map[site.atom_type] = site.name

    # Symbol to Z mapping
    SYM_TO_Z = {v[0]: k for k, v in ATOM_DATA.items()}

    targets = []
    for atype in sorted(type_map.keys()):
        name = type_map[atype]
        Z = SYM_TO_Z.get(name, 0)
        if Z == 0:
            for n in [name[:2], name[:1]]:
                if n in SYM_TO_Z:
                    if n != name:
                        warnings.warn(
                            f"auto_targets_from_crystal: atom name "
                            f"'{name}' not in ATOM_DATA; falling back "
                            f"to '{n}' (Z={SYM_TO_Z[n]}). If this is "
                            f"the wrong element, add the correct one "
                            f"to ATOM_DATA.",
                            stacklevel=2)
                    Z = SYM_TO_Z[n]
                    break
        if Z == 0:
            raise ValueError(
                f"Cannot determine Z for atom '{name}'. Add an entry "
                f"to ATOM_DATA: Z: ('{name}', mass_amu, u_300_K).")
        sym, mass, _ = get_atom_data(Z)
        targets.append(TargetAtom(Z=Z, A=mass, name=name))

    return targets


def auto_u_therm(crystal: CrystalStructure, temperature_K: float = 300.0,
                 per_element: bool = True):
    """
    Estimate thermal vibration amplitudes.
    Uses high-temperature Debye scaling u(T) = u(300) * sqrt(T/300).
    """
    SYM_TO_Z = {v[0]: k for k, v in ATOM_DATA.items()}
    type_to_u = {}
    type_to_count = {}
    for site in crystal.sites:
        Z = SYM_TO_Z.get(site.name, SYM_TO_Z.get(site.name[:2],
                                                 SYM_TO_Z.get(site.name[:1], 14)))
        _, _, u300 = get_atom_data(Z)
        if u300 <= 0:
            continue
        u = u300 * np.sqrt(temperature_K / 300.0)
        atype = int(site.atom_type)
        prev = type_to_u.get(atype, 0.0)
        type_to_u[atype] = max(prev, u)
        type_to_count[atype] = type_to_count.get(atype, 0.0) + site.occupation

    if not type_to_u:
        return {} if per_element else 0.0

    if per_element:
        return type_to_u

    u_sq_sum = sum(u ** 2 * type_to_count[k] for k, u in type_to_u.items())
    n = sum(type_to_count.values())
    return float(np.sqrt(u_sq_sum / max(n, 1.0)))


def auto_electronic_stopping(ion_Z: int, targets: List['TargetAtom'],
                             crystal: CrystalStructure) -> 'ElectronicStoppingModel':
    """
    Auto-compute electronic stopping parameters from ion and target data.
    Uses Lindhard-Scharff formula with Bragg's rule for compounds.
    """
    z1 = float(ion_Z)
    a1 = get_atom_data(ion_Z)[1]

    # Weighted average over target atoms (Bragg's rule)
    se_sum = 0.0
    n_atoms = 0.0
    for t in targets:
        z2 = float(t.Z)
        a2 = t.A
        count = sum(s.occupation for s in crystal.sites if s.name == t.name)

        # Lindhard-Scharff coefficient
        zp = z1 ** (1.0 / 6.0)
        zs = (z1 ** (2.0 / 3.0) + z2 ** (2.0 / 3.0)) ** 1.5
        am = np.sqrt(a1 / (a1 + a2))

        # Z1-oscillation correction
        z_osc = 1.0
        if 2 <= z1 <= 36:
            z_osc = 1.0 + 0.13 * np.sin(PI * z1 / 10) * np.exp(-0.1 * z1 / 10)

        se_atom = 0.045 * z1 * z2 * zp * am * z_osc / zs
        se_sum += se_atom * count
        n_atoms += count

    se_base = se_sum / max(n_atoms, 1.0)

    # Channel parameters from geometry
    ch_width = crystal.auto_channel_spacing()
    # Wider channel → more reduction, narrower → less
    alpha_ch = min(0.5, 0.15 * ch_width)  # empirical scaling

    return ElectronicStoppingModel(
        se_base=se_base,
        alpha_channel=alpha_ch,
        sigma_channel=ch_width * 0.6,  # σ ≈ 60% of channel width
    )


def auto_neutralization(ion_Z: int, crystal_name: str,
                        enabled: bool = False) -> 'NeutralizationModel':
    """Auto-determine neutralization parameters from ion and material type."""
    ion_sym = get_atom_data(ion_Z)[0]
    mat_class = MATERIAL_CLASS.get(crystal_name, 'semi')

    key_specific = (ion_sym, crystal_name)
    if key_specific in NeutralizationModel.V0_TABLE:
        v0 = NeutralizationModel.V0_TABLE[key_specific]
    else:
        key_class = (ion_sym, mat_class)
        v0 = NeutralizationModel.V0_TABLE.get(key_class, 1.0e5)

    return NeutralizationModel(v0=v0, enabled=enabled)


def auto_setup(ion_Z: int, crystal_name: str, E0: float = 2000.0,
               psi_deg: float = 3.0, xi_deg: float = 0.0,
               temperature_K: float = 300.0,
               vacancy_fraction: float = 0.0,
               enable_neutralization: bool = False,
               enable_detector: bool = False,
               detector_angle: float = 136.0,
               detector_acceptance: float = 3.0,
               detector_azimuth: float = 0.0,
               detector_azimuth_accept: float = 180.0,
               n_trajectories: int = 1000) -> dict:
    """
    ONE-CALL SETUP: Creates everything needed for simulation
    from just ion Z, crystal name, and energy.

    Returns dict with: ion, targets, crystal, params, engine, pgr, sim

    Usage:
        # ring detector, default 3 deg acceptance:
        setup = auto_setup(ion_Z=10, crystal_name='SiO2', E0=2000,
                           enable_detector=True, detector_angle=136)
    """
    # Crystal
    crystal = get_crystal(crystal_name)

    # Ion
    ion = auto_ion(ion_Z)

    # Targets from crystal
    targets = auto_targets_from_crystal(crystal)

    # Thermal vibrations
    u_therm = auto_u_therm(crystal, temperature_K)

    # Electronic stopping (auto-calibrated)
    stopping = auto_electronic_stopping(ion_Z, targets, crystal)

    # Neutralization (auto-calibrated)
    neutralization = auto_neutralization(ion_Z, crystal_name,
                                         enabled=enable_neutralization)

    # Detector
    detector = Detector(
        theta_deg=detector_angle,
        acceptance_deg=detector_acceptance,
        azimuth_deg=detector_azimuth,
        azimuth_accept_deg=detector_azimuth_accept,
    )

    # Defects
    defects = DefectMap(vacancy_fraction=vacancy_fraction)

    # Grid size from trajectory count
    n_side = max(2, int(np.sqrt(n_trajectories)))

    params = SimulationParams(
        E0=E0, n_x=n_side, n_y=n_side,
        use_detector=enable_detector, detector=detector,
        neutralization=neutralization, stopping=stopping,
        defects=defects,
    )

    # Engine
    engine = BCAEngine(ion, targets, crystal, psi_deg, xi_deg, u_therm, params)

    # PGR
    pgr = BoundaryParameterCalculator(engine)
    pgr.compute(verbose=False)

    # Simulation
    sim = ChannelingSimulation(engine, pgr)

    return {
        'ion': ion, 'targets': targets, 'crystal': crystal,
        'params': params, 'engine': engine, 'pgr': pgr, 'sim': sim,
        'u_therm': u_therm, 'stopping': stopping,
    }


def main():
    print("=" * 65)
    print("  BCA_MODERN — Universal Crystal BCA Simulation")
    print("  NLH + SRIM + Se(v,rho) + neutralization + defects")
    print("=" * 65)
    print(f"\n  Built-in crystals: {', '.join(sorted(CRYSTAL_DB.keys()))}")

    # ===================================================================
    #  EXAMPLE 1: Full auto-setup — Ne+ → SiO₂, just 3 parameters!
    # ===================================================================
    print("\n" + "=" * 60)
    print("  EXAMPLE 1: Ne+ → SiO₂ (full auto-setup)")
    print("=" * 60)

    setup = auto_setup(
        ion_Z=10,  # Ne+
        crystal_name='SiO2',  # beta-cristobalite
        E0=2000.0,  # 2 keV
        psi_deg=3.0,  # glancing angle
        n_trajectories=200,  # quick test
    )

    print(f"  Ion: {setup['ion'].name}")
    print(f"  Crystal: {setup['crystal'].name}")
    print(f"  Targets: {', '.join(t.name + f'(Z={t.Z})' for t in setup['targets'])}")
    u = setup['u_therm']
    if isinstance(u, dict):
        u_str = ', '.join(f"{k}:{float(v):.4f}" for k, v in sorted(u.items()))
        print(f"  Thermal u (per type) = {{{u_str}}} A (auto)")
    else:
        print(f"  Thermal u = {float(u):.4f} A (auto)")
    print(f"  Se_base = {setup['stopping'].se_base:.4f} (auto)")
    print(f"  alpha_ch = {setup['stopping'].alpha_channel:.3f} (auto)")

    results = setup['sim'].run_simulation()
    back = [r for r in results if r.backscattered]
    print(f"  → Backscattered: {len(back)}")
    if back:
        print(f"  → E_mean = {np.mean([r.final_energy for r in back]):.0f} eV")

    # ===================================================================
    #  EXAMPLE 2: Ar+ → CdTe — different ion, different crystal
    # ===================================================================
    print("\n" + "=" * 60)
    print("  EXAMPLE 2: Ar+ → CdTe (auto-setup)")
    print("=" * 60)

    setup2 = auto_setup(
        ion_Z=18,  # Ar+
        crystal_name='CdTe',
        E0=3000.0,  # 3 keV
        psi_deg=2.0,
        n_trajectories=200,
    )

    print(f"  Ion: {setup2['ion'].name}")
    print(f"  Crystal: {setup2['crystal'].name}")
    print(f"  Targets: {', '.join(t.name + f'(Z={t.Z})' for t in setup2['targets'])}")
    print(f"  Se_base = {setup2['stopping'].se_base:.4f} (auto)")

    results2 = setup2['sim'].run_simulation()
    back2 = [r for r in results2 if r.backscattered]
    print(f"  → Backscattered: {len(back2)}")

    # ===================================================================
    #  EXAMPLE 3: Ne+ → In0.3Ga0.7As (ternary alloy!)
    # ===================================================================
    print("\n" + "=" * 60)
    print("  EXAMPLE 3: Ne+ → In0.3Ga0.7As (ternary alloy)")
    print("=" * 60)

    crystal3 = make_ternary_alloy('GaAs', 2, 'In', 0.3, which_sublattice=0)
    targets3 = [
        TargetAtom(Z=31, A=69.723, name="Ga"),
        TargetAtom(Z=33, A=74.922, name="As"),
        TargetAtom(Z=49, A=114.818, name="In"),
    ]
    ion3 = auto_ion(10)

    print(f"  Crystal: {crystal3.name}")
    print(f"  Sites: {len(crystal3.sites)} (with partial occupation)")

    stopping3 = auto_electronic_stopping(10, targets3, crystal3)
    params3 = SimulationParams(
        E0=2000.0, n_x=10, n_y=20,
        stopping=stopping3,
    )
    engine3 = BCAEngine(ion3, targets3, crystal3, 2.5, 0.0, 0.07, params3)
    pgr3 = BoundaryParameterCalculator(engine3)
    pgr3.compute(verbose=False)
    sim3 = ChannelingSimulation(engine3, pgr3)
    results3 = sim3.run_simulation()
    back3 = [r for r in results3 if r.backscattered]
    print(f"  → Backscattered: {len(back3)}")

    print("\n" + "=" * 65)
    print("  All examples_1 completed successfully.")
    print("  Usage: setup = auto_setup(ion_Z=10, crystal_name='GaP', E0=2000)")
    print("         results = setup['sim'].run_simulation()")
    print("=" * 65)


if __name__ == '__main__':
    main()
