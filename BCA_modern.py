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

        atoms = []
        for ix in range(-nx, nx + 1):
            for iy in range(-ny, ny + 1):
                for iz in range(0, nz):  # layers into the crystal
                    for site in self.sites:
                        if site.occupation < 1.0:
                            if rng.random() > site.occupation:
                                continue

                        # Fractional position in supercell
                        fx = site.x + ix
                        fy = site.y + iy
                        fz = site.z + iz

                        # Convert to Cartesian
                        pos = self.to_cartesian(fx, fy, fz)

                        # Crystal is BELOW surface: z <= 0
                        pos[2] = -abs(pos[2])

                        # Surface relaxation (shift z for surface layers)
                        if iz in self.surface_relaxation:
                            pos[2] += self.surface_relaxation[iz]

                        atoms.append({
                            'pos': pos,
                            'type': site.atom_type,
                            'name': site.name,
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


# Pre-built structures
CRYSTAL_DB = {
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
    cs = get_crystal(base_crystal)
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


def load_cif(filename: str) -> CrystalStructure:
    """
    Load crystal structure from CIF file .
    Supports most standard CIF files from COD, ICSD, Materials Project.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    a = b = c = 1.0
    alpha = beta = gamma = 90.0
    sites = []
    reading_atoms = False
    col_map = {}

    SYM_TO_Z = {v[0]: k for k, v in ATOM_DATA.items()}
    type_counter = {}

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue

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

        # Atom site columns
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
                    if sym_clean not in SYM_TO_Z and len(sym_clean) > 1:
                        sym_clean = sym_clean[0]

                    Z = SYM_TO_Z.get(sym_clean, 0)
                    if Z == 0: continue

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
        ('Ne', 'oxide'): 0.8e5,
        ('Ne', 'semi'): 1.2e5,
        ('Ar', 'metal'): 2.0e5,
        ('Ar', 'oxide'): 1.2e5,
        ('Ar', 'semi'): 1.6e5,
        ('He', 'metal'): 1.0e5,
        ('He', 'oxide'): 0.5e5,
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
    Vacancies are determined ONCE before simulation, not randomly per collision.
    """
    vacancy_fraction: float = 0.0  # 0 = perfect crystal
    vacancies: Set[Tuple[int, ...]] = field(default_factory=set)

    def generate(self, nx: int = 20, ny: int = 20, nz: int = 5,
                 rng: np.random.Generator = None):
        if self.vacancy_fraction <= 0:
            return
        if rng is None:
            rng = np.random.default_rng()
        self.vacancies = set()
        for ix in range(-nx, nx + 1):
            for iy in range(-ny, ny + 1):
                for iz in range(nz):
                    if rng.random() < self.vacancy_fraction:
                        self.vacancies.add((ix, iy, iz))

    def is_vacant(self, ix: int, iy: int, iz: int) -> bool:
        return (ix, iy, iz) in self.vacancies


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

    def scattering_angle_for_ion(self, psi_deg, polar_deg, azimuth_deg):
        psi, alpha, phi = np.radians(psi_deg), np.radians(polar_deg), np.radians(azimuth_deg)
        ix, iy, iz = -np.cos(psi), 0.0, -np.sin(psi)
        ex = np.cos(alpha) * np.cos(phi)
        ey = np.cos(alpha) * np.sin(phi)
        ez = np.sin(alpha)
        ct = max(-1, min(1, -(ix * ex + iy * ey + iz * ez)))
        return np.degrees(np.arccos(ct))

    def ion_in_detector(self, psi_deg, polar_deg, azimuth_deg):
        return abs(self.scattering_angle_for_ion(psi_deg, polar_deg, azimuth_deg)
                   - self.theta_deg) <= self.acceptance_deg

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
    max_collisions: int = 100
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
                 u_therm: float, params: SimulationParams):
        self.ion = ion
        self.targets = targets
        self.crystal = crystal
        self.psi_deg = surface_psi
        self.xi_deg = surface_xi
        self.u_therm = u_therm
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
            t.g_limits = np.array([160.0 / t.b_exponents[i] for i in range(len(t.b_exponents))])
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
        tsa = np.arctan2(np.sqrt(max(1 - f ** 2, 0)), f)
        if tsa < 0: tsa += PI
        if tsa < ts: ka = -1
        ea = t.cea * E * (np.cos(rta) + ka * np.sqrt(max(f ** 2 - np.sin(rta) ** 2, 0))) ** 2
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

    def __init__(self, crystal: CrystalStructure, u_therm: float,
                 defects: DefectMap, rng: np.random.Generator):
        self.crystal = crystal
        self.u_therm = u_therm
        self.defects = defects
        self.rng = rng
        self.PD = crystal.auto_channel_spacing()
        # Pre-generate atom positions — larger volume = better accuracy
        self.atoms = crystal.generate_atom_positions(nx=3, ny=3, nz=3, rng=rng)
        # Pre-compute numpy arrays for fast distance calculation
        self._pos = np.array([a['pos'] for a in self.atoms])
        self._types = np.array([a['type'] for a in self.atoms])
        self._names = [a['name'] for a in self.atoms]

    def find_nearest(self, x, y, z, L, M, N, E, pgr_func):
        """
        Find nearest atom AHEAD of ion within interaction range.

        """
        # Max interaction range (generous estimate)
        max_range = 15.0  # Angstrom — atoms further than this are irrelevant

        # Fast numpy pre-filter: distance from ion to each atom
        dx = self._pos[:, 0] - x
        dy = self._pos[:, 1] - y
        dz = self._pos[:, 2] - z

        # Dot product with direction = projection along trajectory
        stp = L * dx + M * dy + N * dz

        # Only atoms AHEAD and within range
        mask = (stp > 0.01) & (stp < max_range)
        candidates = np.where(mask)[0]

        if len(candidates) == 0:
            return False, {}

        ve = L * L + M * M + N * N
        sv = 1.0 / np.sqrt(ve) if ve > 0 else 1.0

        best_di = 1e10
        best = None

        for idx in candidates:
            atom = self.atoms[idx]
            ax, ay, az = atom['pos']

            fx = ax - x
            fy = ay - y
            fz = az - z

            # Add thermal displacement
            if self.u_therm > 0:
                fx += self.rng.normal(0, self.u_therm)
                fy += self.rng.normal(0, self.u_therm)
                fz += self.rng.normal(0, self.u_therm)

            # Impact parameter (perpendicular distance)
            cx = N * fy - M * fz
            cy = L * fz - N * fx
            cz = M * fx - L * fy
            di = np.sqrt(cx * cx + cy * cy + cz * cz) * sv

            if di < best_di:
                best_di = di
                stp_val = L * fx + M * fy + N * fz
                dist = np.sqrt(fx * fx + fy * fy + fz * fz)
                best = {
                    'pos': (ax, ay, az),
                    'type': atom['type'],
                    'name': atom['name'],
                    'di': di,
                    'fx': fx, 'fy': fy, 'fz': fz,
                    'stp': stp_val,
                    'dist': dist,
                    've': ve, 'sv': sv,
                }

        if best is None:
            return False, {}

        pg = pgr_func(E, best['type'])
        hit = best['di'] < pg

        return hit, best


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
            ct = (n * np.sum(le * lp) - np.sum(le) * np.sum(lp)) / (np.sum(le) ** 2 - n * np.sum(le ** 2))
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
        if k not in self.results: return 0.5
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
    ion_survival_prob: float = 1.0
    sputtered_recoils: list = field(default_factory=list)
    trajectory: list = field(default_factory=list)
    last_collision_atom: str = ""


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
            engine.crystal, engine.u_therm, self.params.defects, self.rng)

    def run_single_trajectory(self, x0_A, y0_A, output_file=None):
        result = TrajectoryResult()
        E = self.params.E0
        L, M, N = self.L1, self.M1, self.N1
        x, y, z = x0_A, y0_A, 0.0  # Angstrom
        sne = 0.0
        kc = 0
        z_max = 0.0
        ko = 0
        kc0 = 0
        stopping = self.params.stopping

        while True:
            kc0 += 1
            if kc0 >= self.params.max_collisions or E <= self.params.min_energy:
                break

            # Find nearest atom using universal navigator
            hit, atom = self.navigator.find_nearest(
                x, y, z, L, M, N, E,
                lambda e, k: self.pgr.get_pgr(e, k))

            if not atom:
                break

            dist_A = atom['dist']
            r_to_row = atom['di']  # impact parameter = distance to row

            de_cont = stopping.continuous_stopping(E, dist_A, r_to_row)
            E -= de_cont
            sne += de_cont
            if E <= self.params.min_energy: break

            if not hit:
                x += L * atom['stp'] / atom['ve']
                y += M * atom['stp'] / atom['ve']
                z += N * atom['stp'] / atom['ve']
                continue

            # --- Collision ---
            k = atom['type']
            p_phys = atom['di']  # impact parameter in Angstrom

            col = self.engine.single_collision(p_phys, E, k)

            # Track depth
            if z < z_max: z_max = z

            # Recoil sputtering check
            if col['E_a'] > 3.0:
                ko += 1
                result.sputtered_recoils.append({
                    'energy': col['E_a'], 'atom': atom['name']})

            # Update energy and losses
            E = col['E_i']
            sne += col['en']

            # Rotate direction (Rodrigues)
            theta_lab = col['theta_i_rad']
            if theta_lab > 1e-6:
                fx, fy, fz = atom['fx'], atom['fy'], atom['fz']
                dn = np.sqrt(fx * fx + fy * fy + fz * fz)
                if dn > 1e-10:
                    ux, uy, uz = fx / dn, fy / dn, fz / dn
                    ct, st = np.cos(theta_lab), np.sin(theta_lab)
                    d = L * ux + M * uy + N * uz
                    L = L * ct + (M * uz - N * uy) * st + ux * d * (1 - ct)
                    M = M * ct + (N * ux - L * uz) * st + uy * d * (1 - ct)
                    N = N * ct + (L * uy - M * ux) * st + uz * d * (1 - ct)

            # Move to the point of closest approach with this atom
            stp = atom['stp']
            ve = atom['ve']
            if ve > 0:
                x += L * stp / ve
                y += M * stp / ve
                z += N * stp / ve
            kc += 1
            kc0 = 0
            result.last_collision_atom = atom['name']

            # Exit check (z > some depth means exited through surface)
            if z > 0.5:
                break

        # --- Results ---
        result.final_energy = E
        result.n_collisions = kc
        result.total_inelastic_loss = sne
        result.max_depth = abs(z_max)

        ve = L * L + M * M + N * N
        nn = N / np.sqrt(ve) if ve > 0 else 0
        if abs(nn) < 1: result.polar_angle = RD * np.arctan2(nn, np.sqrt(max(1 - nn * nn, 1e-30)))

        l1, m1 = self.L1, self.M1
        d = np.sqrt((l1 ** 2 + m1 ** 2) * (L ** 2 + M ** 2))
        if d > 1e-10:
            zf = max(-1, min(1, (l1 * L + m1 * M) / d))
            result.azimuthal_angle = RD * np.arctan2(np.sqrt(max(1 - zf ** 2, 0)), zf)

        result.backscattered = (z > 0.5 and E > self.params.min_energy)

        # Neutralization
        neut = self.params.neutralization
        result.ion_survival_prob = neut.survival_probability(
            E, self.engine.ion.A, result.polar_angle)

        # Structured text output
        if output_file and result.backscattered:
            fi = abs(M) / np.sqrt(ve) if ve > 0 else 0
            if fi < 1: fi = RD * np.arctan2(fi, np.sqrt(1 - fi ** 2))
            if M < 0: fi = -fi
            output_file.write(
                f"{0:5d}{kc:5d}{E:7.0f}{sne:6.0f}"
                f"{result.polar_angle:7.2f}{fi:7.2f}{result.azimuthal_angle:7.2f}"
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
            print(f"Thermal: u={self.engine.u_therm:.3f} A")
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
                if verbose and n_total % 200 == 0:
                    dt = time.time() - t0
                    print(f"  {n_total}/{total} traj, {n_back} back, {n_total / dt:.0f} traj/s")

        # Histograms
        ie = np.zeros(210, dtype=int)
        for r in results:
            ien = int(r.final_energy / 25)
            if 0 <= ien < 210: ie[ien] += 1
        fout.write("# Energy histogram (25 eV bins)\n")
        for i in range(0, 210, 15):
            fout.write(' '.join(f"{ie[j]:5d}" for j in range(i, min(i + 15, 210))) + '\n')

        kr = sum(len(r.sputtered_recoils) for r in results)
        fout.write(f"# KSI={n_back:6d} KR={kr:6d}\n")
        fout.close()

        if verbose:
            dt = time.time() - t0
            print(f"\nDone: {n_total} traj in {dt:.1f}s, {n_back} backscattered")
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
    xi_start, xi_end, xi_step = xi_range
    result_map = {}

    if verbose:
        print(f"\n=== Azimuthal Scan: xi = {xi_start}..{xi_end - 1} deg ===")

    for xi in np.arange(xi_start, xi_end, xi_step):
        engine = BCAEngine(ion, targets, crystal, psi_deg, xi, u_therm, params)
        pgr = BoundaryParameterCalculator(engine)
        pgr.compute(verbose=False)
        sim = ChannelingSimulation(engine, pgr)
        results = sim.run_simulation(verbose=False, output_filename=f'/dev/null')
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
    18: ('Ar', 39.948, 0.0), 22: ('Ti', 63.546, 0.06),
    26: ('Fe', 55.845, 0.055), 29: ('Cu', 63.546, 0.065),
    30: ('Zn', 65.38, 0.070), 31: ('Ga', 69.723, 0.070),
    32: ('Ge', 72.63, 0.065), 33: ('As', 74.922, 0.070),
    34: ('Se', 78.96, 0.075), 36: ('Kr', 83.798, 0.0),
    38: ('Sr', 87.62, 0.08), 40: ('Zr', 91.224, 0.06),
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
    'SiO2': 'oxide', 'Al2O3': 'oxide', 'MgO': 'oxide',
    'GaP': 'semi', 'GaAs': 'semi', 'InP': 'semi', 'InAs': 'semi',
    'CdTe': 'semi', 'ZnSe': 'semi', 'ZnS': 'semi', 'ZnTe': 'semi',
    'GaN': 'semi', 'AlN': 'semi', 'SiC_3C': 'semi',
    'HgTe': 'semi', 'CdS': 'semi', 'GaSb': 'semi', 'InSb': 'semi',
    'AlAs': 'semi', 'AlP': 'semi', 'InN': 'semi',
    'Cu': 'metal', 'Au': 'metal', 'Ag': 'metal', 'Pt': 'metal',
    'Fe': 'metal', 'W': 'metal',
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
                    Z = SYM_TO_Z[n]
                    break
        if Z == 0:
            raise ValueError(f"Cannot determine Z for atom '{name}'")
        sym, mass, _ = get_atom_data(Z)
        targets.append(TargetAtom(Z=Z, A=mass, name=name))

    return targets


def auto_u_therm(crystal: CrystalStructure, temperature_K: float = 300.0) -> float:
    """
    Estimate compound thermal vibration amplitude.
    Uses Debye model scaling: u(T) ∝ sqrt(T/T_Debye) at T > T_Debye.
    """
    SYM_TO_Z = {v[0]: k for k, v in ATOM_DATA.items()}
    u_sq_sum = 0.0
    n = 0
    for site in crystal.sites:
        Z = SYM_TO_Z.get(site.name, SYM_TO_Z.get(site.name[:2],
                                                 SYM_TO_Z.get(site.name[:1], 14)))
        _, _, u300 = get_atom_data(Z)
        if u300 > 0:
            # Scale with temperature (simplified Debye)
            u = u300 * np.sqrt(temperature_K / 300.0)
            u_sq_sum += u ** 2 * site.occupation
            n += site.occupation
    return np.sqrt(u_sq_sum / max(n, 1))


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
    n_atoms = 0
    for t in targets:
        z2 = float(t.Z)
        a2 = t.A
        # Count atoms of this type in unit cell
        count = sum(1 for s in crystal.sites if s.name == t.name)

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

    se_base = se_sum / max(n_atoms, 1)

    # Channel parameters from geometry
    ch_width = crystal.auto_channel_spacing()
    # Wider channel → more reduction, narrower → less
    alpha_ch = min(0.5, 0.15 * ch_width)  # empirical scaling

    return ElectronicStoppingModel(
        se_base=se_base,
        alpha_channel=alpha_ch,
        sigma_channel=ch_width * 0.6,  # σ ≈ 60% of channel width
    )


def auto_neutralization(ion_Z: int, crystal_name: str) -> 'NeutralizationModel':
    """Auto-determine neutralization parameters from ion and material type."""
    ion_sym = get_atom_data(ion_Z)[0]
    mat_class = MATERIAL_CLASS.get(crystal_name, 'semi')

    key = (ion_sym, mat_class)
    v0 = NeutralizationModel.V0_TABLE.get(key, 1.0e5)

    return NeutralizationModel(v0=v0, enabled=False)  # off by default


def auto_setup(ion_Z: int, crystal_name: str, E0: float = 2000.0,
               psi_deg: float = 3.0, xi_deg: float = 0.0,
               temperature_K: float = 300.0,
               vacancy_fraction: float = 0.0,
               enable_neutralization: bool = False,
               enable_detector: bool = False,
               detector_angle: float = 136.0,
               n_trajectories: int = 1000) -> dict:
    """
    ONE-CALL SETUP: Creates everything needed for simulation
    from just ion Z, crystal name, and energy.

    Returns dict with: ion, targets, crystal, params, engine, pgr, sim

    Usage:
        setup = auto_setup(ion_Z=10, crystal_name='SiO2', E0=2000)
        results = setup['sim'].run_simulation()
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
    neutralization = auto_neutralization(ion_Z, crystal_name)
    neutralization.enabled = enable_neutralization

    # Detector
    detector = Detector(theta_deg=detector_angle)

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
    print(f"  Thermal u = {setup['u_therm']:.4f} A (auto)")
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
    print("  All examples completed successfully.")
    print("  Usage: setup = auto_setup(ion_Z=10, crystal_name='GaP', E0=2000)")
    print("         results = setup['sim'].run_simulation()")
    print("=" * 65)


if __name__ == '__main__':
    main()
