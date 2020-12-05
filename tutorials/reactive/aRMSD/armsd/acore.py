"""
aRMSD core functions
(c) 2017 by Arne Wagner
"""

# Authors: Arne Wagner
# License: MIT

from __future__ import absolute_import, division, print_function

import os
from io import open

#import aplot as ap
from builtins import range, input
from functools import reduce

try:

    import numpy as np

    has_np, np_version = True, np.__version__

    np.set_printoptions(suppress=True, precision=8)  # Print only 8 digits in numpy arrays
    np.seterr(all='ignore')  # Ignore all error warnings

except ImportError:

    has_np, np_version = False, 'Module not available'

try:

    import pybel
    from openbabel import OBReleaseVersion

    has_pybel, pyb_version = True, OBReleaseVersion()

except ImportError:

    has_pybel, pyb_version = False, 'Module not available'

try:

    import uncertainties.unumpy as unp
    from uncertainties import ufloat, ufloat_fromstr
    from uncertainties import __version__ as uc_version

    has_uc, uc_version = True, uc_version

except ImportError:

    has_uc, uc_version = False, 'Module not available'

###############################################################################
# CONSTANTS AND CONVERSION FACTORS
###############################################################################

b2a = 5.2917721092E-01  # Conversion factor: Bohr -> Angstrom
NA = 6.02214129E+23  # Avogadro number
x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float)  # x axis
y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float)  # y axis
z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float)  # z axis

###############################################################################
# ATOMIC PROPERTIES
###############################################################################

# Atomic symbols ('H' to 'Rn') as dictionary
pse_symbol = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
              'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
              'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
              'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
              'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46,
              'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
              'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
              'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73,
              'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
              'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86}

# Numpy array of element symbols ('H' to 'Rn')
pse_sym_chg = np.array(['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
                        'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
                        'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
                        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
                        'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',
                        'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'])

# Numpy array of element masses ('H' to 'Rn')
pse_mass = np.array([1.00794, 4.002602, 6.941, 9.012182, 10.811, 12.0107, 14.0067, 15.9994,
                     18.9984032, 20.1797, 22.98976928, 24.3050, 26.9815386, 28.0855, 30.973762,
                     32.065, 35.453, 39.948, 39.0983, 40.078, 44.955912, 47.867, 50.9415,
                     51.9961, 54.938045, 55.845, 58.933195, 58.6934, 63.546, 65.409, 69.723, 72.64,
                     74.92160, 78.96, 79.904, 83.798, 85.4678, 87.62, 88.90585, 91.224, 92.90638,
                     95.94, 98.0, 101.07, 102.90550, 106.42, 107.8682, 112.411, 114.818, 118.710,
                     121.760, 127.60, 126.90447, 131.293, 132.9054519, 137.327, 138.90547, 140.116,
                     140.90765, 144.242, 145.0, 150.36, 151.964, 157.25, 158.92535, 162.500,
                     164.93032, 167.259, 168.93421, 173.04, 174.967, 178.49, 180.94788, 183.84,
                     186.207, 190.23, 192.217, 195.084, 196.966569, 200.59, 204.3833, 207.2,
                     208.98040, 209.0, 210.0, 222.0], dtype=np.float)

# Numpy array of mass density in g cm**(-3) ('H' to 'Rn')
pse_mass_dens = np.array([0.00008988, 0.0001785, 0.534, 1.85, 2.34, 2.267, 0.0012506, 0.001429,
                          0.001696, 0.0008999, 0.971, 1.738, 2.698, 2.3296, 1.82, 2.067, 0.003214,
                          0.0017837, 0.862, 1.54, 2.989, 4.54, 6.11, 7.15, 7.44, 7.874, 8.86,
                          8.912, 8.96, 7.134, 5.907, 5.323, 5.776, 4.809, 3.122, 0.003733, 1.532,
                          2.64, 4.469, 6.506, 8.57, 10.22, 11.5, 12.37, 12.41, 12.02, 10.501,
                          8.69, 7.31, 7.287, 6.685, 6.232, 4.93, 0.005887, 1.873, 3.594, 6.145,
                          6.77, 6.773, 7.007, 7.26, 7.52, 5.243, 7.895, 8.229, 8.55, 8.795, 9.066,
                          9.321, 6.965, 9.84, 13.31, 16.654, 19.25, 21.02, 22.61, 22.56, 21.46,
                          19.282, 13.5336, 11.85, 11.342, 9.807, 9.32, 7.0, 0.00973], dtype=np.float)

# Numpy array of covalent radii ('H' to 'Rn')
pse_cov_radii = np.array([0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41,
                          1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39,
                          1.50, 1.42, 1.38, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16,
                          2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44,
                          1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01,
                          1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75,
                          1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40,
                          1.50, 1.50], dtype=np.float)

# Numpy array of atomic radii ('H' to 'Rn')
pse_atm_radii = np.array([0.79, 0.49, 2.05, 1.40, 1.17, 0.91, 0.75, 0.65, 0.57, 0.51, 2.23, 1.72,
                          1.82, 1.46, 1.23, 1.09, 0.97, 0.88, 2.77, 2.23, 2.09, 2.00, 1.92, 1.85,
                          1.79, 1.72, 1.67, 1.62, 1.57, 1.53, 1.81, 1.52, 1.33, 1.22, 1.12, 1.03,
                          2.98, 2.45, 2.27, 2.16, 2.08, 2.01, 1.95, 1.89, 1.83, 1.79, 1.75, 1.71,
                          2.00, 1.72, 1.53, 1.42, 1.32, 1.24, 3.34, 2.78, 2.74, 2.70, 2.67, 2.64,
                          2.62, 2.59, 2.56, 2.54, 2.51, 2.49, 2.47, 2.45, 2.42, 2.40, 2.25, 2.16,
                          2.09, 2.02, 1.97, 1.92, 1.87, 1.83, 1.79, 1.76, 2.08, 1.81, 1.63, 1.53,
                          1.43, 1.34], dtype=np.float)

# Numpy array of Pauling electronegativities ('H' to 'Rn')
pse_en_array = np.array([2.20, 0.00, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0.00, 0.93, 1.31,
                         1.61, 1.90, 2.19, 2.58, 3.16, 0.00, 0.82, 1.00, 1.36, 1.54, 1.63, 1.66,
                         1.55, 1.83, 1.88, 1.91, 1.90, 1.65, 1.81, 2.01, 2.18, 2.55, 2.96, 3.00,
                         0.82, 0.95, 1.22, 1.33, 1.60, 2.16, 1.90, 2.20, 2.28, 2.20, 1.93, 1.69,
                         1.78, 1.96, 2.05, 2.10, 2.66, 2.60, 0.79, 0.89, 1.10, 1.12, 1.13, 1.14,
                         1.13, 1.17, 1.20, 1.20, 1.10, 1.22, 1.23, 1.24, 1.25, 1.10, 1.27, 1.30,
                         1.15, 2.36, 1.90, 2.20, 2.20, 2.28, 2.54, 2.00, 1.62, 1.33, 2.02, 2.00,
                         2.20, 2.20], dtype=np.float)

# Numpy array of element groups ('H' to 'Rn', Lanthanides treated as 'group 19')
pse_groups = np.array([1, 18,
                       1, 2, 13, 14, 15, 16, 17, 18,
                       1, 2, 13, 14, 15, 16, 17, 18,
                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                       1, 2, 3, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                       4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=np.int)

# Numpy array of 'Group symbols' (G + group number or Ln for Lanthanides)
pse_group_sym = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11',
                          'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18', 'Ln'])

# X-ray atomic scattering factors f' and f'' for lambda = 0.71073 A (MoK-alpha) E = 17445 eV, ('H' to 'Rn')
pse_mo_xsf_1 = np.array([0.999974000, 1.999920000, 2.999967910, 4.000413720, 5.001413250,
                         6.003466500, 7.006533470, 8.011579960, 9.017938090, 10.02661155,
                         11.03664874, 12.05138592, 13.06440217, 14.08207401, 15.10433466,
                         16.12422997, 17.14872528, 18.18242058, 19.21173683, 20.23319495,
                         21.27311043, 22.29126288, 23.31843214, 24.32643683, 25.34778340,
                         26.35851372, 27.35654874, 28.35663032, 29.34125848, 30.31207508,
                         31.25330576, 32.18165052, 33.07003967, 33.91895229, 34.71661653,
                         35.45149520, 36.05507312, 36.41433221, 36.00220711, 37.01404050,
                         38.91069552, 40.31740802, 41.57105965, 42.76023830, 43.88382894,
                         45.01149680, 46.12251809, 47.19463686, 48.29400220, 49.37886502,
                         50.39356754, 51.58400004, 52.58809787, 53.69413974, 54.74660725,
                         55.84292819, 56.89504970, 57.90677259, 58.90921699, 59.96203107,
                         60.90569172, 62.07508955, 62.99151770, 64.10426047, 64.92260903,
                         65.90665812, 66.98193537, 67.87271262, 68.84219456, 69.78989961,
                         70.68863786, 71.63652883, 72.41532520, 73.33183231, 74.13242229,
                         74.97072119, 75.69623669, 76.36229730, 77.15513157, 77.72152069,
                         78.33406107, 78.79183573, 79.01674319, 78.88371081, 76.32290487,
                         77.68380199], dtype=np.float)

pse_mo_xsf_2 = np.array([0.000000170, 0.000006470, 0.000054730, 0.000260980, 0.000674200,
                         0.001610370, 0.003230590, 0.006114140, 0.010329240, 0.016483590,
                         0.024669150, 0.037884490, 0.051707560, 0.070598030, 0.096708650,
                         0.124426070, 0.160286520, 0.209689230, 0.265602230, 0.316323370,
                         0.384148380, 0.453062580, 0.543587620, 0.616325310, 0.732599200,
                         0.849293570, 0.965723840, 1.124864300, 1.289417660, 1.466609840,
                         1.651749230, 1.867983420, 2.093024840, 2.308258380, 2.559813480,
                         2.719243450, 2.978327240, 3.325059460, 3.604080260, 0.582739560,
                         0.638275270, 0.703359270, 0.766832170, 0.835164610, 0.939584970,
                         1.014690590, 1.098206290, 1.228485750, 1.292559320, 1.422247600,
                         1.487531210, 1.763866350, 1.832569490, 2.047205270, 2.189501410,
                         2.403845540, 2.627398410, 2.740622700, 2.911668370, 3.140953900,
                         3.308165570, 3.727824930, 3.776412800, 4.272521360, 4.179548270,
                         4.428787770, 5.009421830, 5.102077290, 5.508233260, 5.799231710,
                         6.238131620, 6.433290800, 6.581230870, 6.772497840, 7.128189080,
                         7.593924980, 7.980841300, 8.578650880, 8.870082880, 9.325896860,
                         9.718822150, 10.15356198, 10.58964941, 11.15808877, 10.44615983,
                         10.50834220], dtype=np.float)

# X-ray atomic scattering factors f' and f'' for lambda = 1.54184 A (CuK-alpha) E = 8041.3 eV, ('H' to 'Rn')
pse_cu_xsf_1 = np.array([0.999984000, 2.000140580, 3.001162920, 4.004108190, 5.009628120,
                         6.019203910, 7.032564960, 8.052393610, 9.075365180, 10.10475201,
                         11.13799294, 12.18013971, 13.21266309, 14.25620677, 15.30359817,
                         16.33537479, 17.36842801, 18.40664032, 19.41580585, 20.38152521,
                         21.36238804, 22.24188305, 23.07629525, 23.83936102, 24.45628026,
                         24.84754919, 24.58955914, 25.02353889, 27.03123582, 28.44372887,
                         29.71521803, 30.90320938, 32.05589367, 33.20790658, 34.32598380,
                         35.46618441, 36.56441425, 37.64252656, 38.72235359, 39.82099219,
                         40.90704531, 41.97604531, 43.02108806, 44.07383638, 45.12046066,
                         46.17446249, 47.17562862, 48.15708368, 49.13683289, 50.08660792,
                         50.98334482, 51.93946443, 52.80793179, 53.63925139, 54.32526637,
                         55.02772471, 55.67425876, 56.16955028, 56.50813993, 56.79982762,
                         57.06379301, 56.43946290, 54.13378952, 54.55835242, 55.83091266,
                         56.15973605, 51.44583598, 58.56828046, 60.87361953, 62.70226239,
                         64.24474058, 65.87137578, 67.51609965, 68.98303895, 70.26366688,
                         71.36574776, 72.60374495, 73.66787199, 74.98908149, 76.10774824,
                         77.19833959, 78.27461986, 79.31748661, 80.24788380, 81.35789793,
                         82.63666651], dtype=np.float)

pse_cu_xsf_2 = np.array([0.000001110, 0.000040270, 0.000334410, 0.001540060, 0.004083640,
                         0.009618100, 0.018392640, 0.033766580, 0.055062310, 0.085184070,
                         0.122452800, 0.181472090, 0.241951210, 0.325446940, 0.437029770,
                         0.551361310, 0.699430450, 0.902784920, 1.105749640, 1.314527650,
                         1.608106750, 1.870692210, 2.196482780, 2.442748030, 2.836298090,
                         3.213092680, 3.566955400, 0.524958450, 0.608610550, 0.704621040,
                         0.803236840, 0.920991410, 1.047709220, 1.182961390, 1.330117010,
                         1.429386500, 1.601947650, 1.847944290, 2.111751880, 2.282858490,
                         2.509146240, 2.748893570, 2.995141680, 3.267983180, 3.644058910,
                         3.990623450, 4.271227380, 4.735048380, 4.964014680, 5.527547690,
                         5.672734660, 6.772299290, 6.944321090, 7.729035380, 8.251531450,
                         9.038631260, 9.782009450, 10.09476312, 10.57806653, 11.36259552,
                         11.95414120, 13.17573718, 11.60607860, 13.26636264, 9.215413320,
                         9.940063910, 4.013604460, 4.105500530, 4.441847770, 4.723525240,
                         5.033621450, 5.285741090, 5.483156570, 5.777077990, 6.090901370,
                         6.437484690, 6.816182270, 7.246360550, 7.728822140, 8.074129010,
                         8.457227470, 8.946724710, 9.311847770, 9.714787250, 10.41474655,
                         11.02345398], dtype=np.float)

# X-ray atomic scattering factors f' and f'' for lambda = 1.79026 A (CoK-alpha) E = 6925.5 eV, ('H' to 'Rn')
pse_co_xsf_1 = np.array([0.999988000, 2.000237400, 3.001662210, 4.005606620, 5.012858040,
                         6.025176470, 7.042235610, 8.067063970, 9.095309730, 10.13094730,
                         11.17120637, 12.22026543, 13.25591347, 14.30281186, 15.35037646,
                         16.37600637, 17.39696617, 18.41394486, 19.37958334, 20.29879659,
                         21.18894415, 21.94487472, 22.58167341, 23.04372987, 22.87886845,
                         22.66387439, 25.01058168, 26.44547048, 27.71719061, 28.91614871,
                         30.08380216, 31.22387691, 32.34262569, 33.47444457, 34.57937447,
                         35.70770437, 36.79486026, 37.85765707, 38.87539949, 40.01185787,
                         41.08333650, 42.13368693, 43.14242376, 44.15544663, 45.16942672,
                         46.15897379, 47.10944611, 48.01758608, 48.92460006, 49.72166897,
                         50.55950109, 51.23963816, 52.00410951, 52.54680321, 52.90703540,
                         53.17545536, 53.28022296, 53.14810600, 52.01691648, 51.54124004,
                         50.12758022, 50.94170183, 47.85664264, 54.04974514, 57.14110795,
                         58.90655937, 60.21445210, 61.94382988, 63.26739634, 64.53887309,
                         65.74629248, 67.10083785, 68.57035502, 69.88961011, 71.06030113,
                         72.07331437, 73.24950482, 74.23304405, 75.43700437, 76.53427055,
                         77.53945855, 78.51917254, 79.49385662, 80.34021865, 81.36163338,
                         82.53725521], dtype=np.float)

pse_co_xsf_2 = np.array([0.000001590, 0.000057390, 0.000472450, 0.002156960, 0.005704070,
                         0.013370990, 0.025394760, 0.046372520, 0.075081730, 0.115465560,
                         0.165215990, 0.243246540, 0.322656940, 0.432755600, 0.578817760,
                         0.727068960, 0.919171690, 1.182276980, 1.446066460, 1.707925100,
                         2.078409980, 2.407547250, 2.804393640, 3.098045620, 3.565686460,
                         0.504586240, 0.579531790, 0.690858660, 0.798021000, 0.921196860,
                         1.049308480, 1.199613280, 1.362810600, 1.538255890, 1.727648180,
                         1.860412200, 2.085076400, 2.393781420, 2.709277120, 2.956202750,
                         3.255023270, 3.559469880, 3.872028980, 4.223541630, 4.703944800,
                         5.153846350, 5.506173200, 6.095507170, 6.379672870, 7.081246920,
                         7.271338450, 8.649318700, 8.878471680, 9.855198990, 10.45268808,
                         11.39648364, 12.35806327, 12.60425396, 13.14180837, 12.47587493,
                         9.535130930, 10.83235634, 3.840533700, 4.373688150, 4.263316700,
                         4.520512620, 5.075529580, 5.203856660, 5.634324340, 5.972522810,
                         6.361743850, 6.689732240, 6.924457720, 7.307287020, 7.697926020,
                         8.130491400, 8.622037410, 9.148644060, 9.849790360, 10.21048882,
                         10.68735988, 11.26657187, 11.74884219, 12.23279512, 13.14385856,
                         13.87197692], dtype=np.float)

# X-ray atomic scattering factors f' and f'' for lambda = 1.93736 A (FeK-alpha) E = 6399.6 eV, ('H' to 'Rn')
pse_fe_xsf_1 = np.array([0.999991000, 2.000302920, 3.001980440, 4.006561320, 5.014911400,
                         6.028926520, 7.048259600, 8.076086470, 9.107403350, 10.14666359,
                         11.19089719, 12.24343078, 13.28005998, 14.32750048, 15.37305558,
                         16.39218839, 17.40154160, 18.39904002, 19.33591284, 20.20750888,
                         21.02720057, 21.66052000, 22.08939006, 22.07229631, 21.41334890,
                         23.95283237, 25.45377177, 26.72096104, 27.92559147, 29.09003149,
                         30.23500950, 31.36615910, 32.47559191, 33.60192911, 34.70249551,
                         35.82587432, 36.90580712, 37.95830633, 38.94409156, 40.09158916,
                         41.14893676, 42.18064894, 43.16407080, 44.14357976, 45.12833245,
                         46.06287810, 46.97221653, 47.81787819, 48.66570264, 49.33853034,
                         50.11912120, 50.56364380, 51.21563351, 51.48045497, 51.52328848,
                         51.27451266, 50.33574116, 49.82735444, 47.13713448, 48.97846544,
                         47.02848135, 52.09395778, 54.93268259, 56.44409467, 58.56105409,
                         59.98505462, 61.12640668, 62.65974318, 63.88758443, 65.06860888,
                         66.20742172, 67.49045937, 68.92762858, 70.19752940, 71.32945224,
                         72.30328707, 73.45008909, 74.39449472, 75.49432762, 76.61171759,
                         77.56216763, 78.48938487, 79.40761372, 80.20186816, 81.14337423,
                         82.25710308], dtype=np.float)

pse_fe_xsf_2 = np.array([0.000001930, 0.000069250, 0.000567090, 0.002577170, 0.006795290,
                         0.015880530, 0.030063670, 0.054743770, 0.088296320, 0.135355940,
                         0.193291000, 0.283617340, 0.375182600, 0.502358130, 0.670463910,
                         0.840142650, 1.060013050, 1.360694060, 1.656056800, 1.957016570,
                         2.368837120, 2.739281400, 3.174136400, 3.491377910, 0.498669510,
                         0.578874320, 0.664160670, 0.795723310, 0.918392770, 1.059211570,
                         1.205821240, 1.377791140, 1.564444930, 1.765932230, 1.982338360,
                         2.137724990, 2.395651960, 2.742660930, 3.089975190, 3.386195230,
                         3.731017550, 4.080284190, 4.429182450, 4.828666600, 5.375567690,
                         5.885064280, 6.283189210, 6.952781190, 7.268719680, 8.042730190,
                         8.270730360, 9.806945730, 10.08204389, 11.17005981, 11.78920220,
                         12.80887775, 13.91771023, 12.57058839, 9.564595880, 10.44721170,
                         3.790331500, 4.271077580, 4.346273900, 4.956327510, 4.818293300,
                         5.104998800, 5.739364930, 5.885831080, 6.373446720, 6.747775610,
                         7.189058170, 7.549600870, 7.827808230, 8.263269350, 8.703194640,
                         9.187937040, 9.755009580, 10.33858909, 11.18024899, 11.54972058,
                         12.08293694, 12.71449490, 13.26755805, 13.80051161, 14.84237071,
                         15.64410106], dtype=np.float)

# X-ray atomic scattering factors f' and f'' for lambda = 2.29100 A (CrK-alpha) E = 5411.8 eV, ('H' to 'Rn')
pse_cr_xsf_1 = np.array([0.999998740, 2.000474820, 3.002846690, 4.009060080, 5.020249760,
                         6.038503190, 7.063517770, 8.098518680, 9.136944770, 10.18418977,
                         11.23735664, 12.29615302, 13.33195664, 14.37565190, 15.40843013,
                         16.40173674, 17.36713914, 18.29097768, 19.12907464, 19.81170952,
                         20.31354041, 20.29424743, 16.66334344, 21.90451294, 23.40043026,
                         24.71510672, 25.95327116, 27.10485127, 28.25799885, 29.39188740,
                         30.51135367, 31.63707957, 32.73211631, 33.84842715, 34.93903799,
                         36.04743016, 37.10183678, 38.12262111, 39.02034458, 40.17580000,
                         41.17740962, 42.14180717, 43.04248308, 43.90584229, 44.77880229,
                         45.52257092, 46.28280341, 46.88892563, 47.51700026, 47.75087966,
                         48.21643015, 47.68331055, 47.44715708, 44.47745435, 43.92969752,
                         44.31519254, 43.28599709, 48.56526842, 50.84122190, 52.61635357,
                         54.27948657, 55.61057866, 57.30083401, 58.35260493, 59.98908031,
                         61.21646704, 62.24535076, 63.57215856, 64.69877542, 65.76248805,
                         66.79365610, 67.97996572, 69.36961631, 70.54473771, 71.59201206,
                         72.47090601, 73.52436623, 74.37198970, 75.08838966, 76.32398963,
                         77.11084079, 77.90570402, 78.62645396, 79.26500513, 79.90603697,
                         80.83428331], dtype=np.float)

pse_cr_xsf_2 = np.array([0.000002880, 0.000103440, 0.000836050, 0.003747460, 0.009823300,
                         0.022780250, 0.042858600, 0.077570470, 0.124079820, 0.188876710,
                         0.268868590, 0.391729870, 0.515126620, 0.687143850, 0.912450260,
                         1.137144710, 1.427951230, 1.824052030, 2.208328380, 2.596911410,
                         3.106189360, 3.558267370, 0.486743630, 0.553722590, 0.660600070,
                         0.775296030, 0.889362390, 1.067465900, 1.231989170, 1.419562550,
                         1.613576300, 1.844440350, 2.092767050, 2.363166520, 2.650429010,
                         2.866668300, 3.210843740, 3.654123820, 4.080577630, 4.505767370,
                         4.969359860, 5.411888300, 5.869794300, 6.383643660, 7.105060080,
                         7.742013420, 8.260874960, 9.141758680, 9.527926100, 10.42694510,
                         10.79355658, 12.66697833, 13.09967174, 12.52466560, 13.36592058,
                         10.69858677, 3.812400430, 4.071627090, 4.319374980, 4.670201380,
                         4.912681510, 5.537179440, 5.626159680, 6.422772070, 6.228791460,
                         6.600222500, 7.435074710, 7.615777970, 8.243499810, 8.712479530,
                         9.290948060, 9.702860840, 10.13402533, 10.69545714, 11.26576924,
                         11.87799819, 12.64273257, 13.36849525, 14.44525675, 14.95491521,
                         15.61635138, 16.38822913, 17.09243578, 17.75485388, 19.09794190,
                         20.11074014], dtype=np.float)

# Numpy array of element colors ('H' to 'Rn')
pse_colors = np.array(['#ffffff', '#ffc8c8', '#a52a2a', '#ff1493', '#00ff00', '#c8c8c8', '#8f8fff',
                       '#f00000', '#c8a518', '#ff1493', '#0000ff', '#2a802a', '#808090', '#c8a518',
                       '#ffa500', '#ffc832', '#00ff00', '#ff1493', '#ff1493', '#808090', '#ff1493',
                       '#808090', '#ff1493', '#808090', '#808090', '#ffa500', '#ff1493', '#a52a2a',
                       '#a52a2a', '#a52a2a', '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#a52a2a',
                       '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#ff1493',
                       '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#808090', '#ff1493', '#ff1493',
                       '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#ff1493',
                       '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#ff1493',
                       '#ff1493', '#ff1493', '#ff1493', '#8f8fff', '#ff1493', '#ff1493', '#ffa500',
                       '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#ff1493',
                       '#ff1493', '#c8a518', '#ff1493', '#ff1493', '#ff1493', '#ff1493', '#ff1493',
                       '#ff1493', '#ff1493'])


def geo_distance(xyz1, xyz2):
    """ Global function for distance calculation - compatible with uncertainties
        coordinates are assumed to be uarrays """

    return np.sum((xyz1 - xyz2) ** 2) ** 0.5


def geo_angle(xyz1, xyz2, xyz3):
    """ Global function for angle calculation - compatible with uncertainties
        coordinates are assumed to be uarrays """

    v1, v2 = xyz1 - xyz2, xyz3 - xyz2

    dv1_dot_dv2 = np.sum(v1 ** 2) ** 0.5 * np.sum(v2 ** 2) ** 0.5

    return (180.0 / np.pi) * unp.arccos(np.dot(v1, v2) / dv1_dot_dv2)


def geo_torsion(xyz1, xyz2, xyz3, xyz4):
    """ Global function for torsion calculation - compatible with uncertainties
        coordinates are assumed to be uarrays """

    b0 = -1.0 * (xyz2 - xyz1)
    b1 = xyz3 - xyz2
    b2 = xyz4 - xyz3

    b0xb1, b1xb2 = np.cross(b0, b1), np.cross(b2, b1)  # Planes defined by the vectors
    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, b1) * (1.0 / np.sum(b1 ** 2) ** 0.5)
    x = np.dot(b0xb1, b1xb2)

    return np.abs((180.0 / np.pi) * unp.arctan2(y, x))  # Ignore sign of the dihedral angle


###############################################################################
# MISC ROUTINES
###############################################################################


def cartesian_product(arrays):
    """ Returns Cartesian product of given arrays (x and y): cartesian_product([x,y]) """

    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
    out = np.empty(rows * cols, dtype=broadcasted[0].dtype)
    start, end = 0, rows

    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows

    # Return value(s)
    return out.reshape(cols, rows).T


def unique(a):
    """ Returns unique 2D array entries of a given array """

    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), dtype=np.bool)
    ui[1:] = (diff != 0).any(axis=1)

    # Return value(s)
    return a[ui]


###############################################################################
# FUNCTIONS FOR MOLECULAR PROPERTIES
###############################################################################


def det_pair_connectivity(molecule, return_distance=False):
    """ Determines the pairwise index array of atoms bound to one another """

    # Calculate all unique combinations
    combinations = unique(np.sort(cartesian_product([np.arange(molecule.n_atoms),
                                                     np.arange(molecule.n_atoms)])))

    xx, yy = np.transpose(combinations)

    # Determine indices of elements excluding the main diagonal
    indices = np.where(xx != yy)[0]

    # Exclude entries from combinations array and 'update' xx and yy
    combinations = np.take(combinations, indices, axis=0)
    xx, yy = np.transpose(combinations)

    # Calculate the distances between identifier pairs
    dist_pairs = np.sqrt(np.sum((np.take(molecule.cor, xx, axis=0) -
                                 np.take(molecule.cor, yy, axis=0)) ** 2, axis=1))

    # Calculate the limit (sum of covalent radii times factor) of the associated bond type
    dist_limit = (np.take(molecule.rad_cov, xx) +
                  np.take(molecule.rad_cov, yy)) * molecule.det_bnd_fct

    # Determine indices where distance is below the sum of the radii
    indices = np.where(dist_pairs <= dist_limit)[0]

    # Generate identifier array
    pair_connectivity = np.take(combinations, indices, axis=0)

    # Return value(s)
    if return_distance:

        return combinations, dist_pairs, pair_connectivity, indices

    else:

        return pair_connectivity


def get_el_config(charge):
    """ Returns the electronic shell structure associated with a nuclear charge """

    # Electronic shells: 1s, 2s, 2p, 3s, 3p, 4s, 3d, 4p, 5s, 4d, 5p, 6s, 4f, 5d, 6p, 7s, 5f, 6d, 7p
    el_shell = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Maximum number of electrons for each shell
    max_el = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 6]

    # Set atomic charge as number of remaining electrons
    rest_electrons = charge

    # Start with first shell (1s)
    shell = 0

    # Until all electrons are depleted:
    while rest_electrons != 0:

        # Compare actual shell with maximum occupation number of current shell
        if el_shell[shell] != max_el[shell]:

            # If current shell is not full, add one electron and deplete it
            el_shell[shell] += 1
            rest_electrons -= 1

        else:  # If the shell is full go to the next one

            shell += 1

    # Return value(s) 
    return el_shell


def get_core_el(charge):
    """ Returns the number of core electrons (electrons below the valence shell) associated with a nuclear charge """

    # If the element is H or He there are no/only core electrons
    if charge == 1 or charge == 2:

        # Return value(s)
        return charge

    else:  # The element is not hydrogen or helium

        # Calculate electronic configuration
        el_config = np.array(get_el_config(charge))

        # Determine valence shell
        valence_shell = np.max(np.nonzero(el_config)[0])

        # Subtract valence electrons and sum all electrons = n_core_el
        return np.sum(el_config[:valence_shell])


###############################################################################
# CONSISTENCY AND MATCHING FUNCTIONS
###############################################################################


def check_for_identical_pos(xyz, eps=0.3):
    """ Returns positions where multiple atoms occupy the same position """

    def get_unique_positions(data):
        """ Stores the unique positions as tuples in a new list """

        pos = []

        for entry in data:

            if tuple(entry) in pos:  # Don't append duplicates

                pass

            else:

                pos.append(tuple(entry))

        # Return value(s)
        return pos

    # Calculate distances for all coordinates
    mult_mol = np.asarray([np.where(np.sqrt(np.sum((xyz - coord) ** 2, axis=1)) < eps)[0] for coord in xyz])

    # Determine positions where more than one atom is below the 'eps' limit
    data_mol = mult_mol[np.asarray([len(entry) > 1 for entry in mult_mol])]

    # Get positions of multiple occupations or return empty list
    pos_mol = get_unique_positions(data_mol)

    # Return value(s)
    return pos_mol


def exclude_identicals(molecule, logger, pos_mol, logg_for='molecule1'):
    """ Function for interactive removal of selected atoms """

    if len(pos_mol) == 0:  # If there are no disordered position - do nothing

        pass

    else:

        logger.lg_multiple_occupation_found(logg_for)
        del_list = []
        question = ">>  Enter identifier of the atom that should be kept: "

        for entry in pos_mol:
            logger.pt_info_multiple_occupation(molecule.sym, molecule.idf, molecule.cor, entry)  # Print information

            remove_list = list(entry)  # Create a list of atoms to remove and get a valid user choice
            pos = logger.get_menu_choice(list(np.asarray(entry) + 1), question, return_type='int') - 1

            # Remove the identifier from the 'removal list' and extend deletion list
            remove_list.remove(pos)
            del_list.extend(remove_list)

        # Delete positions from the charge and coordinate arrays
        molecule.chg = np.delete(molecule.chg, np.asarray(del_list))
        molecule.cor = np.delete(molecule.cor, np.asarray(del_list), 0)
        molecule.cor_std = np.delete(molecule.cor_std, np.asarray(del_list), 0)


def make_consistent(molecule1, molecule2, logger, settings):
    """ Subroutine that establishes consistency between molecule 1 and molecule 2
        by subsequent removal of hydrogen atoms, group treatment, etc.,
        delta: difference in angstrom for identical positions """

    # --------- Basic outline for consistency establishment ---------
    # Check for atoms on identical positions
    # Check if consistent
    # If not: Check for groups
    # If not: Remove hydrogen atoms on carbons
    # If not: Check for groups
    # If not: Remove group-14 H atoms
    # If not: Check for groups
    # If not: Remove all hydrogen atoms
    # If not: Check for groups
    # If not: ERROR
    # If consistent and hydrogens left: ask user to remove them
    # ---------------------------------------------------------------

    def _n_hydrogen_atoms(chg):
        """ Returns the number of hydrogen atoms in the molecule """

        return len(np.where(chg == 1)[0])

    def _hydrogens_left(chg1, chg2):
        """ Returns if there are hydrogen atoms in the molecules """

        return 1 in chg1 or 1 in chg2

    def _if_carbons(chg1, chg2):
        """ Returns if there are carbon atoms in the molecules """

        return len(np.where(chg1 == 6)[0]) > 0 or len(np.where(chg2 == 6)[0]) > 0

    def _are_consistent(chg1, chg2):
        """ Checks if the structures are consistent based on their charge arrays """

        return np.allclose(np.sort(chg1), np.sort(chg2)) if len(chg1) == len(chg2) else False

    def _check_for_identicals(molecule, logger, settings, logg_for='molecule1'):
        """ Checks for identical occupations in the molecule """

        pos_mol = check_for_identical_pos(molecule.cor, settings.delta)
        exclude_identicals(molecule, logger, pos_mol, logg_for)

    def _try_groups(molecule1, molecule2):
        """ Checks if matching problem can be solved with groups """

        molecule1.get_group_charges(), molecule2.get_group_charges()  # Get group properties

        if _are_consistent(molecule1.chg_grp, molecule2.chg_grp):

            return True

        else:

            molecule1.chg_grp, molecule2.chg_grp = None, None

            return False

    def _use_groups(molecule1, molecule2, logger):
        """ Uses groups in the matching process """

        molecule1.get_group_charges(), molecule2.get_group_charges()  # Get group properties
        molecule1.get_group_symbols(), molecule2.get_group_symbols()  # Get group properties
        logger.lg_group_algorithm()
        logger.lg_consistent()  # Logg event

    def _can_remove_carbon_hydrogens(molecule1, molecule2, settings, full_group_14=False):
        """ Checks if hydrogen atoms bound to carbon can be removed within consistency restrictions """

        return len(molecule1.fnd_mult_H_atoms_bte(settings, full_group_14)), len(
            molecule2.fnd_mult_H_atoms_bte(settings, full_group_14))

    def _remove_carbon_hydrogens(molecule, settings, full_group_14=False):
        """ Removes hydrogen atoms bound to carbon in the molecule and returns number of removed atoms """

        old_n_atoms = molecule.n_atoms

        molecule.rem_H_atoms_btc(settings, full_group_14)  # Remove hydrogen atoms

        return old_n_atoms - molecule.n_atoms

    def _remove_all_hydrogens(molecule):
        """ Removes all hydrogen atoms in the molecule and returns number of removed atoms """

        old_n_atoms = molecule.n_atoms

        molecule.rem_H_atoms_all()  # Remove hydrogen atoms

        return old_n_atoms - molecule.n_atoms

    # Check for multiple atoms on identical positions and remove selected atoms
    _check_for_identicals(molecule1, logger, settings, logg_for='molecule1')
    _check_for_identicals(molecule2, logger, settings, logg_for='molecule2')

    # Determine initial number of atoms and hydrogen atoms
    init_n_atoms1, init_n_atoms2 = molecule1.n_atoms, molecule2.n_atoms
    n_atoms1_init, n_atoms2_init = molecule1.n_atoms, molecule2.n_atoms

    init_n_hydro1, init_n_hydro2 = _n_hydrogen_atoms(molecule1.chg), _n_hydrogen_atoms(molecule2.chg)

    # Log number of atoms
    logger.cons_init_at_mol1, logger.cons_init_at_mol2 = n_atoms1_init, n_atoms2_init
    logger.cons_init_at_H_mol1, logger.cons_init_at_H_mol2 = init_n_hydro1, init_n_hydro2

    # Print information about number of atoms and hydrogen atoms in the molecules
    logger.pt_consistency_start(init_n_atoms1, init_n_atoms2)
    logger.pt_number_h_atoms(init_n_hydro1, init_n_hydro2)

    # Begin with algorithm
    if not _are_consistent(molecule1.chg, molecule2.chg):

        if _try_groups(molecule1, molecule2):  # CONSISTENT (GROUP)

            _use_groups(molecule1, molecule2, logger)

        else:

            n_atoms1_no_Hc = _remove_carbon_hydrogens(molecule1, settings, full_group_14=False)
            n_atoms2_no_Hc = _remove_carbon_hydrogens(molecule2, settings, full_group_14=False)

            logger.lg_rem_H_btc(n_atoms1_no_Hc, n_atoms2_no_Hc)  # Log event + print number of removed H atoms

            hydrogens_left = _hydrogens_left(molecule1.chg, molecule2.chg)  # Check for hydrogen atoms

            if not _are_consistent(molecule1.chg, molecule2.chg):

                if _try_groups(molecule1, molecule2):  # CONSISTENT (GROUP)

                    _use_groups(molecule1, molecule2, logger)

                elif hydrogens_left:

                    n_atoms1_no_Hg14 = _remove_carbon_hydrogens(molecule1, settings, full_group_14=True)
                    n_atoms2_no_Hg14 = _remove_carbon_hydrogens(molecule2, settings, full_group_14=True)

                    logger.lg_rem_H_btg14(n_atoms1_no_Hg14, n_atoms2_no_Hg14)  # Log event + print removed H atoms

                    hydrogens_left = _hydrogens_left(molecule1.chg, molecule2.chg)  # Check for hydrogen atoms

                    if not _are_consistent(molecule1.chg, molecule2.chg):

                        if _try_groups(molecule1, molecule2):  # CONSISTENT (GROUP)

                            _use_groups(molecule1, molecule2, logger)

                        elif hydrogens_left:

                            n_atoms1_no_H = _remove_all_hydrogens(molecule1)
                            n_atoms2_no_H = _remove_all_hydrogens(molecule2)

                            logger.lg_rem_H_all(n_atoms1_no_H, n_atoms2_no_H)  # Log event + print removed H atoms

                            if not _are_consistent(molecule1.chg, molecule2.chg):

                                if _try_groups(molecule1, molecule2):  # CONSISTENT (GROUP)

                                    _use_groups(molecule1, molecule2, logger)

                                else:

                                    logger.pt_consistency_error()  # Unsolvable problem

                            else:  # CONSISTENT (REGULAR)

                                logger.lg_consistent()  # Log event

                        else:

                            logger.pt_consistency_error()  # Unsolvable problem

                    else:  # CONSISTENT (REGULAR)

                        logger.lg_consistent()  # Log event

                else:

                    logger.pt_consistency_error()  # Unsolvable problem

            else:  # CONSISTENT (REGULAR)

                logger.lg_consistent()  # Log event

    else:  # CONSISTENT (REGULAR)

        logger.lg_consistent()  # Log event

    # At this point we have established consistency between the molecules
    if logger.consist:

        init_n_atoms1, init_n_atoms2 = molecule1.n_atoms, molecule2.n_atoms
        init_n_hydro1, init_n_hydro2 = _n_hydrogen_atoms(molecule1.chg), _n_hydrogen_atoms(molecule2.chg)
        n_hydro1_c, n_hydro2_c = _can_remove_carbon_hydrogens(molecule1, molecule2, settings, full_group_14=False)
        n_hydro1_full, n_hydro2_full = _can_remove_carbon_hydrogens(molecule1, molecule2, settings, full_group_14=True)
        hydrogens_left = _hydrogens_left(molecule1.chg, molecule2.chg)

        logger.can_rem_H_btc = n_hydro1_c == n_hydro2_c
        logger.can_rem_H_btg14 = n_hydro1_full == n_hydro2_full and n_hydro1_full != n_hydro1_c

        # Log info after molecules are consistent
        logger.cons_at_mol1, logger.cons_at_mol2 = init_n_atoms1, init_n_atoms2
        logger.cons_at_H_mol1, logger.cons_at_H_mol2 = init_n_hydro1, init_n_hydro2

        if not hydrogens_left:  # No hydrogen atoms left ... nothing can be done

            pass

        # Exclude all hydrogen atoms if desired
        elif not logger.rem_H_all and _n_hydrogen_atoms(molecule1.chg) == _n_hydrogen_atoms(molecule2.chg):

            choices = logger.pt_possibilities(init_n_hydro1, n_hydro2_c, n_hydro2_full)

            while True:

                operation = logger.get_menu_choice(choices, question="\n>>  Enter your choice: ", return_type='int')

                if operation == 0:  # Remove all hydrogen atoms

                    n_atoms1_no_H = _remove_all_hydrogens(molecule1)
                    n_atoms2_no_H = _remove_all_hydrogens(molecule2)
                    logger.lg_rem_H_all(n_atoms1_no_H, n_atoms2_no_H)  # Log event + print number of removed H atoms
                    logger.user_choice_rem_all_H = True

                    if logger.use_groups:  # Get group properties

                        molecule1.get_group_charges(), molecule2.get_group_charges()
                        molecule1.get_group_symbols(), molecule2.get_group_symbols()

                    break

                elif operation == 1:  # Remove hydrogens bound to carbon

                    n_atoms1_no_Hc = _remove_carbon_hydrogens(molecule1, settings, full_group_14=False)
                    n_atoms2_no_Hc = _remove_carbon_hydrogens(molecule2, settings, full_group_14=False)
                    logger.lg_rem_H_btc(n_atoms1_no_Hc, n_atoms2_no_Hc)  # Log event + print number of removed H atoms
                    logger.user_choice_rem_btc_H = True

                    if logger.use_groups:  # Get group properties

                        molecule1.get_group_charges(), molecule2.get_group_charges()
                        molecule1.get_group_symbols(), molecule2.get_group_symbols()

                    break

                elif operation == 2:  # Remove hydrogens bound to group-14 elements

                    n_atoms1_no_Hg14 = _remove_carbon_hydrogens(molecule1, settings, full_group_14=True)
                    n_atoms2_no_Hg14 = _remove_carbon_hydrogens(molecule2, settings, full_group_14=True)
                    logger.lg_rem_H_btg14(n_atoms1_no_Hg14, n_atoms2_no_Hg14)  # Log event + print # removed H atoms
                    logger.user_choice_rem_btg14_H = True

                    if logger.use_groups:  # Get group properties

                        molecule1.get_group_charges(), molecule2.get_group_charges()
                        molecule1.get_group_symbols(), molecule2.get_group_symbols()

                    break

                elif operation == 3:  # Do nothing

                    break

                else:

                    logger.pt_invalid_input()


def check_substructure(molecule1, molecule2, logger, pos1, pos2):
    """ Checks if proper substructures were defined by the user """

    if len(pos1) == len(pos2) and len(pos1) >= 3:  # If a proper substructure was defined

        # Save charges, coordinates and identifiers for both molecules
        molecule1.chg_sub = np.take(molecule1.chg, pos1)
        molecule1.cor_sub = np.take(molecule1.cor, pos1, axis=0)
        molecule1.n_atoms_sub = len(molecule1.chg_sub)
        molecule1.idf_sub = np.arange(1, molecule1.n_atoms_sub + 1, dtype=np.int)

        molecule2.chg_sub = np.take(molecule2.chg, pos2)
        molecule2.cor_sub = np.take(molecule2.cor, pos2, axis=0)
        molecule2.n_atoms_sub = len(molecule2.chg_sub)
        molecule2.idf_sub = np.arange(1, molecule2.n_atoms_sub + 1, dtype=np.int)

        if np.allclose(molecule1.chg_sub, molecule2.chg_sub):

            logger.lg_substructure()  # Log event

        else:

            logger.lg_wrong_substructure()
            reset_substructure(molecule1, molecule2, logger)

    else:

        logger.lg_wrong_substructure()
        reset_substructure(molecule1, molecule2, logger)


def set_initial_substructure(molecule1, molecule2):
    """ Defines the whole molecules as substructures """

    molecule1.chg_sub, molecule1.cor_sub, molecule1.n_atoms_sub = np.copy(molecule1.chg), np.copy(molecule1.cor), int(
        np.copy(molecule1.n_atoms))
    molecule2.chg_sub, molecule2.cor_sub, molecule2.n_atoms_sub = np.copy(molecule2.chg), np.copy(molecule2.cor), int(
        np.copy(molecule2.n_atoms))

    molecule1.idf_sub = np.arange(1, molecule1.n_atoms_sub + 1, dtype=np.int)
    molecule2.idf_sub = np.arange(1, molecule2.n_atoms_sub + 1, dtype=np.int)


def reset_substructure(molecule1, molecule2, logger):
    """ Resets the substructure to the full coordinate set """

    set_initial_substructure(molecule1, molecule2)  # Set initial substructures
    logger.lg_reset_substructure()  # Log event


def calc_permutation_matrix(xyz1, xyz2, logger, do_permute):
    """ Calculates all possible combinations of two sets of coordinates according to the defined formula """

    def distance_from_center(vector):
        """ Returns array with distance from origin for a n-dim. vector (n>1) """

        return np.sqrt(np.sum(vector ** 2, axis=1))

    def distance_from_all_atoms(pos_in_vector, vector):
        """ Returns the total absolute distance of the given atom position 
            from all atoms in the molecule """

        return np.sum(np.sqrt(np.sum((vector - vector[pos_in_vector]) ** 2, axis=1)))

    def delta_func_below(vector, delta_value):
        """ Returns array with '1.0' for results <= delta_value, '0.0' otherwise """

        # Create zero array with same shape as original array
        return_array = np.zeros(np.shape(vector))

        # Set array at positions where the value in 'array' is <= delta_value to '1.0'
        return_array[np.where(np.array(vector, dtype=np.float) <= delta_value)[0]] = 1.0

        return return_array

    def delta_func_above(vector, delta_value):
        """ Returns array with '1.0' for results >= delta_value, '0.0' otherwise """

        # Create zero array with same shape as original array
        return_array = np.ones(np.shape(vector))

        # Set array at positions where the value in 'array' is >= delta_value to '0.0'
        return_array[np.where(np.array(vector, dtype=np.float) >= delta_value)[0]] = 0.0

        return return_array

    n_atoms1, n_atoms2 = len(xyz1), len(xyz2)

    # Generate list of permutation indices via cartesian product
    permut_indices = cartesian_product([np.arange(n_atoms1), np.arange(n_atoms2)])

    permutation_matrix = None

    xx, yy = np.transpose(permut_indices)

    if logger.match_alg == 'distance':

        # Calculate matrix elements: distance difference between two atoms and generate matrix in
        # the shape of the number of atoms
        if do_permute == True:
            permutation_matrix = np.reshape(np.sqrt(np.sum((np.take(xyz1, xx, axis=0) -
                                                        np.take(xyz2, yy, axis=0)) ** 2, axis=1)),
                                        (n_atoms1, n_atoms2))
        elif do_permute == False:
            permutation_matrix = np.zeros(shape=(len(xyz1),len(xyz1)))
            for i in range(len(xyz1)):
                for j in range(len(xyz1)):
                    if i == j:
                        permutation_matrix[i,j] = 0.0
                    else:
                        permutation_matrix[i,j] = 1.0

    elif logger.match_alg == 'combined':

        distance_limit = 1.0  # Use delta function to exclude separated atoms (in A)

        # Calculate distances from center
        distances_element_xyz1 = distance_from_center(xyz1)
        distances_element_xyz2 = distance_from_center(xyz2)

        # Calculate distances from all atoms
        dist_all_atoms_xyz1 = np.array([distance_from_all_atoms(entry, xyz1)
                                        for entry in range(n_atoms1)], dtype=np.float)
        dist_all_atoms_xyz2 = np.array([distance_from_all_atoms(entry, xyz2)
                                        for entry in range(n_atoms2)], dtype=np.float)

        # Calculate contributions to matrix elements
        # Distance between atoms
        d_ij_sq = np.sqrt(np.sum((np.take(xyz1, xx, axis=0) -
                                  np.take(xyz2, yy, axis=0)) ** 2, axis=1))

        # Distance from center
        d_ele = np.abs(np.take(distances_element_xyz1, xx, axis=0) -
                       np.take(distances_element_xyz2, yy, axis=0))

        d_ele += delta_func_above(d_ele, distance_limit) * 1.0E+06

        # Distance from all other atoms (relative position in molecule)
        d_all_atoms = np.abs(np.take(dist_all_atoms_xyz1, xx, axis=0) -
                             np.take(dist_all_atoms_xyz2, yy, axis=0))
        d_all_atoms += delta_func_above(d_all_atoms, distance_limit) * 1.0E+06

        # Generate matrix in the shape of the number of atoms
        if do_permute == True:
            permutation_matrix = np.reshape(d_ij_sq + d_ele + d_all_atoms, (n_atoms1, n_atoms2))

        elif do_permute == False:
            permutation_matrix = np.identity(len(xyz1))

    return permutation_matrix


def decompose_molecule(chg, cor, idf):
    """ Decomposes a set of coordinates associated with nuclear charges
        and identifiers into a subset of arrays each one corresponding
        to one nuclear charge """

    #
    # decomposed_molecule = [coordinates(charge1), identifiers(charge1),
    #                        coordinates(charge2), identifiers(charge2),
    #                        ...                                        ]
    #

    def per_atom_type(atom_type, chg, cor, idf):
        """ Returns tuple for each atom type """

        # Find all positions of actual element type in substructure charge array 
        pos = np.where(chg == atom_type)[0]

        # Return all corresponding identifiers and coordinate tuples
        return np.take(cor, pos, axis=0), np.take(idf, pos)

    overwrite_decomp = False

    atom_types = np.unique(chg)  # Determine iteration values

    occurrences = np.take(np.bincount(chg), atom_types)  # Count occurrences of all elements

    # Decompose molecule
    decomposed_molecule = np.asarray([per_atom_type(atom_type, chg, cor, idf)
                                      for atom_type in atom_types])

    if overwrite_decomp:

        print('overwriting decomposition')

        atom_types = np.array([1], dtype=np.int)
        occurrences = 1
        decomposed_molecule = [[cor, idf], [cor, idf]]

    return decomposed_molecule, atom_types, occurrences


def match_molecules(molecule1, molecule2, logger, onlyH):
    """ Matches the coordinate sequences of molecules with identical atom types """

    if logger.use_groups:  # If group charges are to be used

        chg1, chg2 = molecule1.chg_grp, molecule2.chg_grp

    else:

        chg1, chg2 = molecule1.chg, molecule2.chg

    # Decompose both molecules [decompose_molecule function]
    [decomposed_molecule1,
     atom_types1, occurrences1] = decompose_molecule(chg1, molecule1.cor, molecule1.idf)

    [decomposed_molecule2,
     atom_types2, occurrences2] = decompose_molecule(chg2, molecule2.cor, molecule2.idf)

    # Determine the number of atom types
    n_atom_types = len(atom_types1)

    # Set up empty variables (lists)
    identifiers_molecule1, identifiers_molecule2 = [], []
    
    if onlyH == False:
        # Iterate through all atom types
        for atom_type in range(n_atom_types):
            # Extract coordinates and identifiers from decomposed molecules
            coords1, id1 = decomposed_molecule1[atom_type]
            coords2, id2 = decomposed_molecule2[atom_type]

            if logger.match_solv == 'standard':

                # Check for number of occurences of the actual atom type
                if occurrences1[atom_type] == 1:

                    # Directly match atoms
                    identifiers_molecule1.append(id1)
                    identifiers_molecule2.append(id2)

                else:  # If more than one atom of the type exists

                    n_atoms = len(coords1)  # Determine numer of atoms

                    # Calculate permutation matrix for this atom type
                    permutation_matrix = calc_permutation_matrix(coords1, coords2, logger)

                    # Iterate through all atoms of this type
                    for atom in range(n_atoms):

                        # Find indices of minimum value in matrix (first index cooresponds to coords1)
                        indices = np.where(permutation_matrix == np.min(permutation_matrix))

                        if len(indices[0]) != 1:  # Check if there is more than one solution
                        
                            # If there is: use the first match
                            indices = np.array((indices[0][0], indices[1][0]))

                        # Append identifiers corresponding to indices
                        identifiers_molecule1.append(id1[indices[0]])
                        identifiers_molecule2.append(id2[indices[1]])

                        # Deplete corresponding row of first index and column of second index from matrix
                        permutation_matrix = np.delete(permutation_matrix, indices[0], 0)
                        permutation_matrix = np.delete(permutation_matrix, indices[1], 1)

                        # Deplete identifier arrays
                        id1 = np.delete(id1, indices[0])
                        id2 = np.delete(id2, indices[1])

            elif logger.match_solv == 'hungarian':

                # Calculate permutation matrix for this atom type
                permutation_matrix = calc_permutation_matrix(coords1, coords2, logger, do_permute=True)

                # Solve the problem with the Hungarian algorithm
                indices_mol1, indices_mol2 = hungarian_solver(permutation_matrix)

                # Collect indices of the assignment
                identifiers_molecule1.extend(id1[indices_mol1])
                identifiers_molecule2.extend(id2[indices_mol2])

    elif onlyH == True:
        for atom_type in range(n_atom_types):
            coords1, id1 = decomposed_molecule1[atom_type]
            coords2, id2 = decomposed_molecule2[atom_type]

            if logger.match_solv == 'standard':

                # Check for number of occurences of the actual atom type
                if occurrences1[atom_type] == 1:

                    # Directly match atoms
                    identifiers_molecule1.append(id1)
                    identifiers_molecule2.append(id2)

                else:  # If more than one atom of the type exists

                    n_atoms = len(coords1)  # Determine numer of atoms
  
                    # Calculate permutation matrix for this atom type
                    permutation_matrix = calc_permutation_matrix(coords1, coords2, logger)

                # Iterate through only H atoms
                for atom in range(n_atoms):

                    # Find indices of minimum value in matrix (first index cooresponds to coords1)
                    indices = np.where(permutation_matrix == np.min(permutation_matrix))

                    if len(indices[0]) != 1:  # Check if there is more than one solution
                        
                        # If there is: use the first match
                        indices = np.array((indices[0][0], indices[1][0]))

                        # Append identifiers corresponding to indices
                        identifiers_molecule1.append(id1[indices[0]])
                        identifiers_molecule2.append(id2[indices[1]])

                        # Deplete corresponding row of first index and column of second index from matrix
                        permutation_matrix = np.delete(permutation_matrix, indices[0], 0)
                        permutation_matrix = np.delete(permutation_matrix, indices[1], 1)

                        # Deplete identifier arrays
                        id1 = np.delete(id1, indices[0])
                        id2 = np.delete(id2, indices[1])

            elif logger.match_solv == 'hungarian':

                # Calculate permutation matrix for this atom type
                if atom_type == 0:
                    permutation_matrix = calc_permutation_matrix(coords1, coords2, logger, do_permute=True)
                else:
                    permutation_matrix = calc_permutation_matrix(coords1, coords2, logger, do_permute=False)
                # Solve the problem with the Hungarian algorithm
                indices_mol1, indices_mol2 = hungarian_solver(permutation_matrix)

                # Collect indices of the assignment
                identifiers_molecule1.extend(id1[indices_mol1])
                identifiers_molecule2.extend(id2[indices_mol2])

    # Transform identifiers to single arrays containing the new positions
    pos_mol1 = np.ravel(np.array(identifiers_molecule1, dtype=np.int) - 1)
    pos_mol2 = np.ravel(np.array(identifiers_molecule2, dtype=np.int) - 1)
    return(pos_mol1, pos_mol2)
    # Update charges (should be redundant but just to be sure...)
    molecule1.chg = np.take(molecule1.chg, pos_mol1)
    molecule2.chg = np.take(molecule2.chg, pos_mol2)

    # Update coordinates of the coordinate arrays whose orientations were not changed
    molecule1.cor = np.take(molecule1.cor, pos_mol1, axis=0)
    molecule2.cor = np.take(molecule2.cor, pos_mol2, axis=0)

    # Update standard deviations
    molecule1.cor_std = np.take(molecule1.cor_std, pos_mol1, axis=0)
    molecule2.cor_std = np.take(molecule2.cor_std, pos_mol2, axis=0)

    if logger.use_groups:  # If group charges were used, update group properties as well

        molecule1.get_group_charges(), molecule2.get_group_charges()
        molecule1.get_group_symbols(), molecule2.get_group_symbols()

    logger.is_matched = True  # Log successful matching
    molecule1.show_mol = True  # Set variable so result is shown


def fast_rmsd(xyz1, xyz2):
    """ Calculates the unweighted RMSD of two arrays with identical coordinate order """

    return np.sqrt(np.sum(np.sum((xyz1 - xyz2) ** 2, axis=1) / len(xyz2)))


def fast_msd(xyz1, xyz2):
    """ Calculates the unweighted MSD of two arrays with identical coordinate order """

    return np.sqrt(np.sum((xyz1 - xyz2) ** 2, axis=1))


def highlight_dev(molecule1, molecule2, logger, settings):
    """ Returns positions of highest deviations in the molecule and sets colors """

    # Reset changes
    molecule1.set_color(settings.col_model_rgb)
    molecule2.set_color(settings.col_refer_rgb)

    # Update info in logger
    xyz1, xyz2 = molecule1.cor, molecule2.cor

    rmsd_values = np.sqrt(fast_msd(xyz1, xyz2))

    molecule1.disord_pos = rmsd_values.argsort()[-settings.n_dev:]
    molecule2.disord_pos = np.copy(molecule1.disord_pos)

    molecule1.disord_rmsd = rmsd_values[molecule1.disord_pos]
    molecule2.disord_rmsd = np.copy(molecule1.disord_rmsd)

    molecule1.col_at_rgb[molecule1.disord_pos] = settings.col_disord_rgb
    molecule2.col_at_rgb[molecule1.disord_pos] = settings.col_disord_rgb

    logger.n_dev = settings.n_dev
    logger.disord_pos, logger.disord_rmsd = molecule2.disord_pos, molecule2.disord_rmsd
    logger.match_rmsd = fast_rmsd(xyz1, xyz2)


def project_radii(radii, spacing, r_min, r_max):
    """ Projects given radii to values between r_min and r_max; good spacing ~ 1000 """

    radii_norm = radii / np.max(radii)  # Normalize radii

    # Determine min and max of array and generate spacing
    radii_to_proj = np.around(np.linspace(np.min(radii_norm), np.max(radii_norm), spacing), 3)
    values_to_proj = np.around(np.linspace(r_min, r_max, spacing), 3)

    # Determine respective array positions
    pos = np.array([np.argmin(np.abs(radii_to_proj -
                                     radii_norm[entry])) for entry in range(len(radii_norm))], dtype=np.int)

    # Determine new radii
    return np.take(values_to_proj, pos)


###############################################################################
# HUNGARIAN (MUNKRES) ALGORITHM - TAKEN FROM SCIPY
###############################################################################


def hungarian_solver(cost_matrix):
    """ Solve the linear assignment problem using the Hungarian algorithm """
    
    idx_row, idx_col = _hungarian(cost_matrix)
    
    return idx_row, idx_col


class _HungarianState(object):
    """ State of one execution of the Hungarian algorithm """

    def __init__(self, cost_matrix):
        
        self.C = cost_matrix.copy()

        n, m = self.C.shape
        self.row_uncovered = np.ones(n, dtype=np.bool)
        self.col_uncovered = np.ones(m, dtype=np.bool)
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = np.zeros((n + m, 2), dtype=np.int)
        self.marked = np.zeros((n, m), dtype=np.int)

    def _clear_covers(self):
        """Clear all covered matrix cells"""
        
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True


def _hungarian(cost_matrix):
    """ Executes actual Hungarian algorithm """

    cost_matrix = np.asarray(cost_matrix)
    
    if len(cost_matrix.shape) != 2:
        
        raise ValueError("expected a matrix (2-d array), got a %r array"
                         % (cost_matrix.shape,))

    # The algorithm expects more columns than rows in the cost matrix
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        
        cost_matrix = cost_matrix.T
        transposed = True
        
    else:
        
        transposed = False

    state = _HungarianState(cost_matrix)

    # No need to bother with assignments if one of the dimensions of the cost matrix is zero-length
    step = None if 0 in cost_matrix.shape else _step1

    while step is not None:
        
        step = step(state)

    if transposed:
        
        marked = state.marked.T
        
    else:
        
        marked = state.marked

    return np.where(marked == 1)


def _step1(state):

    # Step 1: For each row of the matrix, find the smallest element and
    # subtract it from every element in its row.
    state.C -= state.C.min(axis=1)[:, np.newaxis]
    # Step 2: Find a zero (Z) in the resulting matrix. If there is no
    # starred zero in its row or column, star Z. Repeat for each element
    # in the matrix.
    for i, j in zip(*np.where(state.C == 0)):
        if state.col_uncovered[j] and state.row_uncovered[i]:
            state.marked[i, j] = 1
            state.col_uncovered[j] = False
            state.row_uncovered[i] = False

    state._clear_covers()
    return _step3


def _step3(state):
    
    marked = (state.marked == 1)
    state.col_uncovered[np.any(marked, axis=0)] = False

    if marked.sum() < state.C.shape[0]:
        return _step4


def _step4(state):
    
    # We convert to int as numpy operations are faster on int
    C = (state.C == 0).astype(int)
    covered_C = C * state.row_uncovered[:, np.newaxis]
    covered_C *= np.asarray(state.col_uncovered, dtype=int)
    n = state.C.shape[0]
    m = state.C.shape[1]

    while True:
        # Find an uncovered zero
        row, col = np.unravel_index(np.argmax(covered_C), (n, m))
        if covered_C[row, col] == 0:
            return _step6
        else:
            state.marked[row, col] = 2
            # Find the first starred element in the row
            star_col = np.argmax(state.marked[row] == 1)
            if state.marked[row, star_col] != 1:
                # Could not find one
                state.Z0_r = row
                state.Z0_c = col
                return _step5
            else:
                col = star_col
                state.row_uncovered[row] = False
                state.col_uncovered[col] = True
                covered_C[:, col] = C[:, col] * (
                    np.asarray(state.row_uncovered, dtype=int))
                covered_C[row] = 0


def _step5(state):
    
    count = 0
    path = state.path
    path[count, 0] = state.Z0_r
    path[count, 1] = state.Z0_c

    while True:
        # Find the first starred element in the col defined by the path
        row = np.argmax(state.marked[:, path[count, 1]] == 1)
        if state.marked[row, path[count, 1]] != 1:
            # Could not find one
            break
        else:
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count - 1, 1]

        # Find the first prime element in the row defined by the
        # first path step
        col = np.argmax(state.marked[path[count, 0]] == 2)
        if state.marked[row, col] != 2:
            col = -1
        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col

    # Convert paths
    for i in range(count + 1):
        if state.marked[path[i, 0], path[i, 1]] == 1:
            state.marked[path[i, 0], path[i, 1]] = 0
        else:
            state.marked[path[i, 0], path[i, 1]] = 1

    state._clear_covers()
    # Erase all prime markings
    state.marked[state.marked == 2] = 0
    return _step3


def _step6(state):
    
    # The smallest uncovered value in the matrix
    if np.any(state.row_uncovered) and np.any(state.col_uncovered):
        
        minval = np.min(state.C[state.row_uncovered], axis=0)
        minval = np.min(minval[state.col_uncovered])
        state.C[~state.row_uncovered] += minval
        state.C[:, state.col_uncovered] -= minval

    return _step4


###############################################################################
# MOLECULE OBJECT
###############################################################################


class Molecule(object):
    """ A molecule object used that contains all relevant properties """

    def __init__(self, name, symbol, coord, std):
        """ Initializes a 'Molecule' object """

        self.name = name  # Name of the molecule
        self.sym = symbol  # Element symbols
        self.cor = coord  # Cart. coordinates
        self.cor_std = std  # Standard deviations of Cart. coordinates (may be zero array)
        self.chg = None  # Nuclear charges
        self.mas = None  # Atomic masses
        self.com = None  # Center of mass
        self.ine_ten = None  # (Normalized) inertia tensor
        self.ine_pa = None  # Principal axes of rotation
        self.ine_pc = None  # Principal components of rotation
        self.idf = None  # Element identifiers
        self.sym_idf = None  # Combined symbols and identifiers
        self.chg_grp = None  # PSE groups of the elements as 'charges'
        self.rad_cov = None  # Covalent radii
        self.rad_atm = None  # Atomic radii
        self.bnd_idx = None  # Bond indices
        self.ang_idx = None  # Angle indices
        self.tor_idx = None  # Torsion indices
        self.bnd_dis = None  # Bond distances
        self.ang_deg = None  # Angle degrees
        self.tor_deg = None  # Torsion degrees
        self.n_atoms = None  # Number of atoms
        self.n_h_atoms = None  # Number of hydrogen atoms
        self.n_bonds = None  # Number of bonds
        self.rad_plt_vtk = None  # Radii for vtk plotting
        self.col_glob_rgb = None  # Global variable for plotting color [RGB]
        self.col_disord_rgb = None  # Global variable for plotting color disordered positions [HEX]
        self.col_at_rgb = None  # Color array for the atoms [RGB]
        self.col_bnd_rgb = None  # Color array for the bonds [RGB]
        self.symOPs = []  # List of applied symmetry operations
        self.tot_symOP = np.identity(3, dtype=np.float)  # Combination of all applied sym OPs
        self.show_mol = True  # If molecule is to be shown again

        self.disord_pos = None  # Positions of highest deviations after matching
        self.disord_rmsd = None  # RMSD of highest deviations after matching

        self.cor_save = None  # Copy used for reset option matching (cart. coordinates)
        self.chg_save = None  # Copy used for reset option matching (nuclear charges)
        self.symOPs_save = []  # Copy used for reset option matching (symmetry operations)
        self.cor_origin = None  # Copy used for reset option consistency (cart. coordinates)
        self.chg_origin = None  # Copy used for reset option consistency (nuclear charges)

        self.has_cam_vtk = False  # If the molecule has an existing VTK camera orientation
        self.cam_vtk_pos = None  # VTK camera position
        self.cam_vtk_wxyz = None  # VTK camera angle/axis
        self.cam_vtk_focal_pt = None  # VTK camera focal point
        self.cam_vtk_view_up = None  # VTK camera view up (?)

        self.outfile_name = None  # Name of the outfile to which the structural data will be written

    ###############################################################################
    # FUNCTIONS FOR ATOMIC/MOLECULAR PROPERTIES
    ###############################################################################
    def get_bonds(self, settings):
        """ Determines the bond indices and distances of the molecule """

        # Calculate all unique combinations
        combinations = unique(np.sort(cartesian_product([np.arange(self.n_atoms),
                                                         np.arange(self.n_atoms)])))

        xx, yy = np.transpose(combinations)

        # Determine indices of elements excluding the main diagonal
        indices = np.where(xx != yy)[0]

        # Exclude entries from combinations array and 'update' xx and yy
        combinations = np.take(combinations, indices, axis=0)
        xx, yy = np.transpose(combinations)

        # Calculate the distances between identifier pairs
        dist_pairs = np.sqrt(np.sum((np.take(self.cor, xx, axis=0) -
                                     np.take(self.cor, yy, axis=0)) ** 2, axis=1))

        # Calculate the limit (sum of covalent radii times factor) of the associated bond type
        dist_limit = (np.take(self.rad_cov, xx) +
                      np.take(self.rad_cov, yy)) * settings.det_bnd_fct

        # Determine indices where distance is below the sum of the radii
        indices = np.where(dist_pairs <= dist_limit)[0]

        # Update bond indices and distances
        self.bnd_idx = np.take(combinations, indices, axis=0)
        self.n_bonds = len(indices)

        #self.col_bnd_rgb = np.transpose(np.repeat(self.col_glob_rgb, self.n_bonds).reshape((3, self.n_bonds)))

    def get_charge(self):
        """ Gets/updates charges from element symbols """

        self.chg = np.array([pse_symbol[symbol] for symbol in self.sym], dtype=np.int)
        self.n_atoms = len(self.chg)
        self.n_h_atoms = len(np.where(self.chg == 1)[0])

    def get_symbol(self):
        """ Gets/updates symbols from nuclear charges """

        self.sym = np.asarray(np.take(pse_sym_chg, self.chg - 1), dtype=np.object)

    def get_mass(self):
        """ Gets/updates atomic masses from nuclear charges """

        # Disabled: masses will fail in PSE group algorithm
        self.mas = np.repeat(1.0, self.n_atoms)
        self.mas_sub = np.copy(self.mas)  # Until added to aRMSD

    def get_cov_radii(self, settings):
        """ Gets/updates covalent radii from nuclear charges """

        self.rad_cov = np.take(pse_cov_radii, self.chg - 1)

        if settings.modify_H:  # Modify radii (from 0.31 to 0.41) for H if requested

            self.rad_cov[np.where(self.chg == 1)[0]] = 0.41

        self.rad_plt_vtk = self.rad_cov

    def get_atm_radii(self):
        """ Gets/updates atomic radii from nuclear charges """

        self.rad_atm = np.take(pse_atm_radii, self.chg - 1)

    def get_identifiers(self):
        """ Generates/updates identifiers """

        self.idf = np.arange(1, len(self.chg) + 1, dtype=np.int)

    def get_sym_idf(self):
        """ Combines symbols and identifiers """

        self.sym_idf = np.asarray([('%s-%d' % (self.sym[atom], self.idf[atom])) for atom in range(self.n_atoms)],
                                  dtype=np.object)

    def get_group_charges(self):
        """ Gets group charges (can be used for the matching of different atom types)
            of the elements from nuclear charges """

        self.chg_grp = np.take(pse_groups, self.chg - 1) + 600

    def get_group_masses(self):
        """ Gets group masses of the elements from nuclear charges """

        self.mas_grp = np.ones(self.n_atoms, dtype=np.float)

    def get_group_symbols(self):
        """ Generates group symbols from group charges """

        self.sym_grp = np.asarray(np.take(pse_group_sym, self.chg_grp - 601), dtype=np.object)

    def update_properties(self, settings, get_bonds=True):
        """ Updates all molecular properties """

        self.n_atoms = len(self.chg)
        self.n_h_atoms = len(np.where(self.chg == 1)[0])
        self.get_symbol()
        self.get_identifiers()
        self.get_sym_idf()
        self.get_mass()
        self.get_atm_radii()
        self.get_cov_radii(settings)
        #self.col_at_rgb = np.transpose(np.repeat(self.col_glob_rgb, self.n_atoms).reshape((3, self.n_atoms)))

        if get_bonds:  # Update bonds if requested

            self.get_bonds(settings)

        #self.col_bnd_rgb = np.transpose(np.repeat(self.col_glob_rgb, self.n_bonds).reshape((3, self.n_bonds)))

    def set_color(self, color_rgb):
        """ Sets color of the molecule in RGB """

        self.col_glob_rgb = color_rgb  # Store RGB value

    def set_color_disordered(self, color_rgb):
        """ Sets color of the disordered positions in the molecule in RGB """

        self.col_disord_rgb = color_rgb

    def make_save_point(self, reset_type='save'):
        """ Saves the current status (nuclear charges and atomic coordinates) """

        if reset_type == 'save':

            self.chg_save = np.copy(self.chg)
            self.cor_save = np.copy(self.cor)
            self.cor_std_save = np.copy(self.cor_std)
            self.symOPs_save = self.symOPs

        elif reset_type == 'origin':

            self.chg_origin = np.copy(self.chg)
            self.cor_origin = np.copy(self.cor)
            self.cor_std_origin = np.copy(self.cor_std)

    def reset_molecule(self, settings, reset_type='save'):
        """ Resets the molecule to the saved status """

        if reset_type == 'save':

            self.chg = np.copy(self.chg_save)
            self.cor = np.copy(self.cor_save)
            self.cor_std = np.copy(self.cor_std_save)
            self.symOPs = self.symOPs_save

        elif reset_type == 'origin':

            self.chg = np.copy(self.chg_origin)
            self.cor = np.copy(self.cor_origin)
            self.cor_std = np.copy(self.cor_std_origin)

        self.update_properties(settings)
        self.get_bonds(settings)
        self.show_mol = True

    ###############################################################################
    # FUNCTIONS FOR CONSISTENCY ESTABLISHMENT
    ###############################################################################

    def rem_atom(self, pos, settings, get_bonds=True):
        """ Removes an atom and updates properties """

        self.chg = np.delete(self.chg, pos, 0)  # Delete the respective entry
        self.cor = np.delete(self.cor, pos, 0)  # Delete the respective entry
        self.cor_std = np.delete(self.cor_std, pos, 0)  # Delete the respective entry

        self.update_properties(settings, get_bonds)

    def fnd_H_atoms_bnd_ele(self, chg_ele, settings):
        """ Finds all hydrogens bound to 'element' atoms """

        def single_entry(h_atom):
            """ Determines if current hydrogen atom is bound to any 'element' atom """

            return np.min(np.sqrt(np.sum((e_cor - h_cor[h_atom]) ** 2, axis=1))) <= bond_dist

        bond_dist = (np.take(pse_cov_radii, 0) + np.take(pse_cov_radii, chg_ele - 1)) * settings.det_bnd_fct

        # Determine positions of 'H' and coordinates of 'E' atoms
        h_atom_pos = np.where(self.chg == 1)[0]
        e_cor = np.take(self.cor, np.where(self.chg == chg_ele)[0], axis=0)

        if len(e_cor) != 0:

            h_cor = np.take(self.cor, h_atom_pos, axis=0)
            n_atoms_h = len(h_cor)

            pos = np.asarray([single_entry(h_atom) for h_atom in range(n_atoms_h)], dtype=np.bool)

            return h_atom_pos[pos]

        else:

            return np.empty(0)

    def fnd_mult_H_atoms_bte(self, settings, full_group_14=False):
        """ Finds all H atoms bound to group-14 elements or carbon """

        return np.asarray(np.hstack((self.fnd_H_atoms_bnd_ele(6, settings), self.fnd_H_atoms_bnd_ele(14, settings),
                                     self.fnd_H_atoms_bnd_ele(32, settings), self.fnd_H_atoms_bnd_ele(50, settings),
                                     self.fnd_H_atoms_bnd_ele(82, settings))), dtype=np.int) \
            if full_group_14 else self.fnd_H_atoms_bnd_ele(6, settings)

    def rem_H_atoms_btc(self, settings, full_group_14=False):
        """ Removes all hydrogens bound to carbon atoms """

        h_bound_to_c = self.fnd_mult_H_atoms_bte(settings, full_group_14)

        # Update charges and coordinates respectively
        self.chg = np.delete(self.chg, h_bound_to_c, 0)
        self.cor = np.delete(self.cor, h_bound_to_c, 0)
        self.cor_std = np.delete(self.cor_std, h_bound_to_c, 0)  # Delete the respective entry
        self.n_atoms = len(self.chg)

    def rem_H_atoms_all(self):
        """ Removes all hydrogen atoms from the molecule """

        # Determine positions of 'non H-atoms'
        h_pos = np.where(self.chg == 1)[0]

        # Update arrays
        self.chg = np.delete(self.chg, h_pos, 0)
        self.cor = np.delete(self.cor, h_pos, 0)
        self.cor_std = np.delete(self.cor_std, h_pos, 0)  # Delete the respective entry
        self.n_atoms = len(self.chg)

    def swap_atoms(self, logger):
        """ Interchanges coordinates based on user input """

        # Ask user for number of interchanges
        n_atom_pairs = int(input(logger.pt_swap_n_pairs()))

        # Generate copy of initial coordinate/charge array that is modified
        xyz_swap = np.copy(self.cor)
        chg_swap = np.copy(self.chg)
        std_swap = np.copy(self.cor_std)

        # Iterate through the number of atom pairs given by user input 
        for pair in range(n_atom_pairs):
            
            # Request two identifiers of the atoms that are to be swapped
            fir_atom = logger.get_menu_choice(range(1, self.n_atoms + 1),
                                              question="\n>>  Enter first identifier: ", return_type='int')
            sec_atom = logger.get_menu_choice(range(1, self.n_atoms + 1),
                                              question=">>  Enter second identifier: ", return_type='int')

            # Determine coordinate positions in the coordinate array
            fir_atom -= 1
            sec_atom -= 1

            # Print out to user
            logger.pt_swap_atoms(self.sym[fir_atom], fir_atom + 1, self.sym[sec_atom], sec_atom + 1)

            # Interchange coordinates/charges of the two given positions
            xyz_swap[fir_atom], xyz_swap[sec_atom] = self.cor[sec_atom], self.cor[fir_atom]
            chg_swap[fir_atom], chg_swap[sec_atom] = self.chg[sec_atom], self.chg[fir_atom]
            std_swap[fir_atom], std_swap[sec_atom] = self.cor_std[sec_atom], self.cor_std[fir_atom]

        # Update original arrays
        self.cor = xyz_swap
        self.chg = chg_swap
        self.cor_std = std_swap

        # Generate new identifiers
        self.get_identifiers()
        self.show_mol = True

    ###############################################################################
    # FUNCTIONS FOR STRUCTURAL PROPERTIES
    ###############################################################################

    def Aget_bonds(self, settings):
        """ Determines the bond indices and distances of the molecule """

        # Calculate all unique combinations
        combinations = unique(np.sort(cartesian_product([np.arange(self.n_atoms),
                                                         np.arange(self.n_atoms)])))

        xx, yy = np.transpose(combinations)

        # Determine indices of elements excluding the main diagonal
        indices = np.where(xx != yy)[0]

        # Exclude entries from combinations array and 'update' xx and yy
        combinations = np.take(combinations, indices, axis=0)
        xx, yy = np.transpose(combinations)

        # Calculate the distances between identifier pairs
        dist_pairs = np.sqrt(np.sum((np.take(self.cor, xx, axis=0) -
                                     np.take(self.cor, yy, axis=0)) ** 2, axis=1))

        # Calculate the limit (sum of covalent radii times factor) of the associated bond type
        dist_limit = (np.take(self.rad_cov, xx) +
                      np.take(self.rad_cov, yy)) * settings.det_bnd_fct

        # Determine indices where distance is below the sum of the radii
        indices = np.where(dist_pairs <= dist_limit)[0]

        # Update bond indices and distances
        self.bnd_idx = np.take(combinations, indices, axis=0)
        self.bnd_dis = np.take(dist_pairs, indices, axis=0)
        self.n_bonds = len(indices)

    def check_for_unit(self, logger, settings):
        """ Checks if coordinates are in Angstrom or Bohr """

        logger.lg_unit_check()  # Log event

        if self.n_atoms <= settings.atom_limit:

            logger.lg_unit_check_warning()  # Print warning to user
            pass

        else:

            if self.n_bonds >= (self.n_atoms / 2):

                logger.pt_unit_is_a()  # Print unit information for user
                pass

            else:

                self.cor *= b2a  # Transform coordinates, update bonds and logg event
                self.get_bonds()
                logger.lg_transform_coord_unit()

    ###############################################################################
    # FUNCTIONS FOR STANDARD ORIENTATION AND SYMMETRY OPERATIONS
    ###############################################################################

    def calc_com(self, calc_for='molecule'):
        """ Calculates the barycenter of the molecule/substructure """

        if calc_for == 'molecule':

            self.com = np.dot(np.transpose(self.cor), self.mas / np.sum(self.mas))

        elif calc_for == 'substructure':

            self.com_sub = np.dot(np.transpose(self.cor_sub), self.mas_sub / np.sum(self.mas_sub))

    def shift_com(self, calc_for='molecule'):
        """ Shifts the barycenter of the molecule/substructure to (0,0,0) """

        if calc_for == 'molecule':

            self.cor -= self.com
            self.com -= self.com

        elif calc_for == 'substructure':

            self.cor_sub -= self.com_sub
            self.com_sub -= self.com_sub

    def calc_ine_tens(self, calc_for='molecule'):
        """ Calculates the normalized inertia tensor of the molecule/substructure """

        if calc_for == 'molecule':

            xyz, mass = self.cor, self.mas

        elif calc_for == 'substructure':

            xyz, mass = self.cor_sub, self.mas_sub

        # Calculate x**2*m, y**2*m and z**2*m
        xyz_sq_m_sum = np.dot(np.transpose(xyz) ** 2, mass)

        # Extract x, y and z coordinates
        x, y, z = np.transpose(xyz)

        # Calculate tensor elements
        tensor_xx = np.take(xyz_sq_m_sum, 1) + np.take(xyz_sq_m_sum, 2)
        tensor_yy = np.take(xyz_sq_m_sum, 2) + np.take(xyz_sq_m_sum, 0)
        tensor_zz = np.take(xyz_sq_m_sum, 0) + np.take(xyz_sq_m_sum, 1)
        tensor_xy = - np.sum(np.multiply(x, y) * mass)
        tensor_xz = - np.sum(np.multiply(x, z) * mass)
        tensor_yz = - np.sum(np.multiply(y, z) * mass)

        # Construct full tensor
        inertia_tensor = np.array([[tensor_xx, tensor_xy, tensor_xz],
                                   [tensor_xy, tensor_yy, tensor_yz],
                                   [tensor_xz, tensor_yz, tensor_zz]], dtype=np.float)

        # Normalize tensor
        inertia_tensor /= np.max(inertia_tensor)

        if calc_for == 'molecule':

            self.ine_ten = inertia_tensor

        elif calc_for == 'substructure':

            self.ine_ten_sub = inertia_tensor

    def standard_orientation(self, logger, calc_for='molecule'):
        """ Orients the molecule/substructure in standard orientation, returns coordinates and rotation matrix """

        def _check_tensor():
            """ Subroutine to check and recalculate the inertia tensor if the PAs are not the x,y and z axes """

            # Principal axes not rotated properly
            if not np.asarray(np.diag(self.ine_pa) == np.ones(3, dtype=np.float)).all():
                
                X = np.linalg.solve(self.ine_pa, identity_matrix)
                self.cor = np.transpose(np.dot(X, np.transpose(self.cor)))
                self.calc_ine_tens(calc_for)
                self.ine_pc, self.ine_pa = np.linalg.eigh(self.ine_ten)

        # Define unity matrix and axes
        identity_matrix = np.identity(3, dtype=np.float)

        # Shift center of mass to cartesian origin
        self.calc_com(calc_for)
        self.shift_com(calc_for)

        # Calculate inertia tensor
        self.calc_ine_tens(calc_for)

        # Diagonalize tensor to obtain principal rotation axes and components (using linalg.eigh)
        principal_components, principal_axes = np.linalg.eigh(self.ine_ten)

        # Calculate exact rotation matrix that aligns principal axes with identity matrix and rotate molecule
        X = np.linalg.solve(principal_axes, identity_matrix)

        # Sort principal components and swap x,y,z columns according to their order
        if calc_for == 'molecule':

            self.cor = np.transpose(np.dot(X, np.transpose(self.cor)))
            self.cor = np.transpose(np.transpose(self.cor)[np.argsort(principal_components)])
            self.cor_std = np.transpose(np.transpose(self.cor_std)[np.argsort(principal_components)])
            self.calc_ine_tens(calc_for)
            self.ine_pc, self.ine_pa = np.linalg.eigh(self.ine_ten)
            _check_tensor()

        elif calc_for == 'substructure':

            self.cor_sub = np.transpose(np.dot(X, np.transpose(self.cor_sub)))
            self.cor_sub = np.transpose(np.transpose(self.cor_sub)[np.argsort(principal_components)])
            self.calc_ine_tens(calc_for)
            self.ine_pc_sub, self.ine_pa_sub = np.linalg.eigh(self.ine_ten_sub)

    def sigma_refl(self, ref_axis):
        """ Reflects molecule at the given axis """

        # z axis == reflects z values -> corresponds to xy plane
        # y axis == reflects y values -> corresponds to xz plane
        # x axis == reflects x values -> corresponds to yz plane

        # Ensure a normalized reflection axis
        ref_axis /= np.linalg.norm(np.array(ref_axis, dtype=np.float))

        # Extract x, y, and z coefficients
        a, b, c = ref_axis

        # Calculate reflection matrix
        ref_matrix = np.array([[1.0 - 2.0 * a ** 2, -2.0 * a * b, -2.0 * a * c],
                               [-2.0 * a * b, 1.0 - 2.0 * b ** 2, -2.0 * b * c],
                               [-2.0 * a * c, -2.0 * b * c, 1.0 - 2.0 * c ** 2]], dtype=np.float)

        self.symOPs.append(ref_matrix)
        self.cor = np.dot(self.cor, ref_matrix)
        self.show_mol = True

    def cn_rotation(self, rot_axis, n):
        """ Performs n-fold rotation around the given axis """

        # Rotation axis can for instance be obtained from the moment of inertia tensor

        # Calculate rotation angle resulting from n
        theta = 2.0 * (np.pi / int(n))

        # Ensure a normalized rotation axis
        rot_axis /= np.linalg.norm(np.array(rot_axis, dtype=np.float))

        # Extract x, y, and z coefficients
        a, b, c = rot_axis

        # Calculate matrix elements
        m11 = 1.0 * np.cos(theta) + a ** 2 * (1.0 - np.cos(theta))
        m12 = a * b * (1.0 - np.cos(theta)) - c * np.sin(theta)
        m13 = a * c * (1.0 - np.cos(theta)) + b * np.sin(theta)
        m21 = a * b * (1.0 - np.cos(theta)) + c * np.sin(theta)
        m22 = 1.0 * np.cos(theta) + b ** 2 * (1.0 - np.cos(theta))
        m23 = b * c * (1.0 - np.cos(theta)) - a * np.sin(theta)
        m31 = a * c * (1.0 - np.cos(theta)) - b * np.sin(theta)
        m32 = b * c * (1.0 - np.cos(theta)) + a * np.sin(theta)
        m33 = 1.0 * np.cos(theta) + c ** 2 * (1.0 - np.cos(theta))

        # Set up rotation matrix
        rot_matrix = np.array([[m11, m12, m13],
                               [m21, m22, m23],
                               [m31, m32, m33]], dtype=np.float)

        self.symOPs.append(rot_matrix)
        self.cor = np.dot(self.cor, rot_matrix)
        self.show_mol = True

    def inversion(self):
        """ Inverts molecule at origin """

        self.symOPs.append(-1.0)
        self.cor *= -1.0
        self.show_mol = True

    def calc_total_symOP(self):
        """ Calculates the combined total symmetry operation from all individual operations """

        for OP in self.symOPs:
            self.tot_symOP = np.dot(self.tot_symOP, OP)

    ###############################################################################
    # FUNCTIONS FOR FILE HANDLING
    ###############################################################################

    def get_export_file(self, logger, example, prefix=None):
        """ Gets a valid outfile name from the user """

        self.outfile_name = get_file_name(logger, example, prefix)

    def change_algorithms(self, logger):
        """ Enables the changing of matching/solving algorithms """

        choices = [0, 1, 2, 3, 4, 5, -1, -10]  # List of accepted choices

        # Dictionary of algorithm number and string correlations
        alg_number = {1: 'distance', 2: 'combined', 3: 'brute_force', 4: 'hungarian', 5: 'standard'}
        alg_string = {1: 'matching', 2: 'solving'}

        question = "\n>>  Enter your choice: "

        while True:

            logger.pt_algorithm_menu(self)
            operation = logger.get_menu_choice(choices, question, return_type='int')

            if operation == -10:  # Exit the menu

                break

            elif operation == -1:  # Info about current solving algorithm

                logger.pt_solve_algorithm_details()

            elif operation == 0:  # Info about current matching algorithm

                logger.pt_match_algorithm_details()

            elif operation in [1, 2, 3]:  # Matching algorithms

                if operation == 3:

                    logger.pt_future_implementation()  # Currently no brute force available

                else:

                    logger.match_alg = alg_number[operation]
                    logger.pt_change_algorithm(alg_string[1])

            elif operation in [4, 5]:  # Solving algorithms

                logger.match_solv = alg_number[operation]
                logger.pt_change_algorithm(alg_string[2])

            else:

                logger.pt_invalid_input()


###############################################################################
# KABSCH OBJECT
###############################################################################

class Kabsch(object):
    """ A Kabsch object used that contains all relevant properties of the matched molecules """

    def __init__(self, molecule1, molecule2, settings):
        """ Initializes a Kabsch object """

        self.name1 = molecule1.name  # Name of molecule 1
        self.name2 = molecule2.name  # Name of molecule 2
        self.n_atoms = molecule2.n_atoms  # Number of atoms (Ref. molecule2)
        self.sym = None  # Element symbols
        self.sym_mol1 = molecule1.sym  # Element symbols for molecule 1
        self.sym_mol2 = molecule2.sym  # Element symbols for molecule 2
        self.sym_idf_mol1 = molecule1.sym_idf  # Combined symbols/identifiers for molecule 1
        self.sym_idf_mol2 = molecule2.sym_idf  # Combined symbols/identifiers for molecule 2
        self.chg = molecule2.chg  # Nuclear charges (Ref. molecule2)
        self.chg_mol1 = molecule1.chg  # Nuclear charges for molecule 1
        self.chg_mol2 = molecule2.chg  # Nuclear charges for molecule 2
        self.n_hydro = len(np.where(self.chg == 1)[0])  # Number of hydrogen atoms (Ref. molecule2)
        self.idf = molecule2.idf  # Atomic identifiers
        self.sym_idf = None  # Combined symbols/identifiers
        self.cor_mol1 = None  # Original cart. coordinates
        self.cor_mol2 = None  # Original cart. coordinates
        self.cor_mol1_kbs = None  # Coordinates after Kabsch alignment (overwritten after alignment)
        self.cor_mol2_kbs = None  # Coordinates after Kabsch alignment (overwritten after alignment)
        self.cor_mol1_kbs_std = None  # Standard deviations for coordinates of molecule 1
        self.cor_mol2_kbs_std = None  # Standard deviations for coordinates of molecule 2
        self.has_kabsch = False  # If Kabsch algorithm has been used
        self.cor = molecule2.cor  # Coordinates for aRMSD plot (overwritten after alignment)
        self.cor_std = molecule2.cor_std  # Standard deviations for coordinates for aRMSD plot (overwritten after alignment)
        self.rad_cov_mol1 = molecule1.rad_cov  # Covalent radii for molecule 1
        self.rad_cov_mol2 = molecule2.rad_cov  # Covalent radii for molecule 2
        self.tot_symOP = molecule1.tot_symOP  # Total symOP applied on molecule 1 in the matching progress
        self.wts_type = 'none'  # Type of the weighting function
        self.calc_mcc = False  # If mcc contributions are to be calculated
        self.wts_mol1 = None  # Weights for molecule 1
        self.wts_mol2 = None  # Weights for molecule 2
        self.wts_comb = None  # Average weights
        self.rot_mat = None  # Kabsch rotation matrix
        self.tot_rot_mat = None  # Total rotation matrix
        self.at_types = None  # Atom types
        self.n_atom_types = None  # Number of different atom types
        self.occ = None  # Occurrences of atom types
        self.rmsd = None  # Weighted RMSD
        self.rmsd_idv = None  # RMSD per atom type
        self.msd = None  # Weighted MSD
        self.msd_sum = None  # Sum of weighted MSD
        self.r_sq = None  # R2 of superposition
        self.gard = None  # GARD score
        self.rad_plt_vtk = None  # aRMSD radii for vtk plotting
        self.col_at_mol1_rgb = None  # Color array for the atoms of molecule 1 [RGB]
        self.col_at_mol2_rgb = None  # Color array for the atoms of molecule 2 [RGB]
        self.col_bnd_mol1_rgb = None  # Color array for the bonds of molecule 1 [RGB]
        self.col_bnd_mol2_rgb = None  # Color array for the bonds of molecule 2 [RGB]
        self.col_at_hex = None  # Color array of the atoms in aRMSD representation [HEX]
        self.col_at_rgb = None  # Color array of the atoms in aRMSD representation [RGB]
        self.chd_bnd_col_rgb = None  # Colors of the changed bonds [RGB] 

        if molecule2.n_atoms >= molecule1.n_atoms:  # Use molecule with larger number of bonds

            self.bnd_idx = molecule2.bnd_idx  # Bond indices
            self.n_bonds = molecule2.n_bonds  # NUmber of bonds

        else:

            self.bnd_idx = molecule1.bnd_idx
            self.n_bonds = molecule1.n_bonds

        self.plt_col_aRMSD = None  # Differentiation colors for aRMSD representation [HEX]
        self.col_short_rgb = None  # Colors for bond intersections in RGB space, short
        self.col_long_rgb = None  # Colors for bond intersections in RGB space, long
        self.reg_bnd_idx = None  # Indices of bond indices with deviations < n
        self.chd_bnd_idx = None  # Indices of bond indices with deviations > n
        self.chd_bnd_col = None  # Colors of these bonds (smaller, larger)
        self.n_chd_bnd = 0  # Number of changed bonds
        self.col_bnd_hex = None  # aRMSD color array for the bonds [HEX]
        self.col_bnd_rgb = None  # aRMSD color array for the bonds [RGB]
        self.bnd_dis_mol1 = None  # Bond distances in molecule 1
        self.bnd_dis_mol2 = None  # Bond distances in molecule 2
        self.ang_idx_mol1 = None  # Angle indices for molecule 1
        self.ang_idx_mol2 = None  # Angle indices for molecule 2
        self.ang_idx = None  # Final angle indices
        self.n_angles = None  # Number of angles
        self.ang_deg_mol1 = None  # Angles in molecule 1
        self.ang_deg_mol2 = None  # Angles in molecule 2
        self.tor_idx_mol1 = None  # Dihedral angle indices for molecule 1
        self.tor_idx_mol2 = None  # Dihedral angle indices for molecule 2
        self.tor_idx = None  # Final dihedral angle indices
        self.n_torsions = None  # Number of dihedral angles
        self.tor_deg_mol1 = None  # Dihedral angles in molecule 1
        self.tor_deg_mol2 = None  # Dihedral angles in molecule 2
        self.z_matrix1 = []  # z-matrix for molecule 1
        self.z_matrix2 = []  # z-matrix for molecule 2
        self.rmsd_z_matrix = None  # RMSD of the z-matrices
        self.c_dis = None  # z-matrix RMSD, contributions of distances
        self.c_ang = None  # z-matrix RMSD, contributions of angles
        self.c_tor = None  # z-matrix RMSD, contributions of torsions

        self.has_stats = False  # If bond/angle/dihedral statistics exist
        self.use_std = settings.use_std  # If standard deviations are to be used in calculations

        self.list_of_picks = []  # List of picked events

        self.has_sub = False  # If a substructure has been defined
        self.has_sub_rmsd = False  # If RMSD/Superposition values were calculated
        self.pos_sub1 = None  # Positions of substructure 1
        self.pos_sub2 = None  # Positions of substructure 2
        self.rmsd_sub1 = None  # Weighted RMSD of substructure 1
        self.rmsd_sub2 = None  # Weighted RMSD of substructure 2
        self.c_sub1 = None  # MSD Contribution of substructure 1
        self.c_sub2 = None  # MSD Contribution of substructure 2
        self.cos_sim_sub1 = None  # Cosine similarity of substructure 1
        self.cos_sim_sub2 = None  # Cosine similarity of substructure 2

        self.interp_struct = None  # Coordinates of interpolated structures
        self.interp_rmsd = None  # RMSD values of interpolated structures

        self.has_cam_vtk = molecule2.has_cam_vtk  # If the molecule has an existing VTK camera orientation
        self.cam_vtk_pos = molecule2.cam_vtk_pos  # VTK camera position
        self.cam_vtk_wxyz = molecule2.cam_vtk_wxyz  # VTK camera angle/axis
        self.cam_vtk_focal_pt = molecule2.cam_vtk_focal_pt  # VTK camera focal point
        self.cam_vtk_view_up = molecule2.cam_vtk_view_up  # VTK camera view up (?)

        self.outfile_name = None  # Name of the outfile to which the structural data will be written

    def get_symbols(self):
        """ Gets the element/group symbols from both molecules of the object """

        pos = np.where(self.chg_mol1 != self.chg_mol2)[0]  # Check for groups
        self.sym = np.copy(self.sym_mol2)  # Copy symbols of molecule 2

        if len(pos) != 0:  # Update/Change symbols at respective positions

            group = np.take(pse_groups, self.chg_mol1[pos] - 1)
            self.sym[pos] = pse_group_sym[group - 1]

        self.get_sym_idf()  # Make symbol-identifier array

    def get_sym_idf(self):
        """ Combines symbols and identifiers """

        self.sym_idf = np.asarray([('%s-%d' % (self.sym[atom], self.idf[atom])) for atom in range(self.n_atoms)],
                                  dtype=np.object)

    def init_coords(self, molecule1, molecule2):
        """ Initializes coordinates and shifts geometric center to origin """

        # Shift geometric center to origin
        self.cor_mol1 = molecule1.cor - np.dot(np.transpose(molecule1.cor), np.repeat(1.0, self.n_atoms) / self.n_atoms)
        self.cor_mol2 = molecule2.cor - np.dot(np.transpose(molecule2.cor), np.repeat(1.0, self.n_atoms) / self.n_atoms)

        self.cor_mol1_kbs = np.copy(self.cor_mol1)
        self.cor_mol2_kbs = np.copy(self.cor_mol2)

        self.cor_mol1_kbs_std = molecule1.cor_std
        self.cor_mol2_kbs_std = molecule2.cor_std

    def add_bond(self, id1, id2):
        """ Adds a bond between two given atomic indices """

        # If the requested bond does not exist, add it and update bonds
        if not self.bond_exists(id1, id2)[0]:
            self.bnd_idx = np.vstack((self.bnd_idx, np.array([id1, id2])))
            self.update_bonds()

    def remove_bond(self, id1, id2):
        """ Removes a bond between two given atomic indices """

        is_bond, pos = self.bond_exists(id1, id2)

        if is_bond:
            self.bnd_idx = np.delete(self.bnd_idx, pos, 0)  # Delete the respective entry
            self.update_bonds()

    def bond_exists(self, id1, id2):
        """ Returns if a bond already exists between the two atom indices """

        def bond_pos(id1, id2):

            return np.where(np.sum(np.abs(self.bnd_idx - np.array([id1, id2])), axis=1) == 0)[0]

        pos_case1, pos_case2 = bond_pos(id1, id2), bond_pos(id2, id1)  # Try both combinations

        if len(pos_case1) != 0:

            return True, pos_case1

        elif len(pos_case2) != 0:

            return True, pos_case2

        else:

            return False, []

    def update_bonds(self):
        """ Updates bond information and recalculates distances """

        def bond_pos():
            bnd_pos = []

            for entry in self.bnd_types:
                bnd_pos.append([np.allclose(self.bnd_chars[pos], entry) for pos in range(self.n_bonds)])

            return np.asarray(bnd_pos)

        self.n_bonds = len(self.bnd_idx)  # Determine number of bonds
        self.bnd_idx = np.take(self.bnd_idx, np.argsort(np.transpose(self.bnd_idx)[1]), axis=0)

        id1, id2 = np.transpose(self.bnd_idx)  # Get indices and calculate distances

        self.bnd_dis_mol1 = np.sqrt(np.sum((np.take(self.cor_mol1_kbs, id1, axis=0) -
                                            np.take(self.cor_mol1_kbs, id2, axis=0)) ** 2, axis=1))

        self.bnd_dis_mol2 = np.sqrt(np.sum((np.take(self.cor_mol2_kbs, id1, axis=0) -
                                            np.take(self.cor_mol2_kbs, id2, axis=0)) ** 2, axis=1))

        # Determine different bond types, and positions of these bonds in the index array
        self.bnd_chars = np.take(self.chg, self.bnd_idx, axis=0)
        self.bnd_types = unique(self.bnd_chars)
        self.n_bnd_types = len(self.bnd_types)

        self.bnd_type_pos = bond_pos()

        # Create bond labels
        chg1, chg2 = np.transpose(self.bnd_types)
        self.bnd_label = np.asarray(pse_sym_chg[chg1 - 1], dtype=np.object) + "-" + np.asarray(pse_sym_chg[chg2 - 1],
                                                                                               dtype=np.object)

    def get_angle_idx(self, settings, calc_for='mol1'):
        """ Calculates all angle indices in a molecule based on bond indices """

        if calc_for == 'mol1':

            cor_xyz = self.cor_mol1_kbs
            rad_cov = self.rad_cov_mol1

        else:

            cor_xyz = self.cor_mol2_kbs
            rad_cov = self.rad_cov_mol2

        # Generate all angle combinations
        a1 = a2 = a3 = np.arange(self.n_atoms)

        # Use cartesian product to calculate all possible coordinate triples
        all_combinations = cartesian_product([a1, a2, a3])

        # Extract individual identifier arrays
        p1, p2, p3 = np.transpose(all_combinations)

        # Extract pair connectivity identifiers
        # pc1, pc2 = np.transpose(self.bnd_idx)

        # Determine all triples that have three different identifiers
        unique_positions = np.where((p1 != p2) & (p1 != p3) & (p2 != p3))[0]

        # Shorten initial combination array to all unique combination
        unique_combinations = all_combinations[unique_positions]

        # Extract respective identifier arrays
        xx, yy, zz = np.transpose(unique_combinations)

        # Calculate the distances between identifier pairs
        dist_pairs1 = np.sqrt(np.sum((cor_xyz[xx] - cor_xyz[yy]) ** 2, axis=1))
        dist_pairs2 = np.sqrt(np.sum((cor_xyz[yy] - cor_xyz[zz]) ** 2, axis=1))

        # Calculate the limit (sum of covalence radii times factor) of the associated bond type
        dist_limit1 = (np.take(rad_cov, xx) + np.take(rad_cov, yy)) * settings.det_bnd_fct
        dist_limit2 = (np.take(rad_cov, yy) + np.take(rad_cov, zz)) * settings.det_bnd_fct

        # Determine indices where both distances are below the sum of the radii
        indices = np.where((dist_pairs1 <= dist_limit1) & (dist_pairs2 <= dist_limit2))[0]

        # Determine all angle triples
        if calc_for == 'mol1':

            self.ang_idx_mol1 = unique_combinations[indices]

        else:

            self.ang_idx_mol2 = unique_combinations[indices]

    def calc_angles(self, settings, calc_for='mol1'):
        """ Calculates all angles associated with angle indices """

        if calc_for == 'mol1':

            cor_xyz = self.cor_mol1_kbs

        else:

            cor_xyz = self.cor_mol2_kbs

        # Extract respective identifier arrays
        a_xx, a_yy, a_zz = np.transpose(self.ang_idx)

        # Calculate bond vectors
        v1 = np.take(cor_xyz, a_xx, axis=0) - np.take(cor_xyz, a_yy, axis=0)
        v2 = np.take(cor_xyz, a_zz, axis=0) - np.take(cor_xyz, a_yy, axis=0)

        # Calculate product of lengths of bond vectors
        dv1_dot_dv2 = np.sqrt(np.sum(v1 ** 2, axis=1)) * np.sqrt(np.sum(v2 ** 2, axis=1))

        # Calculate the respective angles: np.einsum('ij,ij->i', v1, v2) is equivalent to row-wise
        # dot product ([np.dot(v1[entry], v2[entry]) for entry in range(len(v1))])
        if calc_for == 'mol1':

            self.ang_deg_mol1 = np.around(np.degrees(np.arccos(np.einsum('ij,ij->i', v1, v2) / dv1_dot_dv2)),
                                          settings.calc_prec_stats)

        else:

            self.ang_deg_mol2 = np.around(np.degrees(np.arccos(np.einsum('ij,ij->i', v1, v2) / dv1_dot_dv2)),
                                          settings.calc_prec_stats)

    def get_all_angles(self, settings):
        """ Determines all angle indices and angles for both molecules """

        self.get_angle_idx(settings, calc_for='mol1')
        self.get_angle_idx(settings, calc_for='mol2')

        if len(self.ang_idx_mol2) >= len(self.ang_idx_mol1):  # Use molecule with larger number of angles

            self.ang_idx = self.ang_idx_mol2

        else:

            self.ang_idx = self.ang_idx_mol1

        self.n_angles = len(self.ang_idx)
        self.calc_angles(settings, calc_for='mol1')
        self.calc_angles(settings, calc_for='mol2')

    def get_torsion_idx(self, settings, calc_for='mol1'):
        """ Calculates all torsion indices in a molecule based on angle indices """

        def tors_per_angle(triple, c_pc0, c_pc2):
            """ Function that returns all valid torsion combinations of current angle triple and candidates """

            # Combine entry combinations
            return np.hstack((np.ravel(np.asarray([np.hstack((pos, triple)) for pos in c_pc0 if pos not in triple],
                                                  dtype=np.int)),
                              np.ravel(np.asarray([np.hstack((triple, pos)) for pos in c_pc2 if pos not in triple],
                                                  dtype=np.int))))

        if calc_for == 'mol1':

            cor_xyz = self.cor_mol1_kbs
            rad_cov = self.rad_cov_mol1

        else:

            cor_xyz = self.cor_mol2_kbs
            rad_cov = self.rad_cov_mol2

        torsions = []  # Set up empty torsion list

        # Calculate all indices of atoms connected to each atom
        candidates = np.asarray([self.calc_coord_pos(settings, cor_xyz, rad_cov, atom) for atom in range(self.n_atoms)])

        # Calculate all torsion/dihedral indices
        [torsions.extend(tors_per_angle(triple, candidates[triple[0]], candidates[triple[1]]))
         for triple in self.ang_idx]

        if calc_for == 'mol1':

            # Transform list to array and reshape it
            self.tor_idx_mol1 = np.asarray(torsions, dtype=np.int).reshape((int(len(torsions) / 4.0), 4))

        else:

            # Transform list to array and reshape it
            self.tor_idx_mol2 = np.asarray(torsions, dtype=np.int).reshape((int(len(torsions) / 4.0), 4))

    def calc_coord_pos(self, settings, cor_xyz, rad_cov, pos):
        """ Calculates the positions of all adjacent atoms to the atoms of the given index """

        # Calculate the distances from all atoms, sort and take the lowest non-zero 'n_neighb' ones
        dist = np.sqrt(np.sum((cor_xyz - cor_xyz[pos]) ** 2, axis=1))

        # Calculate difference of radii and distance
        below_rad_lim = dist - (rad_cov + rad_cov[pos]) * settings.det_bnd_fct

        # Determine positions where the difference if < 0 (no self interaction)
        pos = np.where((below_rad_lim < 0.0) & (np.abs(below_rad_lim) < (rad_cov + rad_cov[pos])[pos]))[0]

        # Return value(s)
        return pos

    def get_all_torsions(self, settings):
        """ Determines all angle indices and angles for both molecules """

        self.get_torsion_idx(settings, calc_for='mol1')
        self.get_torsion_idx(settings, calc_for='mol2')

        if len(self.tor_idx_mol2) >= len(self.tor_idx_mol1):  # Use molecule with larger number of angles

            self.tor_idx = self.tor_idx_mol2

        else:

            self.tor_idx = self.tor_idx_mol1

        self.recalc_torsions(settings)  # Calculates torsion angles rounded to 'precision'

    def check_for_nan(self, settings):
        """ Checks for nan values in calculated angles and removes respective values """

        self.tor_idx = self.tor_idx[~np.isnan(self.tor_deg_mol1 * self.tor_deg_mol2)]
        self.tor_idx_mol1, self.tor_idx_mol2 = np.copy(self.tor_idx), np.copy(self.tor_idx)

        self.recalc_torsions(settings)

    def recalc_torsions(self, settings):
        """ (Re)calculates torsion angles """

        self.n_torsions = len(self.tor_idx)
        self.calc_torsions(settings, calc_for='mol1')
        self.calc_torsions(settings, calc_for='mol2')

        # candidates = np.where(np.abs(self.tor_deg_mol1 - self.tor_deg_mol2) > 10.0)[0]
        # t1, t2 = self.tor_deg_mol1[candidates], self.tor_deg_mol2[candidates]
        t1, t2 = np.copy(self.tor_deg_mol1), np.copy(self.tor_deg_mol2)
        invert_at = np.where(np.abs(t2 - t1) > np.abs((180.0 - t2) - t1))[0]

        t1[invert_at] -= 180.0
        self.tor_deg_mol1 = np.abs(t1)

    def calc_torsions(self, settings, calc_for='mol1'):
        """ Calculates all dihedral angles of the given atomic indices """

        def calc_entry(coords):
            """ Calculates a single entry """

            b = coords[:-1] - coords[1:]
            b[0] *= -1.0
            v = np.asarray([v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]], dtype=np.float)

            # Normalize vectors
            v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)

            b1 = b[1] / np.linalg.norm(b[1])
            x = np.dot(v[0], v[1])
            m = np.cross(v[0], b1)
            y = np.dot(m, v[1])

            return np.around(np.abs(np.degrees(np.arctan2(y, x))), settings.calc_prec_stats)

        # Calculate the respective torsions for all torsion indices
        if calc_for == 'mol1':

            self.tor_deg_mol1 = np.asarray([calc_entry(np.take(self.cor_mol1_kbs, np.take(self.tor_idx, entry, axis=0),
                                                               axis=0)) for entry in range(self.n_torsions)])

        else:

            self.tor_deg_mol2 = np.asarray([calc_entry(np.take(self.cor_mol2_kbs, np.take(self.tor_idx, entry, axis=0),
                                                               axis=0)) for entry in range(self.n_torsions)])

    def get_max_diff_prop(self, settings, prop='distance'):
        """ Returns the positions in the property arrays with highest intermolecular deviations """

        if prop == 'distance':

            quant1, quant2 = self.bnd_dis_mol1, self.bnd_dis_mol2
            indices = self.bnd_idx

        elif prop == 'angle':

            quant1, quant2 = self.ang_deg_mol1, self.ang_deg_mol2
            indices = self.ang_idx

        elif prop == 'torsion':

            quant1, quant2 = self.tor_deg_mol1, self.tor_deg_mol2
            indices = self.tor_idx

        # Determine the positions of maximum deviations (n_max_dev) and the respective values
        pos = np.argsort(np.abs(np.abs(quant1) - np.abs(quant2)))[::-1][:settings.n_max_diff]
        val_mol1, val_mol2 = np.around(np.take(quant1, pos), settings.calc_prec_stats), \
                             np.around(np.take(quant2, pos), settings.calc_prec_stats)
        delta = val_mol2 - val_mol1

        # Get the atomic indices and allocate the sym_idf information
        idx = np.take(indices, pos, axis=0)
        sym_idf = np.take(self.sym_idf, idx)
        descriptor = ['[' + ', '.join(entry) + ']' for entry in sym_idf]  # Combines sym_idf information

        # Return result
        return descriptor, val_mol1, val_mol2, delta

    def calc_z_matrix_rmsd(self):
        """ Calculates the z-matrix RMSD and the contributions of distances, angles and torsions """

        self.z_matrix1, self.z_matrix2 = [], []  # Delete existing matrices

        self.get_z_matrices()  # Get z-matrices

        msd_z_dis, msd_z_ang, msd_z_tor = np.transpose((self.z_matrix1 - self.z_matrix2) ** 2)

        # Normalize delta MSD arrays
        msd_z_dis /= np.max(msd_z_dis)
        msd_z_ang /= np.max(msd_z_ang)
        msd_z_tor /= np.max(msd_z_tor)

        n_dis, n_ang, n_tor = self.n_atoms - 1, self.n_atoms - 2, self.n_atoms - 3
        n_total = n_dis + n_ang + n_tor

        # Calculate RMSD value of the z-matrix
        self.rmsd_z_matrix = ((np.sum(msd_z_dis) + np.sum(msd_z_ang) + np.sum(msd_z_tor)) / n_total) ** 0.5

        # Contributions of distances, angles and torsions
        self.c_dis = np.sum(msd_z_dis) / (np.sum(msd_z_dis) + np.sum(msd_z_ang) + np.sum(msd_z_tor))
        self.c_ang = np.sum(msd_z_ang) / (np.sum(msd_z_dis) + np.sum(msd_z_ang) + np.sum(msd_z_tor))
        self.c_tor = np.sum(msd_z_tor) / (np.sum(msd_z_dis) + np.sum(msd_z_ang) + np.sum(msd_z_tor))

    def get_z_matrices(self):
        """ Gets the z-matrices for both molecules from the Cart. coordinates """

        self.make_z_matrix(calc_for='mol1')
        self.make_z_matrix(calc_for='mol2')

        self.z_matrix1 = np.asarray(self.z_matrix1)
        self.z_matrix2 = np.asarray(self.z_matrix2)

    def make_z_matrix(self, calc_for='mol1'):
        """ Generates a z-matrix from Cartesian coordinates:
            Entries are in order of the element symbols """

        if calc_for == 'mol1':

            z_mat = self.z_matrix1

            if self.use_std:

                cor = unp.uarray(self.cor_mol1_kbs, self.cor_mol1_kbs_std)

            else:

                cor = self.cor_mol1_kbs

        else:

            z_mat = self.z_matrix2

            if self.use_std:

                cor = unp.uarray(self.cor_mol2_kbs, self.cor_mol2_kbs_std)

            else:

                cor = self.cor_mol2_kbs

        self.add_initial_atoms_zmatrix(z_mat, calc_for)

        [self.add_atom_to_zmatrix(z_mat, cor[atom], calc_for) for atom in range(3, self.n_atoms)]

    def add_initial_atoms_zmatrix(self, z_mat, calc_for):
        """ Adds the first atoms to start the z-matrix """

        if calc_for == 'mol1':

            if self.use_std:

                pos1, pos2, pos3 = unp.uarray(self.cor_mol1_kbs[:3], self.cor_mol1_kbs_std[:3])

            else:

                pos1, pos2, pos3 = self.cor_mol1_kbs[:3]

        else:

            if self.use_std:

                pos1, pos2, pos3 = unp.uarray(self.cor_mol2_kbs[:3], self.cor_mol2_kbs_std[:3])

            else:

                pos1, pos2, pos3 = self.cor_mol2_kbs[:3]

        if self.use_std:

            # First atom
            z_mat.append(np.asarray([ufloat(0.0, 0.0), ufloat(0.0, 0.0), ufloat(0.0, 0.0)]))

            # Second atom
            z_mat.append(np.asarray([geo_distance(pos1, pos2), ufloat(0.0, 0.0), ufloat(0.0, 0.0)]))

            # Third atom
            z_mat.append(np.asarray([geo_distance(pos1, pos3), geo_angle(pos1, pos2, pos3), ufloat(0.0, 0.0)]))

        else:

            # First atom
            z_mat.append(np.asarray([0.0, 0.0, 0.0]))

            # Second atom
            z_mat.append(np.asarray([geo_distance(pos1, pos2), 0.0, 0.0]))

            # Third atom
            z_mat.append(np.asarray([geo_distance(pos1, pos3), geo_angle(pos1, pos2, pos3), 0.0]))

    def add_atom_to_zmatrix(self, z_mat, cor, calc_for):
        """ Adds remaining atoms to z-matrix """

        if calc_for == 'mol1':

            if self.use_std:

                pos1, pos2, pos3 = unp.uarray(self.cor_mol1_kbs[:3], self.cor_mol1_kbs_std[:3])

            else:

                pos1, pos2, pos3 = self.cor_mol1_kbs[:3]

        else:

            if self.use_std:

                pos1, pos2, pos3 = unp.uarray(self.cor_mol2_kbs[:3], self.cor_mol2_kbs_std[:3])

            else:

                pos1, pos2, pos3 = self.cor_mol2_kbs[:3]

        # Add z-matrix entry
        z_mat.append(np.asarray([geo_distance(pos1, cor),
                                 geo_angle(pos1, pos2, cor),
                                 geo_torsion(pos1, pos2, pos3, cor)]))

    def check_substructure(self, logger, pos_sub1, pos_sub2):
        """ Checks if proper substructures were defined """

        if len(pos_sub1) >= 2 and len(pos_sub2) >= 2:  # Each substructure should at least contain two atoms

            self.has_sub = True
            self.pos_sub1 = pos_sub1
            self.pos_sub2 = pos_sub2

        else:

            logger.pt_no_proper_rmsd_sub()

    def get_wigner_seitz_radii(self, calc_for='mol1'):
        """ Calculate Wigner-Seitz radii from nuclear charges """

        if calc_for == 'mol1':

            chg = self.chg_mol1

        else:

            chg = self.chg_mol2

        # Wigner-Seitz Radius in A
        w_s_r = (((3.0 * chg) /
                  (4.0 * np.pi * np.take(pse_mass_dens, chg - 1) * NA)) ** (1.0 / 3.0) * 0.01) / 1.0E-10

        # Return result(s)
        return w_s_r

    def get_xsf_stored(self, logger, charge, xsf_type='MoKa'):

        # Set up dictionary of the prestored scattering factors
        xsf_dict = {'MoKa': (pse_mo_xsf_1, pse_mo_xsf_2), 'CuKa': (pse_cu_xsf_1, pse_cu_xsf_2),
                    'CoKa': (pse_co_xsf_1, pse_co_xsf_2), 'FeKa': (pse_fe_xsf_1, pse_fe_xsf_2),
                    'CrKa': (pse_cr_xsf_1, pse_cr_xsf_2)}

        if not xsf_type in ['MoKa', 'CuKa', 'CoKa', 'FeKa', 'CrKa']:  # Check for valid user input

            logger.pt_xsf_wrong_source()
            xsf_type = 'MoKa'

        # Get scattering factors from nuclear charge
        chosen_xsf_1, chosen_xsf_2 = xsf_dict[xsf_type]
        xsf1, xsf2 = np.take(chosen_xsf_1, charge - 1), np.take(chosen_xsf_2, charge - 1)

        # Return value(s)
        return xsf1, xsf2

    def get_xsf_from_file(self, logger, xsf_type='MoKa', calc_for='mol1'):
        """ Returns interpolated scattering factors from file or default values """

        def get_xsf_atom_type(self, logger, charge, xsf_type):
            """ Returns scattering factors for a given atom type (charge) and energy """

            # X-ray energies are listed in: http://xdb.lbl.gov/Section1/Table_1-2.pdf

            symbol = pse_sym_chg[charge - 1]  # Get symbol from given charge

            try:

                # Determine absolute file path (assume all .nff files are located in the 'xsf' subdirectory)
                filepath = os.path.join(os.environ.get('_MEIPASS2', os.path.abspath('.')), 'xsf',
                                        str(symbol) + str('.nff'))

                # Read in data using optimized Numpy routine: skip first row and use all three columns
                content = np.loadtxt(filepath, dtype=np.float, skiprows=1, usecols=(0, 1, 2))

                # Find nearest scattering factors by interpolation
                xsf1 = np.interp(logger.xfs_energy, np.transpose(content)[0], np.transpose(content)[1])
                xsf2 = np.interp(logger.xfs_energy, np.transpose(content)[0], np.transpose(content)[2])
                success = 1.0

            except (IOError, NameError, SyntaxError, ValueError):

                # Use prestored xsf factors given by 'xsf_type'
                xsf1, xsf2 = self.get_xsf_stored(logger, charge, xsf_type)
                success = 0.0

            return xsf1, xsf2, success

        if calc_for == 'mol1':

            chg = self.chg_mol1

        else:

            chg = self.chg_mol2

        # Determine atom types
        charge_set = np.unique(chg)
        n_element_types = len(charge_set)

        # Read energies and both scattering factors for element types from file
        xsf1_type, xsf2_type, success = np.transpose(np.array([get_xsf_atom_type(self, logger, charge, xsf_type)
                                                               for charge in charge_set], dtype=np.float))

        if 0.0 in success:
            logger.pt_xsf_import_error()  # Track error

        # Set up empty arrays for all supported elements ('H' to 'Rn')
        max_element = 86
        xsf1_elements, xsf2_elements = np.zeros(max_element), np.zeros(max_element)

        # Iterate through all elements types
        for element_type in range(n_element_types):
            # Write values at the atomic position in empty arrays
            xsf1_elements[charge_set[element_type] - 1] = xsf1_type[element_type]
            xsf2_elements[charge_set[element_type] - 1] = xsf2_type[element_type]

        # Generate arrays in accordance with 'element_charge'
        xsf1, xsf2 = np.take(xsf1_elements, self.chg - 1), np.take(xsf2_elements, self.chg - 1)

        return xsf1, xsf2

    def get_weights(self, logger):
        """ Generates normalized weighting functions for Kabsch alignment"""

        def calc_densities(chg, rad):
            """ Calculates a density """

            return chg / (4.0 / 3.0 * np.pi * rad ** 3)

        def calc_mcc(quant, xyz):
            """ Calculates the contributions from all other centers """

            return np.sum([quant[entry] * np.exp(-np.sqrt(np.sum((xyz - xyz[entry]) ** 2, axis=1)))
                           for entry in range(self.n_atoms)], axis=0)

        if self.wts_type == 'none':

            self.wts_mol1 = np.ones(self.n_atoms, dtype=np.float)
            self.wts_mol2 = np.copy(self.wts_mol1)

        elif self.wts_type == 'mass':

            self.wts_mol1 = np.take(pse_mass, self.chg_mol1 - 1)
            self.wts_mol2 = np.take(pse_mass, self.chg_mol2 - 1)

        elif self.wts_type == 'n_electrons':

            self.wts_mol1 = np.asarray(self.chg_mol1, dtype=np.float)
            self.wts_mol2 = np.asarray(self.chg_mol2, dtype=np.float)

        elif self.wts_type == 'n_core_electrons':

            self.wts_mol1 = np.asarray([get_core_el(charge) for charge in self.chg_mol1], dtype=np.float)
            self.wts_mol2 = np.asarray([get_core_el(charge) for charge in self.chg_mol2], dtype=np.float)

        elif self.wts_type == 'rho_sph':

            dens_mol1 = calc_densities(self.chg_mol1, self.rad_cov_mol1)
            dens_mol2 = calc_densities(self.chg_mol2, self.rad_cov_mol2)

            if self.calc_mcc:

                self.wts_mol1 = calc_mcc(dens_mol1, self.cor_mol1)
                self.wts_mol2 = calc_mcc(dens_mol2, self.cor_mol2)

            else:

                self.wts_mol1 = dens_mol1
                self.wts_mol2 = dens_mol2

        elif self.wts_type == 'rho_lda':

            # Calculate Wigner-Seitz radii and densities
            ws_radii_mol1 = self.get_wigner_seitz_radii(calc_for='mol1')
            ws_radii_mol2 = self.get_wigner_seitz_radii(calc_for='mol2')

            dens_mol1 = calc_densities(self.chg_mol1, ws_radii_mol1)
            dens_mol2 = calc_densities(self.chg_mol2, ws_radii_mol2)

            if self.calc_mcc:

                self.wts_mol1 = calc_mcc(dens_mol1, self.cor_mol1)
                self.wts_mol2 = calc_mcc(dens_mol2, self.cor_mol2)

            else:

                self.wts_mol1 = dens_mol1
                self.wts_mol2 = dens_mol2

        elif self.wts_type == 'xsf_sq':

            xsf1_mol1, xsf2_mol1 = self.get_xsf_from_file(logger, xsf_type='MoKa', calc_for='mol1')
            xsf1_mol2, xsf2_mol2 = self.get_xsf_from_file(logger, xsf_type='MoKa', calc_for='mol2')

            if self.calc_mcc:

                self.wts_mol1 = calc_mcc(xsf1_mol1 + xsf2_mol1, self.cor_mol1) ** 2
                self.wts_mol2 = calc_mcc(xsf1_mol2 + xsf2_mol2, self.cor_mol2) ** 2

            else:

                self.wts_mol1 = (xsf1_mol1 + xsf2_mol1) ** 2
                self.wts_mol2 = (xsf1_mol2 + xsf2_mol2) ** 2

                # Normalize weight functions
        self.wts_mol1 = self.wts_mol1 / np.sum(self.wts_mol1) * len(self.wts_mol1)
        self.wts_mol2 = self.wts_mol2 / np.sum(self.wts_mol2) * len(self.wts_mol2)

        self.wts_comb = (self.wts_mol1 + self.wts_mol2) / 2.0

    def get_w_function(self, logger):
        """ Gets a valid weighting function from the user """

        choices = [0, 1, 2, 3, 4, 5, 6, -10]  # List of accepted choices

        wf_number = {0: 'none', 1: 'xsf_sq', 2: 'mass', 3: 'n_electrons',
                     4: 'n_core_electrons', 5: 'rho_sph', 6: 'rho_lda'}  # WF types as dict

        question = "\n>>  Enter your choice: "

        while True:

            logger.pt_w_function_menu(self)
            wf_type = logger.get_menu_choice(choices, question, return_type='int')

            if wf_type == -10:

                break

            elif wf_type in [0, 1, 2, 3, 4, 5, 6]:  # Set WF accordingly

                if wf_type in [1, 5, 6]:  # Ask user for MCC

                    if wf_type == 1:
                        logger.xfs_energy = eval(
                            input("\nEnter energy of X-ray radiation in eV (e.g. 17479.34 for MoK(alpha)): "))

                    self.calc_mcc = logger.get_menu_choice([True, False],
                                                           question="\n>>  Calculate mcc contributions (True / False): ",
                                                           return_type='bool')

                else:

                    self.calc_mcc = False

                self.wts_type = wf_number[wf_type]

            else:

                logger.pt_invalid_input()

    def full_rmsd(self, logger):
        """ Calculates full RMSD with decomposition of two arrays with identical coordinate order """

        def _prep_sub_data(calc_for='sub1'):
            """ Prepare data to include standard deviations if uncertainties are available """

            if calc_for == 'sub1':

                pos = self.pos_sub1

            else:

                pos = self.pos_sub2

            if logger.has_uc and logger.use_std:  # Combine data as array

                return unp.uarray(np.take(self.cor_mol1_kbs, pos, axis=0),
                                  np.take(self.cor_mol1_kbs_std, pos, axis=0)), \
                       unp.uarray(np.take(self.cor_mol2_kbs, pos, axis=0),
                                  np.take(self.cor_mol2_kbs_std, pos, axis=0))

            else:  # Return the nominal coordinates

                return np.take(self.cor_mol1_kbs, pos, axis=0), np.take(self.cor_mol2_kbs, pos, axis=0)

        def _prep_mol_data(calc_for='mol1'):
            """ Prepare data to include standard deviations if uncertainties are available """

            if calc_for == 'mol1':

                coords, stds = self.cor_mol1_kbs, self.cor_mol1_kbs_std

            else:

                coords, stds = self.cor_mol2_kbs, self.cor_mol2_kbs_std

            return unp.uarray(coords, stds) if logger.has_uc and logger.use_std else coords

        self.at_types = np.unique(self.sym)  # Determine atom types (unique elements of 'element symbol' array)
        self.n_atom_types = len(self.at_types)  # Determine number of atom types

        # Determine occurrences and positions of element types
        pos = np.asarray([np.where(self.sym == self.at_types[entry])[0] for entry in range(self.n_atom_types)])
        self.occ = np.asarray([len(entry) for entry in pos])

        # Handle uncertainties
        xyz1, xyz2 = _prep_mol_data(calc_for='mol1'), _prep_mol_data(calc_for='mol2')

        # Calculate atomic MSD (wx**2, wy**2, wz**2)
        self.msd = (xyz1 - xyz2) ** 2 * np.reshape(self.wts_comb, (-1, 1))

        # Sum up atomic MSD values (wx**2 + wy**2 + wz**2)
        self.msd_sum = np.sum(self.msd, axis=1)

        # Calculate RMSD ((MSD/wts)**0.5)
        self.rmsd = np.sum(self.msd_sum / np.sum(self.wts_comb)) ** 0.5
        
        # RMSD values for all different atom types combined
        self.rmsd_idv = np.asarray([(np.sum(np.take(self.msd_sum, entry) /
                                            np.sum(np.take(self.wts_comb, entry)))) ** 0.5 for entry in pos])

        # --------- Calculate similarity descriptors

        # Calculate 'cosine similarity' of the superposition
        a, b = np.sum(xyz1, axis=1), np.sum(xyz2, axis=1)

        self.cos_sim = np.dot(a, b) / (np.sum(a ** 2) ** 0.5 * np.sum(b ** 2) ** 0.5)

        # Calculate 'R^2' of the superposition
        SStot, SSres = np.sum((xyz1 - xyz1.mean()) ** 2), np.sum((xyz1 - xyz2) ** 2)

        self.r_sq = 1.0 - (SSres / SStot)

        # Calculate GARD
        gard_val = np.ones(self.n_atoms, dtype=np.float)

        if logger.has_uc and logger.use_std:  # Change to uarray in case of uncertainties

            gard_val = unp.uarray(np.ones(self.n_atoms), np.zeros(self.n_atoms))

        d = (np.sum((xyz1 - xyz2) ** 2, axis=1)) ** 0.5
        pos_mid = np.where((d >= logger.d_min) & (d <= logger.d_max))[0]
        pos_max = np.where(d >= logger.d_max)[0]
        gard_val[pos_mid] = (d[pos_mid] - logger.d_min) / (logger.d_max - logger.d_min)
        gard_val[pos_max] = 0.0

        self.gard = np.sum(gard_val * self.wts_comb) / np.sum(self.wts_comb)

        # --------- Evaluate substructures

        if self.has_sub:  # Check for substructures

            wts_sub1, wts_sub2 = np.take(self.wts_comb, self.pos_sub1), np.take(self.wts_comb, self.pos_sub2)

            # Handle uncertainties
            mol1_sub1, mol2_sub1 = _prep_sub_data(calc_for='sub1')
            mol1_sub2, mol2_sub2 = _prep_sub_data(calc_for='sub2')

            msd_sum_sub1 = np.sum((mol1_sub1 - mol2_sub1) ** 2 * np.reshape(wts_sub1, (-1, 1)), axis=1)
            msd_sum_sub2 = np.sum((mol1_sub2 - mol2_sub2) ** 2 * np.reshape(wts_sub2, (-1, 1)), axis=1)

            self.rmsd_sub1 = (np.sum(msd_sum_sub1 / np.sum(wts_sub1))) ** 0.5
            self.rmsd_sub2 = (np.sum(msd_sum_sub2 / np.sum(wts_sub2))) ** 0.5

            # MSD can be written as a linear combination of the two substructures
            # Contributions are given by:
            # msd_sub1 * x + msd_sub2 * (1.0 - x) = msd
            # x = (msd - msd_sub2) / (msd_sub1 - msd_sub2)
            self.c_sub1 = (self.rmsd ** 2 - self.rmsd_sub2 ** 2) / (self.rmsd_sub1 ** 2 - self.rmsd_sub2 ** 2)
            self.c_sub2 = 1.0 - self.c_sub1

            # 'Cosine similarities'
            a1, b1 = np.sum(mol1_sub1, axis=1), np.sum(mol2_sub1, axis=1)
            a2, b2 = np.sum(mol1_sub2, axis=1), np.sum(mol2_sub2, axis=1)

            self.cos_sim_sub1 = np.dot(a1, b1) / (np.sum(a1 ** 2) ** 0.5 * np.sum(b1 ** 2) ** 0.5)
            self.cos_sim_sub2 = np.dot(a2, b2) / (np.sum(a2 ** 2) ** 0.5 * np.sum(b2 ** 2) ** 0.5)

            # Superposition 'R^2'
            SStot_sub1 = np.sum((mol1_sub1 - mol1_sub1.mean()) ** 2)
            SSres_sub1 = np.sum((mol1_sub1 - mol2_sub1) ** 2)

            SStot_sub2 = np.sum((mol1_sub2 - mol1_sub2.mean()) ** 2)
            SSres_sub2 = np.sum((mol1_sub2 - mol2_sub2) ** 2)

            self.r_sq_sub1 = 1.0 - (SSres_sub1 / SStot_sub1)
            self.r_sq_sub2 = 1.0 - (SSres_sub2 / SStot_sub2)

            self.has_sub_rmsd = True

    def kabsch_algorithm(self, logger):
        """ Computes the optimal rotation matrix, performs alignment and returns coordinates and rotation matrix """

        self.kabsch_rot_matrix()  # Perform Kabsch algorithm for weighted coordinates
        self.has_kabsch = True
        self.full_rmsd(logger)  # Calculate RMSD and similarity descriptors

    def kabsch_rot_matrix(self):
        """ Computes the optimal rotation matrix and aligns P with Q """

        # Literature
        # http://scripts.iucr.org/cgi-bin/paper?S0567739476001873
        # http://scripts.iucr.org/cgi-bin/paper?S0567739478001680

        # Calculate the contributions of the combined weighting function to the center of masses
        ma = np.dot(self.wts_comb, self.cor_mol1) / np.sum(self.wts_comb)
        mb = np.dot(self.wts_comb, self.cor_mol2) / np.sum(self.wts_comb)

        # Correct the center of masses wrt weights
        self.cor_mol1_kbs = self.cor_mol1 - ma
        self.cor_mol2_kbs = self.cor_mol2 - mb

        # Reshape the functions for multiplication
        wf_reshape = self.wts_comb.reshape((-1, 1))

        # Calculate the covariance matrix
        A = np.dot(np.transpose(self.cor_mol2_kbs * wf_reshape), self.cor_mol1_kbs * wf_reshape)

        # Perform the SVD of A
        V, S, Wt = np.linalg.svd(A)

        # Calculate transposed matrices
        Vt = np.transpose(V)
        W = np.transpose(Wt)

        # Check if reflection for right-handed coordinate system is needed (d = sign(det(dot(W Vt))))
        reflect_yn = np.sign(np.linalg.det(np.dot(W, Vt)))

        if reflect_yn == -1.0:  # Reflect if sign is negative

            S[-1] = -S[-1]  # Change sign of last value in S

            V[:, -1] = -V[:, -1]  # Change respective signs in V

            Vt = np.transpose(V)  # Update Vt

        # Calculate optimal rotation matrix U for weighted coordinate overlap
        self.rot_mat = np.dot(W, Vt)
        self.tot_rot_mat = np.dot(self.tot_symOP, self.rot_mat)

        # Rotate first coordinate array
        self.cor_mol1_kbs = np.dot(self.cor_mol1_kbs, self.rot_mat)

        self.cor = self.cor_mol2  # Use molecule 2 as reference for subsequent plots

    def set_colors(self, settings):
        """ Sets RGB colors for the superposition plot """

        self.col_at_mol1_rgb = np.transpose(np.repeat(settings.col_model_fin_rgb,
                                                      self.n_atoms).reshape((3, self.n_atoms)))
        self.col_at_mol2_rgb = np.transpose(np.repeat(settings.col_refer_fin_rgb,
                                                      self.n_atoms).reshape((3, self.n_atoms)))

        self.col_bnd_mol1_rgb = np.transpose(np.repeat(settings.col_model_fin_rgb,
                                                       self.n_bonds).reshape((3, self.n_bonds)))
        self.col_bnd_mol2_rgb = np.transpose(np.repeat(settings.col_refer_fin_rgb,
                                                       self.n_bonds).reshape((3, self.n_bonds)))

        self.col_glob_rgb = settings.col_refer_fin_rgb  # Global variable

    def clean_aRMSD_rep(self):
        """ Resets the aRMSD colors and radii """

        self.plt_col_aRMSD = None  # Differentiation colors for aRMSD representation [HEX]
        self.col_short_rgb = None  # Colors for bond intersections in RGB space, short
        self.col_long_rgb = None  # Colors for bond intersections in RGB space, long
        self.reg_bnd_idx = None  # Indices of bond indices with deviations < n
        self.chd_bnd_idx = None  # Indices of bond indices with deviations > n
        self.chd_bnd_col = None  # Colors of these bonds (smaller, larger)
        self.chd_bnd_col_hex = None  # Color array for the changed bonds [HEX]
        self.chd_bnd_col_rgb = None  # Color array for the changed bonds [RGB]
        self.col_bnd_hex = None  # aRMSD color array for the bonds [HEX]
        self.col_bnd_rgb = None  # aRMSD color array for the bonds [RGB]

    def get_aRMSD_rep(self, settings, project_rad=False):
        """ Uses the RMSD to generate aRMSD colors and radii """

        # Calculate atomic RMSDs (distances between the corresponding atoms)
        RMSD_per_atom = np.sqrt(unp.nominal_values(self.msd_sum))

        # Check the deviation distribution between the two molecules and determine
        # radii (proportional to relative deviation)
        if np.sum(RMSD_per_atom) < settings.eps:

            # Use smaller sphere sizes
            radii = np.ones(self.n_atoms, dtype=np.float) * 1500

        else:

            RMSD_per_atom /= np.max(RMSD_per_atom)  # Normalize result

            if project_rad:  # Project radii if requested

                RMSD_per_atom = project_radii(RMSD_per_atom, 1000, 0.02, 1.0)

            # Check if RMSD is rather small in the molecule
            if np.max(RMSD_per_atom) <= settings.id_lim:

                radii = RMSD_per_atom ** 0.5 * 1000

            else:

                radii = RMSD_per_atom * 1800

        # Recalculate RMSD per atom in case it got normalized
        RMSD_per_atom = np.sqrt(unp.nominal_values(self.msd_sum))

        # Project radii
        self.rad_plt_vtk = project_radii(radii, 1000, 0.75, 4.0)

        if settings.use_aRMSD_col:  # If aRMSD colors are requested

            # Make the plotting colors from green to red that are used to indicate quality of atomic alignment
            self.plt_col_aRMSD = make_color_scale(settings.n_col_aRMSD, as_rgb=False)

            # Generate spacing list [0.0, ..., max_RMSD_diff] with length of 'n_col_aRMSD' based on length of color array
            diff_col = np.linspace(0.0, settings.max_RMSD_diff, settings.n_col_aRMSD, endpoint=True, dtype=np.float)

            # Find best matching color in the defined HTML colors based on "RMSD_per_atom" values
            # Use: "atomic_RMSD_norm" for relative deviation (1.0 to 0.0) or "RMSD_per_atom" for an absolute deviation
            col_pos = np.array([np.argmin(np.abs(diff_col - RMSD_per_atom[atom]))
                                for atom in range(self.n_atoms)], dtype=np.int)

            self.col_at_hex = np.take(self.plt_col_aRMSD, col_pos)

        else:  # Color by element default

            self.col_at_hex = np.take(pse_colors, self.chg - 1)

        # Create color_arrays in [HEX] and [RGB] space
        self.col_at_rgb = np.array([hex2rgb(color_hex, normalize=True) for color_hex in self.col_at_hex])
        self.col_bnd_hex = np.repeat(settings.col_bnd_glob_hex, self.n_bonds)
        self.col_bnd_rgb = np.transpose(np.repeat(settings.col_bnd_glob_rgb, self.n_bonds).reshape((3, self.n_bonds)))

        self.chd_bnd_col_rgb = np.copy(self.col_bnd_rgb)

        # Calculate bond length differences
        bond_lengths_diff = self.bnd_dis_mol1 - self.bnd_dis_mol2

        # Determine positions of shorter and longer bonds in structure 1 with respect to structure 2
        pos_changed_bonds = []

        if settings.thresh != 0.0:
            pos_changed_bonds = np.where(np.abs(bond_lengths_diff) >= settings.thresh)[0]
            self.reg_bnd_idx = np.where(np.abs(bond_lengths_diff) < settings.thresh)[0]

        # If there are shorter and/or longer bonds
        if len(pos_changed_bonds) != 0:
            # Determine positions of longer and shorted bonds
            pos_longer_bonds = np.where((bond_lengths_diff > 0.0) & (np.abs(bond_lengths_diff) > settings.thresh))[0]
            pos_shorter_bonds = np.where((bond_lengths_diff < 0.0) & (np.abs(bond_lengths_diff) > settings.thresh))[0]

            self.chd_bnd_col_rgb[pos_longer_bonds] = settings.col_long_rgb
            self.chd_bnd_col_rgb[pos_shorter_bonds] = settings.col_short_rgb

            # Combine indices of shorter and longer bonds in this sequence
            self.chd_bnd_idx = np.sort(np.hstack((pos_shorter_bonds, pos_longer_bonds)))
            self.n_chd_bnd = len(self.chd_bnd_idx)

            # Set up color array
            self.chd_bnd_col_hex = np.hstack((np.repeat(settings.col_short_hex, len(pos_shorter_bonds)),
                                              np.repeat(settings.col_long_hex, len(pos_longer_bonds))))

    def interpolate_structures(self, settings, substructure):
        """ Returns a list of linear interpolated structures between the two molecules or a substructure """

        if len(substructure) != self.n_atoms:

            # Set up zero array, determine substructure (default: all atoms) and create diff array
            diff = np.zeros((np.shape(self.cor_mol2_kbs)))

            diff[substructure] = np.take((self.cor_mol2_kbs - self.cor_mol1_kbs) / settings.n_steps_interp,
                                         substructure, axis=0)

        else:

            diff = (self.cor_mol2_kbs - self.cor_mol1_kbs) / settings.n_steps_interp

        interpol_list = [self.cor_mol2_kbs]  # Start with reference structure (molecule 2) at list position 0
        interpol_rmsd = [0.0]  # RMSD of the interpolated structure with respect to the reference structure

        # Append interpolated structures (last structure is molecule 1)
        [interpol_list.append(self.cor_mol2_kbs - diff * step) for step in range(1, settings.n_steps_interp + 1)]

        self.interp_struct = np.asarray(interpol_list)

        # Append RMSD values of interpolated structures (last structure is molecule 1)
        [interpol_rmsd.append(fast_rmsd(self.interp_struct[0], self.interp_struct[entry]))
         for entry in range(1, settings.n_steps_interp + 1)]

        self.interp_rmsd = np.asarray(interpol_rmsd)

    def make_sum_formula(self, pos=None):
        """ Generates a sum formula based on """

        def make_entry(sym, occ):
            """ Returns a proper single entry """

            if occ == 1:  # Avoid a 1 in sum formula

                return str(sym)

            elif sym[-1] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:  # For normal symbols

                return str(sym) + str(occ)

            else:  # For group symbols

                return str(sym) + '_' + str(occ)

        if pos is None:

            sym_types, occ = self.at_types, self.occ

        else:

            sym = np.take(self.sym, pos)
            sym_types = np.unique(sym)
            occ = np.asarray([len(np.where(sym == sym_types[entry])[0]) for entry in range(len(sym_types))])

        per_atom = []

        [per_atom.append(make_entry(sym_types[entry], occ[entry])) for entry in range(len(sym_types))]

        return ''.join(per_atom)

    ###############################################################################
    # FUNCTIONS FOR FILE HANDLING
    ###############################################################################

    def get_export_file(self, logger, example, prefix=None):
        """ Gets a valid outfile name from the user """

        self.outfile_name = get_file_name(logger, example, prefix)

    def write_interpol_structures(self, logger, settings):
        """ Writes the results of the interpolation to separate outfiles """

        def write_single_file(outfile_name, sym, cor, rmsd):
            """ Writes one file """

            n_atoms = len(sym)
            comment = '; RMSD: ' + '{:3.5f}'.format(rmsd)

            output = open(outfile_name, 'wb')

            # Write number of atoms and comment string - RMSD as additional information
            output.write(str(n_atoms) + '\n' + logger.wrt_comment(comment))

            # Write element symbol and coordinates to file
            [output.write(str(sym[atom]) + '\t' + write_coord(cor[atom][0]) + '\t' +
                          write_coord(cor[atom][1]) + '\t' +
                          write_coord(cor[atom][2]) + '\n') for atom in range(n_atoms)]

            output.close()  # Close output file and echo to user

        prefix = 'interp_'
        suffix = '.xyz'

        [write_single_file(prefix + str(entry).zfill(2) + suffix, self.sym_mol2, self.interp_struct[entry],
                           self.interp_rmsd[entry])
         for entry in range(settings.n_steps_interp)]

        # Echo to user
        logger.pt_interpol_write_success(self)

    def export_kabsch(self, logger, settings):
        """ Handles the export of structural data (final alignment) to outfiles """

        choices = [0, 1, 2, -10]  # List of accepted choices

        question = "\n>>  Enter your choice: "

        while True:

            logger.pt_export_kabsch_menu()
            operation = logger.get_menu_choice(choices, question, return_type='int')

            if operation == -10:  # Exit the menu

                break

            elif operation == 0:  # Export superposition as two simple xyz file

                self.get_export_file(logger, example='myfile.xyz', prefix='first')
                write_xyz_file(logger, self.outfile_name, self.sym_mol1, self.cor_mol1_kbs)

                self.get_export_file(logger, example='myfile.xyz', prefix='second')
                write_xyz_file(logger, self.outfile_name, self.sym_mol2, self.cor_mol2_kbs)

            elif operation in [1, 2]:

                self.get_export_file(logger, example='myfile.xyzs', prefix=None)  # Get file name from user

                pos_dis, col_dis_rgb = None, None

                if operation == 1:  # Export superposition as one extended aRMSD xyz file

                    pos_inter, col_inter_rgb = None, None

                    rad_plt_vtk_exp1, rad_plt_vtk_exp2 = np.repeat(settings.rad_bnd, self.n_atoms), np.repeat(
                        settings.rad_bnd, self.n_atoms)

                    write_xyzs_file(logger, self.outfile_name, np.hstack((self.sym_mol1, self.sym_mol2)),
                                    np.vstack((self.cor_mol1_kbs, self.cor_mol2_kbs)),
                                    np.vstack((self.cor_mol1_kbs_std, self.cor_mol2_kbs_std)),
                                    np.vstack((self.col_at_mol1_rgb, self.col_at_mol2_rgb)),
                                    np.hstack((rad_plt_vtk_exp1, rad_plt_vtk_exp2)),
                                    np.vstack((self.bnd_idx, self.bnd_idx + self.n_atoms)),
                                    np.vstack((self.col_bnd_mol1_rgb, self.col_bnd_mol2_rgb)),
                                    np.repeat(settings.rad_bnd, self.n_bonds * 2),
                                    pos_dis, col_dis_rgb, pos_inter, col_inter_rgb, self.cam_vtk_pos, self.cam_vtk_wxyz,
                                    settings.scale_glob)

                elif operation == 2:  # Export aRMSD representation as one extended aRMSD xyz file

                    pos_inter, col_inter_rgb = self.chd_bnd_idx, np.take(self.chd_bnd_col_rgb, self.chd_bnd_idx, axis=0)

                    rad_plt_vtk = self.rad_plt_vtk * settings.scale_at  # Scale radii

                    write_xyzs_file(logger, self.outfile_name, self.sym_mol2, self.cor_mol2_kbs, self.cor_mol2_kbs_std,
                                    self.col_at_rgb, rad_plt_vtk,
                                    self.bnd_idx, self.col_bnd_rgb, np.repeat(settings.rad_bnd, self.n_bonds),
                                    pos_dis, col_dis_rgb, pos_inter, col_inter_rgb, self.cam_vtk_pos, self.cam_vtk_wxyz,
                                    settings.scale_glob)

            else:

                logger.pt_invalid_input()


###############################################################################
# SETTINGS AND PLOT STYLES
###############################################################################

class settings(object):
    """ An object container for the settings and plotting styles """

    def __init__(self):
        """ Initializes object """

        self.name_regular = 'Ball and Stick'  # Name of the standard plotting style
        self.name_wire = 'Wireframe'  # Name of the Wireframe plotting style

    def parse_settings(self, file_name):
        """ Uses settings from a settings file given by file_name' """

        def _open_file(file_name):
            """ Parses information from style file """

            try:

                # Determine absolute file path
                filepath = os.path.join(os.environ.get('_MEIPASS2', os.path.abspath('.')), file_name)

                infile = open(filepath, "r")
                data = infile.readlines()
                infile.close()

                # Return value(s)
                return data

            except (IOError, NameError, SyntaxError, ValueError, UnboundLocalError):  # Error case: return None

                return None

        def _check_int(new_var, options):
            """ Checks if variable is an integer within options """

            return True if type(new_var) == int and new_var in range(options[0], options[1] + 1) else False

        def _check_float(new_var, options):
            """ Checks if variable is a float within options """

            return True if type(new_var) == float and new_var >= min(options) and new_var <= max(options) else False

        def _check_string(new_var, options):
            """ Checks if variable is a string within options """

            return True if type(new_var) == str and new_var in options else False

        def _check_tuple(new_var, options):
            """ Checks if the variable is a tuple """

            return True if type(new_var) == tuple and new_var[0] in range(options[0], options[1] + 1) and new_var[
                                                                                                              1] in range(
                options[0], options[1] + 1) else False

        def _check_bool(new_var):
            """ Checks if variable is a boolean """

            return True if type(new_var) == bool else False

        def _check_html(new_var):
            """ Checks if variable is a proper HTML string """

            def if_entry_in_dict(entry):

                html_dict = ['F', 'f', 'E', 'e', 'D', 'd', 'B', 'b', 'A',
                             'a', 'C', 'c', '0', '1', '2', '3', '4', '5',
                             '6', '7', '8', '9']

                return entry in html_dict

            if type(new_var) != str or len(new_var) != 7 or new_var[0] != '#':  # Formal requirements for HTML strings

                return False

            else:

                string_in_list = [if_entry_in_dict(new_var[entry]) for entry in range(1, len(new_var))]

                return False not in string_in_list

        def _is_proper_var(new_var, var_type, options):
            """ Checks if the new variable is a properly defined integer, float,
            string, tuple, boolean or html value """

            if var_type == 'int':

                return _check_int(new_var, options)

            elif var_type == 'float':

                return _check_float(new_var, options)

            elif var_type == 'string':

                return _check_string(new_var, options)

            elif var_type == 'tuple':

                return _check_tuple(new_var, options)

            elif var_type == 'bool':

                return _check_bool(new_var)

            elif var_type == 'html':

                return _check_html(new_var)

        def _set_variable(data, keyword, var_type, default, options):
            """ Overwrites default variable if the new one if in order """

            new_var = _find_variable(data, keyword)

            if new_var is None:

                return default

            elif _is_proper_var(new_var, var_type, options):  # Only if the new variable is ok

                return new_var

            else:

                return default

        def _find_variable(data, keyword):
            """ Finds value of given keyword in data or returns None """

            if data is None:

                return None

            else:

                for line in data:

                    if keyword in line:

                        try:

                            # Keyword is assumed to be at second position
                            return eval(line.split()[1])

                        # Error case: return None
                        except (IOError, NameError, SyntaxError, ValueError, UnboundLocalError):

                            return None

        data = _open_file(file_name)  # Open the settings file

        # Variables that cannot be changed
        self.use_std = True

        # Set all variables: 1. Colors
        self.col_xray_hex = _set_variable(data, 'color_xray_mol=', 'html', default='#484848', options=None)
        self.col_xray_rgb = hex2rgb(self.col_xray_hex, normalize=True)
        self.col_model_hex = _set_variable(data, 'color_model_mol=', 'html', default='#CD0000', options=None)
        self.col_model_rgb = hex2rgb(self.col_model_hex, normalize=True)
        self.col_refer_hex = _set_variable(data, 'color_reference_mol=', 'html', default='#006400', options=None)
        self.col_refer_rgb = hex2rgb(self.col_refer_hex, normalize=True)
        self.col_disord_hex = _set_variable(data, 'color_disorderd_pos=', 'html', default='#FFCC00', options=None)
        self.col_disord_rgb = hex2rgb(self.col_disord_hex, normalize=True)
        self.col_model_fin_hex = _set_variable(data, 'color_model_mol_final=', 'html', default='#484848', options=None)
        self.col_model_fin_rgb = hex2rgb(self.col_model_fin_hex, normalize=True)
        self.col_refer_fin_hex = _set_variable(data, 'color_reference_mol_final=', 'html', default='#A8A8A8',
                                               options=None)
        self.col_refer_fin_rgb = hex2rgb(self.col_refer_fin_hex, normalize=True)
        self.col_bnd_glob_hex = _set_variable(data, 'color_bond_aRMSD=', 'html', default='#FFCC00', options=None)
        self.col_bnd_glob_rgb = hex2rgb(self.col_bnd_glob_hex, normalize=True)
        self.col_short_hex = _set_variable(data, 'color_bond_comp_short=', 'html', default='#006400', options=None)
        self.col_short_rgb = hex2rgb(self.col_short_hex, normalize=True)
        self.col_long_hex = _set_variable(data, 'color_bond_comp_long=', 'html', default='#CD0000', options=None)
        self.col_long_rgb = hex2rgb(self.col_long_hex, normalize=True)
        self.picker_col_hex = _set_variable(data, 'color_picker=', 'html', default='#00CCFF', options=None)
        self.picker_col_rgb = hex2rgb(self.picker_col_hex, normalize=True)
        self.arrow_col_hex = _set_variable(data, 'color_arrows=', 'html', default='#0000FF', options=None)
        self.arrow_col_rgb = hex2rgb(self.arrow_col_hex, normalize=True)
        self.backgr_col_hex = _set_variable(data, 'color_background=', 'html', default='#FFFFFF', options=None)
        self.backgr_col_rgb = hex2rgb(self.backgr_col_hex, normalize=True)
        self.col_model_inertia_hex = _set_variable(data, 'color_model_inertia=', 'html', default='#FF8000',
                                                   options=None)
        self.col_model_inertia_rgb = hex2rgb(self.col_model_inertia_hex, normalize=True)
        self.col_refer_inertia_hex = _set_variable(data, 'color_reference_inertia=', 'html', default='#00FF80',
                                                   options=None)
        self.col_refer_inertia_rgb = hex2rgb(self.col_refer_inertia_hex, normalize=True)

        # 2. General settings
        self.use_grad = _set_variable(data, 'gen_use_gradient=', 'bool', default=True, options=None)
        self.modify_H = _set_variable(data, 'gen_mod_H_radius=', 'bool', default=True, options=None)
        self.det_bnd_fct = _set_variable(data, 'gen_bond_tolerance_factor=', 'float', default=1.12, options=[1.0, 1.5])
        self.n_dev = _set_variable(data, 'gen_n_deviations_matching=', 'int', default=5, options=[0, 5])
        self.atom_limit = _set_variable(data, 'gen_n_atoms_coord_type=', 'int', default=4, options=[0, 8])
        self.delta = _set_variable(data, 'gen_delta_identical=', 'float', default=0.3, options=[0.0, 1.0])
        self.use_aRMSD_col = _set_variable(data, 'gen_armsd_colors= ', 'bool', default=True, options=None)
        

        # 3. VTK variables
        self.window_size = _set_variable(data, 'vtk_window_size=', 'tuple', default=(512, 512), options=[128, 1024])
        self.use_light = _set_variable(data, 'vtk_use_lighting=', 'bool', default=True, options=None)
        self.use_depth_peel = _set_variable(data, 'vtk_use_depth_peeling=', 'bool', default=True, options=None)
        self.draw_labels = _set_variable(data, 'vtk_draw_labels=', 'bool', default=True, options=None)
        self.draw_arrows = _set_variable(data, 'vtk_draw_arrows=', 'bool', default=True, options=None)
        self.draw_legend = _set_variable(data, 'vtk_draw_legend=', 'bool', default=True, options=None)
        self.draw_col_map = _set_variable(data, 'vtk_draw_color_map=', 'bool', default=True, options=None)
        self.magnif_fact = _set_variable(data, 'vtk_export_magnification=', 'int', default=4, options=[1, 20])

        self.res_atom = _set_variable(data, 'vtk_atom_resolution=', 'int', default=50, options=[10, 500])
        self.res_bond = _set_variable(data, 'vtk_bond_resolution=', 'int', default=50, options=[10, 500])
        self.scale_glob = _set_variable(data, 'vtk_global_scale=', 'float', default=2.0, options=[0.001, 20.0])
        self.scale_atom = _set_variable(data, 'vtk_atom_scale_regular=', 'float', default=0.3, options=[0.001, 10.0])
        self.scale_atom_wf = _set_variable(data, 'vtk_atom_scale_wireframe=', 'float', default=0.2435,
                                           options=[0.001, 10.0])
        self.rad_bond = _set_variable(data, 'vtk_bond_radius_regular=', 'float', default=0.075, options=[0.001, 10.0])
        self.rad_bond_wf = _set_variable(data, 'vtk_bond_radius_wireframe=', 'float', default=0.185,
                                         options=[0.001, 10.0])
        self.alpha_at = _set_variable(data, 'vtk_atom_bond_alpha=', 'float', default=1.0, options=[0.0, 1.0])
        self.alpha_arrow = _set_variable(data, 'vtk_arrow_alpha=', 'float', default=1.0, options=[0.0, 1.0])
        self.label_type = _set_variable(data, 'vtk_label_type=', 'string', default='full',
                                        options=['full', 'symbol_only'])
        self.std_type = _set_variable(data, 'vtk_picker_std_type=', 'string', default='simple',
                                      options=['simple', 'advanced'])

        # 4. RMSD variables
        self.max_RMSD_diff = _set_variable(data, 'rmsd_max_RMSD_diff=', 'float', default=0.7, options=[0.3, 2.0])
        self.n_col_aRMSD = _set_variable(data, 'rmsd_n_colors=', 'int', default=19, options=[3, 511])
        self.n = _set_variable(data, 'rmsd_n=', 'float', default=0.45, options=[0.01, 0.5])
        self.thresh = _set_variable(data, 'rmsd_bond_threshold=', 'float', default=0.02, options=[0.0, 2.0])
        self.eps = _set_variable(data, 'rmsd_comp_eps=', 'float', default=1.0E-06, options=[1.0E-12, 1.0E-02])
        self.id_lim = _set_variable(data, 'rmsd_sphere_size_limit=', 'float', default=0.3, options=[0.0, 1.0])
        self.calc_prec = _set_variable(data, 'rmsd_calc_precision=', 'int', default=3, options=[1, 7])
        self.n_max_diff = _set_variable(data, 'rmsd_n_max_diff_prop=', 'int', default=3, options=[0, 10])
        self.n_steps_interp = _set_variable(data, 'rmsd_n_interpolation=', 'int', default=10, options=[1, 20])
        self.gard_d_min = _set_variable(data, 'gard_d_min=', 'float', default=0.3, options=[0.0, 1.0])
        self.gard_d_max = _set_variable(data, 'gard_d_max=', 'float', default=1.2, options=[1.0, 3.0])

        # 5. Defaults for statistics plot
        self.new_black = _set_variable(data, 'stats_plot_color_black=', 'html', default='#484848', options=None)
        self.new_red = _set_variable(data, 'stats_plot_color_red=', 'html', default='#DE3D3D', options=None)
        self.new_blue = _set_variable(data, 'stats_plot_color_blue=', 'html', default='#0029A3', options=None)
        self.new_green = _set_variable(data, 'stats_plot_color_green=', 'html', default='#007A00', options=None)
        self.title_pt = _set_variable(data, 'stats_fontsize_title=', 'int', default=16, options=[3, 20])
        self.ax_pt = _set_variable(data, 'stats_fontsize_axis=', 'int', default=12, options=[3, 20])
        self.error_prop = _set_variable(data, 'stats_error_property=', 'string', default='std', options=['std', 'var'])
        self.stats_draw_legend = _set_variable(data, 'stats_draw_legend=', 'bool', default=True, options=None)
        self.stats_draw_grid = _set_variable(data, 'stats_show_grid=', 'bool', default=False, options=None)
        self.legend_pos = _set_variable(data, 'stats_legend_position=', 'string', default='upper left',
                                        options=['upper left', 'upper right', 'lower left', 'lower right'])
        self.splitter = _set_variable(data, 'stats_splitter=', 'float', default=1.0, options=[0.0, 10.0])
        self.calc_prec_stats = _set_variable(data, 'stats_calc_precision=', 'int', default=3, options=[1, 7])

        # This decides between 'Ball and Stick' and 'Wireframe'
        self.name = self.name_regular
        self.scale_at = self.scale_atom
        self.rad_bnd = self.rad_bond

    def change_plot_style(self):
        """ Interchanges between the two plot styles """

        self.use_wireframe() if self.name == self.name_regular else self.use_ball_and_stick()

    def change_True_False(self, var):
        """ Interchanges boolean variables """

        return False if var else True

    def change_html_string(self, var):
        """ Changes a valid HTML string """

        return var, hex2rgb(var, normalize=True)

    def change_two_settings(self, var, options):
        """ Interchanges between two given possibilities """

        return options[0] if var == options[1] else options[1]

    def use_ball_and_stick(self):
        """ Ensures the use of the Ball and Stick style """

        self.name = self.name_regular
        self.scale_at = self.scale_atom
        self.rad_bnd = self.rad_bond

    def use_wireframe(self):
        """ Ensures the use of the Wireframe style """

        self.name = self.name_wire
        self.scale_at = self.scale_atom_wf
        self.rad_bnd = self.rad_bond_wf

    def change_rmsd_settings(self, logger):
        """ Change general RMSD settings and representation details """

        choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -5, -10]  # List of accepted choices

        question = "\n>>  Enter your choice: "

        while True:

            logger.pt_change_rmsd_settings_menu(self)
            operation = logger.get_menu_choice(choices, question, return_type='int')

            if operation == -10:  # Exit the menu

                break

            elif operation == -5:  # Change between RYG and atom-based coloring scheme for aRMSD rep.

                self.use_aRMSD_col = self.change_True_False(self.use_aRMSD_col)

            elif operation == 0:  # Maximum value for color projection

                self.max_RMSD_diff = logger.get_menu_choice([0.3, 3.0],
                                                            question="\n>>  Enter a value between 0.3 and 3.0: ",
                                                            return_type='float')

            elif operation == 1:  # Number of colors for aMRSD representation

                self.n_col_aRMSD = logger.get_menu_choice(range(3, 513),
                                                          question="\n>>  Enter a value between 3 and 512: ",
                                                          return_type='int')

            elif operation == 2:  # Threshold for bond comparison

                self.thresh = logger.get_menu_choice([0.0, 0.5], question="\n>>  Enter a value between 0.0 and 0.5: ",
                                                     return_type='float')

            elif operation in [3, 4, 5]:  # Colors for bonds

                html_string = logger.get_menu_choice([], question="\n>>  Enter a HTML string (e.g. '#006400'): ",
                                                     return_type='HTML')

                if operation == 3:

                    self.col_bnd_glob_hex, self.col_bnd_glob_rgb = self.change_html_string(html_string)

                elif operation == 4:

                    self.col_short_hex, self.col_short_rgb = self.change_html_string(html_string)

                elif operation == 5:

                    self.col_long_hex, self.col_long_rgb = self.change_html_string(html_string)

            elif operation == 6:  # Length of the bond intersection

                self.n = (1.0 - logger.get_menu_choice([0.0, 1.0], question="\n>>  Enter a value between 0.0 and 1.0: ",
                                                       return_type='float')) / 2

            elif operation == 7:  # Precision for the aRMSD picker

                self.calc_prec = logger.get_menu_choice(range(1, 5), question="\n>>  Enter a value between 1 and 5: ",
                                                        return_type='int')

            elif operation == 8:  # Set number of highest property deviations

                self.n_max_diff = logger.get_menu_choice(range(0, 5), question="\n>>  Enter a value between 2 and 5: ",
                                                         return_type='int')

            elif operation == 9:  # Number of points for structure interpolations

                self.n_steps_interp = logger.get_menu_choice(range(1, 20),
                                                             question="\n>>  Enter a value between 1 and 20: ",
                                                             return_type='int')

            else:

                logger.pt_invalid_input()

    def change_rmsd_vtk_plot_settings(self, align, logger):
        """ Allows the user to change RMSD/VTK plot settings """

        choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, -10]  # List of accepted choices

        question = "\n>>  Enter your choice: "

        while True:

            logger.pt_change_rmsd_vtk_settings_menu(self, align)
            operation = logger.get_menu_choice(choices, question, return_type='int')

            if operation == -10:  # Exit the menu

                break

            elif operation == 0:  # Draw labels

                self.draw_labels = self.change_True_False(self.draw_labels)

            elif operation == 1:  # Change label type

                self.label_type = self.change_two_settings(self.label_type, ['full', 'symbol_only'])

            elif operation == 2:  # Change global scale factor

                self.scale_glob = logger.get_menu_choice([0.1, 8.0],
                                                         question="\n>>  Enter a value between 0.1 and 8.0: ",
                                                         return_type='float')

            elif operation == 3:  # Change plot resolution

                self.res_atom = logger.get_menu_choice(range(10, 201),
                                                       question="\n>>  Enter a value between 10 and 200: ",
                                                       return_type='int')

            elif operation in [4, 5]:  # Colors of model and reference

                html_string = logger.get_menu_choice([], question="\n>>  Enter a HTML string (e.g. '#006400'): ",
                                                     return_type='HTML')

                if operation == 4:

                    self.col_model_fin_hex, self.col_model_fin_rgb = self.change_html_string(html_string)

                elif operation == 5:

                    self.col_refer_fin_hex, self.col_refer_fin_rgb = self.change_html_string(html_string)

                align.set_colors(settings)  # Update properties

            elif operation == 6:  # Use lighting

                self.use_light = self.change_True_False(self.use_light)

            elif operation == 7:  # Change export magnification

                self.magnif_fact = int(
                    logger.get_menu_choice([1.0, 20.0], question="\n>>  Enter a value between 1.0 and 20.0: ",
                                           return_type='float'))

            elif operation == 8:  # Draw color bar

                self.draw_col_map = self.change_True_False(self.draw_col_map)

            else:

                logger.pt_invalid_input()

    def change_vtk_plot_settings(self, molecule1, molecule2, logger):
        """ Allows the user to change VTK plot settings """

        choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -10]  # List of accepted choices

        question = "\n>>  Enter your choice: "

        while True:

            logger.pt_change_vtk_settings_menu(self, molecule1, molecule2)
            operation = logger.get_menu_choice(choices, question, return_type='int')

            if operation == -10:  # Exit the menu

                break

            elif operation == -1:  # Change plotting style

                self.change_plot_style()

            elif operation == 0:  # Draw labels

                self.draw_labels = self.change_True_False(self.draw_labels)

            elif operation == 1:  # Change label type

                self.label_type = self.change_two_settings(self.label_type, ['full', 'symbol_only'])

            elif operation == 2:  # Draw arrows

                self.draw_arrows = self.change_True_False(self.draw_arrows)

            elif operation == 3:  # Draw legend

                self.draw_legend = self.change_True_False(self.draw_legend)

            elif operation == 4:  # Change global scale factor

                self.scale_glob = logger.get_menu_choice([0.1, 8.0],
                                                         question="\n>>  Enter a value between 0.1 and 8.0: ",
                                                         return_type='float')

            elif operation == 5:  # Change atom scale factor

                self.scale_atom = logger.get_menu_choice([0.01, 2.0],
                                                         question="\n>>  Enter a value between 0.01 and 2.0: ",
                                                         return_type='float')

            elif operation == 6:  # Change plot resolution

                self.res_atom = logger.get_menu_choice(range(10, 201),
                                                       question="\n>>  Enter a value between 10 and 200: ",
                                                       return_type='int')

            elif operation in [7, 8]:  # Colors of model and reference

                html_string = logger.get_menu_choice([], question="\n>>  Enter a HTML string (e.g. '#006400'): ",
                                                     return_type='HTML')

                if operation == 7:

                    self.col_model_hex, self.col_model_rgb = self.change_html_string(html_string)
                    molecule1.set_color(self.col_model_rgb)
                    molecule1.update_properties(self, get_bonds=False)  # Update properties - bonds are not needed

                elif operation == 8:

                    self.col_refer_hex, self.col_refer_rgb = self.change_html_string(html_string)
                    molecule2.set_color(self.col_refer_rgb)
                    molecule2.update_properties(self, get_bonds=False)  # Update properties - bonds are not needed

            elif operation == 9:  # Use lighting

                self.use_light = self.change_True_False(self.use_light)

            elif operation == 10:  # Change export magnification

                self.magnif_fact = int(
                    logger.get_menu_choice([1.0, 20.0], question="\n>>  Enter a value between 1.0 and 20.0: ",
                                           return_type='float'))

            else:

                logger.pt_invalid_input()


###############################################################################
# FUNCTIONS FOR COLOR TRANSFORMATIONS
###############################################################################

def rgb2hex(r, g, b, normalized=False):
    """ Converts RGB to HEX color """

    # Check if RGB triplett is normalized to unity
    if normalized:
        r, g, b = r * 255.0, g * 255.0, b * 255.0

    return '#%02X%02X%02X' % (r, g, b)


def hex2rgb(hexcolor, normalize=False):
    """ Converts HEX to RGB color """

    # Check if hex string starts with '#' or '0x'
    if hexcolor[0] == '#':

        hexcolor = hexcolor[1:]

    elif hexcolor[:2] == '0x' or hexcolor[:2] == '0X':

        hexcolor = hexcolor[2:]

    # Form RGB triplet
    r, g, b = int(hexcolor[0:2], 16), int(hexcolor[2:4], 16), int(hexcolor[4:6], 16)

    # Normalize to unity if requested
    if normalize:
        r, g, b = round(r / 255.0, 5), round(g / 255.0, 5), round(b / 255.0, 5)

    return np.array([r, g, b], dtype=np.float)


def make_color_scale(n_colors, as_rgb=False):
    """ Creates a color scale: Red - Yellow - Green """

    def get_all_colors(as_rgb=False):
        """ Creates all RBG colors from Red to Green """

        row_a = np.hstack((np.arange(0, 256), np.repeat(255, 256)))
        row_b = row_a[::-1]
        row_c = np.repeat(0, 512)

        all_colors_rgb = np.transpose(np.vstack((row_a, row_b, row_c)))  # Array of RGB triplets

        if as_rgb:

            return all_colors_rgb

        else:

            return np.asarray([rgb2hex(entry[0], entry[1], entry[2]) for entry in all_colors_rgb])

    col_idx = np.linspace(0, 512, n_colors, endpoint=True, dtype=np.int)
    col_idx[-1] = 511  # Last entry must be 511

    return get_all_colors(as_rgb=False)[col_idx]


###############################################################################
# FUNCTIONS FOR FILE HANDLING
###############################################################################


def parse_files(logger, number, settings):
    """ Main wrapper for reading molecular data """

    def contains_path(input_file):
        """ Checks if the file name contains a path """

        return '\\' in input_file

    # First file (model)
    if number == 1:

        number_str = 'first file (comp./model)'
        mol_name = 'Model'

    # Second file (reference)
    elif number == 2:

        number_str = 'second file (exp./reference)'
        mol_name = 'Reference'

    # Stay in loop until correct file is provided by user
    while True:

        try:

            print("\nEnter the file name with extension for the " + str(number_str))
            input_file = input("\n>> ")

            try:

                if contains_path(input_file):
                    
                    name_start = len(input_file) - [entry for entry, count in enumerate(input_file[::-1])
                                                    if count == '\\'][0]

                    input_file = input_file[name_start::]  # Remove the path from file name

                # Determine file type by evaluating its ending from the reversed file name (last .)
                file_ending_start = len(input_file) - [entry for entry, count in enumerate(input_file[::-1])
                                                       if count == '.'][0]

                filetype = input_file[file_ending_start:len(input_file)]

                # Get element symbols and coordinate from file
                element_symbol, element_xyz, element_xyz_std = get_data(logger, input_file, filetype, settings)

                molecule = Molecule(mol_name, element_symbol, element_xyz, element_xyz_std)  # Create a molecule object

                # Print out information
                logger.pt_import_success(input_file, element_symbol)

                if number == 1:

                    logger.file_mol1 = input_file

                elif number == 2:

                    logger.file_mol2 = input_file

                break

            except (IndexError, TypeError):

                logger.pt_no_ob_support()

        except IOError:

            logger.pt_file_not_found()

    # Return values(s)
    return molecule, input_file


def get_data(logger, input_file, filetype, settings):
    """ Wrapper for the extraction of coordinates and symbols/charges
        from different file formats. Charges are assigned based on symbols """

    def _openfile(input_file):
        """ Opens file and returns data """

        infile = open(input_file, 'r')
        data = infile.readlines()
        infile.close()

        return data

    def _get_data_openbabel(input_file, filetype):
        """ Wrapper for data extraction with openbabel """

        try:

            mol = pybel.readfile(filetype, input_file).next()  # Parse data with openbabel

            # Extract charges and coordinates from data, transform charges to symbols
            element_charge = np.asarray([atom.atomicnum for atom in mol], dtype=np.int)
            element_xyz = np.asarray([atom.coords for atom in mol], dtype=np.float)

            element_symbol = np.asarray(np.take(pse_sym_chg, element_charge - 1), dtype=np.object)

            return element_symbol, element_xyz

        except ValueError:

            logger.pt_no_pybel_file()

            return None, None

    def _determine_QM_prog(data):
        """ Determine the program behind the .out file """

        prg_type = None  # No software type established so far

        if ' * O   R   C   A *' in data[2]:

            prg_type = 'orca'

        else:

            for index, line in enumerate(data):

                if 'Cite this work as:' in line:

                    index += 1

                    if data[index].split()[0] == 'Gaussian':
                        prg_type = 'gaussian'

        return prg_type

    data = _openfile(input_file)  # Extract raw data and file type from input file
    element_symbol, element_xyz = None, None  # Initial values for symbols and coordinates
    element_xyz_std = None  # Assume no standard deviations

    # Internally supported file types
    internal_file_types = ['xyz', 'cif', 'res', 'ins', 'lst', 'xyzs', 'mol2',
                           'mol', 'sdf', 'sd', 'out']

    if filetype in internal_file_types:

        if filetype == 'xyz':

            element_symbol, element_xyz = read_xyz_file(logger, data)

        elif filetype == 'mol2':

            element_symbol, element_xyz = read_mol2_file(logger, data)

        elif filetype in ['mol', 'sdf', 'sd']:

            element_symbol, element_xyz = read_sdf_file(logger, data)

        # Files which may contain coordinate uncertainties (except .res)
        elif filetype in ['cif', 'res', 'ins', 'lst', 'xyzs', 'out'] and logger.has_uc:

            if filetype in ['cif', 'res', 'ins', 'lst']:  # X-ray structure formats

                xray = Xray_structure(settings)

                if filetype == 'cif':

                    xray.read_cif_file(logger, data, settings)

                elif filetype in ['res', 'ins', 'lst']:  # ShelX format

                    xray.read_res_file(logger, data, filetype, settings)

                xray.user_expansion(logger, settings)  # Interactive coordinate expansion

                element_symbol, element_xyz, element_xyz_std = xray.sym, xray.cor, xray.std

            elif filetype == 'xyzs':  # aRMSD file type

                element_symbol, element_xyz, element_xyz_std = read_xyzs_file(data)

            elif filetype == 'out':  # QC outfiles (only Orca and Gaussian are supported so far)

                prg_type = _determine_QM_prog(data)

                if prg_type == 'orca':

                    element_symbol, element_xyz, element_xyz_std = read_orca_out_file(logger, data)

                elif prg_type == 'gaussian':

                    element_symbol, element_xyz, element_xyz_std = read_gaussian_out_file(logger, data)

                else:

                    print("Program type '" + str(prg_type) + "' is not supported")

                if element_xyz_std is not None:
                    element_xyz_std *= 1.0 / b2a  # Transform gradient to Hartree/Angstrom

                if not settings.use_grad:  # If gradient information is not to be used

                    element_xyz_std = None

    # Other formats with openbabel/pybel
    elif logger.has_pybel:

        if filetype in pybel.informats:  # Check if file type is supported by openbabel

            element_symbol, element_xyz = _get_data_openbabel(input_file, filetype)

        else:  # Unsupported by openbabel

            logger.pt_unsupported_file_type(internal_file_types)

    else:  # Openbabel is not availabel

        logger.pt_no_pybel()
        logger.pt_unsupported_file_type(internal_file_types)

    # Check for supported element symbols
    if element_symbol is not None:

        check_array = np.asarray([symbol in pse_symbol for symbol in element_symbol])
        check_problem = False in check_array

    else:

        check_array = np.asarray([False])
        check_problem = True

    if not check_problem:  # Check and adjust arrays and return value(s)

        if element_xyz_std is None:
            
            element_xyz_std = np.zeros((len(element_symbol), 3))

    elif element_symbol is not None:

        logger.pt_unsupported_atom(list(element_symbol[np.where(check_array == False)[0]])[0])

    return element_symbol, element_xyz, element_xyz_std


def read_orca_out_file(logger, data):
    """ Reads .out files from cora and returns symbols and coordinates (gradient as standard deviations) """

    element_xyz_std = None  # In case that the job is a single point calculation

    for index, line in enumerate(data):

        # Start at line containing 'CARTESIAN COORDINATES (ANGSTROEM)'
        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:

            index += 0

            if data[index + 1].split()[0] == '---------------------------------':

                element_symbol, element_xyz = [], []

                index += 0
                f = 2

                while len(data[index + f].split()) == 4:
                    
                    element_symbol.append(data[index + f].split()[0])
                    element_xyz.append(data[index + f].split()[-3:])

                    f += 1

        # Start at line containing 'CARTESIAN GRADIENT' (assumed to be in Hartree/Bohr)
        if 'CARTESIAN GRADIENT' in line:

            index += 0

            if data[index + 1].split()[0] == '------------------':

                element_xyz_std = []

                index += 0
                f = 3

                while len(data[index + f].split()) == 6:
                    
                    element_xyz_std.append(data[index + f].split()[-3:])

                    f += 1

    # Transform lists to arrays and reshape coordinate array
    element_symbol = np.asarray(element_symbol, dtype=np.object)
    element_xyz = np.asarray(element_xyz, dtype=np.float)
    element_xyz = np.reshape(element_xyz, (len(element_symbol), 3))

    if element_xyz_std is not None:  # Reshape gradient if it exists

        element_xyz_std = np.asarray(element_xyz_std, dtype=np.float)
        element_xyz_std = np.abs(np.reshape(element_xyz_std, (len(element_symbol), 3)))

    # Return value(s)
    return element_symbol, element_xyz, element_xyz_std


def read_gaussian_out_file(logger, data):
    """ Reads .out files from gaussian and returns symbols and coordinates (gradient as standard deviations) """

    element_xyz_std = None  # In case that the job is a single point calculation

    for index, line in enumerate(data):

        if 'Coordinates (Angstroms)' in line:

            element_charge, element_xyz = [], []

            index += 3
            f = 0

            while len(data[index + f].split()) > 1:
                
                element_charge.append(data[index + f].split()[1])
                element_xyz.append(data[index + f].split()[-3:])

                f += 1

        # Start at line containing 'Forces (Hartrees/Bohr)'
        if 'Forces (Hartrees/Bohr)' in line:

            element_xyz_std = []

            index += 3
            f = 0

            while len(data[index + f].split()) == 5:
                
                element_xyz_std.append(data[index + f].split()[-3:])

                f += 1

    # Transform lists to arrays and get symbols from charges
    element_charge = np.asarray(element_charge, dtype=np.int)
    element_symbol = np.asarray(np.take(pse_sym_chg, element_charge - 1), dtype=np.object)
    element_xyz = np.asarray(element_xyz, dtype=np.float)
    element_xyz = np.reshape(element_xyz, (len(element_symbol), 3))

    if element_xyz_std is not None:  # Reshape gradient if it exists

        element_xyz_std = np.asarray(element_xyz_std, dtype=np.float)

    # Return value(s)
    return element_symbol, element_xyz, element_xyz_std


def read_sdf_file(logger, data):
    """ Reads .mol / .sd / .sdf files and returns symbols and coordinates -
        parses only one structure """

    element_symbol, element_xyz = [], []  # Empty variables for symbol and coordinates

    start_at = 4  # Line starting with coordinates

    n_atoms = int(data[3].split()[0])  # Take number of atoms from the first value given in line 4

    for index in range(start_at, n_atoms + start_at):

        element_symbol.append(data[index].split()[3].lower().capitalize())
        element_xyz.append(data[index].split()[0:3])

    # Transform lists to arrays
    element_symbol = np.asarray(element_symbol, dtype=np.object)
    element_xyz = np.asarray(element_xyz, dtype=np.float)

    # Return value(s
    return element_symbol, element_xyz


def read_mol2_file(logger, data):
    """ Reads .mol2 files and returns symbols and coordinates """

    element_symbol, element_xyz = [], []  # Empty variables for symbol and coordinates

    for index, line in enumerate(data):

        # Start at line containing '<TRIPOS>ATOM'
        if '<TRIPOS>ATOM' in line:

            index += 0
            f = 0

            coordinates_left_to_read = True

            while coordinates_left_to_read:

                f += 1

                # Append information: First character is the atomic symbol
                if len(data[index + f].split()[1]) in [1, 2]:

                    element_symbol.append(data[index + f].split()[1].lower().capitalize())

                elif len(data[index + f].split()[1]) > 2:

                    logger.pt_unsupported_element(data[line].split()[0])
                    raise ValueError(logger.pt_exiting())

                element_xyz.append(data[index + f].split()[2:5])

                if len(data[index + f + 1].split()) == 1:
                    
                    coordinates_left_to_read = False

    # Transform lists to arrays
    element_symbol = np.asarray(element_symbol, dtype=np.object)
    element_xyz = np.asarray(element_xyz, dtype=np.float)

    # Return value(s
    return element_symbol, element_xyz


def read_xyz_file(logger, data):
    """ Reads .xyz files and returns symbols and coordinates """

    # Set up empty variables for symbol and coordinates
    element_symbol, element_xyz = [], []

    start_at = 2  # Where the parsing begins

    if len(data[0].split()) == 4:  # Check if the first line is actual data

        start_at = 0

    for line in range(start_at, len(data)):

        if len(data[line].split()) == 4:

            # Append information: First character is the atomic symbol
            # If a two character symbol is found append capitalized first character
            if len(data[line].split()[0]) in [1, 2]:

                element_symbol.append(data[line].split()[0].lower().capitalize())

            elif len(data[line].split()[0]) > 2:

                logger.pt_unsupported_element(data[line].split()[0])
                raise ValueError(logger.pt_exiting())

            # Last three characters are xyz coordinates
            element_xyz.append(data[line].split()[-3:])

    # Transform lists to arrays
    element_symbol = np.asarray(element_symbol, dtype=np.object)
    element_xyz = np.asarray(element_xyz, dtype=np.float)

    # Return value(s)
    return element_symbol, element_xyz


def read_xyzs_file(data):
    """ Reads .xyzs files and returns symbols and coordinates with standard deviations """

    # Set up empty variables for symbol and coordinates
    element_symbol, element_xyz, element_xyz_std = [], [], []

    for index, line in enumerate(data):

        if len(data[index].split()) == 11:
            f = 0

            element_symbol.append(data[index + f].split()[0])
            element_xyz.append(data[index + f].split()[1:4])
            element_xyz_std.append(data[index + f].split()[4:7])

            f += 1

    # Transform lists to arrays
    element_symbol = np.asarray(element_symbol, dtype=np.object)
    element_xyz = np.asarray(element_xyz, dtype=np.float)
    element_xyz_std = np.asarray(element_xyz_std, dtype=np.float)

    return element_symbol, element_xyz, element_xyz_std


def write_coord(coord):
    """ Adjusts whitespace for coordinates """

    return "{:06.8f}".format(coord) if coord < 0.0 else " " + "{:06.8f}".format(coord)


def write_xyz_file(logger, outfile_name, sym, cor):
    """ Writes coordinates to .xyz outfile """

    extension = '.xyz'

    output = open(outfile_name + extension, 'wb')
    output.write(str(len(sym)) + '\n' + logger.wrt_comment())

    # Write element symbol and coordinates to file
    [output.write(str(sym[atom]) + '\t' +
                  write_coord(cor[atom][0]) + '\t' +
                  write_coord(cor[atom][1]) + '\t' +
                  write_coord(cor[atom][2]) + '\n') for atom in range(len(sym))]

    output.close()  # Close output file and echo to user
    logger.pt_write_success(outfile_name + extension)


def write_xyzs_file(logger, outfile_name, sym, cor, cor_std, col_at_rgb, rad_at_plt,
                    bnd_idx, col_bnd_rgb, rad_bnd_plt, pos_dis=None, col_dis_rgb=None,
                    pos_inter=None, col_inter_rgb=None, cam_vtk_pos=None, cam_vtk_wxyz=None,
                    scale_fact=1.0):
    """ Writes an extended .xyzs file """

    extension = '.xyzs'

    # Comment strings for the file
    comments = (
        '# Atoms - [element symbol] [x,y,z coordinates] [x,y,z standard deviations] [norm. RGB color] [radius]\n' +
        '# Bonds - [connection IDs.] [norm. RGB color] [radius]\n' +
        '# Disordered connections - [connection IDs.] [norm. RGB color]\n' +
        '# Intersections - [bond IDs.] [norm. RGB color]\n' +
        '# Camera - [position] [angle (radians), rotation axis]\n\n' +
        '# Info: Enumeration begins with 0, the molecule is scaled by the following factor:\n\n' +
        'scale_factor ' + str(scale_fact) + '\n\n')

    atoms_str = '[Atoms]'
    bonds_str = '[Bonds]'
    matches_str = '[Disordered]'
    intersec_str = '[Intersections]'
    camera_str = '[Camera]'

    n_atoms, n_bonds = len(sym), len(bnd_idx)

    output = open(outfile_name + extension, 'wb')  # Generate outfile and write comment string
    output.write(logger.wrt_comment() + comments + atoms_str + ' ' + str(n_atoms) + '\n')

    # Write out element symbol, coordinates (with standard deviations), colors and radii to file
    [output.write(str(sym[atom]) +
                  '\t' + write_coord(cor[atom][0]) +
                  '\t' + write_coord(cor[atom][1]) +
                  '\t' + write_coord(cor[atom][2]) +
                  '\t' + write_coord(cor_std[atom][0]) +
                  '\t' + write_coord(cor_std[atom][1]) +
                  '\t' + write_coord(cor_std[atom][2]) +
                  '\t' + '{:03.3f}'.format(col_at_rgb[atom][0]) +
                  '\t' + '{:03.3f}'.format(col_at_rgb[atom][1]) +
                  '\t' + '{:03.3f}'.format(col_at_rgb[atom][2]) +
                  '\t' + '{:03.3f}'.format(rad_at_plt[atom]) + '\n') for atom in range(n_atoms)]

    output.write('\n' + bonds_str + ' ' + str(n_bonds) + '\n')  # Write [bond section]

    # Write out connectivities and color of all bonds
    [output.write(str(bnd_idx[bond][0]) + '\t' + str(bnd_idx[bond][1]) + '\t' +
                  '{:03.3f}'.format(col_bnd_rgb[bond][0]) + '\t' +
                  '{:03.3f}'.format(col_bnd_rgb[bond][1]) + '\t' +
                  '{:03.3f}'.format(col_bnd_rgb[bond][2]) + '\t' +
                  '{:03.3f}'.format(rad_bnd_plt[bond]) + '\n') for bond in range(n_bonds)]

    if pos_dis is not None:
        n_disordered = len(pos_dis)

        output.write('\n' + matches_str + ' ' + str(n_disordered) + '\n')  # Write [disordered section]

        # Write out disordered connections
        [output.write(str(pos_dis[dev][0]) + '\t' + str(pos_dis[dev][1]) + '\t' +
                      '{:03.3f}'.format(col_dis_rgb[0]) + '\t' +
                      '{:03.3f}'.format(col_dis_rgb[1]) + '\t' +
                      '{:03.3f}'.format(col_dis_rgb[2]) + '\n') for dev in range(n_disordered)]

    if pos_inter is not None:
        n_intersections = len(pos_inter)

        output.write('\n' + intersec_str + ' ' + str(n_intersections) + '\n')  # Write [disordered section]

        # Write out intersection positions
        [output.write(str(pos_inter[inter]) + '\t' +
                      '{:03.3f}'.format(col_inter_rgb[inter][0]) + '\t' +
                      '{:03.3f}'.format(col_inter_rgb[inter][1]) + '\t' +
                      '{:03.3f}'.format(col_inter_rgb[inter][2]) + '\n') for inter in range(n_intersections)]

    if cam_vtk_pos is not None:
        output.write('\n' + camera_str + '\n')  # Write [camera section]

        # Write out camera properties
        output.write('Position:    ' +
                     write_coord(cam_vtk_pos[0]) + '\t' +
                     write_coord(cam_vtk_pos[1]) + '\t' +
                     write_coord(cam_vtk_pos[2]) + '\n')

        output.write('WXYZ:        ' +
                     write_coord(cam_vtk_wxyz[0] * np.pi / 180.0) + '\t' +
                     write_coord(cam_vtk_wxyz[1]) + '\t' +
                     write_coord(cam_vtk_wxyz[2]) + '\t' +
                     write_coord(cam_vtk_wxyz[3]))

    output.close()  # Close output file and echo to user
    logger.pt_write_success(outfile_name + extension)


def get_file_name(logger, example, prefix=None):
    """ Returns a valid file name with extension from user input """

    output_file = logger.inp_file_export(example, prefix)  # Request user to specify outfile name

    # Check if extension is given
    pos_dot = [entry for entry, count in enumerate(output_file) if count == '.']

    if len(pos_dot) == 0:  # No file type given, use default

        return output_file

    else:

        return output_file[0:pos_dot[0]]


class Xray_structure(object):
    """ Handler for crystallographic data """

    def __init__(self, settings):
        """ Initializes the object """

        self.a = None  # First cell axis
        self.b = None  # Second cell axis
        self.c = None  # Third cell axis
        self.alpha = None  # First cell angle [degrees]
        self.beta = None  # Second cell angle [degrees]
        self.gamma = None  # Third cell angle [degrees]
        self.cell_vol = None  # Cell volume
        self.trans_mat = None  # Transformation matrix for fractional to Cart. coords

        self.symOPs = None  # List of symmetry operations for coordinate expansion
        self.cor_frac = None  # Fractional coordinates
        self.std_frac = None  # Standard deviations of fractional coordinates
        self.cor = None  # Cartesian coordinates
        self.std = None  # Standard deviations of Cartesian coordinates
        self.sym = None  # Element symbols
        self.idf = None  # Element identifiers
        self.sym_idf = None  # Combined symbols and identifiers
        self.n_atoms = None  # Number of atoms
        self.rad_cov = None  # Covalent radii
        self.rad_plt_vtk = None  # Radii for VTK plots
        self.bnd_idx = None  # Bond indices
        self.n_bonds = None  # Number of bonds

        self.col_glob_rgb = settings.col_xray_rgb  # Color of the structure
        self.col_at_rgb = None  # Color array for the atoms [RGB]
        self.col_bnd_rgb = None  # Color array for the bonds [RGB]

        self.has_cam_vtk = False  # If the molecule has an existing VTK camera orientation
        self.cam_vtk_pos = None  # VTK camera postion
        self.cam_vtk_focal_pt = None  # VTK camera focal point
        self.cam_vtk_view_up = None  # VTK camera view up (?)

        self.show_mol = True  # If a plot is to be shown
        self.finalized = False  # If a final transformation was carried out

        self.outfile_name = None  # File for structure export

    def get_charge(self):
        """ Gets/updates charges from element symbols """

        self.chg = np.array([pse_symbol[symbol] for symbol in self.sym], dtype=np.int)
        self.n_atoms = len(self.chg)

    def get_mass(self):
        """ Gets/updates atomic masses from nuclear charges """

        self.mas = np.take(pse_mass, self.chg - 1)

    def get_identifiers(self):
        """ Generates/updates identifiers """

        self.idf = np.arange(1, len(self.chg) + 1, dtype=np.int)

    def get_sym_idf(self):
        """ Combines symbols and identifiers """

        self.sym_idf = np.asarray([('%s-%d' % (self.sym[atom], self.idf[atom])) for atom in range(self.n_atoms)],
                                  dtype=np.object)

    def get_cov_radii(self, settings):
        """ Gets/updates covalent radii from nuclear charges """

        self.rad_cov = np.take(pse_cov_radii, self.chg - 1)

        if settings.modify_H:  # Modify radii (from 0.31 to 0.41) for H if requested

            self.rad_cov[np.where(self.chg == 1)[0]] = 0.41

        self.rad_plt_vtk = np.repeat(1.5, self.n_atoms)

    def read_res_file(self, logger, data, filetype, settings):
        """ Reads .res/.ins or .lst files and returns symbols, cart. coordinates and coordinate uncertainties """

        def _res_symOP(line):
            """ Handles symmetry operations in .res/.ins/.lst files """

            symOP = str(line[5::])  # Exclude 'SYMM ' prefix

            # Replace capital symbols with small ones
            symOP = symOP.replace('X', 'x')
            symOP = symOP.replace('Y', 'y')
            symOP = symOP.replace('Z', 'z')
            symOP = symOP.replace('\n', '')

            return symOP

        def _clean_up_symbol(symbol):
            """ Cleans up an element symbol in .res/.ins/.lst file """

            new_sym = ''.join([' ' if char.isdigit() else char for char in symbol])  # Replace digits

            # Split string and take the first entry, ensure correct formatting
            new_sym = str(new_sym.split()[0])
            new_sym = new_sym.lower()
            new_sym = new_sym.capitalize()

            if len(new_sym) > 2:
                logger.pt_unsupported_element(new_sym)
                raise ValueError(logger.pt_exiting())

            return new_sym

        element_symbol, element_xyz_frac = [], []
        list_of_symOPs = []

        # Different keyword after coordinates for .res/.ins and .lst files
        breaker = 'HKLF' if filetype != 'lst' else 'REM'

        for index, line in enumerate(data):

            if 'CELL' in line:  # Start at line containing 'CELL'

                index += 0
                self.a = ufloat(data[index].split()[2], data[index + 1].split()[2])
                self.b = ufloat(data[index].split()[3], data[index + 1].split()[3])
                self.c = ufloat(data[index].split()[4], data[index + 1].split()[4])
                self.alpha = ufloat(data[index].split()[5], data[index + 1].split()[5])
                self.beta = ufloat(data[index].split()[6], data[index + 1].split()[6])
                self.gamma = ufloat(data[index].split()[7], data[index + 1].split()[7])

            if 'SYMM' in line:  # SymOPs are in lines starting with 'SYMM', no identity transformation is given

                index += 0
                list_of_symOPs.append(_res_symOP(line))

            if 'FVAR' in line:  # Below this line are the atoms in the parameter list

                index += 1
                f = 0

                while True:

                    # This discriminates the lines with the coordinates from the additional information
                    if len(data[index + f].split()) >= 7:
                        element_symbol.append(data[index + f].split()[0])
                        element_xyz_frac.extend(data[index + f].split()[2:5])

                    # .res/.ins files "end" their coordinate sections with 'HKLF', 'lst' files with 'REM'
                    if data[index + f].split()[0] == breaker:
                        break

                    f += 1  # Increase increment by 1

        # Clean up symbols and transform the list to an array
        self.sym = np.asarray([_clean_up_symbol(entry) for entry in element_symbol], dtype=np.object)

        self.get_charge()  # Get charges and number of atoms
        self.get_cov_radii(settings)  # Get covalent radii
        self.get_identifiers()
        self.get_sym_idf()
        self.col_at_rgb = np.transpose(np.repeat(self.col_glob_rgb, self.n_atoms).reshape((3, self.n_atoms)))

        self.cor_frac = np.asarray(element_xyz_frac, dtype=np.float)  # Transform strings to floats
        self.cor_frac = np.reshape(self.cor_frac, (self.n_atoms, 3))  # Reshape coordinate array

        self.symOPs = list_of_symOPs

        if filetype == 'lst':  # .lst files contain coordinate uncertainties

            element_xyz_frac_std = []

            for index, line in enumerate(data):

                if 'ATOM' in line:  # This is the unique identifier for the parameter section with standard deviations

                    index += 3
                    f = 0

                    for atom in range(self.n_atoms):  # The number of atoms was previously established

                        if len(data[index + f].split()) < 3:  # No uncertainties (fixed atomic positions)

                            element_xyz_frac_std.extend([0.0, 0.0, 0.0])

                        else:

                            element_xyz_frac_std.extend(data[index + f].split()[1:4])

                        f += 3  # Increase increment by 3

        else:

            element_xyz_frac_std = np.zeros((self.n_atoms, 3))  # No uncertainties in .res/.ins files

        element_xyz_frac_std = np.asarray(element_xyz_frac_std, dtype=np.float)  # Transform strings to floats
        self.std_frac = np.reshape(element_xyz_frac_std, (self.n_atoms, 3))  # Reshape coordinate array

        self.calc_trans_mat()  # Calculate transformation matrix

    def read_cif_file(self, logger, data, settings):
        """ Reads .cif files and returns symbols, cart. coordinates and coordinate uncertainties """

        def _transform_to_float(value):
            """ Splits the standard deviation of the value """

            return ufloat(np.float(value), 0.0) if value.find('(') == -1 else ufloat_fromstr(value)

        def _multi_transform_to_float(values):
            """ Transform all items in a list """

            a, b = [], []

            for value in values:
                value_as_ufloat = _transform_to_float(value)
                a.append(value_as_ufloat.nominal_value)
                b.append(value_as_ufloat.std_dev)

            return np.asarray(a, dtype=np.float), np.asarray(b, dtype=np.float)

        element_symbol, element_xyz_frac, element_u_val = [], [], []
        list_of_symOPs = []

        for index, line in enumerate(data):

            if '_cell_length_a' in line:  # Start at line containing '_cell_length_a'

                index += 0
                self.a = _transform_to_float(data[index].split()[-1])
                self.b = _transform_to_float(data[index + 1].split()[-1])
                self.c = _transform_to_float(data[index + 2].split()[-1])
                self.alpha = _transform_to_float(data[index + 3].split()[-1])
                self.beta = _transform_to_float(data[index + 4].split()[-1])
                self.gamma = _transform_to_float(data[index + 5].split()[-1])

            if '_symmetry_equiv_pos_as_xyz' in line:

                index += 1
                f = 0

                while len(data[index + f].split()) != 0:

                    if len(data[index + f].split()) == 2:  # Number + symOPs as one string

                        list_of_symOPs.append(data[index + f].split()[1])

                    elif len(data[index + f].split()) == 3:  # Only SymOPs as individual strings

                        comb_string = data[index + f].split()[0] + data[index + f].split()[1] + \
                                      data[index + f].split()[2]

                        list_of_symOPs.append(eval(comb_string))

                    f += 1  # Increase increment by 1

            if '_atom_site_fract_x' in line:

                index += 0
                f = 0

                # Go to the line which contains the first atom
                while len(data[index + f].split()) == 1:

                    index += 1

                    if len(data[index + f].split()) != 1:
                        break

                while len(data[index + f].split()) > 2:

                    if len(data[index + f].split()[1]) in [1, 2]:

                        element_symbol.append(data[index + f].split()[1].lower().capitalize())

                    elif len(data[index + f].split()[1]) > 2:

                        logger.pt_unsupported_element(data[index + f].split()[1])
                        raise ValueError(logger.pt_exiting())

                    # Append fractional coordinates and anisotropy values
                    element_xyz_frac.extend(data[index + f].split()[2:5])
                    element_u_val.append(data[index + f].split()[5])

                    f += 1  # Increase increment by 1

        # Transform symbol list to array
        self.sym = np.asarray(element_symbol)

        self.get_charge()  # Get charges and number of atoms
        self.get_cov_radii(settings)  # Get covalent radii
        self.get_identifiers()
        self.get_sym_idf()
        self.col_at_rgb = np.transpose(np.repeat(self.col_glob_rgb, self.n_atoms).reshape((3, self.n_atoms)))

        element_xyz_frac, element_xyz_frac_std = _multi_transform_to_float(element_xyz_frac)

        # Reshape coordinate arrays
        self.cor_frac = np.reshape(element_xyz_frac, (self.n_atoms, 3))  # Reshape coordinate array
        self.std_frac = np.reshape(element_xyz_frac_std, (self.n_atoms, 3))  # Reshape coordinate array

        self.symOPs = list_of_symOPs[1::]  # Exclude unity transformation

        self.calc_trans_mat()  # Calculate transformation matrix

    def expand_by_symOP(self, symOP, settings):
        """ Expands the fractional coordinates with the given symOP """

        def _symOP_to_float(my_string):
            """ Transforms .cif symmetry operations to executable form """

            if '.' in my_string:

                pass

            else:

                # Arrays for integer -> float conversion
                int_num = np.arange(10, dtype=np.int)
                flt_num = np.arange(10, dtype=np.float)

                for number in range(10):  # Update all possible numbers

                    my_string = my_string.replace(str(int_num[number]), str(flt_num[number]))

            # Split string and return value(s)
            return my_string.split(',')

        def _gen_equiv_pos(symOP):
            """ Returns fractional coordinates after symmetry operation """

            x, y, z = np.transpose(self.cor_frac)

            return np.transpose(np.vstack((eval(symOP[0]), eval(symOP[1]), eval(symOP[2]))))

        def _no_duplicate(pos):
            """ Returns 'True' if the coordinate is not in the coordinate array """

            return len(np.where(np.sum(np.abs(cor_ex[pos] - cor_new), axis=1) < 0.01)[0]) == 0

        sym_new, cor_new, cor_std_new = np.copy(self.sym), np.copy(self.cor_frac), np.copy(self.std_frac)

        cor_ex = _gen_equiv_pos(_symOP_to_float(symOP))  # Expand fractional coordinates

        for pos in range(self.n_atoms):

            if _no_duplicate(pos):  # If the position is not already occupied

                sym_new = np.hstack((sym_new, self.sym[pos]))
                cor_new = np.vstack((cor_new, cor_ex[pos]))
                cor_std_new = np.vstack((cor_std_new, self.std_frac[pos]))

        self.sym = sym_new
        self.cor_frac = cor_new
        self.std_frac = cor_std_new
        self.get_charge()  # Get charges and number of atoms
        self.get_identifiers()
        self.get_sym_idf()
        self.get_cov_radii(settings)  # Update covalent radii
        self.frac_to_cart_bonds(settings)  # Update bonds

        # Update atomic colors
        self.col_at_rgb = np.transpose(np.repeat(self.col_glob_rgb, self.n_atoms).reshape((3, self.n_atoms)))

    def delete_positions(self, settings, del_list):
        """ Deletes given positions and update properties """

        self.sym = np.delete(self.sym, np.asarray(del_list))
        self.cor_frac = np.delete(self.cor_frac, np.asarray(del_list), 0)
        self.std_frac = np.delete(self.std_frac, np.asarray(del_list), 0)
        self.chg = np.delete(self.chg, np.asarray(del_list))
        self.cor = np.delete(self.cor, np.asarray(del_list), 0)

        self.get_charge()  # Get charges and number of atoms
        self.get_identifiers()
        self.get_sym_idf()
        self.get_cov_radii(settings)  # Update covalent radii
        self.frac_to_cart_bonds(settings)  # Update bonds

        # Update atomic colors
        self.col_at_rgb = np.transpose(np.repeat(self.col_glob_rgb, self.n_atoms).reshape((3, self.n_atoms)))

    def frac_to_cart_bonds(self, settings):
        """ Calculates regular cartesian coordinates from fractional coordinates including bonds """

        self.cor = np.transpose(np.dot(self.trans_mat, np.transpose(self.cor_frac)))  # Transform coordinates
        self.cor = np.asarray(unp.nominal_values(self.cor), dtype=np.float)
        self.get_bonds(settings)

    def final_transformation(self):
        """ Final transformation of the (expanded) coordinates """

        # Determine positions of fixed/unrefined H/heavy atoms
        pos_h_fix = np.where(np.sum(self.std_frac, axis=1) == 0.0)[0]

        # Convert fractional to cartesian coordinates
        element_coords = np.transpose(np.dot(self.trans_mat, np.transpose(unp.uarray(self.cor_frac, self.std_frac))))
        self.cor = np.asarray(unp.nominal_values(element_coords), dtype=np.float)
        self.std = np.asarray(unp.std_devs(element_coords), dtype=np.float)

        self.std[pos_h_fix] = np.array([0.0, 0.0, 0.0])  # Set stds of fixed/unrefined H/heavy atoms to zero
        self.finalized = True

    def calc_trans_mat(self):
        """ Calculates the transformation matrix for cartesian coordinates from fractional coordinates,
            assumes cell axes in Angstrom and angles in degrees """

        # Transform degrees to radians and calculate cell volume
        alpha, beta, gamma = unp.radians(self.alpha), unp.radians(self.beta), unp.radians(self.gamma)

        self.cell_vol = self.a * self.b * self.c * (1.0 - unp.cos(alpha) ** 2 - unp.cos(beta) ** 2 - unp.cos(
            gamma) ** 2 +
                                                    2.0 * unp.cos(alpha) * unp.cos(beta) * unp.cos(gamma)) ** 0.5

        # Calculate matrix elements
        # http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm
        # http://www.pymolwiki.org/index.php/Cart_to_frac
        # http://physics.bu.edu/~erikl/research/tools/crystals/read_cif.py
        m11 = self.a
        m12 = self.b * unp.cos(gamma)
        m13 = self.c * unp.cos(beta)
        m21 = ufloat(0.0, 0.0)
        m22 = self.b * unp.sin(gamma)
        m23 = (self.c * (unp.cos(alpha) - unp.cos(beta) * unp.cos(gamma))) / unp.sin(gamma)
        m31 = ufloat(0.0, 0.0)
        m32 = ufloat(0.0, 0.0)
        m33 = self.cell_vol / (self.a * self.b * unp.sin(gamma))

        # Setup and return transformation matrix
        self.trans_mat = unp.matrix([[m11, m12, m13],
                                     [m21, m22, m23],
                                     [m31, m32, m33]])

        # Round values and recombine to transformation matrix        
        nom = np.around(unp.nominal_values(self.trans_mat), 10)
        dev = np.around(unp.std_devs(self.trans_mat), 10)

        self.trans_mat = unp.uarray(nom, dev)

    def get_bonds(self, settings):
        """ Determines the bond indices and distances of the molecule """

        # Calculate all unique combinations
        combinations = unique(np.sort(cartesian_product([np.arange(self.n_atoms),
                                                         np.arange(self.n_atoms)])))

        xx, yy = np.transpose(combinations)

        # Determine indices of elements excluding the main diagonal
        indices = np.where(xx != yy)[0]

        # Exclude entries from combinations array and 'update' xx and yy
        combinations = np.take(combinations, indices, axis=0)
        xx, yy = np.transpose(combinations)

        # Calculate the distances between identifier pairs
        dist_pairs = np.sqrt(np.sum((np.take(self.cor, xx, axis=0) -
                                     np.take(self.cor, yy, axis=0)) ** 2, axis=1))

        # Calculate the limit (sum of covalent radii times factor) of the associated bond type
        dist_limit = (np.take(self.rad_cov, xx) +
                      np.take(self.rad_cov, yy)) * settings.det_bnd_fct

        # Determine indices where distance is below the sum of the radii
        indices = np.where(dist_pairs <= dist_limit)[0]

        # Update bond indices and distances
        self.bnd_idx = np.take(combinations, indices, axis=0)
        self.n_bonds = len(indices)

        self.col_bnd_rgb = np.transpose(np.repeat(self.col_glob_rgb, self.n_bonds).reshape((3, self.n_bonds)))

    def get_export_file(self, logger, example, prefix=None):
        """ Gets a valid outfile name from the user """

        self.outfile_name = get_file_name(logger, example, prefix)

    def user_expansion(self, logger, settings):
        """ Handles the interactive user expansion of Xray structures """

        question = "\n>>  Enter your choice: "
        picker_type = 'cluster'

        # Calculate initial set of Cartesian coordinates and bonds
        self.frac_to_cart_bonds(settings)

        while True:

            choices = logger.pt_x_ray_menu(self.n_atoms, self.symOPs, picker_type)

            if self.show_mol:

                viewmol_vtk = ap.Molecular_Viewer_vtk(settings)
                viewmol_vtk.make_fractional_plot(self, settings, picker_type)
                logger.pt_plotting()  # Inform user of the new plot
                logger.pt_plotting_screenshot()
                del_list = viewmol_vtk.show(self, self, settings)
                self.show_mol = True

                if len(del_list) > 0:
                    self.delete_positions(settings, del_list)
                    logger.pt_plotting_deleted(len(del_list))
                    self.show_mol = True

            operation = logger.get_menu_choice(choices, question, return_type='int')

            if operation == -10:  # Exit the menu

                self.final_transformation()
                break

            elif operation in [-4, -5]:  # Write structure to .xyz or .xyzs outfile

                if not self.finalized:  # Make final transformation if needed

                    self.final_transformation()

                if operation == -4:  # .xyz export

                    self.get_export_file(logger, example='myfile.xyz', prefix=None)

                    write_xyz_file(logger, self.outfile_name, self.sym, self.cor)

                elif operation == -5:  # .xyzs export

                    pos_dis, col_dis_rgb, pos_inter, col_inter_rgb = None, None, None, None

                    self.get_export_file(logger, example='myfile.xyzs', prefix=None)

                    write_xyzs_file(logger, self.outfile_name, self.sym, self.cor, self.std, self.col_at_rgb,
                                    self.rad_plt_vtk, self.bnd_idx, self.col_bnd_rgb,
                                    np.repeat(settings.rad_bnd, self.n_bonds), pos_dis, col_dis_rgb, pos_inter,
                                    col_inter_rgb, self.cam_vtk_pos, self.cam_vtk_wxyz, settings.scale_glob)

            elif operation == -2:  # Change picker type

                if picker_type == 'cluster':

                    picker_type = 'normal'

                else:

                    picker_type = 'cluster'

            elif operation == -1:  # Show molecule again

                self.show_mol = True

            elif operation in choices:

                if operation == max(choices):  # Custom symmetry operation

                    symOP = logger.get_menu_choice(choices,
                                                   question="\n> Enter a valid symmetry operation (e.g. 'x+1, -y, z+1/3'): ",
                                                   return_type='symOP')

                else:

                    logger.pt_sym_expand(self.symOPs[operation])

                    symOP = self.symOPs[operation]

                # Expand original coordinates by the symOP
                self.expand_by_symOP(symOP, settings)
                self.show_mol = True

            else:

                logger.pt_invalid_input()
