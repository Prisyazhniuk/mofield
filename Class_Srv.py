import numpy as np


# some static Service functions and constants
class Srv:
    # Bohr magneton (erg/G, i.e. emu)
    muB = 9.27401e-21

    # Boltzmann constant (erg/K)
    kB = 1.380649e-16

    # Avogadro number (1/mol)
    NA = 6.02214076e23

    # helpful const c1 = muB / kB
    c1 = muB / kB
    c2 = 9.27401 / 1.380649

    # transform vector from spherical to cartesian
    @staticmethod
    def to_cartesian(m, cos_t, cos_p):
        sin_t = np.sqrt(1.0 - cos_t * cos_t)
        m_x = m * sin_t * cos_p
        m_y = 0.0  # m * sin_t * np.sqrt(1.0 - cos_p * cos_p)
        m_z = m * cos_t
        return m_x, m_y, m_z

    # calculates angle cosine between 2 vectors
    @staticmethod
    def calculate_cos(a, b):
        a_abs = np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
        b_abs = np.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
        if a_abs > 0.0 and b_abs > 0:
            return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (a_abs * b_abs)
        else:
            return np.NaN

# ================================================================================================================
# предочтения
# a (octa):   Sc, Fe(2+), In, Ti(4+), Pt
# d (tetra):  Ga, Al, Si, Ge
# ================================================================================================================
    atom_wight = {
        'O'  : 15.99977,
        'Al' : 26.981539,
        'Ca' : 40.078,
        'Sc' : 44.955912,
        'Ti' : 47.867,
        'V'  : 50.9415,
        'Cr' : 51.9961,
        'Mn' : 54.938045,
        'Fe' : 55.845,
        'Co' : 58.933194,
        'Cu' : 63.546,
        'Ga' : 69.723,
        'Y'  : 88.90585,
        'Zr' : 91.224,
        'In' : 114.818,
        'La' : 138.90547,
        'Pr' : 140.90765,
        'Gd' : 157.25,
        'Tm' : 168.93421,
        'Lu' : 174.9668,
        'Pt' : 195.084,
        'Pb' : 207.2,
        'Bi' : 208.98040,
    }

    # YIG: Y_3 Fe_5 O_12
    mat_YIG = {
        'Y': 3,
        'Fe': 5,
        'O': 12
    }

    # GdIG: Gd_3 Fe_5 O_12
    mat_GdIG = {
        'Gd': 3,
        'Fe': 5,
        'O': 12
    }

    # Bi_0,95 Y_0,95 Lu_1,1 Fe_3,75 Ga_1,25 O_12
    mat_T_comp_01 = {
        'Bi': 0.95,
        'Y': 0.95,
        'Lu': 1.1,
        'Fe': 3.75,
        'Ga': 1.25,
        'O': 12
    }

    @staticmethod
    def get_mu(mat):
        mu_result = 0.0
        for a in mat:
            mu_result += mat[a] * Srv.atom_wight[a]
        return mu_result


# class to present the result of magnetization optimization
class MagnetizationEnergyResult:
    energy = np.NaN
    magn = [np.NaN, np.NaN, np.NaN]
    cos_theta = [np.NaN, np.NaN, np.NaN]
    success = False

    def __init__(self, energy, magn, cos_theta, success):
        self.energy = energy
        self.magn = magn
        self.cos_theta = cos_theta
        self.success = success
