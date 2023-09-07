import numpy as np
from Class_Srv import Srv


# material properties definitions, default values for GdIG
class Material:

    # Molecular filed constants according to Dionne, 1976
    # exchange const for octa Fe (2 ions per a f.u.), Oe g / emu
    l_aa = -61292.6  # GdIG, 79D
    # exchange const for FeA-FeD, Oe g / emu
    l_ad = -91467.5  # GdIG, 79D
    # exchange const for FeA-Rc, Oe g / emu
    l_ac = -3243.8 # GdIG, 79D
    # exchange const for tetra Fe (3 ions per a f.u.), Oe g / emu
    l_dd = -28288.9 # GdIG, 79D
    # exchange const for FeD-Rc, Oe g / emu
    l_dc = -5657.8 # GdIG, 79D
    # exchange const for Rc-Rc, Oe g / emu
    l_cc = 0.0  # GdIG, 79D

    # f.u. molecular weight, g/mol
    mu = 942.96336  # Gd3Fe5O12

    # ions count per a f.u.
    na = 2.0 # Octa # GdIG, 79D
    nd = 3.0 # Tetra
    nc = 3.0  # R

    # uniaxial anisotropy for lattices
    ku = 0.0

    # film normal vector given by cos of angles between vector and external field
    film_normal = [0.0, 0.0, 1.0]

    cos_phi = [-1.0, 1.0, -1.0]

    # helpful calculations
    alpha_a = mu / (Srv.NA * Srv.muB * na)
    alpha_d = mu / (Srv.NA * Srv.muB * nd)
    alpha_c = mu / (Srv.NA * Srv.muB * nc)

    def __init__(self, material_data = None):
        if material_data is not None:
            self.l_aa = material_data['lambdaAA']
            self.l_ad = material_data['lambdaAD']
            self.l_ac = material_data['lambdaAC']
            self.l_dd = material_data['lambdaDD']
            self.l_dc = material_data['lambdaDC']
            self.l_cc = material_data['lambdaCC']
            self.mu = material_data['mu']
            self.na = material_data['nA']
            self.nd = material_data['nD']
            self.nc = material_data['nC']
            self.ku = material_data['ku']
            self.film_normal = material_data['film_normal']
        if self.film_normal[0] ** 2 + self.film_normal[1] ** 2 + self.film_normal[2] ** 2 != 1:
            raise Exception('Wrong film normal')
        if self.film_normal[1] != 0.0:
            raise Exception('Film normal cannot have non-zero y-component')
        self.alpha_a = self.mu / (Srv.NA * Srv.muB * self.na)
        self.alpha_d = self.mu / (Srv.NA * Srv.muB * self.nd)
        if self.nc > 0.0:
            self.alpha_c = self.mu / (Srv.NA * Srv.muB * self.nc)
        else:
            self.alpha_c = 0.0

    def update_compound(self, mu = np.NaN, na = np.NaN, nd = np.NaN, nc = np.NaN):
        if not np.isnan(mu):
            self.l_aa = self.l_aa * mu / self.mu
            self.l_ad = self.l_ad * mu / self.mu
            self.l_ac = self.l_ac * mu / self.mu
            self.l_dd = self.l_dd * mu / self.mu
            self.l_dc = self.l_dc * mu / self.mu
            self.l_cc = self.l_cc * mu / self.mu
            self.mu = mu
            if self.na > 0.0:
                alpha_a = mu / (Srv.NA * Srv.muB * self.na)
            if self.nd > 0.0:
                alpha_d = mu / (Srv.NA * Srv.muB * self.nd)
            if self.nc > 0.0:
                alpha_c = mu / (Srv.NA * Srv.muB * self.nc)
        if not np.isnan(na):
            self.na = na
            self.alpha_a = self.mu / (Srv.NA * Srv.muB * self.na)
        if not np.isnan(nd):
            self.nd = nd
            self.alpha_d = self.mu / (Srv.NA * Srv.muB * self.nd)
        if not np.isnan(nc):
            self.nc = nc
            if self.nc > 0.0:
                self.alpha_c = self.mu / (Srv.NA * Srv.muB * self.nc)

    def get_max_magnetization(self):
        factor = Srv.NA * Srv.muB / self.mu
        ma_max = factor * 5.0 * self.na
        md_max = factor * 5.0 * self.nd
        mc_max = factor * 7.0 * self.nc
        return [ma_max, md_max, mc_max]