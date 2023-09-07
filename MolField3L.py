import sys
import numpy as np
from scipy import optimize
from Class_Srv import Srv
from Class_Srv import MagnetizationEnergyResult
from Class_Material import Material


# class to calculate the properties of 3 sub-lattice ferrimagnet
class MolField3L:
    version = '1.0-12.10.200'

    eps = 1.0e-8

    material = None

    # constructor
    def __init__(self, material_definition = None):
        if material_definition is not None:
            material_module = __import__(material_definition)
            self.material = Material(material_module.data)
        else:
            self.material = Material()

    # calculate magnetization for given h_extern - external field and t - temperature
    # returns: array of 3 MagnetizationEnergyResult.
    # magn0 - initial magnetization, is an array of sublattice magnetizations
    # each sub-lattice has a number: a = 0, d = 1, c = 2
    def calculate_magnetization(self, h_extern, t, magn0):
        work_array = []

        cos_theta_m = [-1.0, 1.0, -1.0]
        res_m = self.check_energy_min_for_extremal_master_angle(h_extern, t, magn0, cos_theta_m)
        if res_m is not None and res_m.success:
            magn0 = res_m.magn
            work_array.append(res_m)

        cos_theta_p = [1.0, -1.0, 1.0]
        res_p = self.check_energy_min_for_extremal_master_angle(h_extern, t, magn0, cos_theta_p)
        if res_p is not None and res_p.success:
            magn0 = res_p.magn
            work_array.append(res_p)

        test = optimize.minimize_scalar(self.calculate_energy, bounds=(-1.0, 1.0), args=(h_extern, t, magn0),
                                        method='bounded')
        if test.success:
            cos_theta = [test.x, -test.x, test.x]
            res_0 = self.get_magnetization_and_energy_for_master_angle(h_extern, t, magn0, cos_theta)
            if res_0.success:
                work_array.append(res_0)

        res_array = []
        arr_len = len(work_array)
        while arr_len > 0:
            i_pos = -1
            e_min = sys.float_info.max
            for i in range(len(work_array)):
                if work_array[i].energy < e_min:
                    i_pos = i
                    e_min = work_array[i].energy
            if i_pos > -1:
                res = work_array[i]
                res_array.append(res)
                work_array.remove(res)
            else:
                arr_len = 0
            arr_len = len(work_array)
        return res_array

    # function to be used in minimize_scalar to find master angle for energy minimum
    def calculate_energy(self, master_cos, h_extern, t, magn0):
        cos_theta = [master_cos, -master_cos, master_cos]
        res = self.get_magnetization_and_energy_for_master_angle(h_extern, t, magn0, cos_theta)
        return res.energy

    # check whether energy minimum exists for master angle cosine equals +1 or -1
    # master angle cosine is cos_theta[0]
    def check_energy_min_for_extremal_master_angle(self, h_extern, t, magn, cos_theta):
        cos_theta_ex = [cos_theta[0], cos_theta[1], cos_theta[2]]
        if cos_theta[0] == -1.0:
            cos_theta_ex[0] += self.eps
        else:
            if cos_theta[0] == 1.0:
                cos_theta_ex[0] -= self.eps
            else:
                raise Exception('Wrong master angle')
        res = self.get_magnetization_and_energy_for_master_angle(h_extern, t, magn, cos_theta)
        res_ex = self.get_magnetization_and_energy_for_master_angle(h_extern, t, magn, cos_theta_ex)
        if res_ex.energy > res.energy: # Energy minimum
            return res
        else:
            return None

    # returns full (zeeman and anisotropy) energy
    def get_energy(self, h_ext, magn, cos_theta):
        e_zeeman = -1.0 * h_ext * (magn[0] * cos_theta[0] + magn[1] * cos_theta[1] + magn[2] * cos_theta[2])
        e_ua = 0.0
        if self.material.ku != 0.0:
            ma_x, ma_y, ma_z = Srv.to_cartesian(magn[0], cos_theta[0], Material.cos_phi[0])
            ma = np.sqrt(ma_x ** 2 + ma_y ** 2 + ma_z ** 2)
            if ma > 0.0:
                cos_mn = Srv.calculate_cos([ma_x, ma_y, ma_z], self.material.film_normal)
                if not np.isnan(cos_mn):
                    e_ua = self.material.ku * cos_mn * cos_mn
        return e_zeeman + e_ua

    # calculates magnetization for master angle and corresponding energy
    # cos_theta_master - is cos_theta[0]
    # h_extern - external magnetic field (along z-axis always)
    # t - temperature
    # magn - initial magnetization array
    # cos_theta - initial array of cosines
    def get_magnetization_and_energy_for_master_angle(self, h_extern, t, magn, cos_theta):
        success, magn, cos_theta = self.optimize_magnetization_for_angle(h_extern, t, magn, cos_theta)
        energy = self.get_energy(h_extern, magn, cos_theta)
        test = self.is_magnetization_along_field(h_extern, magn, cos_theta)
        if not test:
            success = False
        return MagnetizationEnergyResult(energy, magn, cos_theta, success)

    # find optimal sub-lattice magnetization values and required angles
    def optimize_magnetization_for_angle(self, h_ext, t, magn, cos_theta):
        max_step_count = 5000
        success = True
        step_count = 0
        do_repeat = True
        while do_repeat:
            res = optimize.root(self.brillion_eq_sys, x0=magn, args=(h_ext, t, cos_theta), method='lm')
            if res.success:
                magn = [res.x[0], res.x[1], res.x[2]]
                d_theta = self.optimize_slave_angles(h_ext, magn, cos_theta)
                if d_theta < self.eps:
                    do_repeat = False
            else:
                do_repeat = False
                success = False
            step_count += 1
            if step_count > max_step_count:
                do_repeat = False
                success = False
        return success, magn, cos_theta

    # returns uniaxial anisotropy field caused by a-sublattice only
    def get_uniaxial_anisotropy_field(self, ma_x, ma_y, ma_z, cos_t):
        hu_x = 0.0
        hu_z = 0.0
        if self.material.ku != 0.0:
            ma = np.sqrt(ma_x ** 2 + ma_y ** 2 + ma_z ** 2)
            if ma > 0.0:
                cos_mn = Srv.calculate_cos([ma_x, ma_y, ma_z], self.material.film_normal)
                if not np.isnan(cos_mn):
                    hu = -2.0 * self.material.ku * cos_mn / ma
                    hu_x = hu * np.sqrt(1.0 - cos_t ** 2)
                    hu_z = hu * cos_t
        return hu_x, hu_z

    # returns molecular fields for each sub-lattice
    def get_molecular_fields(self, h_extern, magn, cos_theta):
        ma_x, ma_y, ma_z = Srv.to_cartesian(magn[0], cos_theta[0], Material.cos_phi[0])
        md_x, md_y, md_z = Srv.to_cartesian(magn[1], cos_theta[1], Material.cos_phi[1])
        if len(magn) > 2:
            mc_x, mc_y, mc_z = Srv.to_cartesian(magn[2], cos_theta[2], Material.cos_phi[2])
        else:
            mc_x, mc_y, mc_z = 0.0, 0.0, 0.0

        hu_x, hu_z = self.get_uniaxial_anisotropy_field(ma_x, ma_y, ma_z, cos_theta[0])

        ha_x = self.material.l_aa * ma_x + self.material.l_ad * md_x + self.material.l_ac * mc_x + hu_x
        ha_z = self.material.l_aa * ma_z + self.material.l_ad * md_z + self.material.l_ac * mc_z + h_extern + hu_z
        ha = (ha_x * ha_x + ha_z * ha_z) ** 0.5

        hd_x = self.material.l_ad * ma_x + self.material.l_dd * md_x + self.material.l_dc * mc_x
        hd_z = self.material.l_ad * ma_z + self.material.l_dd * md_z + self.material.l_dc * mc_z + h_extern
        hd = (hd_x * hd_x + hd_z * hd_z) ** 0.5

        hc_x = self.material.l_ac * ma_x + self.material.l_dc * md_x + self.material.l_cc * mc_x
        hc_z = self.material.l_ac * ma_z + self.material.l_dc * md_z + self.material.l_cc * mc_z + h_extern
        hc = (hc_x * hc_x + hc_z * hc_z) ** 0.5

        h_abs = [ha, hd, hc]
        h_z = [ha_z, hd_z, hc_z]
        h_x = [ha_x, hd_x, hc_x]
        return h_abs, h_z, h_x

    # system of 3 brillion equations for each sub-lattice
    def brillion_eq_sys(self, magn, h_extern, t, cos_theta):

        h_abs, h_z, h_x = self.get_molecular_fields(h_extern, magn, cos_theta)

        beta_a = Srv.c1 * h_abs[0] / t
        beta_d = Srv.c1 * h_abs[1] / t
        beta_c = Srv.c1 * h_abs[2] / t

        th_a = np.tanh(beta_a)
        th_a6 = np.tanh(6.0 * beta_a)

        th_d = np.tanh(beta_d)
        th_d6 = np.tanh(6.0 * beta_d)

        th_c = np.tanh(beta_c)
        th_c8 = np.tanh(8.0 * beta_c)

        fa = (6.0 * th_a - th_a6 - magn[0] * self.material.alpha_a * th_a * th_a6) * 1.0e7
        fd = (6.0 * th_d - th_d6 - magn[1] * self.material.alpha_d * th_d * th_d6) * 1.0e7
        fc = (8.0 * th_c - th_c8 - magn[2] * self.material.alpha_c * th_c * th_c8) * 1.0e7

        return [fa, fd, fc]

    def brillion_eq_sys_zero_field(self, x, t):
        m_a = x[0]
        m_d = x[1]
        m_c = x[2]

        h_a = self.material.l_aa * m_a + self.material.l_ad * m_d + self.material.l_ac * m_c
        h_d = self.material.l_ad * m_a + self.material.l_dd * m_d + self.material.l_dc * m_c
        h_c = self.material.l_ac * m_a + self.material.l_dc * m_d + self.material.l_cc * m_c

        beta_a = Srv.c1 * h_a / t
        beta_d = Srv.c1 * h_d / t
        beta_c = Srv.c1 * h_c / t

        th_a = np.tanh(beta_a)
        th_a6 = np.tanh(6.0 * beta_a)

        th_d = np.tanh(beta_d)
        th_d6 = np.tanh(6.0 * beta_d)

        th_c = np.tanh(beta_c)
        th_c8 = np.tanh(8.0 * beta_c)

        fa = (6.0 * th_a - th_a6 - m_a * self.material.alpha_a * th_a * th_a6) * 1.0e10
        fd = (6.0 * th_d - th_d6 - m_d * self.material.alpha_d * th_d * th_d6) * 1.0e10
        fc = (8.0 * th_c - th_c8 - m_c * self.material.alpha_c * th_c * th_c8) * 1.0e10

        return [fa, fd, fc]

    # equation system which is used to optimize angles (find increment)
    def angles_eq_system(self, h_extern, magn, cos_theta):
        h_abs, h_z, h_x = self.get_molecular_fields(h_extern, magn, cos_theta)

        k0 = self.material.l_ad * magn[0]
        k1 = self.material.l_dd * magn[1] - h_abs[1]
        k2 = self.material.l_dc * magn[2]
        f1 = k0 * cos_theta[0] + k1 * cos_theta[1] + k2 * cos_theta[2] + h_extern

        k0 = self.material.l_ac * magn[0]
        k1 = self.material.l_dc * magn[1]
        k2 = self.material.l_cc * magn[2] - h_abs[2]
        f2 = k0 * cos_theta[0] + k1 * cos_theta[1] + k2 * cos_theta[2] + h_extern

        return [f1, f2]

    # build jacobian and find inrement for slave angles from linear equation system
    def angle_increment_from_jacobian(self, h_extern, magn, cos_theta):
        max_delta = 0.05
        jacob = np.zeros((2, 2))
        for i_col in range(2):
            cos_theta_m = [cos_theta[0], cos_theta[1], cos_theta[2]]
            cos_theta_p = [cos_theta[0], cos_theta[1], cos_theta[2]]
            cos_theta_m[i_col + 1] = cos_theta_m[i_col + 1] - self.eps
            cos_theta_p[i_col + 1] = cos_theta_p[i_col + 1] + self.eps
            d = cos_theta_m[i_col + 1] + 1.0
            if d < 0.0:
                cos_theta_m[i_col + 1] = cos_theta_m[i_col + 1] - d
                cos_theta_p[i_col + 1] = cos_theta_p[i_col + 1] - d
            d = cos_theta_p[i_col + 1] - 1.0
            if d > 0.0:
                cos_theta_m[i_col + 1] = cos_theta_m[i_col + 1] - d
                cos_theta_p[i_col + 1] = cos_theta_p[i_col + 1] - d

            fm = self.angles_eq_system(h_extern, magn, cos_theta_m)
            fp = self.angles_eq_system(h_extern, magn, cos_theta_p)
            for i_row in range(2):
                jacob[i_row][i_col] = 0.5 * (fp[i_row] - fm[i_row]) / self.eps
        right_part = self.angles_eq_system(h_extern, magn, cos_theta)
        right_part[0] = -1.0 * right_part[0]
        right_part[1] = -1.0 * right_part[1]
        delta = np.linalg.solve(jacob, right_part)
        dx_max = 0.0
        for i in range(2):
            if np.abs(delta[i]) > dx_max:
                dx_max = np.abs(delta[i])
        if dx_max > max_delta:
            for i in range(2):
                delta[i] = max_delta * delta[i] / dx_max
        return delta

    # optimizing slave-angles
    def optimize_slave_angles(self, h_extern, magn, cos_theta):
        delta = self.angle_increment_from_jacobian(h_extern, magn, cos_theta)
        d_theta = 0.0
        for i in range(2):
            d = cos_theta[i + 1] + delta[i]
            if d > 1.0:
                delta[i] = 1.0 - cos_theta[i + 1]
            if d < -1.0:
                delta[i] = -1.0 - cos_theta[i + 1]
            if np.fabs(delta[i]) > d_theta:
                d_theta = np.fabs(delta[i])
            cos_theta[i + 1] = cos_theta[i + 1] + delta[i]
        return d_theta

    # check whether each sub-lattice magnetization i directed along it's molecular field
    def is_magnetization_along_field(self, h_extern, magn, cos_theta):
        h_abs, h_z, h_x = self.get_molecular_fields(h_extern, magn, cos_theta)
        ma_x, ma_y, ma_z = Srv.to_cartesian(magn[0], cos_theta[0], Material.cos_phi[0])
        md_x, md_y, md_z = Srv.to_cartesian(magn[1], cos_theta[1], Material.cos_phi[1])
        mc_x, mc_y, mc_z = Srv.to_cartesian(magn[2], cos_theta[2], Material.cos_phi[2])
        criterium = 1.0
        cos = Srv.calculate_cos([h_x[0], 0.0, h_z[0]], [ma_x, ma_y, ma_z])
        if cos < criterium:
            criterium = cos
        cos = Srv.calculate_cos([h_x[1], 0.0, h_z[1]], [md_x, md_y, md_z])
        if cos < criterium:
            criterium = cos
        cos = Srv.calculate_cos([h_x[2], 0.0, h_z[2]], [mc_x, mc_y, mc_z])
        if cos < criterium:
            criterium = cos
        if criterium > 0.9:
            return True
        else:
            return False


# ================================================================================================================
# The code block to demonstrate the usage of MolField3L class
# ================================================================================================================
def build_dependence_on_master_angle(model_obj, h_extern, t, magn, cos_master_min, cos_master_max, cos_master_step):
    cos_theta_master = cos_master_min
    while cos_theta_master <= cos_master_max:
        cos_theta = [cos_theta_master, -cos_theta_master, cos_theta_master]
        res = model_obj.get_magnetization_and_energy_for_master_angle(h_extern, t, magn, cos_theta)
        mz = 0.0
        for i in range(3):
            mz += res.magn[i] * res.cos_theta[i]
        print(cos_theta_master, res.energy, mz, res.success)
        cos_theta_master += cos_master_step


def build_temperature_dependence(model_obj, h_extern, magn, t_min, t_max, t_step):
    t = t_min
    while t <= t_max:
        test_result = model_obj.calculate_magnetization(h_extern, t, magn)
        if len(test_result) > 0:
            m = test_result[0]
            mz_total = 0.0
            mx_total = 0.0
            for i in range(3):
                mx_total += m.magn[i] * np.sqrt(1.0 - m.cos_theta[i] * m.cos_theta[i])
                mz_total += m.magn[i] * m.cos_theta[i]
            m_fe_z = m.magn[0] * m.cos_theta[0] + m.magn[1] * m.cos_theta[1]
            magn = m.magn
        if len(test_result) > 1:
            m1 = test_result[1]
            mz_total_ex = 0.0
            mx_total_ex = 0.0
            for i in range(3):
                mz_total_ex = m1.magn[i] * m1.cos_theta[i]
                mx_total_ex += m.magn[i] * np.sqrt(1.0 - m.cos_theta[i] * m.cos_theta[i])
            m_fe_z_ex = m1.magn[0] * m1.cos_theta[0] + m1.magn[1] * m1.cos_theta[1]
            magn = m.magn

        if len(test_result) > 1:
            print(t, mz_total, mx_total, m_fe_z, mz_total_ex, mx_total_ex, m_fe_z_ex)
        if len(test_result) == 1:
            print(t, mz_total, mx_total, m_fe_z)
        t += t_step


def build_zero_field_temperature_dependence(model_obj, magn, t_min, t_max, t_step):
    magn[1] = -1.0 * magn[1]
    t = t_min
    if t <= 0.0:
        t = t_step
    while t <= t_max:
        res = optimize.root(model_obj.brillion_eq_sys_zero_field, x0=magn, args=(t), method='lm')
        magn = [res.x[0], res.x[1], res.x[2]]
        mz_total = res.x[0] + res.x[1] + res.x[2]
        # print(t, np.fabs(mz_total), magn[0], magn[1], magn[2])
        print(t, np.fabs(mz_total), magn[0], magn[1], magn[2]) #  - 273
        t += t_step


# ================================================================================================================
#   Example to call
# ================================================================================================================
material_def_name = 'Material_YIG'    # material definition for pure IYG
model = MolField3L(material_def_name) # create MolField3L object for the material
# new magnetic ions content per f. u.
delta_Bi = 0.0
n_Bi = 0.95 - delta_Bi
n_Y = 0.95 + delta_Bi
delta_Lu = 0.0
n_Lu = 1.1
n_Fe = 3.5
n_Ga = 1.25 # 5.0 - n_Fe
n_Ga = 1.5 # 5.0 - n_Fe

# new_na = 1.948
new_na = 2.0 - delta_Lu
# new_nd = n_Fe - new_na
new_nd = 1.5
new_nc = 0.0
# define material formula unit (f. u.)
new_material_fu ={
    'Bi' : n_Bi,
    'Y'  : n_Y,
    'Lu' : n_Lu,
    'Fe' : n_Fe,
    'Ga' : n_Ga,
    'O' : 12.0
 }
new_mu = Srv.get_mu(new_material_fu)
model.material.update_compound(mu = new_mu, na = new_na, nd = new_nd, nc = new_nc)
# correct molecular field constant for iron delution
n_x = 2.0 - new_na
n_y = 3.0 - new_nd
model.material.l_aa = model.material.l_aa * (1.0 - 0.42 * n_y)
model.material.l_dd = model.material.l_dd * (1.0 - 0.43 * n_x)
model.material.l_ad = model.material.l_ad * (1.0 - 0.125 * n_x - 0.127 * n_y)
# correct molecular field constant for A_D interaction due to Bi
model.material.l_ad = model.material.l_ad * (1.0 + 0.032 * n_Bi)
model.material.l_ad = model.material.l_ad * (1.0 - 0.009 * n_Lu)
# model.material.l_ad = model.material.l_ad * 0.875
m_init= model.material.get_max_magnetization() # sub-lattice magnetizations at T = 0
build_zero_field_temperature_dependence(model, m_init, 1.0, 600.0, 0.5)