import sys
import numpy as np
from scipy import optimize
from Class_Srv import Srv
from Class_Srv import MagnetizationEnergyResult
from Class_Material import Material
import MolField3L


# ================================================================================================================
# Пример работы с пленкой С компенсацией по железу
# ================================================================================================================
material_def_name = 'Material_YIG'    # material definition for pure IYG
model = MolField3L(material_def_name) # create MolField3L object for the material
# new magnetic ions content per f. u.
n_Bi = 0.95
n_Y = 0.95
n_Lu = 1.1
n_Fe = 3.75
n_Ga = 5.0 - n_Fe

new_na = 1.915
new_nd = n_Fe - new_na
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
m_init= model.material.get_max_magnetization() # sub-lattice magnetizations at T = 0
# build_zero_field_temperature_dependence(model, m_init, 1.0, 600.0, 0.5)
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
material_file = 'Material_BiGd_Film'
model = MolField3L(material_file) # create MolField3L object for GdIG
m_init= model.material.get_max_magnetization()
print (model.material.l_ad)
new_ad = model.material.l_ad * (1.0 + 0.035 * model.material.nc)
model.material.l_ad = new_ad
print (model.material.l_ad)
model.material.update_compound(na=1.95)
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
model = MolField3L() # create MolField3L object for the GdIG - default definition in Material class
# new magnetic ions content per f. u.
# Gd_2.4Bi_0.6Fe_4.22Ga_0.59Al_0.05Pt_0.11Cr_0.02Cu_0.02O_12
n_Ga = 0.59 # Gallium content
n_Ga_a = 0.0 # Gallium content in octa.
n_Ga_d = n_Ga - n_Ga_a
n_Fe = 4.22 # Ferrum content
n_Pt = 0.099 # 0.095 # 0.11 # ?
n_Cr = 0.02
n_Cu = 0.02
n_Al = 0.05
n_a_other = n_Cr + n_Cu + n_Pt
new_na = 2.0 - n_a_other - n_Ga_a # octa-Fe
new_nd = n_Fe - new_na # 2.22 # tetra-Fe
print(new_na, new_nd)
new_nc = 2.4 # dode-Gd
n_Bi = 3.0 - new_nc
# define material formula unit (f. u.)
new_material_fu ={
    'Bi' : n_Bi,
    'Gd' : new_nc,
    'Fe' : n_Fe, #4.22,
    'Pt' : n_Pt,
    'Cr' : n_Cr,
    'Cu' : n_Cu,
    'Ga' : n_Ga,
    'Al' : n_Al,
    'O' : 12.0
 }
new_mu = Srv.get_mu(new_material_fu)
model.material.update_compound(mu = new_mu, na = new_na, nd = new_nd, nc = new_nc)
# correct molecular field constant dor A_D interaction
# according to Dionne (1970)
n_x = 2.0 - new_na
n_y = 3.0 - new_nd
model.material.l_aa = model.material.l_aa * (1.0 - 0.42 * n_y)
model.material.l_dd = model.material.l_dd * (1.0 - 0.43 * n_x)
model.material.l_ad = model.material.l_ad * (1.0 - 0.125 * n_x - 0.127 * n_y)
# according to Hansen. 1983 foe Gd(3-x)Bi(x)Fe(5)O(12)
model.material.l_ad = model.material.l_ad * (1.0 + 0.035 * n_Bi)
m_init= model.material.get_max_magnetization() # sub-lattice magnetization at T = 0
# build_zero_field_temperature_dependence(model, m_init, 1.0, 600.0, 0.5)

# ================================================================================================================
# Simple model for Gd-film =======================================================================================
# ================================================================================================================
model = MolField3L() # create MolField3L object for the GdIG - default definition in Material class
# new magnetic ions content per f. u.
# simple f.u.^ Gd_2.4Bi_0.6Fe_2.0-xGa_xFe_2.22+xGa_0.78+xO_12
n_Ga = 0.78 # Gallium content
n_Ga_a = 0.12 # Gallium content in octa.
n_Ga_d = n_Ga - n_Ga_a
n_Fe = 4.22 # Ferrum content
new_na = 2.0 - n_Ga_a # octa-Fe
new_nd = n_Fe - new_na # 2.22 # tetra-Fe
new_nc = 2.4 # dode-Gd
n_Bi = 3.0 - new_nc
# define material formula unit (f. u.)
new_material_fu ={
    'Bi' : n_Bi,
    'Gd' : new_nc,
    'Fe' : n_Fe, #4.22,
    'Ga' : n_Ga,
    'O' : 12.0
 }
new_mu = Srv.get_mu(new_material_fu)
model.material.update_compound(mu = new_mu, na = new_na, nd = new_nd, nc = new_nc)
# correct molecular field constant dor A_D interaction
# according to Dionne (1970)
n_x = 2.0 - new_na
n_y = 3.0 - new_nd
model.material.l_aa = model.material.l_aa * (1.0 - 0.42 * n_y)
model.material.l_dd = model.material.l_dd * (1.0 - 0.43 * n_x)
model.material.l_ad = model.material.l_ad * (1.0 - 0.125 * n_x - 0.127 * n_y)
# according to Hansen. 1983 foe Gd(3-x)Bi(x)Fe(5)O(12)
model.material.l_ad = model.material.l_ad * (1.0 + 0.035 * n_Bi)
m_init= model.material.get_max_magnetization() # sub-lattice magnetization at T = 0
# build_zero_field_temperature_dependence(model, m_init, 1.0, 600.0, 0.5)