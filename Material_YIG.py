# molecular field parameters for YIG
# Y3Fe5O12
# Roschmann, P., Hansen, P.: J. Appl. Phys. 52 (1981) 6257
data = {
    'lambdaAA': -50030.0, # exchange const for octa Fe (2 ions per a f.u.), Oe g / emu
    'lambdaAD': -72320.0,
    'lambdaAC': 0.0,
    'lambdaDD': -22430.0,
    'lambdaDC': 0.0,
    'lambdaCC': 0.0,
    'nA': 2.0,  # Octa
    'nD': 3.0, # Tetra
    'nC': 0,  # R
    'mu': 738.0, # f.u. molecular weight, g/mol
    'ku': 0.0,   # uniaxial anisotropy for master sub-lattices, i. e. a-ions
    'film_normal': [0.0, 0.0, 1.0] # film normal vector given by cos of angles between vector and external field
}