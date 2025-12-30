import numpy as np

# ===========================================
# Site / geometry constants / magnetic field
# ===========================================

refAlt = 1264  # DAQ room altitude [m]
InjectionHeight = 1.0e7 # upper atmosphere [m]
c_light = 2.997924580e8


B_dec = 0.
B_inc = np.pi/2. + 1.0609856522873529
Bn = np.array([np.sin(B_inc)*np.cos(B_dec),np.sin(B_inc)*np.sin(B_dec),np.cos(B_inc)])
print(Bn)

# ==========================
# Analysis cuts / thresholds
# ==========================
cuts = {
    "width_rec_min": 1.25,
    "width_rec_max": 2.99,
    "scaling_factor_A_min": 1e6,
    "scaling_factor_A_max": 1e10,
    "theta_min": 63,
    "theta_max": 80,            # max zenith angle (degrees)
    "reduced_chi2_adf_max": 25, 
    "reduced_chi2_swf_max": 20,
    "reduced_chi2_pwf_max": 25,
    "antenna_number": 6
}

Energy = 'energy_em'

Simulation = True