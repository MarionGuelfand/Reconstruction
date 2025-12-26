import numpy as np

# ===========================================
# Site / geometry constants / magnetic field
# ===========================================

refAlt = 1231  # DAQ room altitude [m]
ShowerCoreHeight = 1231
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
    "w_rec_max": 3.0,           # max width
    "theta_max": 80,            # max zenith angle (degrees)
    "reduced_chi2_adf_max": 20, 
    "reduced_chi2_swf_max": 20,
    "reduced_chi2_pwf_max": 25,
    "bin_width": 0.3           # bin width for omega   
}