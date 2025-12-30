"""
This script processes a set of simulated cosmic ray events to compute correction coefficients
[f(air density, sin(alpha)] used in energy reconstruction.


The corrected energy estimator is defined as:
    E* = A / (sin(alpha) * f(air_density, sin(alpha)))

Main steps:
1. Load simulation input, ADF reconstruction, and spherical reconstruction data.
2. Merge the datasets and apply quality cuts (zenith angle, number of antennas, width, amplitude, etc.).
3. Compute geomagnetic factor (sin(alpha)) for each event.
4. Compute air density at the source position (X_source).
5. Compute normalized scaling factor: A / (E * sin(alpha)).
6. Split events randomly into training and test sets.
7. Fit a polynomial regression model (default degree=3) to correct for second-order dependencies 
   of the normalized scaling factor on geomagnetic factor and air density.
8. Save the polynomial features and regression model as a pickle file.
9. Save the regression coefficients (intercept + polynomial coefficients) as a CSV for reproducibility.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import utils as utils
import config as conf
sys.path.append(str(conf.analysis_dir))
import functions_analysis as func
import cuts as cuts
import pickle

disp=True
do_split = True

df_input = pd.read_csv(f"{conf.input_simus_file}", sep='\s+',
                        names=['eventname', 'zenith', 'azimuth', 'energy', 'energy_em', 'primary', 'xmaxdist',
                                'xmaxgrammage', 'x_xmax', 'y_xmax', 'z_xmax', 'x_core', 'y_core', 'z_core', 'corealt',
                                'antennasnumber', 'antennathreshold', 'amplitudethreshold'])

df_rec_adf = pd.read_csv(f"{conf.rec_adf_file}", sep='\s+',
                          names=['eventname', 'antennasnumber', 'zenithrec', 'errorzenith', 'azimuthrec', 'errorazimuth',
                                 'chi2_adf', 'sign', 'widthrec', 'amprec'])

df_rec_sphere = pd.read_csv(f"{conf.rec_sphere_file}", sep='\s+',
                          names=['eventname', 'antennasnumber', 'chi2_sphere', 'NaN', 'x_Xsource', 'y_Xsource',
                                 'z_Xsource', 'distance_Xsource', 't_source', 'zenith_rec_sphere', 'azimuth_rec_sphere'])

# -------------------------
# Merge tables
# -------------------------
df_merge = df_input.merge(df_rec_adf, on=['eventname', 'antennasnumber'])
df_merge = pd.merge(df_merge, df_rec_sphere, on=['eventname', 'antennasnumber'])

# Convert primary to numeric codes: proton=1, iron=2, other=0
df_merge['primary'] = df_merge['primary'].astype(str)
df_merge['primary'] = np.where(df_merge['primary'] == '2212', 1, 
                                np.where(df_merge['primary'].str.startswith('Fe'), 2, 0))

reduced_chi2_adf = func.compute_reduced_chi2(df_merge['chi2_adf'], 
                                    df_merge['antennasnumber'], n_params=4)

reduced_chi2_sphere = func.compute_reduced_chi2(df_merge['chi2_sphere'], 
                                    df_merge['antennasnumber'], n_params=4)

df_merge['chi2_adf_reduced'] = reduced_chi2_adf
df_merge['chi2_sphere_reduced'] = reduced_chi2_sphere


# -------------------------
# Apply quality cuts
# -------------------------
df_merge = df_merge.loc[df_merge['zenithrec'] >= cuts.cuts["theta_min"]]  
df_merge = df_merge.loc[df_merge['antennasnumber'] >= cuts.cuts["antenna_number"]]  
df_merge = df_merge.loc[df_merge['z_Xsource'] > 0]  
df_merge = df_merge.loc[df_merge['widthrec'] > cuts.cuts["width_rec_min"]]  
df_merge = df_merge.loc[df_merge['widthrec'] < cuts.cuts["width_rec_max"]]  
df_merge = df_merge.loc[df_merge['amprec'] != cuts.cuts["scaling_factor_A_min"]]  
df_merge = df_merge.loc[df_merge['amprec'] != cuts.cuts["scaling_factor_A_max"]] 
df_merge = df_merge.loc[df_merge['chi2_adf_reduced'] < cuts.cuts["reduced_chi2_adf_max"]] 
#df_merge = df_merge.loc[df_merge['chi2_sphere_reduced'] < cuts.cuts["reduced_chi2_swf_max"]] 

# -------------------------
# Compute geomagnetic factor
# -------------------------
geomagnetic_factor = func.get_sin_alpha(
    df_merge['zenithrec'], df_merge['azimuthrec'], cuts.B_inc, cuts.B_dec
)

df_merge['sin_alpha'] = geomagnetic_factor

# -------------------------
# Compute air density at source (obtained with SWF)
# -------------------------
source_positions = df_merge[['x_Xsource', 'y_Xsource', 'z_Xsource']].values
air_density = []

for pos in source_positions:
    height = func.height_Xsource(pos)
    density = func.get_density(height, model='grand_atm')
    air_density.append(density)

df_merge['air_density'] = np.array(air_density) * 1e3  # convert to kg/m^3

# -------------------------
# Some illustrative plots
# -------------------------
if disp == True:

    fig = plt.figure()
    plt.scatter(df_merge[cuts.Energy], df_merge['amprec'], c=df_merge['zenith'], cmap='jet',zorder=2, s=20)
    plt.colorbar(label=r'$\theta$ [°]')
    plt.grid()
    plt.xlabel(r'$E_{\rm primary}$ [EeV]')
    plt.ylabel(r'A')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()


    fig = plt.figure()
    plt.scatter(df_merge[cuts.Energy], df_merge['amprec']/df_merge['sin_alpha'], c=df_merge['zenith'], cmap='jet',zorder=2, s=20)
    plt.colorbar(label=r'$\theta$ [°]')
    plt.grid()
    plt.xlabel(r'$E_{\rm primary}$ [EeV]')
    plt.ylabel(r'A/$ \rm sin(\alpha)$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()

    fig = plt.figure()
    plt.scatter(df_merge[cuts.Energy], df_merge['widthrec'], c=df_merge['zenith'], cmap='jet',zorder=2, s=20)
    plt.colorbar(label=r'$\theta$ [°]')
    plt.grid()
    plt.xlabel(r'$E_{\rm primary}$ [EeV]')
    plt.ylabel(r'Width $\delta \omega$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()


# -------------------------
# Compute normalized scaling factor A / (sin(alpha)*E)
# -------------------------
norm_scaling_factor = df_merge['amprec'] / (geomagnetic_factor * df_merge[cuts.Energy])
df_merge['norm_scaling_factor'] = norm_scaling_factor


# -------------------------
# Split events for training and test
# -------------------------

if do_split:
    train_event_names, test_event_names = utils.split_events(df_merge['eventname'].values)
    df_train = df_merge[df_merge['eventname'].isin(train_event_names)]
    df_test = df_merge[df_merge['eventname'].isin(test_event_names)]
else:
    df_train = df_merge.copy()  # tout le dataset comme "train"
    df_test = None

# -------------------------
# Compute correction coefficients
# -------------------------
poly_features, linreg_model, predicted_correction = utils.compute_correction_coefficients(
    geomagnetic_factor[df_merge['eventname'].isin(train_event_names)],
    df_train['air_density'].values,
    df_train['norm_scaling_factor'].values
)

df_train['coherence_factor'] = predicted_correction

# -------------------------
# Save coefficients to file
# -------------------------
coeff_file = conf.coefficients_file_pkl 
with open(coeff_file, 'wb') as f:
    pickle.dump({'poly_features': poly_features, 'linreg_model': linreg_model}, f)

print(f"Correction coefficients saved to {coeff_file}")

# -------------------------
# Create a DataFrame for CSV saving and save the coefficients DataFrame as CSV
# -------------------------
intercept = linreg_model.intercept_
coef = linreg_model.coef_


all_coef = np.hstack([intercept, coef])

df_coef = pd.DataFrame(all_coef, columns=['value'])
df_coef['name'] = ['intercept'] + [f'coef_{i}' for i in range(len(coef))]
df_coef['poly_degree'] = poly_features.degree  # ajouter le degré du polynôme

df_coef.to_csv(conf.coefficients_file_csv, index=False)
print(f"Correction coefficients saved to {conf.coefficients_file_csv}")



df_train['A_estimator'] = df_train['norm_scaling_factor'] * df_train[cuts.Energy] / df_train['coherence_factor']

df_train['resolution'] = (df_train['A_estimator'] - df_train[cuts.Energy]) / df_train[cuts.Energy] * 100
resolution = np.std(df_train['resolution'])
mean = np.mean(df_train['resolution'])
median = np.median(df_train['resolution'])

# Print results
print(f"Résolution totale : {resolution}%")
print(f"Moyenne totale : {mean}%")
print(f"Médiane totale : {median}%")

# -------------------------
# Some illustrative plots
# -------------------------

if disp == True:

    fig = plt.figure()
    plt.hist(df_train['resolution'], bins = 20)
    plt.title(f"resolution: {resolution:.2f} %")
    plt.xlabel(r"($E_{\rm recons}$-$E_{\rm primary}$)/$E_{\rm primary}$")
    plt.show()


