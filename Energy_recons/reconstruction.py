"""
Apply precomputed correction coefficients to a dataset (simulations or real events)
and compute reconstructed energy or scaling factors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle

import config as conf
import utils as utils
sys.path.append(str(conf.analysis_dir))
import functions_analysis as func
import cuts as cuts

disp=True
# --------------------------------------------------
# Load precomputed correction coefficients
# --------------------------------------------------
coeff_file = conf.coefficients_file_pkl
with open(coeff_file, 'rb') as f:
    coeffs = pickle.load(f)

poly_features = coeffs['poly_features']
linreg_model = coeffs['linreg_model']

print(linreg_model)

print(f"Loaded correction coefficients from {coeff_file}")

if cuts.Simulation:
    df_input = pd.read_csv(f"{conf.output_dir}/input_simus.txt", sep='\s+',
                        names=['eventname', 'zenith', 'azimuth', 'energy', 'energy_em', 'primary', 'xmaxdist',
                                'xmaxgrammage', 'x_xmax', 'y_xmax', 'z_xmax', 'x_core', 'y_core', 'z_core', 'corealt',
                                'antennasnumber', 'antennathreshold', 'amplitudethreshold'])


df_rec_adf = pd.read_csv(f"{conf.recons_dir}/Rec_adf_recons.txt", sep='\s+',
                          names=['eventname', 'antennasnumber', 'zenithrec', 'errorzenith', 'azimuthrec', 'errorazimuth',
                                 'chi2_adf', 'sign', 'widthrec', 'amprec'])

df_rec_sphere = pd.read_csv(f"{conf.recons_dir}/Rec_sphere_wave_recons.txt", sep='\s+',
                          names=['eventname', 'antennasnumber', 'chi2_sphere', 'NaN', 'x_Xsource', 'y_Xsource',
                                 'z_Xsource', 'distance_Xsource', 't_source', 'zenith_rec_sphere', 'azimuth_rec_sphere'])


# -------------------------
# Merge tables
# -------------------------
df_data = pd.merge(df_input, df_rec_adf, on=['eventname', 'antennasnumber'])
df_data = pd.merge(df_data, df_rec_sphere, on=['eventname', 'antennasnumber'])


# Convert primary to numeric codes: proton=1, iron=2, other=0
df_data['primary'] = df_data['primary'].astype(str)
df_data['primary'] = np.where(df_data['primary'] == '2212', 1, 
                                np.where(df_data['primary'].str.startswith('Fe'), 2, 0))

reduced_chi2_adf = func.compute_reduced_chi2(df_data['chi2_adf'], 
                                    df_data['antennasnumber'], n_params=4)

reduced_chi2_sphere = func.compute_reduced_chi2(df_data['chi2_sphere'], 
                                    df_data['antennasnumber'], n_params=4)

df_data['chi2_adf_reduced'] = reduced_chi2_adf
df_data['chi2_sphere_reduced'] = reduced_chi2_sphere


# -------------------------
# Apply quality cuts
# -------------------------
df_data = df_data.loc[df_data['zenithrec'] >= cuts.cuts["theta_min"]]  
df_data = df_data.loc[df_data['antennasnumber'] >= cuts.cuts["antenna_number"]]  
df_data = df_data.loc[df_data['z_Xsource'] > 0]  
df_data = df_data.loc[df_data['widthrec'] > cuts.cuts["width_rec_min"]]  
df_data = df_data.loc[df_data['widthrec'] < cuts.cuts["width_rec_max"]]  
df_data = df_data.loc[df_data['amprec'] != cuts.cuts["scaling_factor_A_min"]]  
df_data = df_data.loc[df_data['amprec'] != cuts.cuts["scaling_factor_A_max"]] 
df_data = df_data.loc[df_data['chi2_adf_reduced'] < cuts.cuts["reduced_chi2_adf_max"]] 
#df_data = df_data.loc[df_data['chi2_sphere_reduced'] < cuts.cuts["reduced_chi2_swf_max"]] 


# -------------------------
# Split events for training and test
# -------------------------
train_event_names, test_event_names = utils.split_events(df_data['eventname'].values)
df_train = df_data[df_data['eventname'].isin(train_event_names)]
df_test = df_data[df_data['eventname'].isin(test_event_names)]

# --------------------------------------------------
# Compute geomagnetic factor
# --------------------------------------------------
geomagnetic_factor = func.get_sin_alpha(
    df_test['zenithrec'], df_test['azimuthrec'], cuts.B_inc, cuts.B_dec
)
geomagnetic_factor = np.array(geomagnetic_factor) 
# --------------------------------------------------
# Compute air density at source
# --------------------------------------------------
source_positions = df_test[['x_Xsource', 'y_Xsource', 'z_Xsource']].values
air_density = []

for pos in source_positions:
    height = func.height_Xsource(pos)
    density = func.get_density(height, model='grand_atm')
    air_density.append(density)

df_test['air_density'] = np.array(air_density) * 1e3  # convert to kg/m^3


# --------------------------------------------------
# Apply precomputed correction
# --------------------------------------------------
df_test['corrected_factor'] = utils.apply_correction_coefficients(
    geomagnetic_factor,
    df_test['air_density'].values,
    poly_features,
    linreg_model
)

pred = utils.apply_correction_coefficients(
    geomagnetic_factor,
    df_test['air_density'].values,
    poly_features,
    linreg_model
)

print("First 10 corrected factors:", pred[:10])
print("Coefficients in model:", linreg_model.coef_)
print("Intercept:", linreg_model.intercept_)

# --------------------------------------------------
# Compute reconstructed energy estimator
# --------------------------------------------------
df_test['energy_estimator'] = df_test['amprec'] / (geomagnetic_factor * df_test['corrected_factor'])

# --------------------------------------------------
# Optional: compute resolution if true energy available
# --------------------------------------------------
if 'energy' in df_test.columns:
    df_test['resolution'] = (df_test['energy_estimator'] - df_test['energy']) / df_test['energy'] * 100
    resolution_std = np.std(df_test['resolution'])
    resolution_mean = np.mean(df_test['resolution'])
    resolution_median = np.median(df_test['resolution'])
    print(f"Resolution: std={resolution_std:.2f}%, mean={resolution_mean:.2f}%, median={resolution_median:.2f}%")


# -------------------------
# Resolution plots
# -------------------------
Energy = cuts.Energy  # colonne correspondant à l'énergie

df_test['norm_scaling_factor'] = df_test['amprec'] / (geomagnetic_factor * df_test[cuts.Energy])
df_test['A_estimator'] = df_test['norm_scaling_factor'] * df_test['energy'] / df_test['corrected_factor']

# Compute relative difference
df_test['rel_diff'] = (df_test['A_estimator'] - df_test[Energy]) / df_test[Energy] * 100

# Overall statistics
resolution_tot = np.std(df_test['rel_diff'])
mean_tot = np.mean(df_test['rel_diff'])
median_tot = np.median(df_test['rel_diff'])

# Scatter plot vs energy
bins_energy = np.logspace(np.log10(df_test[Energy].min()), np.log10(df_test[Energy].max()), 5)
bin_centers_energy = (bins_energy[:-1] + bins_energy[1:]) / 2

if disp==True:
    plt.figure()
    plt.scatter(df_test[Energy], df_test['rel_diff'], s=10, alpha=0.8)
    plt.errorbar(bin_centers_energy, 
                [df_test[(df_test[Energy]>=bins_energy[i]) & (df_test[Energy]<bins_energy[i+1])]['rel_diff'].mean() for i in range(len(bin_centers_energy))],
                yerr=[df_test[(df_test[Energy]>=bins_energy[i]) & (df_test[Energy]<bins_energy[i+1])]['rel_diff'].std() for i in range(len(bin_centers_energy))],
                fmt='o', linestyle='-', alpha=1, linewidth=2,
                label=f"Mean: {mean_tot:.2f}%, Median: {median_tot:.2f}%, Std: {resolution_tot:.2f}%")
    plt.xscale('log')
    plt.xlabel('True Energy [EeV]')
    plt.ylabel(r'(A* - E)/E [%]')
    plt.legend(frameon=False)
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Histogram of resolution
    plt.figure()
    plt.hist(df_test['rel_diff'], bins=50)
    plt.title(f"Resolution: std={resolution_tot:.2f}%, mean={mean_tot:.2f}%, median={median_tot:.2f}%")
    plt.xlabel('(A* - E)/E [%]')
    plt.ylabel('Counts')
    plt.show()

    # Histogram by zenith bins
    num_bins_zenith = 3
    zenith_edges = np.linspace(df_test['zenithrec'].min(), df_test['zenithrec'].max(), num_bins_zenith+1)
    df_test['zenith_bins'] = pd.cut(df_test['zenithrec'], bins=zenith_edges, include_lowest=True)

    colors = ['lightgreen', 'lightcoral', 'skyblue']

    plt.figure()
    for i, zenith_bin in enumerate(df_test['zenith_bins'].unique()):
        subset = df_test[df_test['zenith_bins'] == zenith_bin]
        rel_diff = subset['rel_diff'].values
        plt.hist(rel_diff, bins=50, alpha=0.3, color=colors[i], edgecolor=None)
        plt.hist(rel_diff, bins=50, histtype='step', linewidth=2, color=colors[i],
                label=f'{zenith_bin.left:.1f}-{zenith_bin.right:.1f}°')

    plt.xlabel('(A* - E)/E [%]')
    plt.ylabel('Counts')
    plt.legend()
    plt.grid()
    plt.show()

