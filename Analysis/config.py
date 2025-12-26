# config.py

import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------
# Paths
# --------------------------
base_dir = '/Users/mguelfan/Documents/GRAND/ADF_DC2/'
data_dir = base_dir + 'output_recons_DC2/CR_candidates/adc_candidates_Jolan/'
output_figures = data_dir + 'Figures_02:12:2025/'
path_antenna_position = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_DC2/CR_candidates/'

# Antenna file
ANTENNA_FILE = path_antenna_position + '_gp13_65_rtksort.txt'

# Paths for months
root_paths = [
    data_dir + 'July_2025/causality/GLOBAL_filtered/',
    data_dir + 'August_2025/causality/GLOBAL_filtered/',
    data_dir + 'September_2025/causality/GLOBAL_filtered/',
    data_dir + 'October_2025/causality/GLOBAL_filtered/'
]
month_names = ['July', 'August', 'September', 'October']

# Colors for plots
colors = plt.cm.coolwarm(np.linspace(0, 1, len(month_names)))

# --------------------------
# Plot settings
# --------------------------
PLOT_PARAMS = {
    "legend.fontsize": 16,
    "axes.labelsize": 23,
    "axes.titlesize": 23,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.figsize": (10, 8),
    "axes.grid": False,
    "pcolor.shading": "auto",
}
plt.rcParams.update(PLOT_PARAMS)
