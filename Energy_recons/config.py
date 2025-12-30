# config.py
from pathlib import Path

# --- Base directories ---
base_dir = Path(__file__).resolve().parent.parent
analysis_dir = base_dir / "Analysis"

#
#output_dir = Path('/Users/mguelfan/Documents/GRAND/ADF_DC2/Reconstruction/Energy_recons/examples/DC2_Training_L1')
output_dir = Path('/Users/mguelfan/Documents/GRAND/ADF_DC2/Reconstruction/Energy_recons/examples/DC2_RF2Alpha_L1/Efield')

#
#recons_dir = Path('/Users/mguelfan/Documents/GRAND/ADF_DC2/Reconstruction/Energy_recons/examples/DC2_Training_L1')
recons_dir= Path('/Users/mguelfan/Documents/GRAND/ADF_DC2/Reconstruction/Energy_recons/examples/DC2_RF2Alpha_L1/Efield')


# Files
input_simus_file = output_dir / 'input_simus.txt'
rec_adf_file = output_dir / 'Rec_adf_recons.txt'
rec_sphere_file = output_dir / 'Rec_sphere_wave_recons.txt'
coefficients_file_pkl = output_dir / 'correction_coefficients.pkl'
coefficients_file_csv = output_dir / 'correction_coefficients.csv'
