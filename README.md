# GRAND Cosmic Ray Processing Pipeline

## Overview
This repository contains the full data processing pipeline for GRAND cosmic-ray detection events. 
It includes modules for:

- **Data extraction** from ROOT files (antenna coordinates, peak amplitudes, times)
- **Reconstruction** of events (with PWF, SWF, ADF)
- **Energy reconstruction** from radio amplitudes with geomagnetic and atmospheric corrections
- **Analysis** of reconstructed events


The pipeline allows you to process flagged events, generate reconstruction tables, 
and check which DUs are alive or dead.

---

## Directory Structure
├── Data_processing/ # Scripts to process flagged events and CD ROOT files

├── Recons/ # Scripts to reconstruct events and write output tables

├── Analysis/ # Analysis scripts and plotting

├── Energy_recons/ # Scrpt to reconstruct energy

├── config.py # Configuration paths and parameters

## Output Files Details

### Data Processing (`Data_processing/`)

- **`main.py`** – processes flagged events and generates:
  - `coord_antennas.txt`
  - `coord_antennas_withID.txt`
  - `Rec_coinctable.txt`
  - `Info_rootfile.txt`
- **`main_cd_process.py`** – processes all CD ROOT file containing the flagged events and generates:
  - `events_CD_all_events.txt` to check which DUs are alive in the corresponding ROOT file

| File | Columns | Notes |
|------|--------|------|
| coord_antennas.txt | Internal counter, x, y, z, Event nr, Run nr | All antennas that triggered |
| coord_antennas_withID.txt | + Real antenna ID | Same as above |
| Rec_coinctable.txt | Internal counter, Unique event ID, Peak time, Peak amplitude, Event nr, Run nr | Coincidence analysis |
| Info_rootfile.txt | Unique event ID, ROOT file, Run nr, Last nr, Event nr | Traceability |
| events_CD_all_events.txt | Unique event ID, Nb antennas triggered, DU IDs triggered | Detector performance |

**Notes on structure:**

- Each file contains **all processed events sequentially**.  
- Lines are ordered first by event, then by antennas involved.  
- `events_CD_all_events.txt` allows to identify **dead DUs**: if a DU never appears in any event, it is considered dead.  

**See examples of these txt files in:** `Data_processing/examples/`


### Reconstruction Outputs (`Recons/`)

- **`recons.py`** – performs reconstruction (plane wave, spherical wave, or ADF) from coincidence tables and antenna coordinates.  
  Run **`main.py`** (and **`main_cd_process.py`**) first to generate the input files (`coord_antennas.txt` and `Rec_coinctable.txt`).
- Use **code PTREND** (Python version) (**TREND** (original C++ version from V. Decoene) for the reconstruction routines.)

| File | Columns | Notes |
|------|--------|------|
| `Rec_plane_wave_recons.txt` | Unique event ID, Nb antennas, Zenith (deg), NaN, Azimuth (deg), Chi2, NaN, NaN | Plane wave reconstruction (PWF): direction fit for each coincidence |
| `Rec_plane_time.txt` | Unique event ID, Nb antennas, Processing time (s) | Time spent per plane wave fit |
| `Rec_sphere_wave_recons.txt` |  Coincidence ID, Nb antennas, Chi², NaN, x (Xsource m), y (Xsource m), z (Xsource m), r_xmax (m), t_s (m), Zenith (deg), Azimuth (deg) | Spherical wave reconstruction (SWF) using PWF as input |
| `Rec_sphere_time.txt` | Unique event ID, Nb antennas, Processing time (s) | Time spent per spherical fit |
| `Rec_adf_recons.txt` | Unique event ID, Nb antennas, Theta (deg), Theta_err (deg), Phi (deg), Phi_err (deg), Chi², NaN, DeltaOmega, Amplitude | ADF 3D reconstruction parameters |
| `Rec_adf_time.txt` | Unique event ID, Nb antennas, Processing time (s) | Time spent per ADF fit |
| `Rec_adf_parameters.txt` | Unique event ID, Nb antennas, Measured amplitude (ADC, uV, uV/m), Reconstructed amplitude (ADC, uV, uV/m), Eta (deg), Omega (deg), Omega_CR (deg), Omega_CR_analytic (toy model), Distance l_ant (antenna-Xsource), x (antenna coordinate, m), y (antenna coordinate, m), z (antenna coordinate, m) | Detailed ADF parameters and reconstructed amplitudes |

**Notes on structure:**

- Each file contains **all processed events sequentially**.  
- Lines are ordered by events.  
- Times are in **seconds**, distances in **meters**, angles in **degrees**.  
- `Rec_adf_parameters.txt` contains arrays per coincidence; each line corresponds to a single antenna’s contribution to the ADF model.  

**See examples of these txt files in:** `Recons/PTREND/nonoise_110uV_5antennas/`

## Energy Reconstruction (`Energy_recons/`)

This directory contains the pipeline used to reconstruct the **cosmic-ray (primary or electromagnetic) energy**
from the reconstructed electric-field amplitude, accounting for geomagnetic and atmospheric effects.

The energy estimator is defined as:

E_recons = A / ( sin(alpha) · f(rho_air, sin(alpha)) )

where:
- A is the reconstructed electric-field scaling factor (from ADF),
- alpha is the geomagnetic angle,
- rho_air is the air density at the reconstructed source position,
- f is a correction function learned from simulations.

⚠️ The reconstruction uses **electric-field amplitudes (µV/m)**, not ADC counts.

---

### Directory Structure
Energy_recons/

├── compute_correction.py # Derive correction coefficients from simulations

├── reconstruction.py # Apply correction and reconstruct energy

├── utils.py # Helper functions (splits, ML, corrections)

├── config.py # Paths, constants, configuration

## Workflow

The energy reconstruction is performed in two steps:

### 1. Compute Correction Coefficients (`compute_correction.py`)

- Uses **simulated events only**
- Applies quality cuts (zenith, number of antennas, χ², amplitudes)
- Computes:
  - geomagnetic factor **sin(α)**
  - air density **ρ_air** at the reconstructed source position
- **Fits a polynomial regression model to compute the correction term**
  
  **f(ρ_air, sin(α))**,  
  which accounts for secondary dependencies of the radio signal beyond the pure geomagnetic scaling.
- Saves the trained model for later use

**Outputs:** 
- `correction_coefficients.pkl` and `correction_coefficients.csv` – regression coefficients to compute *f(ρ_air, sin(α))*


### 2. Energy Reconstruction (`reconstruction.py`)

- Applies precomputed correction coefficients
- Can be used on:
  - simulations (validation and resolution)
  - real data (energy reconstruction only)
- Computes:
  - corrected energy estimator
  - (simulations only) resolution and diagnostic plots

A **train/test split** can be enabled for simulations:
- correction may be applied to both subsets,
- performance metrics are computed **only on the test set**.

## Alternative Energy Reconstruction (Voltage-Based)

In addition to the standard energy reconstruction based on electric-field amplitudes,
an alternative method (first proxy) is implemented using the **measured voltage amplitude or ADC amplitude** directly.

This approach relies on a linear calibration of the form:

E_recons = ( A / sin(α) − b ) / a

where:
- A is the reconstructed **voltage amplitude or ADC amplitude**,
- α is the geomagnetic angle,
- a and b are calibration coefficients derived from simulations.

This method provides a fast and simple energy estimator,
without explicit atmospheric or source-position corrections.

The voltage-based reconstruction is less accurate than the full correction method,
but offers a first proxy directly on voltage or ADC data.


### Notes

- Angles in **degrees**
- Distances in **meters**
- Air density in **kg/m³**
- Energy typically in **EeV**
