# GRAND Cosmic Ray Processing Pipeline

## Overview
This repository contains the full data processing pipeline for GRAND cosmic-ray detection events. 
It includes modules for:

- **Data extraction** from ROOT files (antenna coordinates, peak amplitudes, times)
- **Reconstruction** of events (with PWF, SWF, ADF)
- **Analysis** of reconstructed events


The pipeline allows you to process flagged events, generate reconstruction tables, 
and check which DUs are alive or dead.

---

## Directory Structure
├── Data_processing/ # Scripts to process flagged events and CD ROOT files

├── Recons/ # Scripts to reconstruct events and write output tables

├── Analysis/ # Analysis scripts and plotting

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

| File | Columns | Notes |
|------|--------|------|
| `Rec_plane_wave_recons.txt` | Coincidence ID, Nb antennas, Zenith (deg), Phi (deg), placeholders, Chi2 | Plane wave reconstruction (PWF): direction fit for each coincidence |
| `Rec_plane_time.txt` | Coincidence ID, Nb antennas, Processing time (s) | Time spent per plane wave fit |
| `Rec_sphere_wave_recons.txt` | Coincidence ID, Nb antennas, Reconstructed theta, phi, r_xmax, t_s, ... | Spherical wave reconstruction (SWF) using PWF as input |
| `Rec_sphere_time.txt` | Coincidence ID, Nb antennas, Processing time (s) | Time spent per spherical fit |
| `Rec_adf_recons.txt` | Coincidence ID, Nb antennas, theta, phi, delta_omega, amplitude | ADF 3D reconstruction parameters |
| `Rec_adf_time.txt` | Coincidence ID, Nb antennas, Processing time (s) | Time spent per ADF fit |
| `Rec_adf_parameters.txt` | Coincidence ID, Nb antennas, amplitude_simu, amplitude_recons, eta, omega, omega_cr, omega_cr_analytic, l_ant_array, coordinates | Detailed ADF parameters and reconstructed amplitudes |

**Notes on structure:**

- Each file contains **all processed events sequentially**.  
- Lines are ordered by events.  
- Times are in **seconds**, distances in **meters**, angles in **degrees**.  
- `Rec_adf_parameters.txt` contains arrays per coincidence; each line corresponds to a single antenna’s contribution to the ADF model.  

**See examples of these txt files in:** `Recons/examples/`
