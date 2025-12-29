# GRAND Cosmic Ray Processing Pipeline

## Overview
This repository contains the full data processing pipeline for GRAND cosmic-ray detection events. 
It includes modules for:

- **Data extraction** from ROOT files (antenna coordinates, peak amplitudes, times)
- **Quality control** of DUs: number of DUs triggered in a given ROOT file
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

- **Data processing**

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


