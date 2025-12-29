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

