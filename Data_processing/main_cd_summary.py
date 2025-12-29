#!/usr/bin/env python3
"""
Script: CD Event Summary and DU Status Check

Purpose:
--------
This script processes flagged CD events to generate a summary that allows
verification of which DUs are alive or dead in a given ROOT file.
For each flagged event, it records:

  - A unique event ID
  - The number of DUs that triggered at least once in the ROOT file
  - The list of unique DU IDs that triggered at least once in the ROOT file

The output is written to a text file (events_CD_all_events.txt), which can be
used to monitor detector performance and identify non-functioning DUs.
If a DU never appears in any processed event, it can be considered dead.

Modes:
------
1. Batch mode (job=True): command-line arguments specify start_line, stop_line, and output folder.
2. Manual mode (job=False): start_line, stop_line, and output folder are set in the script.
"""

import os
import sys
import re
import functions_processing as func
import config as conf

def main_cd_summary(job=False):
    # --------------------------
    # Mode parameters
    # --------------------------
    if job:  # Batch mode: use command-line arguments
        start_line = int(sys.argv[1])
        stop_line_arg = sys.argv[2]
        output_recons = sys.argv[3]

        stop_line = None if stop_line_arg == "None" else int(stop_line_arg)
        flagged_txt = conf.flagged_txt

    else:  # Manual mode: parameters set in script
        start_line = 1
        stop_line = 100  # Set to None to process until end
        flagged_txt = conf.flagged_txt
        output_recons = conf.output_recons

    # --------------------------
    # Output file
    # --------------------------
    output_file = conf.output_file
    os.makedirs(output_recons, exist_ok=True)
    output_path = os.path.join(output_recons, output_file)

    # Open file in append mode
    with open(output_path, 'a') as f:

        # Loop over flagged events
        for rootfile, event in func.read_event_list(flagged_txt, start_line=start_line, stop_line=stop_line):
            path_rootfile = f'{conf.path_rootfile_base}{rootfile}'

            # Extract last number and run number from filename
            matches = re.findall(r'\d+', rootfile)
            last_number = int(matches[-1]) if matches else None

            # Process the event and get unique eventID
            eventID = func.read_event_ID(event, path_rootfile, last_number)

            # Get number of antennas and unique DU IDs for this CD run
            number_antennas_in_cd_run, unique_antennas_ID = func.process_CD(path_rootfile)

            antennas_str = " ".join(map(str, unique_antennas_ID))

            # Write summary line: eventID, number of antennas, list of antenna IDs
            f.write(f"{eventID}\t{number_antennas_in_cd_run}\t{antennas_str}\n")
            f.flush()

if __name__ == "__main__":
    main_cd_summary()
