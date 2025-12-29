import sys
import os
import re
from glob import glob
import grand.dataio.root_files as gdr
import grand.dataio.root_trees as groot 
from grand import ECEF, Geodetic, GRANDCS, LTP
import grand.manage_log as mlg
import config as conf
import functions_processing as func

"""
Main script for processing flagged GRAND cosmic-ray events.

This script reads a list of flagged events from a text file (Flagged_events.txt),
which contains ROOT file names and specific event numbers identified as interesting
or worth analyzing (e.g., candidate cosmic-ray events). Only these flagged events
are processed.

Reconstruction Output Files:

1. coord_antennas.txt
   - Antenna positions (x, y, z) for all antennas detecting the event.
   - Includes event number and run number.
2. coord_antennas_withID.txt
   - Same as coord_antennas.txt, but includes the real antenna ID.
3. Rec_coinctable.txt
   - Stores peak times and amplitudes per DU.
4. Info_rootfile.txt
   - Metadata about the ROOT file and event.
   - Contains unique event ID, ROOT file name, run number, last number in filename, and event number.


Parameters:
-----------
job : bool
    If True, the script runs in batch mode using command-line arguments.
    If False, it runs manually with predefined start/stop lines.

start_line : int
    First line of the Flagged_events.txt file to process (1-based index). 
    Only relevant if job=False or when manually specifying a range.

stop_line : int or None
    Last line of the Flagged_events.txt file to process (1-based index).
    If None, processing continues until the end of the file.

output_recons : str
    Path to the directory where reconstruction tables will be written.
    Can be specified via command-line in job mode or taken from config.py in manual mode.

flagged_txt : str
    Path to the text file containing the list of flagged events (rootfile names and event numbers).

Usage:
------
- Batch mode (job=True):
    python main.py <start_line> <stop_line> <output_recons>
    Example: python main.py 1 100 /path/to/output/

- Manual mode (job=False):
    Adjust start_line, stop_line, and output_recons directly in the script.
"""


def main(job = False):
    # --------------------------
    # Batch mode: read from command-line arguments
    # --------------------------
    if job:
        start_line = int(sys.argv[1])
        stop_line_arg = sys.argv[2]
        output_recons = sys.argv[3]
        stop_line = None if stop_line_arg == "None" else int(stop_line_arg)
        flagged_txt = conf.flagged_txt

    # --------------------------
    # Manual mode: parameters set in script
    # --------------------------
    else:
        start_line = 1
        stop_line = 2
        output_recons = conf.output_recons
        flagged_txt = conf.flagged_txt

    # --------------------------
    # Loop over flagged events
    # --------------------------
    for rootfile, event in func.read_event_list(flagged_txt,
                                                start_line=start_line,
                                                stop_line=stop_line):
        path_rootfile = os.path.join(conf.path_rootfile_base, rootfile)

        # Extract numbers from filename
        matches = re.findall(r'\d+', rootfile)
        last_number = int(matches[-1]) if matches else None
        match = re.search(r'RUN(\d+)', rootfile)
        run_number = int(match.group(1)) if match else None

        # Process the event and write reconstruction tables
        func.process_single_event(rootfile, event, path_rootfile,
                                  run_number, last_number, output_recons)


if __name__ == "__main__":
    main()