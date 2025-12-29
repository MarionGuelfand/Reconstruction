import os
import numpy as np
from scipy.signal import hilbert
import grand.dataio.root_trees as groot 
import pandas as pd

########################################################################################
#Reconstruction utilities for GRAND data processing.

#This module provides helper functions to:
#- Read lists of flagged events
#- Process GRAND ROOT files 
#- Write reconstruction output tables (antenna coordinates, coincidences, file info)
########################################################################################


c = 3e8
########################################################################################
#RTK positions
path_antenna_position = '/sps/grand/mguelfand/DC2/Tests_codes_Marion/CR_candidates/'

file_path = path_antenna_position + '_gp13_65_rtksort.txt'
column_names = ['antenna_ID', 'x', 'y', 'z'] #'x' is East, 'y' is North, 'z' DAQ level (1231m) 
antenna_position = pd.read_csv(file_path, sep='\s+', names=column_names, header=None)
antenna_position['antenna_ID'] = antenna_position['antenna_ID'].astype(int) 
#print(antenna_position)
#########################################################################################

def get_last_idx(filename_):
    """
    Return the last event index written in a text file.

    Parameters
    ----------
    filename : str
        Path to the output file.

    Returns
    -------
    int
        Last index found in the file, or -1 if the file does not exist or is empty.
    """
    if not os.path.exists(filename_):
        return -1  # No existing index
    with open(filename_, 'r') as f:
        lines = f.readlines()
        if not lines:
            return -1
        last_line = lines[-1]
        last_idx = int(last_line.split()[0])
        return last_idx


def WriteReconsTables(folder_name_, recons_inputs_):
    """
    Write all reconstruction output tables in a given folder.

    Parameters
    ----------
    folder_name_ : str
        Output directory.
    recons_inputs_ : ndarray
        Array containing reconstruction quantities.
    """

    if not os.path.exists(folder_name_): os.makedirs(folder_name_)
    WriteCoordAntTable(folder_name_+'/coord_antennas.txt', recons_inputs_)
    WriteCoordAntTableWithID(folder_name_+'/coord_antennas_withID.txt', recons_inputs_)
    WriteRecCoincTable(folder_name_+'/Rec_coinctable.txt', recons_inputs_)
    WriteFileInfoTable(folder_name_+'/Info_rootfile.txt', recons_inputs_)

    return 0


def WriteFileInfoTable(filename_, recons_inputs_):
    """
    Write ROOT file and event metadata.

    Output format:
    eventID rootfile run_number last_number event_number
    """
   
    with open(filename_, 'a') as file:
        rootfile      = recons_inputs_[9,0]   # string
        eventID = int(recons_inputs_[10,0])
        run_number = int(recons_inputs_[11,0])
        last_number  = int(recons_inputs_[12,0])
        event_number = int(recons_inputs_[13,0])
        
        file.write(f"{eventID} {rootfile} {run_number} {last_number} {event_number}\n")

    return 0

def WriteCoordAntTable(filename_, recons_inputs_):
    
    start_idx = get_last_idx(filename_) + 1
    _idx = np.arange(start_idx, start_idx + len(recons_inputs_[0,:]))

    lines = [f"{_idx[i]} {recons_inputs_[1,i]} {recons_inputs_[2,i]} {recons_inputs_[3,i]} {int(recons_inputs_[7,i])} {int(recons_inputs_[8,i])}\n"
             for i in range(len(_idx))]

    with open(filename_, 'a') as file:
        file.writelines(lines)

    return 0


def WriteCoordAntTableWithID(filename_, recons_inputs_):
    """
    Write antenna coordinates including the real antenna ID.
    """

    start_idx = get_last_idx(filename_) + 1
    _idx = np.arange(start_idx, start_idx + len(recons_inputs_[0,:]))
    lines = [f"{_idx[i]} {recons_inputs_[1,i]} {recons_inputs_[2,i]} {recons_inputs_[3,i]} {recons_inputs_[6,i]:.0f} {int(recons_inputs_[7,i])} {int(recons_inputs_[8,i])}\n"
             for i in range(len(_idx))]
    with open(filename_, 'a') as file:
        file.writelines(lines)
   
    return 0

def WriteRecCoincTable(filename_, recons_inputs_):
   
    start_idx = get_last_idx(filename_) + 1
    _idx = np.arange(start_idx, start_idx + len(recons_inputs_[0,:]))
    lines = [
        f"{_idx[i]} {int(recons_inputs_[0,i])} {recons_inputs_[4,i]} {recons_inputs_[5,i]} {int(recons_inputs_[7,i])} {int(recons_inputs_[8,i])}\n"
        for i in range(len(_idx))
    ]
    with open(filename_, 'a') as file:
        file.writelines(lines)
    return 0

def WriteFileInfoTable(filename_, recons_inputs_):
   
    with open(filename_, 'a') as file:
        rootfile      = recons_inputs_[9,0]   # string
        eventID = int(recons_inputs_[10,0])
        run_number = int(recons_inputs_[11,0])
        last_number  = int(recons_inputs_[12,0])
        event_number = int(recons_inputs_[13,0])
        
        file.write(f"{eventID} {rootfile} {run_number} {last_number} {event_number}\n")

    return 0


def read_event_list(txt_file, start_line, stop_line):
    """
    Generator function to read flagged events from a text file.
    Skips empty lines and comments (#).

    Parameters:
    txt_file (str): Path to the input text file (Flagged event containing root file and corresponding event).
    start_line (int): First line number to read (1-based index: begins with 1).
    stop_line (int or None): Last line number to read (1-based index). If None, read until the end.

    Yields:
    tuple: (runfile, event) for each line in the specified range.

    This function only reads the text file and produces a sequence of (runfile, event) tuples.
    """
    valid_line_count = 0
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # ignore empty or comment lines

            valid_line_count += 1

            # Skip lines before start_line
            if valid_line_count < start_line:
                continue

            # Stop if we exceed stop_line, stop_line is included in the count
            if stop_line is not None and valid_line_count > stop_line:
                return

            parts = line.split()
            if len(parts) < 2:
                continue  # skip malformed lines

            runfile, evt_str = parts[:2]
            yield runfile, int(evt_str)

def process_CD(path):
    """
    Process a full CD ROOT file and extract unique DU IDs.

    Parameters
    ----------
    path : str
        Path to the ROOT file.

    Returns
    -------
    int
        Number of unique DUs.
    ndarray
        Array of unique DU IDs.
    """

    adc_file = path
    root_file = groot.DataFile(adc_file)
    t_adc = root_file.tadc
    t_run = root_file.trun
    t_rawvoltage = root_file.trawvoltage
    n_events = t_adc.get_number_of_entries() 
    all_du_ids = []

    for event in range(n_events):
        t_adc.get_entry(event)
        t_run.get_entry(event)
        t_rawvoltage.get_entry(event)
        du_ids = np.array(t_adc.du_id)
        all_du_ids.append(du_ids)

    all_du_ids = np.concatenate(all_du_ids) 
    unique_du_ids = np.unique(all_du_ids)
   
    return len(unique_du_ids), unique_du_ids

def read_event_ID(event, path, last_number):
    """
    Process a single flagged event in a ROOT file:
    1. Extract ADC traces per DU.
    2. Compute peak amplitude and peak time using Hilbert transform.
    3. Gather DU positions and event info.
    4. Save results to output reconstruction tables.

    Parameters:
    rootfile
    event: eventID
    path: rootfile path
    run_number:
    last_number: lastnumber in the name of the rootfile (because different rootfiles have same run number)
    output_recons: where we save the output txt files
    """    
    adc_file = path

    root_file = groot.DataFile(adc_file)
    t_adc = root_file.tadc
    t_run = root_file.trun
    t_rawvoltage = root_file.trawvoltage
    t_adc.get_entry(event)
    t_run.get_entry(event)
    t_rawvoltage.get_entry(event)
    events_list = t_adc.get_list_of_events()
    run_numbers = [run_number for _, run_number in events_list]
    unique_run_numbers = np.unique(run_numbers)[0]
    eventID = int(str(f"{unique_run_numbers}{last_number}{event}"))
    return eventID

def process_single_event(rootfile, event, path, run_number, last_number, output_recons):
    """
    Process a single flagged event from a root file for reconstruction.

    Steps performed:
    1. Load ADC traces for all DUs (detector units) for the given event.
    2. Compute the signal modulus and Hilbert transform to extract peak amplitudes and times.
    3. Identify unique and duplicated DU IDs and apply causality checks for duplicated DUs.
    4. Convert DU coordinates to the local North/West/Up reference frame.
    5. Prepare arrays of peak times, amplitudes, positions, DU IDs, event info for reconstruction tables.
    6. Write reconstruction tables to output text files.

    Parameters
    ----------
    rootfile : str
        Name of the root file containing the event.
    event : int
        Event index in the root file.
    path : str
        Path to the root file.
    run_number : int
        Run number corresponding to the root file.
    last_number : int
        Last number in the root file name (needed for unique event ID).
    output_recons : str
        Path to the folder where reconstruction tables are saved.
    """

    recons_inputs = []    
    adc_file = path

    #print(f"File: {adc_file}")
    #print(f"Event: {event}")

    root_file = groot.DataFile(adc_file)
    t_adc = root_file.tadc
    t_run = root_file.trun
    t_rawvoltage = root_file.trawvoltage

    # Load the specific event
    t_adc.get_entry(event)
    t_run.get_entry(event)
    t_rawvoltage.get_entry(event)

    # Get unique run numbers for this event and construct a unique event ID
    events_list = t_adc.get_list_of_events()
    run_numbers = [run_number for _, run_number in events_list]
    unique_run_numbers = np.unique(run_numbers)[0]
    eventID = int(str(f"{unique_run_numbers}{last_number}{event}"))
    #print(eventID)

    # Calculate t0 for ADC traces to align time in seconds
    event_second = np.array(t_adc.du_seconds).min()
    event_nano = np.array(t_adc.du_nanoseconds).min()
    t0_adc = (t_adc.du_seconds-event_second)*1e9  - event_nano + t_adc.du_nanoseconds
    
    # Load ADC traces and DU IDs
    trace_adc = np.array(t_adc.trace_ch)
    trace_shape = trace_adc.shape  # (nb_du, 4, tbins of a trace)
    nb_du = trace_shape[0]
    du_ids = np.array(t_adc.du_id) #print total number of DUs
    du_ids = du_ids.astype(int) 

    # Identify unique and duplicated DUs
    (unique_ids, counts) = np.unique(du_ids, return_counts=True) #gives unique_ids + number of times the ID appear
    du_duplicated = unique_ids[counts > 1]
    du_non_duplicated = unique_ids[counts == 1]

     # Get DU positions and convert to North/West/Up frame
    du_positions = antenna_position[antenna_position['antenna_ID'].isin(du_ids)]
    du_positions = du_positions.set_index('antenna_ID').loc[du_ids]
    x_coords = du_positions['y'].astype(float).values #North
    y_coords = - du_positions['x'].astype(float).values #West
    z_coords = du_positions['z'].astype(float).values + 1231

    # Only process events with enough unique DUs
    if len(unique_ids) >= 5 and len(du_non_duplicated) > 0:  
        peaktime_list = []
        peakamp_list = []
        xant_list = []
        yant_list = []
        zant_list = []
        du_id_list = []
 
        first_non_duplicated_idx = None

        # Loop over all DUs and compute peak times/amplitudes for non-duplicated DUs
        for du_idx in range(nb_du):
            du_id = du_ids[du_idx]
            if du_id in du_non_duplicated:
                #print(t_adc.du_id[du_idx])
                trace_adc_x = trace_adc[du_idx,1]
                trace_adc_y = trace_adc[du_idx,2]
                trace_adc_z = trace_adc[du_idx,3]
                Emodulus = np.sqrt(trace_adc_x**2+trace_adc_y**2+trace_adc_z**2)
                #print('Emodulus', Emodulus)
                hilbert_amp = np.abs(hilbert(Emodulus))
                peakamp = np.max(hilbert_amp)
                #print('peak amplitude', peakamp)
                peaktime = np.argmax(hilbert_amp)*2 + t0_adc[du_idx] #from samples to ns
                #print('trace time', np.argmax(hilbert_amp))
                peaktime = peaktime*1e-9 #ns in s
                #print('peak time', peaktime)
                #print('DU position:', t_rawvoltage.gps_lat[du_idx])
                #print(t_rawvoltage.gps_long[du_idx])
                #print(t_rawvoltage.gps_alt[du_idx])

                # Store values
                peaktime_list.append(peaktime)
                peakamp_list.append(peakamp)
                xant_list.append(x_coords[du_idx])
                yant_list.append(y_coords[du_idx])
                zant_list.append(z_coords[du_idx])
                du_id_list.append(du_ids[du_idx]) 

            if first_non_duplicated_idx is None:
                first_non_duplicated_idx = du_idx
                #print('first_non_duplicated_idx', first_non_duplicated_idx)
                #print('peaktime', peaktime_list)
        
        # Loop over duplicated DUs and select the most causal occurrence
        multi_causal_count = 0

        for du_id in du_duplicated:
            min_delta_t = np.inf
            best_du_idx = None
            causal_occurrences = 0 
            
            # Loop over all occurrences of this duplicated DU ID to identify the most causally consistent instance
            for du_idx in np.where(du_ids == du_id)[0]:
                trace_x = trace_adc[du_idx, 1]
                trace_y = trace_adc[du_idx, 2]
                trace_z = trace_adc[du_idx, 3]

                Emodulus = np.sqrt(trace_x**2 + trace_y**2 + trace_z**2)
                hilbert_amp = np.abs(hilbert(Emodulus))
                peakamp = np.max(hilbert_amp)
                peaktime = np.argmax(hilbert_amp)*2 + t0_adc[du_idx]
                peaktime = peaktime * 1e-9  # ns -> s

                # Check causality with first non-duplicated DU
                if first_non_duplicated_idx is not None:
                    delta_t = abs(peaktime - peaktime_list[0])
                    L = np.linalg.norm(
                        np.array([x_coords[du_idx], y_coords[du_idx], z_coords[du_idx]]) -
                        np.array([xant_list[0], yant_list[0], zant_list[0]])
                    )

                    if delta_t <= L / c:
                        #print(f"DU {du_id} (idx {du_idx}) respecte la causalité : Δt={delta_t:.2e} s < L/c={L/c:.2e} s")
                        
                        # keep most causal occurence
                        if delta_t < min_delta_t:
                            min_delta_t = delta_t
                            best_du_idx = du_idx
                    #else:
                        #print(f"DU {du_id} (idx {du_idx}) NE respecte PAS la causalité : Δt={delta_t:.2e} s > L/c={L/c:.2e} s")
                
            if causal_occurrences > 1:
                multi_causal_count += 1

            if best_du_idx is not None:
                # Store the most causal occurrence
                trace_x = trace_adc[best_du_idx, 1]
                trace_y = trace_adc[best_du_idx, 2]
                trace_z = trace_adc[best_du_idx, 3]
                Emodulus = np.sqrt(trace_x**2 + trace_y**2 + trace_z**2)
                hilbert_amp = np.abs(hilbert(Emodulus))
                peakamp = np.max(hilbert_amp)
                peaktime = np.argmax(hilbert_amp)*2 + t0_adc[best_du_idx]
                peaktime = peaktime * 1e-9

                peaktime_list.append(peaktime)
                peakamp_list.append(peakamp)
                xant_list.append(x_coords[best_du_idx])
                yant_list.append(y_coords[best_du_idx])
                zant_list.append(z_coords[best_du_idx])
                du_id_list.append(du_id)

        peaktime_array = np.array(peaktime_list)
        peakamp_array = np.array(peakamp_list)
        xant_array = np.array(xant_list)
        yant_array = np.array(yant_list)
        zant_array = np.array(zant_list)
        du_id_array = np.array(du_id_list)  
        
        print(f"Number of duplicated DUs with multiple causal occurrences: {multi_causal_count} for {rootfile} {event}")

        
        event_number_col = np.repeat(event, len(xant_array)).astype(np.int64)
        run_number_col = np.repeat(unique_run_numbers, len(xant_array)).astype(np.int64)
        recons_inputs.append(np.array([
        np.repeat(eventID, len(xant_array)),
        xant_array,
        yant_array,
        zant_array,
        peaktime_array,
        peakamp_array,
        du_id_array,
        event_number_col,
        run_number_col,
        np.array([rootfile]*len(xant_array), dtype=object),      # <-- string here
        np.array([eventID]*len(xant_array)),
        np.array([run_number]*len(xant_array)),
        np.array([last_number]*len(xant_array)),
        np.array([event]*len(xant_array))
    ], dtype=object))
        
    

        recons_inputs = np.concatenate(recons_inputs, axis=1)
        print("-> writing reconstruction tables")
        #output_recons = '/sps/grand/mguelfand/DC2/Tests_codes_Marion/CR_candidates/output/adc_candidates_Jolan/test/'
        WriteReconsTables(output_recons, recons_inputs)
        print("-> Done")