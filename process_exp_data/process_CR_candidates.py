import sys
import os
import numpy as np
import grand.dataio.root_files as gdr
import grand.dataio.root_trees as groot 
from grand import ECEF, Geodetic, GRANDCS, LTP
import grand.manage_log as mlg
import matplotlib.pyplot as plt
import scipy.signal as ssi
import random
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt, lfilter



################################################################################
##Functions
def WriteReconsTables(folder_name_, recons_inputs_):

    if not os.path.exists(folder_name_): os.makedirs(folder_name_)
    WriteCoordAntTable(folder_name_+'/coord_antennas.txt', recons_inputs_)
    WriteCoordAntTableWithID(folder_name_+'/coord_antennas_withID.txt', recons_inputs_)
    WriteRecCoincTable(folder_name_+'/Rec_coinctable.txt', recons_inputs_)

    return 0

def WriteCoordAntTable(filename_, recons_inputs_):

    _idx = np.arange(0, len(recons_inputs_[0,:]))
    with open(filename_, 'w') as file: [file.write(f"{_idx[i]} {recons_inputs_[1,i]} {recons_inputs_[2,i]} {recons_inputs_[3,i]} {int(recons_inputs_[7,i])} {int(recons_inputs_[8,i])}\n") for i in range(len(_idx))]

    return 0

#add real ID of the antenna
def WriteCoordAntTableWithID(filename_, recons_inputs_):

    _idx = np.arange(0, len(recons_inputs_[0,:]))
    with open(filename_, 'w') as file: [file.write(f"{_idx[i]} {recons_inputs_[1,i]} {recons_inputs_[2,i]} {recons_inputs_[3,i]} {recons_inputs_[6,i]:.0f}  {int(recons_inputs_[7,i])} {int(recons_inputs_[8,i])}\n") for i in range(len(_idx))]

    return 0

def WriteRecCoincTable(filename_, recons_inputs_):

    _idx = np.arange(0, len(recons_inputs_[0,:]))
    with open(filename_, 'w') as file: [file.write(f"{_idx[i]} {int(recons_inputs_[0,i])} {recons_inputs_[4,i]} {recons_inputs_[5,i]}  {int(recons_inputs_[7,i])} {int(recons_inputs_[8,i])}\n") for i in range(len(_idx))]

    return 0

coord_daq  = Geodetic(latitude=40.99746387, longitude=93.94868871, height=0)   # lat, lon of the center station (from FEB@rocket) # z=0 @ sea level
#coord_1078 = Geodetic(latitude=40.99434, longitude=93.94177, height=1205.9284000000027)
#def get_DU_coord(lat, long, alt, obstime='1970-01-01', origin=coord_1078):
def  get_DU_coord(lat, long, alt, obstime='2024-01-01', origin=coord_daq):
  # From GPS to Cartisian coordinates
  geod = Geodetic(latitude=lat, longitude=long, height=alt)
  gcs = GRANDCS(geod, obstime=obstime, location=origin)
  return gcs

################################################################################
##Main
rootsim_path = '/sps/grand/mguelfand/DC2/Tests_codes_Marion/CR_candidates/'

f_adc = [rootsim_path + 'adc_20250519_113216_0_L1_0000.root']
f_run = [rootsim_path + 'run_00000_L0_0000.root']
f_rawvoltage = [rootsim_path + 'rawvoltage_20250519_113216_0_L1_0000.root']

root_file = groot.DataFile(f_adc)
root_run = groot.DataFile(f_run)
root_rawvoltage = groot.DataFile(f_rawvoltage)
t_adc = root_file.tadc
t_run = root_run.trun
t_rawvoltage = root_rawvoltage.trawvoltage
n_events = t_adc.get_number_of_entries() 
events_list = t_adc.get_list_of_events()

event_run_list = []

for event_number, run_number in events_list:
    event_run_list.append((event_number, run_number))
event_run_array = np.array(event_run_list)

recons_inputs, simu_inputs = [], []

for event_idx in range(n_events): #loop over all events
#for event_number,run_number in events_list:
    #print(f"->{(event_number+1)/n_events*100:.1f}%")
    #print(event_number)
    #print(run_number)

    event_number = int(str(event_run_array[event_idx][0]))
    run_number = int(str(event_run_array[event_idx][1]))

    t_adc.get_entry(event_idx)
    t_run.get_entry(event_idx)
    t_rawvoltage.get_entry(event_idx)
    trace_adc = np.array(t_adc.trace_ch)
    #eventID = int(str(event_idx))
    eventID = int(str(f"{run_number}{event_number}"))
    

    #t0 calculations
    event_second = np.array(t_adc.du_seconds).min()
    event_nano = np.array(t_adc.du_nanoseconds).min()
    t0_adc = (t_adc.du_seconds-event_second)*1e9  - event_nano + t_adc.du_nanoseconds
  
    trace_shape = trace_adc.shape  # (nb_du, 4, tbins of a trace)
    sig_size = trace_shape[-1]
    nb_du = trace_shape[0]
    print(nb_du)
    du_ids = np.array(t_adc.du_id) #print total number of DUs
    #print(du_ids)
    du_xyzs = get_DU_coord(np.array(list(t_rawvoltage.gps_lat)), np.array(list(t_rawvoltage.gps_long)), np.array(list(t_rawvoltage.gps_alt)))
    #print(du_xyzs)
    xant, yant, zant = du_xyzs
    #coreAlt = 1264. #m -> from event_files.tt_run.origin_geoid
    #zant +=coreAlt
    dt_ns = np.full(len(du_ids), t_run.t_bin_size[0]) # sampling time in ns, sampling freq = 1e9/dt_ns.
 

    peaktime_list = []
    peakamp_list = []
    xant_list = []
    yant_list = []
    zant_list = []
    du_id_list = []
    
    for du_idx in range(nb_du):
        print(nb_du)
        #du_idx = 0
        trace_adc_x = trace_adc[du_idx,1]
        trace_adc_y = trace_adc[du_idx,2]
        trace_adc_z = trace_adc[du_idx,3]
        trace_adc_time = np.arange(len(trace_adc_z)) * dt_ns[du_idx] 
        #print(trace_adc_time)
        trace_adc_time = np.arange(len(trace_adc_z)) * dt_ns[du_idx] 
        fs = 1 / np.mean(np.diff(trace_adc_time))*1e9 
        #print(fs)
        Emodulus = np.sqrt(trace_adc_x**2+trace_adc_y**2+trace_adc_z**2)
        hilbert_amp = np.abs(hilbert(Emodulus))
        peakamp = np.max(hilbert_amp)
        print('peak amplitude', peakamp)
        peaktime = trace_adc_time[np.where(hilbert_amp == peakamp)[0][0]] + t0_adc[du_idx]
        peaktime = peaktime*1e-9 #ns in s
        print('peak time', peaktime)

        peaktime_list.append(peaktime)
        peakamp_list.append(peakamp)
        xant_list.append(xant[du_idx])
        yant_list.append(yant[du_idx])
        zant_list.append(zant[du_idx])
        du_id_list.append(du_ids[du_idx]) 
    
    peaktime_array = np.array(peaktime_list)
    peakamp_array = np.array(peakamp_list)
    xant_array = np.array(xant_list)
    yant_array = np.array(yant_list)
    zant_array = np.array(zant_list)
    du_id_array = np.array(du_id_list)

    event_number_col = np.repeat(event_number, len(xant_array)).astype(np.int64)
    run_number_col = np.repeat(run_number, len(xant_array)).astype(np.int64)
    recons_inputs.append([np.repeat(eventID, len(xant_array)), xant_array, yant_array, zant_array, peaktime_array, peakamp_array, du_id_array, event_number_col, run_number_col])

recons_inputs = np.concatenate(recons_inputs, axis=1)

print("-> writing reconstruction tables")
#WriteReconsTables(rootfile_name, recons_inputs)
output_recons = '/sps/grand/mguelfand/DC2/Tests_codes_Marion/CR_candidates/output/voltage_refdaq/'
WriteReconsTables(output_recons, recons_inputs)
print("-> Done")
