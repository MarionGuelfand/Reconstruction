import sys
import os
import numpy as np
import grand.dataio.root_files as gdr
import grand.dataio.root_trees as groot 
import grand.manage_log as mlg
import matplotlib.pyplot as plt
import scipy.signal as ssi
import random
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt, lfilter


######################################################################################################################################################
# Code to process the old L1 DC2 simulations in ROOT format 
#(adjustments needed for new DC2 simulations, in particular check reference frames for antenna coordinates, Xmax, shower core, and units for energy) 

#This code generates the necessary text files required for reconstruction: coord_antennas.txt and Rec_coinctable.txt,
#as well as input_simus.txt (which contains all the true parameters used for the simulations, and is used for comparison with reconstruction results).

#Only keep antennas above the AmplitudeThreshold and events for which the antenna multiplicity is greater than the AntennaThreshold when generating the text files.

#Timing and amplitudes are processed by computing the Hilbert envelope, which is crucial (for example for timing accuracy in SWF.)

#This processing can also be applied to the true data.
######################################################################################################################################################

##Functions#
def WriteReconsTables(folder_name_, recons_inputs_, simu_inputs_):

    if not os.path.exists(folder_name_): os.makedirs(folder_name_)
    WriteCoordAntTable(folder_name_+'/coord_antennas.txt', recons_inputs_)
    WriteCoordAntTableWithID(folder_name_+'/coord_antennas_withID.txt', recons_inputs_)
    WriteRecCoincTable(folder_name_+'/Rec_coinctable.txt', recons_inputs_)
    WriteInputSimu(folder_name_+'/input_simus.txt', simu_inputs_)

    return 0

def WriteCoordAntTable(filename_, recons_inputs_):

    _idx = np.arange(0, len(recons_inputs_[0,:]))
    with open(filename_, 'w') as file: [file.write(f"{_idx[i]} {recons_inputs_[1,i]} {recons_inputs_[2,i]} {recons_inputs_[3,i]}\n") for i in range(len(_idx))]

    return 0

#add real ID of the antenna
def WriteCoordAntTableWithID(filename_, recons_inputs_):

    _idx = np.arange(0, len(recons_inputs_[0,:]))
    with open(filename_, 'w') as file: [file.write(f"{_idx[i]} {recons_inputs_[1,i]} {recons_inputs_[2,i]} {recons_inputs_[3,i]} {recons_inputs_[6,i]:.0f}\n") for i in range(len(_idx))]

    return 0

def WriteRecCoincTable(filename_, recons_inputs_):

    _idx = np.arange(0, len(recons_inputs_[0,:]))
    with open(filename_, 'w') as file: [file.write(f"{_idx[i]} {int(recons_inputs_[0,i])} {recons_inputs_[4,i]} {recons_inputs_[5,i]}\n") for i in range(len(_idx))]

    return 0

def WriteInputSimu(filename_, simu_inputs_):

    _idx = np.arange(0, len(simu_inputs_[0,:]))
    with open(filename_, 'w') as file: [file.write(f" {simu_inputs_[0,i]} {simu_inputs_[1,i]} {simu_inputs_[2,i]} {simu_inputs_[3,i]} {simu_inputs_[4,i]} {simu_inputs_[5,i]} {simu_inputs_[6,i]} {simu_inputs_[7,i]} {simu_inputs_[8,i]} {simu_inputs_[9,i]} {simu_inputs_[10,i]} {simu_inputs_[11,i]} {simu_inputs_[12,i]} {simu_inputs_[13,i]} {simu_inputs_[14,i]} {simu_inputs_[15,i]} {simu_inputs_[16,i]} {simu_inputs_[17,i]} \n") for i in range(len(_idx))]

    return 0


################################################################################
##Main
#rootsim_path = '/sps/grand/DC2Training/ZHAireS-NJ/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/efield_29-24992_L0_0000.root'

rootsim_path = sys.argv[1] #path to root file, see example above
AmplitudeThreshold = float(sys.argv[2]) #in muV/m (typically 1 sigma = 22 muV/m -> 3 sigma = 66 and 5 sigma = 110 muV/m)
AntennaThreshold = float(sys.argv[3]) # minimal number of antennas (typically 5)



my_dir = '/sps/grand/mguelfand/DC2/Tests_codes_Marion/output_recons/'
rootfile_name = my_dir+rootsim_path.split('/')[-1]
event_files = groot.DataDirectory(rootsim_path) #open the root file with all events

#Get the trees L1 
#for tshower, always use L0
trun = event_files.trun_l1
trunefieldsim=event_files.trunefieldsim_l1  
tshower = event_files.tshower_l0
tefield = event_files.tefield_l1

#get total number of events
n_events = tefield.get_number_of_events()

recons_inputs, simu_inputs = [], []

lowcut = 50e6  # 50 MHz
highcut = 200e6  # 200 MHz

for event_number in range(n_events-1): #loop over all events
    print(f"->{(event_number+1)/n_events*100:.1f}%")

    tefield.get_entry(event_number)
    trun.get_entry(event_number)
    tshower.get_entry(event_number) 
    trunefieldsim.get_entry(event_number)  

    zenith = tshower.zenith
    azimuth= tshower.azimuth
    energy_primary = tshower.energy_primary
    energy_primary = energy_primary*1e-9 # from GeV to EeV (For ZHaires only, already in EeV for Coreas)
    energy_em = tshower.energy_em #for now = 0 (empty tree)
    energy_em = energy_em*1e-9 # from GeV to EeV (For ZHaires only, already in EeV for Coreas)
    xmax = tshower.xmax_pos_shc #Xmax location in shower core frame (because doesn't work in detector frame!)
    #xmax = tshower.xmax_pos
    xmax_dist = np.linalg.norm(xmax) #Xmax distance to shower core
    shower_core =tshower.shower_core_pos #in detector frame
    coreAlt = tshower.core_alt
    MagneticField=tshower.magnetic_field #B_inc, B_dec, B_tot
    primary = tshower.primary_type
    xmax_gram = tshower.xmax_grams
    eventID = int(str(tefield.event_number)) 

    trace_efield = np.array(tefield.trace)

    #t0 calculations
    event_second = tshower.core_time_s
    event_nano = tshower.core_time_ns
    t0_efield = (tefield.du_seconds-event_second)*1e9  - event_nano + tefield.du_nanoseconds
    #time window parameters. time windows go from t0-t_pre to t0+t_post
    t_pre=trunefieldsim.t_pre

    trace_shape = trace_efield.shape  # (nb_du, 3, tbins of a trace)
    sig_size = trace_shape[-1]
    nb_du = trace_shape[0]

    event_dus_indices = tefield.get_dus_indices_in_run(trun)
    dt_ns = np.asarray(trun.t_bin_size)[event_dus_indices] # sampling time in ns, sampling freq = 1e9/dt_ns. 

    du_xyzs = np.asarray(trun.du_xyz)[event_dus_indices] 
    xant, yant, zant = du_xyzs.T

    ##corrections to be fixed later to have the same referential 
    coreAlt = 1264. #m -> from event_files.tt_run.origin_geoid
    zant +=coreAlt #for ZHaiRES simulations only, not for CoREAS one
    shower_core[2] += coreAlt #move at correct altitude
    xmax[0:2] += shower_core[0:2] #move into detector frame but zxmax is already from sea level...

    print(f"Zenith = {zenith:.1f}°, Azimuth = {azimuth:.1f}°, Primary energy = {energy_primary:.3e} EeV, EM energy = {energy_em:.3e} GeV")
    print(f"Xmax location ({xmax[0]:.2f}, {xmax[1]:.2f}, {xmax[2]:.2f}), Xmax grammage = {xmax_gram:.3f} g/cm^2, Primary type {primary}")
    print(f"Shower core location ({shower_core[0]:.2f}, {shower_core[1]:.2f}, {shower_core[2]:.2f}), core altitude = {coreAlt:.2f} m")

    peaktime_list = []
    peakamp_list = []
    peaktime_filter_list = []
    peakamp_filter_list = []
    xant_list = []
    yant_list = []
    zant_list = []
    du_id_list = []
    
    # loop over all stations.  
    for du_idx in range(nb_du):
        trace_efield_x = trace_efield[du_idx,0]
        trace_efield_y = trace_efield[du_idx,1]
        trace_efield_z = trace_efield[du_idx,2]
        trace_efield_time = np.arange(0,len(trace_efield_z)) * dt_ns[du_idx] - t_pre

        fs = 1 / np.mean(np.diff(trace_efield_time))*1e9  # Hz

        Emodulus = np.sqrt(trace_efield_x**2+trace_efield_y**2+trace_efield_z**2)
        hilbert_amp = np.abs(hilbert(Emodulus))
        peakamp = np.max(hilbert_amp)
        peaktime = trace_efield_time[np.where(hilbert_amp == peakamp)[0][0]] + t0_efield[du_idx]
        peaktime = peaktime*1e-9 #ns in s

        peaktime_list.append(peaktime)
        peakamp_list.append(peakamp)
        xant_list.append(xant[du_idx])
        yant_list.append(yant[du_idx])
        zant_list.append(zant[du_idx])
        du_id_list.append(du_idx)

    peaktime_array = np.array(peaktime_list)
    peakamp_array = np.array(peakamp_list)

    xant_array = np.array(xant_list)
    yant_array = np.array(yant_list)
    zant_array = np.array(zant_list)
    du_id_array = np.array(du_id_list)

    #only keep antennas above AmplitudeThreshold and events for which antenna multiplicity > AntennaThreshold
    sel = peakamp_array >= AmplitudeThreshold
    if len(xant_array[sel]) > AntennaThreshold:
        #print('**************')
        #print('**************')
        recons_inputs.append([np.repeat(eventID, len(xant_array[sel])), xant_array[sel], yant_array[sel], zant_array[sel], peaktime_array[sel], peakamp_array[sel], du_id_array[sel]])
        simu_inputs.append([eventID, zenith, azimuth, energy_primary, energy_em, primary, xmax_dist, xmax_gram, xmax[0], xmax[1], xmax[2], shower_core[0], shower_core[1], shower_core[2], coreAlt, len(xant_array[sel]), AmplitudeThreshold, AntennaThreshold])
        
        
    else:
        print(f"Not enough antennas above {AmplitudeThreshold:.1f} (muV/m)")
       

recons_inputs = np.concatenate(recons_inputs, axis=1)
simu_inputs = np.array(simu_inputs, dtype=object).T

print("-> writing reconstruction tables")
WriteReconsTables(rootfile_name, recons_inputs, simu_inputs)
print("-> Done")
