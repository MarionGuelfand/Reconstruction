import sys
import os
import glob
#########################################################################################
# Use this script to run process_DC2_L1_write_txt_files.py
# You can adjust the AmplitudeThreshold, AntennaThreshold, and the input folder 
#containing the data to be processed.
##########################################################################################
rootfile_main_dir = '/sps/grand/DC2Training/ZHAireS/'

rootfile_paths = glob.glob(rootfile_main_dir)


AmplitudeThreshold = 110 #66 #muV/m
AntennaThreshold = 5

print("***Starting to process all rootfiles***")
print(f"{len(rootfile_paths):d} files to process")
for rootfile in rootfile_paths:
    os.system("python process_DC2_L1_write_txt_files.py "+rootfile+" "+str(AmplitudeThreshold)+" "+str(AntennaThreshold))
print("***All rootfiles have beenn processed***")
