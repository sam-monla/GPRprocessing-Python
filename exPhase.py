"""
Example of automatic picking of the soil surface using the algorithm based on phase analysis (The functions of phasePicking_tools.py)
"""
import os
import numpy as np
import pandas as pd
import file_mgmt as fmgmt
import basic_proc as bp

### - Import the LIDAR data from a .las file - ##############################################################
#############################################################################################################

las = "/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/Blandford_GPR_2017/GPR-Blandford-H2017/GPR-Blandford-H2017/LIDAR/LiDAR_13-4175121F08_LAS_MTM_TEL/13_4175121F08_de.las"
coordLas = fmgmt.readLas(las)

### - Make a list with all .DZT file (GPR data) - ###########################################################
#############################################################################################################

ini_path = '/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/Blandford_GPR_2017/GPR-Blandford-H2017/GPR-Blandford-H2017/GPR'
file_list = []
for root, dirs, files in os.walk(ini_path, topdown=False):
    for name in files:
        # To avoid selecting other types of files
        if not name.endswith(".DZT"):
            break
        file_list.append(os.path.join(root, name))

### - Make a list with all .dst file (GPS data) - ###########################################################
#############################################################################################################

dst_path = '/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/Blandford_GPR_2017/GPS_headers'
dst_list = []
for root, dirs, files in os.walk(dst_path, topdown=False):
    for name in files:
        if not name.endswith(".dst"):
            break
        dst_list.append(os.path.join(root,name))

### - Initialization of variables for later use - ###########################################################
#############################################################################################################

gridno_b = 0
Inversion_GPR = False
vertical_list = []
vert_list_dst = []
list_temp_vtk = []
list_samp_min = []

### - Iteration over file_list (GPR_data) - #################################################################
#############################################################################################################

# Problems with files 0 and 28 to 32
#for fich in file_list[1:28] + file_list[33:]:
for fich in file_list[1:2]:
    # Finds wich DZT file are aquired in Grid mode. The GPR as a grid mode in wich the radargrams are always put in the same direction regardless of the direction of the equipement. 

    # The DZT files are grouped in grids. The following lines find wich of the files are taken from the same grid
    # Example: ... GRID____086.3DS/FILE____001.DZT
    num = fich.find('GRID____')
    # gridno is a string of the form GRID____086 (the files names always have the same number of characters)
    gridno = fich[num:num+11]
    # fich_nom is a string of the form GRID____086/FILE____001
    fich_nom = fich[num:num+27]
    # A file is part of a grid if there is more than 1 file with the same gridno
    isGRID = True if sum(x.count(gridno+'.3DS') for x in file_list) > 1 else False

    # Imports data from single DZT and dst file
    head, dat = fmgmt.readDZT(fich, [2])
    ntrace_ini = dat.shape[1]
    # Selects the dst file assuming they are placed in order
    GPS = pd.read_csv(dst_list[file_list.index(fich)], sep='\t')
    # Makes a 3 columns (X,Y,Z) matrix with the GPS data
    GPS = np.hstack((np.array([GPS['Xshot']]).T, np.array([GPS['Yshot']]).T, np.array([GPS['Zshot']]).T))

    # Checks the direction in wich the data were taken (North/South OR East/West)
    dif_x = abs(GPS[0][0] - GPS[-1][0])
    dif_y = abs(GPS[0][1] - GPS[-1][1])
    # If dif_y (northing of last-first coordinate) is more than 10 times diff_x (easting of last-first coordinate), the line was aquired in the North/South direction (parallel to dikes)
    # We separate these files because of the dikes
    isNorth = True if dif_y >= 10*dif_x else False

    if isNorth:
        # We put lines acquired in the North/South direction in a special list. This list will be analysed later.
        vertical_list.append(fich)
        vert_list_dst.append(dst_list[file_list.index(fich)])
        # Ends the loop, continue to the next file
        continue

    # Checks the direction of GPS datas for a given grid
    # Since we only deal with East/West files in this loop
    isNorth = False 
    # If the current file is not in the same grid as the previous one, checks if we need a flip 
    if isGRID and (gridno != gridno_b):
        Inversion_GPR, ignore = bp.flip_rad(GPS, isNorth)
        # Store the current gridno to compare it to the next one on the next iteration
        gridno_b = gridno

    # Flips GPS datas if necessary
    Inversion, GPS = bp.flip_rad(GPS, isNorth)
    # Flips GPR datas if necessary
    # For grids that contains a single file, isGRID=False. These files needs to be flipped if GPS datas are decreasing.
    if Inversion == True and isGRID == False:
        dat = np.flip(dat,axis=1)
    # Flipping of GPR data when GPS datas are decreasing and when the file is part of a grid
    elif Inversion_GPR == True and isGRID == True:
        dat = np.flip(dat,axis=1)

    ### - Basic Processing - ################################################################################
    #########################################################################################################

    # Removes empty traces
    start_t, end_t = bp.rem_empty(dat)
    dat = dat[:,start_t:end_t]
    # Saves a copy of the original matrix before processing
    dat_copy = np.copy(dat)
    # Removes empty traces from GPS data
    GPS = GPS[start_t:end_t]

    # For the picking of soil surface, we don't need the full matrix. The first 100 samples are enough
    dat = dat[:100,:]
    # GPR data takes values between 0 and 65536 (16 bits). 32768 is the middle value. Here we center the values around 0 for better result with the phase transformation
    dat = dat - 32768
    # Dewow and air waves removal on the original matrix
    dat_copy = dat_copy - 32768
    dat_copy = bp.deWOW(dat_copy, 18)
    dat_copy = bp.rem_mean_trace(dat_copy, dat_copy.shape[1]/2)

    # Temporal width of a phase estimation from air waves
    # Gets the average value of every sample
    moy = np.mean(dat,axis=1)
    # Normalization
    moy = moy/np.max(np.abs(moy))
    # Finds big values. Air waves have a very high amplitude
    big_val = np.where(moy > 0.1)
    # Function to get the width of air waves.
    def ranges(nums):
        nums = sorted(set(nums))
        gaps = [[s,e] for s,e in zip(nums,nums[1:]) if s+1 < e]
        edges = iter(nums[:1] + sum(gaps,[]) + nums[-1:])
        return list(zip(edges,edges))
    # seqs is a list of tuples containing the starting and ending samples of the positive phases of air waves.
    seqs = ranges(big_val[0])
    # Finds the width from the first phase
    Tph = seqs[0][1] - seqs[0][0]
    # Removes data that are above air wave,except for 5 samples
    dat = dat[big_val[0][0]-5:,:]

    # Dewow and air waves removal from data
    dat = bp.deWOW(dat,18)
    dat_trmoy = bp.rem_mean_trace(dat,dat.shape[1]/2)
    # Filtre SVD pour ondes directes et bruit
    #dat1 = dat_trmoy[:,:int(dat.shape[1]/2)]
    #dat2 = dat_trmoy[:,int(dat.shape[1]/2):]
    #dat1 = phaseTool.quick_SVD(dat1,coeff=[None,4])
    #dat2 = phaseTool.quick_SVD(dat2,coeff=[None,4])
    #dat_filtreSVD = np.concatenate((dat1,dat2),axis=1)

    # Use LIDAR data for elevations
    GPS_corr = bp.lidar2gps_elev(coordLas,GPS)
    # Find dikes' positions
    dikes, diff_clone = bp.find_dikes(GPS_corr)
    print("dikes' positions:",dikes)
    