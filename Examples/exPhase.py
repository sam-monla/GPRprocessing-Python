"""
Example of automatic picking of the soil surface using the algorithm based on phase analysis (The functions of phasePicking_tools.py)
"""
import os
import numpy as np
import pandas as pd
import file_mgmt as fmgmt

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
GPR_Reverso = False
vertical_list = []
vert_list_dst = []
list_temp_vtk = []
list_samp_min = []

### - Iteration over file_list (GPR_data) - #################################################################
#############################################################################################################

# Problems with files 0 and 28 to 32
for fich in file_list[1:28] + file_list[33:]:
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

    # Déterminer le sens des radargrammes pour une grille
    isNorth = False  
    if isGRID and (gridno != gridno_b):
        Inversion_GPR, ignore = pflt.change_side(GPS, isNorth)
        gridno_b = gridno

    