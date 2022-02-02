"""
Example of automatic picking of the soil surface using the algorithm based on amplitude analysis (The functions of ampPicking_tools.py)
"""
import os
import file_mgmt as fmgmt
import basic_proc as bp
import ampPicking_tools as ampTools
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

### - Importing data - ######################################################################################
#############################################################################################################

# Imports las file
las = "/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/Blandford_GPR_2017/GPR-Blandford-H2017/GPR-Blandford-H2017/LIDAR/LiDAR_13-4175121F08_LAS_MTM_TEL/13_4175121F08_de.las"
coordLas = fmgmt.readLas(las)

# Lists every DZT files
file_path = '/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/Blandford_GPR_2017/GPR-Blandford-H2017/GPR-Blandford-H2017/GPR'
file_list = []
for root, dirs, files in os.walk(file_path, topdown=False):
    for name in files:
        if not name.endswith(".DZT"):
            break
        file_list.append(os.path.join(root, name))

# Lists every dst files
dst_path = '/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/Blandford_GPR_2017/GPS_headers'
dst_list = []
for root, dirs, files in os.walk(dst_path, topdown=False):
    for name in files:
        if not name.endswith(".dst"):
            break
        dst_list.append(os.path.join(root,name))

# Initialization of objects for later use
gridno_b = 0
Inversion_GPR = False
invent_hori = {}
vertical_list = []

# Iteration on every file
# This code doesn't deal with vertical (North/South directions) lines
for file in file_list[1:31]:
    print(file)
    # Finds wich DZT file are aquired in Grid mode. The GPR as a grid mode in wich the radargrams are always put in the same direction regardless of the direction of the equipement. 

    # The DZT files are grouped in grids. The following lines find wich of the files are taken from the same grid
    # Example: ... GRID____086.3DS/FILE____001.DZT
    num = file.find('GRID____')
    # gridno is a string of the form GRID____086 (the files names always have the same number of characters)
    gridno = file[num:num+11]
    # file_name is a string of the form GRID____086/FILE____001
    file_name = file[num:num+27]
    # A file is part of a grid if there is more than 1 file with the same gridno
    isGRID = True if sum(x.count(gridno+'.3DS') for x in file_list) > 1 else False

    # Imports data from DZT (GPR) and dst (GPS) files
    head, dat = fmgmt.readDZT(file, [2])
    ntrace_ini = dat.shape[1]
    # Selects the dst file assuming they are placed in order
    GPS = pd.read_csv(dst_list[file_list.index(file)], sep='\t')
    # Makes a 3 columns (X,Y,Z) matrix with the GPS data
    GPS = np.hstack((np.array([GPS['Xshot']]).T, np.array([GPS['Yshot']]).T, np.array([GPS['Zshot']]).T))

    # Checks the direction in wich the data were taken (North/South OR East/West)
    dif_x = abs(GPS[0][0] - GPS[-1][0])
    dif_y = abs(GPS[0][1] - GPS[-1][1])
    # If dif_y (northing of last -(minus) first coordinate) is more than 10 times diff_x (easting of last-first coordinate), the line was aquired in the North/South direction (parallel to dikes)
    # We separate these files because of the dikes
    isNorth = True if dif_y >= 10*dif_x else False

    if isNorth:
        # We put lines acquired in the North/South direction in a special list. This list will be analysed later.
        # The amplitude algorithm doesn't perform well with these lines, so the list is not used in this example
        vertical_list.append(file)
        continue
    
    # Checks the direction of GPS datas for a given grid
    # Since we only deal with East/West files in this loop
    if isGRID and (gridno != gridno_b):
        Inversion_GPR, ignore = bp.flip_rad(GPS, isNorth)
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

    # Removing of empty traces
    start_tr, end_tr = bp.rem_empty(dat)
    dat = dat[:,start_tr:end_tr]
    # Dewow
    dat = bp.deWOW(dat,18)
    # Saves a copy of the original matrix before processing
    dat_copy = np.copy(dat)
    # Removes empty traces from GPS data
    GPS = GPS[start_tr:end_tr]

    # Use LIDAR data for elevations
    GPS_corr = bp.lidar2gps_elev(coordLas,GPS)

    if isNorth == False:
        cx_list = []
        cy_list = []
        cx_brut_list = []
        cy_brut_list = []
        # Finds the positions of the dikes
        dikes, diff_clone = bp.find_dikes(GPS_corr[:,2])
        print("dikes' positions:",dikes)
        # Preliminary picking
        prelp,prelp_b,recon,deb_dig = ampTools.prel_picking(dat[:100,:],dikes,head,100)
        # Horizontal smoothing
        # For some reasons I put the results of preliminary picking in a dictionnary.
        field_smooth = []
        for clef in prelp_b:
            field_smooth.append(ampTools.hori_smooth(350,0.3,np.array(prelp_b[clef][1]),1))
        # Final picking
        for i in range(len(field_smooth)):
            clefs = list(recon.keys())
            if deb_dig:
                cx,cy,cx_brut,cy_brut = ampTools.final_picking(field_smooth[i], 100, recon[clefs[i]], ntrace_ini, head, dikes[i][1])
            elif i == 0 and not deb_dig:
                cx,cy,cx_brut,cy_brut = ampTools.final_picking(field_smooth[i], 100, recon[clefs[i]], ntrace_ini, head, 0)
            else:
                cx,cy,cx_brut,cy_brut = ampTools.final_picking(field_smooth[i], 100, recon[clefs[i]], ntrace_ini, head, dikes[i-1][1])
            cx_brut_list.append(cx_brut)
            cy_brut_list.append(cy_brut)
            cx_list.append(cx)
            cy_list.append(cy)

    dat_copy = bp.deWOW(dat_copy,18)
    dat_copy = bp.rem_mean_trace(dat_copy,2048)
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    maxi = np.amax(dat_copy)/3
    mini = np.amin(dat_copy)/3
    plt.imshow(dat_copy[:100,:], cmap='bwr', vmin=mini, vmax=maxi)
    if isNorth == False:
        plt.plot(np.hstack(cx_brut_list),np.hstack(cy_brut_list), '.g', MarkerSize=0.4)
    ax.set_aspect(8)
    plt.xlabel("Traces")
    plt.ylabel("Échantillons")
    plt.show()


    

