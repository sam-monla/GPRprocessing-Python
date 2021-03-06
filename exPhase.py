"""
Example of automatic picking of the soil surface using the algorithm based on phase analysis (The functions of phasePicking_tools.py)
"""
import os
from turtle import position
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import file_mgmt as fmgmt
import basic_proc as bp
import phasePicking_tools as phaseTools

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

# Problems with files 0, 6, 28 to 32
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
    dat = dat[big_val[0][0]-5:100,:]

    # Dewow and air waves removal from data
    dat = bp.deWOW(dat,18)
    dat_trmoy = bp.rem_mean_trace(dat,dat.shape[1]/2)
    # Filtre SVD pour ondes directes et bruit
    dat1 = dat_trmoy[:,:int(dat.shape[1]/2)]
    dat2 = dat_trmoy[:,int(dat.shape[1]/2):]
    dat1 = bp.quick_SVD(dat1,coeff=[None,4])
    dat2 = bp.quick_SVD(dat2,coeff=[None,4])
    dat_filtreSVD = np.concatenate((dat1,dat2),axis=1)

    # Use LIDAR data for elevations
    GPS_corr = bp.lidar2gps_elev(coordLas,GPS)
    # Find dikes' positions
    dikes, diff_clone = bp.find_dikes(GPS_corr[:,2])
    print("dikes' positions:",dikes)

    ### - Picking - #########################################################################################
    #########################################################################################################

    # Calculates the quadrature trace using the Hilbert transform
    dat_prim = phaseTools.quad_trace(dat_filtreSVD)
    # Calculates instantaneous phase
    teta_mat = np.arctan2(dat_prim,dat_filtreSVD)
    # Cosine toonly get values between 0 and 1
    costeta_mat = np.cos(teta_mat)
    # Calculates the Cij and twtij matrix
    twtij, Cij = phaseTools.Ctwt_matrix(costeta_mat)

    ### - Case 0 - ##########################################################################################
    ### - If there is dikes at the very beginning AND at the very end - #####################################

    if len(dikes) > 0:
        # Initialization of the final lists
        horizons_end = []
        xice_end =[]
        dikes_end = []
        xdike_end = []

        # If there is dikes at the very beginning AND at the very end
        if (dikes[0][0] == 0) and (dikes[-1][1] == len(GPS_corr[:,2])-1):
            print("Case 0")
            for i in range(len(dikes)-1):
                if len(dikes) == 2:
                    print("There is dikes only at the extremes of the radargram")
                    # Defines the zone between the 2 dikes as the field
                    field = costeta_mat[:,dikes[0][1]:dikes[-1][0]]
                    # Isolates the portion of Cij and twtij that are associated with the field
                    Cij_field = Cij[:,dikes[0][1]:dikes[-1][0]]
                    twtij_field = twtij[:,dikes[0][1]:dikes[-1][0]]
                    # Isolates the portion of original GPR data that is associated with the field
                    dat_brut_field = dat_filtreSVD[:,dikes[0][1]:dikes[-1][0]]
                    GPS_field = GPS_corr[:,2][dikes[0][1]:dikes[-1][0]]
                    # Horizons detection
                    hp, hp_tp, C_clone, t_clone, nosign = phaseTools.horipick(Cij_field,twtij_field,Tph,tol_C=0.1,tolmin_t=0.6,tolmax_t=2)

                # If there is more than 2 dikes
                else:
                    # Isolates traces between each pair of dikes
                    field = costeta_mat[:,dikes[i][1]:dikes[i+1][0]]
                    Cij_field = Cij[:,dikes[i][1]:dikes[i+1][0]]
                    twtij_field = twtij[:,dikes[i][1]:dikes[i+1][0]]
                    dat_brut_field = dat_filtreSVD[:,dikes[i][1]:dikes[i+1][0]]
                    GPS_field = GPS_corr[:,2][dikes[i][1]:dikes[i+1][0]]
                    # Horizons detection
                    hp, hp_tp, C_clone, t_clone, nosign = phaseTools.horipick(Cij_field,twtij_field,Tph,tol_C=0.1,tolmin_t=0.6,tolmax_t=2)

                # Removal of horizons that are too short
                min_length = 25
                long_hp = [hori for hori in hp if len(hori) > min_length]
                # Horizons junctions
                longer_hp, longer_hpt, signs = phaseTools.horijoin(long_hp,C_clone,t_clone,Lg=200,Tj=5)

                # Filters horizons by sign and by length to detect the soil/ice surface. This surface is supposed to be longest and to be early on the radargram
                # Keeps only the horizons of positive sign
                sign_hpt = [longer_hpt[k] for k in range(len(signs)) if signs[k] > 0]
                # Keeps only the horizons with a length at least equal to half the field
                long_reflect = [hori for hori in sign_hpt if len(hori) > (field.shape[1]/2)]
                list_mean = []
                # Of the remaining horizons, we keep the one that comes the earliest
                for reflect in long_reflect:
                    list_mean.append(np.mean([ref[0] for ref in reflect]))
                ice_cold = np.where(list_mean == np.min(list_mean))[0]

                # For the horizon associated with soil/ice surface, we interpolate so that every trace gets a picked coordinate
                # We add +dikes[i][1] to get the true position, relative to the whole radargram
                ice_tr = [m[1]+dikes[i][1] for m in long_reflect[ice_cold[0]]]
                ice_samp = [n[0] for n in long_reflect[ice_cold[0]]]
                # Makes a numpy array containing the number of every trace
                x_ice = np.linspace(ice_tr[0],(ice_tr[-1])-1,ice_tr[-1]-ice_tr[0])
                # Interpolate a vertical position at every horizontal position
                f_ice = interpolate.interp1d(ice_tr,ice_samp,kind="linear")
                new_ice = f_ice(x_ice)
                # Smoothing of the soil/ice surface
                tottraces = len(x_ice)
                win = 75
                mov_av = np.zeros(len(x_ice))
                halfwid = int(np.ceil(win/2))
                mov_av[:halfwid+1] = np.mean(new_ice[:halfwid+1])
                mov_av[tottraces-halfwid:] = np.mean(new_ice[tottraces-halfwid:])
                for z in range(halfwid,tottraces-halfwid+1):
                    mov_av[z] = np.mean(new_ice[z-halfwid:z+halfwid])

                # Picking on the dikes
                if len(dikes) <= 2:
                    x_dike = [None]
                    new_dike = [None]
                elif i < (len(dikes)-2):
                    dike_case = 0
                    x_dike,new_dike = phaseTools.pick_dike(dike_case,i,dikes,Cij,twtij,Tph,mov_av)

                # If dikes are just at the extremes of the radargram, adds the soil/ice surface to the final lists
                if (any(elem is None for elem in x_dike)) and (any(elem is None for elem in new_dike)):
                        xice_end.append(x_ice)
                        xdike_end = None
                        horizons_end.append(mov_av)
                        dikes_end = None
                else:
                    xice_end.append(x_ice)
                    xdike_end.append(x_dike)
                    horizons_end.append(mov_av)
                    dikes_end.append(new_dike)

        ### - Case 1 - ######################################################################################
        ### - If there is 1 dike at the very beginning of the radargram - ###################################

        elif (dikes[0][0] == 0) and (dikes[-1][1] != len(GPS_corr[:,2])-1):
            print("Case 1")
            for i in range(len(dikes)):
                # Isolates traces between each pair of dikes
                if i == len(dikes)-1:
                    field = costeta_mat[:,dikes[i][1]:]
                    Cij_field = Cij[:,dikes[i][1]:]
                    twtij_field = twtij[:,dikes[i][1]:]
                    dat_brut_field = dat_filtreSVD[:,dikes[i][1]:]
                    GPS_field = GPS_corr[:,2][dikes[i][1]:]
                    # Horizons detection
                    hp, hp_tp, C_clone, t_clone, nosign = phaseTools.horipick(Cij_field,twtij_field,Tph,tol_C=0.1,tolmin_t=0.6,tolmax_t=2)
                else:
                    field = costeta_mat[:,dikes[i][1]:dikes[i+1][0]]
                    Cij_field = Cij[:,dikes[i][1]:dikes[i+1][0]]
                    twtij_field = twtij[:,dikes[i][1]:dikes[i+1][0]]
                    dat_brut_field = dat_filtreSVD[:,dikes[i][1]:dikes[i+1][0]]
                    GPS_field = GPS_corr[:,2][dikes[i][1]:dikes[i+1][0]]
                    # Horizons detection
                    hp, hp_tp, C_clone, t_clone, nosign = phaseTools.horipick(Cij_field,twtij_field,Tph,tol_C=0.1,tolmin_t=0.6,tolmax_t=2)

                # Removal of horizons that are too short
                min_length = 25
                long_hp = [hori for hori in hp if len(hori) > min_length]
                # Horizons junctions
                longer_hp, longer_hpt, signs = phaseTools.horijoin(long_hp,C_clone,t_clone,Lg=200,Tj=5)
                
                # Filters horizons by sign and by length to detect the soil/ice surface. This surface is supposed to be longest and to be early on the radargram
                # Keeps only the horizons of positive sign
                sign_hpt = [longer_hpt[i] for k in range(len(signs)) if signs[k] > 0]
                print(len(sign_hpt))
                # Keeps only the horizons with a length at least equal to half the field
                long_reflect = [hori for hori in sign_hpt if len(hori) > (field.shape[1]/2)]
                print(len(long_reflect))
                list_mean = []
                # Of the remaining horizons, we keep the one that comes the earliest
                for reflect in long_reflect:
                    list_mean.append(np.mean([ref[0] for ref in reflect]))
                ice_cold = np.where(list_mean == np.min(list_mean))[0]

                # For the horizon associated with soil/ice surface, we interpolate so that every trace gets a picked coordinate
                # We add +dikes[i][1] to get the true position, relative to the whole radargram
                ice_tr = [m[1]+dikes[i][1] for m in long_reflect[ice_cold[0]]]
                ice_temp = [n[0] for n in long_reflect[ice_cold[0]]]
                # Makes a numpy array containing the number of every trace
                x_ice = np.linspace(ice_tr[0],(ice_tr[-1])-1,ice_tr[-1]-ice_tr[0])
                # Interpolate a vertical position at every horizontal position
                f_ice = interpolate.interp1d(ice_tr,ice_temp,kind="linear")
                new_ice = f_ice(x_ice)
                # Smoothing of the soil/ice surface
                tottraces = len(x_ice)
                fen = 75
                mov_av = np.zeros(len(x_ice))
                halfwid = int(np.ceil(fen/2))
                mov_av[:halfwid+1] = np.mean(new_ice[:halfwid+1])
                mov_av[tottraces-halfwid:] = np.mean(new_ice[tottraces-halfwid:])
                for z in range(halfwid,tottraces-halfwid+1):
                    mov_av[z] = np.mean(new_ice[z-halfwid:z+halfwid])

                # Picking on the dikes
                if i < (len(dikes)-1):
                    dike_case = 1
                    x_dike,new_dike = phaseTools.pick_dike(dike_case,i,dikes,Cij,twtij,Tph,mov_av)

                # Préparation des horizons finaux
                xice_end.append(x_ice)
                xdike_end.append(x_dike)
                horizons_end.append(mov_av)
                dikes_end.append(new_dike)

        ### - Case 2 - ######################################################################################
        ### - If there is a dike at the very end - ##########################################################

        elif (dikes[0][0] != 0) and (dikes[-1][1] == len(GPS_corr[:,2])-1):
            print("Case 2")
            for i in range(len(dikes)):
                # Isolates traces between each pair of dikes
                if i == 0:
                    field = costeta_mat[:,:dikes[i][0]]
                    Cij_field = Cij[:,:dikes[i][0]]
                    twtij_field = twtij[:,:dikes[i][0]]
                    dat_brut_field = dat_filtreSVD[:,:dikes[i][0]]
                    GPS_field = GPS_corr[:,2][:dikes[i][0]]
                    # Horizons detection   
                    hp, hp_tp,C_clone, t_clone, nosign = phaseTools.horipick(Cij_field,twtij_field,Tph,tol_C=0.1,tolmin_t=0.6,tolmax_t=2)
                else:
                    field = costeta_mat[:,dikes[i-1][1]:dikes[i][0]]
                    Cij_field = Cij[:,dikes[i-1][1]:dikes[i][0]]
                    twtij_field = twtij[:,dikes[i-1][1]:dikes[i][0]]
                    dat_brut_field = dat_filtreSVD[:,dikes[i-1][1]:dikes[i][0]]
                    GPS_field = GPS_corr[:,2][dikes[i-1][1]:dikes[i][0]]
                    # Horizons detection   
                    hp, hp_tp,C_clone, t_clone, nosign = phaseTools.horipick(Cij_field,twtij_field,Tph,tol_C=0.1,tolmin_t=0.6,tolmax_t=2)

                # Removal of horizons that are too small
                min_length = 25
                long_hp = [hori for hori in hp if len(hori) > min_length]
                # Horizons junctions
                longer_hp, longer_hpt, signs = phaseTools.horijoin(long_hp,C_clone,t_clone,Lg=200,Tj=5)

                # Display of horizons
                #fig = plt.figure(figsize=(10, 6))
                #ax = fig.add_subplot(211)
                #ax.set_title("GPR data after basic processing")
                #maxi = 1
                #mini = -1
                #plt.imshow(field, cmap='bwr', vmin=mini, vmax=maxi)
                #for hori in range(len(longer_hpt)):
                #    samp = [i[0] for i in longer_hpt[hori]]
                #    traces = [j[1] for j in longer_hpt[hori]]
                #    plt.plot(traces,samp,'.-k',MarkerSize=1,LineWidth=1)
                #ax.set_aspect(8)
                #plt.xlabel("Traces")
                #plt.ylabel("Samples")
                #plt.show()

                # Filters horizons by sign and by length to detect the soil/ice surface. This surface is supposed to be longest and to be early on the radargram
                # Keeps only the horizons of positive sign
                sign_hpt = [longer_hpt[i] for k in range(len(signs)) if signs[k] > 0]
                # Keeps only the horizons with a length at least equal to half the field
                long_reflect = [hori for hori in sign_hpt if len(hori) > (field.shape[1]/2)]
                list_mean = []
                # Of the remaining horizons, we keep the one that comes the earliest
                for reflect in long_reflect:
                    list_mean.append(np.mean([ref[0] for ref in reflect]))
                print(list_mean)
                ice_cold = np.where(list_mean == np.min(list_mean))[0]
                
                # For the horizon associated with soil/ice surface, we interpolate so that every trace gets a picked coordinate
                # We add +dikes[i][1] to get the true position, relative to the whole radargram
                if i == 0:
                    ice_tr = [m[1] for m in long_reflect[ice_cold[0]]]
                else:
                    ice_tr = [m[1] + dikes[i-1][1] for m in long_reflect[ice_cold[0]]]
                ice_temp = [n[0] for n in long_reflect[ice_cold[0]]]
                # Makes a numpy array containing the number of every trace
                x_ice = np.linspace(ice_tr[0],(ice_tr[-1])-1,ice_tr[-1]-ice_tr[0])
                # Interpolate a vertical position at every horizontal position
                f_ice = interpolate.interp1d(ice_tr,ice_temp,kind="linear")
                new_ice = f_ice(x_ice)
                # Smoothing of the soil/ice surface
                tottraces = len(x_ice)
                fen = 75
                mov_av = np.zeros(len(x_ice))
                halfwid = int(np.ceil(fen/2))
                mov_av[:halfwid+1] = np.mean(new_ice[:halfwid+1])
                mov_av[tottraces-halfwid:] = np.mean(new_ice[tottraces-halfwid:])
                for z in range(halfwid,tottraces-halfwid+1):
                    mov_av[z] = np.mean(new_ice[z-halfwid:z+halfwid])

                # Picking on the dikes
                if i < (len(dikes)-1):
                    dike_case = 2
                    x_dike,new_dike = phaseTools.pick_dike(dike_case,i,dikes,Cij,twtij,Tph,mov_av)

                xice_end.append(x_ice)
                xdike_end.append(x_dike)
                horizons_end.append(mov_av)
                dikes_end.append(new_dike)

        ### - Case 3 - ######################################################################################
        ### - No dikes at the extremes - ####################################################################
    
        elif (dikes[0][0] != 0) and (dikes[-1][1] != len(GPS_corr[:,2])-1):
            print("Case 3")
            for i in range(len(dikes)):
                # Isolates traces between each pair of dikes
                if i == 0:
                    field = costeta_mat[:,:dikes[i][0]]
                    Cij_field = Cij[:,:dikes[i][0]]
                    twtij_field = twtij[:,:dikes[i][0]]
                    dat_brut_field = dat_filtreSVD[:,:dikes[i][0]]
                    GPS_field = GPS_corr[:,2][:dikes[i][0]]
                    hp, hp_tp,C_clone, t_clone, nosign = phaseTools.horipick(Cij_field,twtij_field,Tph,tol_C=0.1,tolmin_t=0.6,tolmax_t=2)
                if i == len(dikes)-1:
                    field = costeta_mat[:,dikes[i][1]:]
                    Cij_field = Cij[:,dikes[i][1]:]
                    twtij_field = twtij[:,dikes[i][1]:]
                    dat_brut_field = dat_filtreSVD[:,dikes[i][1]:]
                    GPS_field = GPS_corr[:,2][dikes[i][1]:]
                    hp, hp_tp,C_clone, t_clone, nosign = phaseTools.horipick(Cij_field,twtij_field,Tph,tol_C=0.1,tolmin_t=0.6,tolmax_t=2)
                else:
                    field = costeta_mat[:,dikes[i-1][1]:dikes[i][0]]
                    Cij_field = Cij[:,dikes[i-1][1]:dikes[i][0]]
                    twtij_field = twtij[:,dikes[i-1][1]:dikes[i][0]]
                    dat_brut_field = dat_filtreSVD[:,dikes[i-1][1]:dikes[i][0]]
                    GPS_field = GPS_corr[:,2][dikes[i-1][1]:dikes[i][0]]
                    hp, hp_tp,C_clone, t_clone, nosign = phaseTools.horipick(Cij_field,twtij_field,Tph,tol_C=0.1,tolmin_t=0.6,tolmax_t=2)

                # Removal of horizons that are too short
                min_length = 25
                long_hp = [hori for hori in hp if len(hori) > min_length]
                # Horizons junctions
                longer_hp, longer_hpt, signs = phaseTools.horijoin(long_hp,C_clone,t_clone,Lg=200,Tj=5)
                
                # Filters horizons by sign and by length to detect the soil/ice surface. This surface is supposed to be longest and to be early on the radargram
                # Keeps only the horizons of positive sign
                sign_hpt = [longer_hpt[k] for k in range(len(signs)) if signs[k] > 0]
                # Keeps only the horizons with a length at least equal to half the field
                long_reflect = [hori for hori in sign_hpt if len(hori) > (field.shape[1]/2)]
                list_mean = []
                # Of the remaining horizons, we keep the one that comes the earliest
                for reflect in long_reflect:
                    list_mean.append(np.mean([ref[0] for ref in reflect]))
                ice_cold = np.where(list_mean == np.min(list_mean))[0]

                # For the horizon associated with soil/ice surface, we interpolate so that every trace gets a picked coordinate
                # We add +dikes[i][1] to get the true position, relative to the whole radargram
                if i == 0:
                    ice_tr = [m[1] for m in long_reflect[ice_cold[0]]]
                else:
                    ice_tr = [m[1] + dikes[i-1][1] for m in long_reflect[ice_cold[0]]]
                ice_temp = [n[0] for n in long_reflect[ice_cold[0]]]
                # Makes a numpy array containing the number of every trace
                x_ice = np.linspace(ice_tr[0],ice_tr[-1]-1,ice_tr[-1]-ice_tr[0])
                # Interpolate a vertical position at every horizontal position
                f_ice = interpolate.interp1d(ice_tr,ice_temp,kind="linear")
                new_ice = f_ice(x_ice)
                # Smoothing of the soil/ice surface
                tottraces = len(x_ice)
                fen = 75
                mov_av = np.zeros(len(x_ice))
                halfwid = int(np.ceil(fen/2))
                mov_av[:halfwid+1] = np.mean(new_ice[:halfwid+1])
                mov_av[tottraces-halfwid:] = np.mean(new_ice[tottraces-halfwid:])
                for z in range(halfwid,tottraces-halfwid+1):
                    mov_av[z] = np.mean(new_ice[z-halfwid:z+halfwid])

                # Picking on the dikes
                dike_case = 3
                x_dike,new_dike = phaseTools.pick_dike(dike_case,i,dikes,Cij,twtij,Tph,mov_av)

                # Préparation des horizons finaux
                xice_end.append(x_ice)
                xdike_end.append(x_dike)
                horizons_end.append(mov_av)
                dikes_end.append(new_dike)

    ### - Topographic correction - Snow layer - ############################################################
    ########################################################################################################

    # This step shift every trace vertically so that the picked surface matches the GPS elevations converted in sample and superposed on the radargram

    if len(dikes) > 0:
        newdata_res,GPS_final,x_tot,totliss,shifts = phaseTools.snow_corr(dat_copy,xdike_end,xice_end,dikes_end,horizons_end,GPS_corr,head,offset=(big_val[0][0]-5),smooth=75,resample=1)
    ### - Display of data after correction for snow layer -  ################################################
    #########################################################################################################

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(211)
    ax.set_title("Avant la correction topographique")
    maxi = 8192
    mini = -8192
    plt.imshow(dat_copy, cmap='bwr', vmin=mini, vmax=maxi)
    #plt.plot(x_tot,totliss,"-g")
    ax.set_aspect(8)
    plt.xlabel("Traces")
    plt.ylabel("Échantillons")

    ax = fig.add_subplot(212)
    ax.set_title("Après la correction topographique")
    maxi = 8192
    mini = -8192
    plt.imshow(newdata_res, cmap='bwr', vmin=mini, vmax=maxi)
    #plt.plot(GPS_final,"-k")
    ax.set_aspect(8)
    plt.xlabel("Traces")
    plt.ylabel("Échantillons")
    plt.tight_layout()
    plt.show()

    ### - Create a .vts file to open in Paraview - ##########################################################
    #########################################################################################################

    outfile = "/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/GPR_vtk/G{}-F{}".format(gridno[8:],fich_nom[24:])
    print("G{}-F{}".format(gridno[8:],fich_nom[24:]))

    # Only keeps GPS positions associated with the soil/ice surface
    positions = GPS_corr[int(x_tot[0]):int(x_tot[-1])+1,:]
    if positions[:,0][0] > positions[:,0][1]:
        posx = np.flip(positions[:,0])
        posz = np.flip(positions[:,2])
    else:
        posx = positions[:,0]
        posz = positions[:,2]

    # Creates a time vector    
    twtt = np.linspace(0,newdata_res.shape[0]*((head["ns_per_zsample"]*1e9)),newdata_res.shape[0])
    # Finds the max GPS elevation
    samp_min = np.min(GPS_final)
    # Removes everything that comes earlier than the max elevation
    newdata_res = newdata_res[int(samp_min)-2:,:]
    newtwtt = twtt[int(samp_min)-2:]
    # Substracts temporal delay caused by empty samples
    newtwtt = newtwtt - newtwtt[0]

    # Deletes some stored data to avoid memory problems when exporting to VTK
    del dat
    del dat_copy
    del GPS
    del moy
    del dat_trmoy
    del GPS_corr
    del new_ice
    del dat_prim
    del costeta_mat
    del teta_mat
    del twtij
    del Cij
    del field
    del Cij_field
    del twtij_field
    del dat_brut_field
    del GPS_field
    del dat_filtreSVD
    del C_clone
    del t_clone
    del x_dike
    del new_dike
    del head
    del nosign
    del ice_cold
    del mov_av

    # Exports processed data to .vts file -> You can open it in Paraview (You can also open every vts files in the same Paraview window)
    bp.exportVTK(outfile,posx,newtwtt,newdata_res,positions)
    # Adds the coordinates of the final data to a list that will be used for North/South (vertical) lines
    list_temp_vtk.append(positions)
    positions[:,2] = GPS_final-samp_min

    del newdata_res

### - Processing of vertical GPR lines (North/South) (Parallel to dikes) (No dikes) - #######################
#############################################################################################################

# The steps are very similar to the horizontal lines

Inversion_GPR = False
for fichN in vertical_list:

    # Checks wich files are aquired in GRID mode
    num = fichN.find('GRID____')
    gridno = fichN[num:num+11]
    fich_nom = fichN[num:num+27]
    isGRID = True if sum(x.count(gridno+'.3DS') for x in vertical_list) > 1 else False

    # Importer les données des fichiers DZT et dst
    # Import data from DZT (GPR) and dst (GPS) files
    head, dat = fmgmt.readDZT(fichN, [2])
    ntrace_ini = dat.shape[1]
    GPS = pd.read_csv(vert_list_dst[vertical_list.index(fichN)], sep="\t")
    # Makes a 3 columns (X,Y,Z) matrix with the GPS data
    GPS = np.hstack((np.array([GPS['Xshot']]).T, np.array([GPS['Yshot']]).T, np.array([GPS['Zshot']]).T))

    # Determines the orientation (North or South) of the radargrams for a given grid
    if isGRID and (gridno != gridno_b):
        Inversion_GPR, ignore = bp.flip_rad(GPS, isNorth)
        gridno_b = gridno

    # Changes the orientation of GPS data if necessary
    Inversion, GPS = bp.flip_rad(GPS, isNorth)
    # Changer le sens des données DZT, si nécessaire
    if Inversion == True and isGRID == False:
        dat = np.flip(dat,axis=1)
    elif Inversion_GPR == True and isGRID == True:
        dat = np.flip(dat,axis=1)

    # Removal of empty traces
    deb, fin = bp.rem_empty(dat)
    dat = dat[:,deb:fin]
    dat_copy = np.copy(dat)
    GPS = GPS[deb:fin]

    # For the picking of soil surface, we don't need the full matrix. The first 100 samples are enough
    dat = dat[:100,:]
    # GPR data takes values between 0 and 65536 (16 bits). 32768 is the middle value. Here we center the values around 0 for better result with the phase transformation
    dat = dat - 32768
    dat_copy = dat_copy - 32768
    dat = bp.deWOW(dat,18)
    dat_copy = bp.deWOW(dat_copy, 18)

    # Use LIDAR data for elevations
    GPS_corr = bp.lidar2gps_elev(coordLas,GPS)

    # Masks every part of data surrounding the soil reflection - Only keeps neighbouring data
    # Look the prep_picking_NS() description for more details
    mask, data_rng = phaseTools.prep_picking_NS(dat)

    # Calculates the quadrature trace using the Hilbert transform
    dat_prim_moy = phaseTools.quad_trace(data_rng)
    # Calculates instantaneous phase
    teta_mat_moy = np.arctan2(dat_prim_moy,data_rng)
    # Cosine toonly get values between 0 and 1
    costeta_mat_moy = np.cos(teta_mat_moy)

    # Apply masks
    data_rng = mask*data_rng
    costeta_mat_moy = mask*costeta_mat_moy

    # Horizons detection
    twtij, Cij = phaseTools.Ctwt_matrix(costeta_mat_moy)
    hp, tphp, C_clone, t_clone, signs = phaseTools.horipick(Cij,twtij,Tph=5,tol_C=0.65,tolmin_t=0.2,tolmax_t=2)
    # Filters horizons of negative phase
    new_hp = []
    for i in range(len(hp)):
        if signs[i] < 0:
            continue
        else:
            new_hp.append(hp[i])
    # Horizons junctions
    new_hp = [hori for hori in new_hp if len(hori) > 25]
    long_hp, long_hpt, signs1 = phaseTools.horijoin(new_hp,C_clone,t_clone,Lg=500,Tj=3)
    # Filters short horizons
    longer_hp = [hori for hori in long_hpt if len(hori) > 8000]

    # Smoothing and interpolation
    ice_tr = [m[1] for m in longer_hp[0]]
    ice_temp = [n[0] for n in longer_hp[0]]
    x_ice = np.linspace(ice_tr[0],ice_tr[-1]-1,ice_tr[-1]-ice_tr[0])
    f_ice = interpolate.interp1d(ice_tr,ice_temp,kind="linear")
    new_ice = f_ice(x_ice)
    # Smoothing
    tottraces = len(x_ice)
    fen = 75
    moy_mobil = np.zeros(len(x_ice))
    halfwid = int(np.ceil(fen/2))
    moy_mobil[:halfwid+1] = np.mean(new_ice[:halfwid+1])
    moy_mobil[tottraces-halfwid:] = np.mean(new_ice[tottraces-halfwid:])
    for z in range(halfwid,tottraces-halfwid+1):
        moy_mobil[z] = np.mean(new_ice[z-halfwid:z+halfwid])

    # Topographic correction - Removal of snow layer
    newdata_res,GPS_final,x_tot,totliss,shifts = phaseTools.snow_corr(dat_copy,None,x_ice,None,moy_mobil,GPS_corr,head,offset=0,smooth=75,resample=1)

    ### - Display of topographic correction - ###############################################################
    #########################################################################################################

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(211)
    ax.set_title("Avant la correction topographique")
    maxi = 8192
    mini = -8192
    plt.imshow(dat_copy, cmap='bwr', vmin=mini, vmax=maxi)
    ax.set_aspect(8)
    plt.xlabel("Traces")
    plt.ylabel("Échantillons")

    ax = fig.add_subplot(212)
    ax.set_title("Après la correction topographique")
    maxi = 8192
    mini = -8192
    plt.imshow(newdata_res, cmap='bwr', vmin=mini, vmax=maxi)
    plt.plot(GPS_final,"-k")
    ax.set_aspect(8)
    plt.xlabel("Traces")
    plt.ylabel("Échantillons")
    #plt.show()

    ### - Create a .vts file to open in Paraview - ##########################################################
    #########################################################################################################

    # Only keeps GPS points associated with the picked soil/ice surface
    positions = GPS_corr[int(x_tot[0]):int(x_tot[-1])+1,:]
    if positions[:,1][0] > positions[:,1][1]:
        posx = np.flip(positions[:,0])
        posy = np.flip(positions[:,1])
        posz = np.flip(positions[:,2])
    else:
        posx = positions[:,0]
        posy = positions[:,1]
        posz = positions[:,2]
    # Creation of a time vector    
    twtt = np.linspace(0,newdata_res.shape[0]*((head["ns_per_zsample"]*1e9)),newdata_res.shape[0]) 

    # Finds the max GPS elevation
    samp_min = np.min(GPS_final)
    # Removes everything that comes before this max elevation
    newdata_res = newdata_res[int(samp_min)-2:,:]
    newtwtt = twtt[int(samp_min)-2:]

    ### - Aligns vertical GPR lines with the horizontal ones - ##############################################
    time_surf = []
    # For every horizontal line, defines a rectangle from the most extreme coordinates
    for hline in list_temp_vtk:
        west = np.min(positions[:,0])
        east = np.max(positions[:,0])
        south = np.min(hline[:,1])
        north = np.max(hline[:,1])

        # Selects every GPS coordinates of the horizontal line that is included in the rectangle
        h_ind_GPSselect = (hline[:,0]>=west) & (hline[:,1]>=south) & (hline[:,0]<=east) & (hline[:,1]<=north)
        hGPS_select = hline[h_ind_GPSselect]
        # Selects every GPS coordinates of the vertical line that is included in the rectangle
        v_ind_GPSselect = (positions[:,0]>=west) & (positions[:,1]>=south) & (positions[:,0]<=east) & (positions[:,1]<=north)
        vGPS_select = positions[v_ind_GPSselect]

        if (hGPS_select.shape[0] > 0) & (vGPS_select.shape[0] > 0):
            # Defines a new rectangle. west and east are defined from the vertical line, north and south from horizontal line. The goal is to define a smaller zone around the point of intersection of the 2 GPR lines
            west = np.min(vGPS_select[:,0])
            east = np.max(vGPS_select[:,0])
            south = np.min(hGPS_select[:,1])
            north = np.max(hGPS_select[:,1])

            # Only keeps the coordinates of hGPS_select that are included in the new smaller rectangle
            hind_fin = (hGPS_select[:,0]>=west) & (hGPS_select[:,1]>=south) & (hGPS_select[:,0]<=east) & (hGPS_select[:,1]<=north)
            hGPS_fin = hGPS_select[hind_fin]
            # Only keeps the coordinates of vGPS_select that are included in the new smaller rectangle
            vind_fin = (vGPS_select[:,0]>=west) & (vGPS_select[:,1]>=south) & (vGPS_select[:,0]<=east) & (vGPS_select[:,1]<=north)
            vGPS_fin = vGPS_select[vind_fin]

            if (hGPS_fin.shape[0] > 0) & (vGPS_fin.shape[0] > 0):
                # Calculates a linear equation from hGPS_fin coordinates
                slop1 = (hGPS_fin[-1,1]-hGPS_fin[0,1])/(hGPS_fin[-1,0]-hGPS_fin[0,0])
                b1 = hGPS_fin[-1,1] - slop1*hGPS_fin[-1,0]
                # Calculates a linear equation from vGPS_fin coordinates
                slop2 = (vGPS_fin[-1,1]-vGPS_fin[0,1])/(vGPS_fin[-1,0]-vGPS_fin[0,0])
                b2 = vGPS_fin[-1,1] - slop2*vGPS_fin[-1,0]
                # Finds the x coordinate of the intersection
                xint = (b2-b1)/(slop1-slop2)
                # Finds the y coordinate of the intersection
                yint = slop1*xint+b1

                # Finds the closest GPS point to the intersection location
                GPS_int = min(hGPS_fin[:,0], key=lambda x:abs(x-xint))
                posGPSint = np.where(hGPS_fin[:,0] == GPS_int)
                # Finds the maximum elevation of the horizontal line
                elev_max = np.min(hline[:,2])
                # Finds the elevation difference between the intersection point and the max elevation
                off_set = hline[posGPSint[0][0],2] - elev_max
                time_surf.append(off_set)
            else:
                continue
        else:
            continue
    
    """
    If we apply the code on a selection of GPR lines only, including some vertical lines, we must insure there is at least 1 crossing between the vertical line and one horizontal line. Otherwise, time_surf remains empty.
    """
    newtwtt = newtwtt - newtwtt[0]
    if len(time_surf) == 0:
        print("time_surf is empty!")
        continue
    else:
        diff_elev_mean = np.mean(time_surf)*(head["ns_per_zsample"]*1e9)
        newtwtt = newtwtt + diff_elev_mean

    outfile = "/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/GPR_vtk/G{}-F{}".format(gridno[8:],fich_nom[24:])
    
    del data_rng
    del dat_prim_moy
    del teta_mat_moy
    del costeta_mat_moy
    del twtij
    del Cij
    del C_clone
    del t_clone
    del dat
    del dat_copy
    del GPS
    del GPS_corr
    del hGPS_select
    del vGPS_select
    del hGPS_fin
    del vGPS_fin

    bp.exportVTK(outfile,posy,newtwtt,newdata_res,positions)

    del newdata_res
 


    

    

    

    


