"""
Functions to apply basic processing steps to GPR data.

The following funcions are taken from the GPRPy repository (https://github.com/NSGeophysics/GPRPy):
- deWOW()
- rem_mean_trace()
- correctTopo()
- prepVTK()
- exportVTK()
"""
import numpy as np
from tqdm import tqdm
from scipy import linalg
import scipy.interpolate as interp
import scipy.signal as signal
from pyevtk.hl import gridToVTK
from scipy import interpolate

def lidar2gps_elev(coordlas,GPS,isNorth=False,dimE=3,dimN=2):
    """
    Function to interpolate elevations at each GPS positions from LIDAR data. The direction of the GPS line is important in the interpolation process because of the dikes situation. The isNorth=True case is the one for data with no dikes. 

    INPUT:
    - coordlas: LIDAR data (Output of readLas() function in file_mgmt.py)
    - GPS: GPS data from the .dst file. Array of 3 columns (X,Y,Z)
    - isNorth: Boolean value. True if the GPS line goes in the North/South (vertical) directions, False if the GPS line goes in the East/West (horizontal) directions.
    - dimE: Width (in meters) of the rectangle used to interpolate an elevation from LIDAR data for a GPS line that goes in the East/West directions (default is 3)
    - dimN: Width (in meters) of the square used to interpolate an elevation from LIDAR data for a GPS line that goes in the North/South directions (default is 2)

    OUTPUT:
    - gps: Numpy array of 3 columns (X,Y,Z) describing the GPS positions of a GPR line. In this output, the GPS elevations are replaced by values estimated from LIDAR data.
    """
    # Finds the smallest rectangle in the field that can contain every single trace of the GPS line
    # Values at the limits
    minEast = np.min(GPS[:,0])
    maxEast = np.max(GPS[:,0])
    minNorth = np.min(GPS[:,1])
    maxNorth = np.max(GPS[:,1])

    # Finds every LIDAR point inside the rectangle
    index = (coordlas[:,0]>=minEast) & (coordlas[:,1]>=minNorth) & (coordlas[:,0]<=maxEast) & (coordlas[:,1]<=maxNorth)
    las_flt = coordlas[index]

    # Sometimes, las_flt doesn't contain enough data to correctly identify the position of the dikes. We recommend having at least 300 points in las_flt. The rectangle is extended.
    if not isNorth:
        spread = 0.5
        while las_flt.shape[0] < 300:
            minNorth_b = minNorth - spread
            maxNorth_b = maxNorth + spread
            index = (coordlas[:,0]>=minEast) & (coordlas[:,1]>=minNorth_b) & (coordlas[:,0]<=maxEast) & (coordlas[:,1]<=maxNorth_b)
            las_flt = coordlas[index]
            spread += 0.5
    
    las_flt = np.asarray(las_flt)
    gps = np.asarray(GPS)
    coords = np.vstack((las_flt[:,0], las_flt[:,1], las_flt[:,2]))

    liste = []
    # For each GPS position, we find:
    # Every LIDAR points that are included in a rectangle dimN meters wide and 20*dimN meters high centered at the GPS position (when isNorth is True)
    # OR
    # Every LIDAR points that are included in a rectangle dimE wide and infinitely high centered at the GPS position (when isNorth is False)
    for pos in gps:
        if isNorth:
            ind = np.where((coords[0]>=pos[0]-dimN/2) & (coords[0]<=pos[0]+dimN/2) & (coords[1]>=pos[1]-dimN*10) & (coords[1]<=pos[1]+dimN*10))
            liste.append(ind[0].tolist())
        else:
            ind = np.where(((coords[0] >= pos[0]-(dimE/2)) & (coords[0] <= pos[0]+(dimE/2))))
            liste.append(ind[0].tolist())
    # Calculates the elevation average for the remaining LIDAR points, and asigns it to the current GPS position
    for i in range(len(liste)):
        gps[i,2] = np.mean(las_flt[liste[i], 2])

    return gps

def rem_empty(data):
    """
    Removes empty traces from the data

    INPUT:
    - data: The data extracted from the .DZT file
    OUTPUT:
    - beg: Index of first trace that is not empty (0 if no empty trace at the beginning)
    - end: Index of last trace that is not empty (data.shape[1] if no empty trace at the end)
    """
    # Find traces in wich all samples = first sample
    empty_traces = np.where(np.all(data==data[0,:],axis=0)==True)
    # Find if the empty traces are at the beginning or at the end of the record
    frst_half = empty_traces[0][empty_traces[0] < (data.shape[1]/2)]
    scd_half = empty_traces[0][empty_traces[0] > (data.shape[1]/2)]

    # Create the objects beg and end
    # beg = index of first trace that is not empty
    # end = index of last trace that is not empty
    beg = 0
    end = data.shape[1]

    if frst_half.size != 0:
        beg = np.amax(frst_half)+1
    if scd_half.size != 0:
        end = np.amin(scd_half)

    return (beg,end)

def deWOW(data,window):
    '''
    Subtracts from each sample along each trace an 
    along-time moving average.
    Can be used as a low-cut filter.
    TAKEN FROM: GPRPy -> https://github.com/NSGeophysics/GPRPy

    INPUT:
    data       data matrix whose columns contain the traces 
    window     length of moving average window 
               [in "number of samples"]
    OUTPUT:
    newdata    data matrix after dewow
    '''
    data=np.asmatrix(data) # Added
    totsamps = data.shape[0]
    # If the window is larger or equal to the number of samples,
    # then we can do a much faster dewow
    if (window >= totsamps):
        newdata = data-np.matrix.mean(data,0)            
    else:
        newdata = np.asarray(np.zeros(data.shape))
        halfwid = int(np.ceil(window/2.0))
        
        # For the first few samples, it will always be the same
        avgsmp=np.matrix.mean(data[0:halfwid+1,:],0)
        newdata[0:halfwid+1,:] = data[0:halfwid+1,:]-avgsmp

        # for each sample in the middle
        for smp in tqdm(range(halfwid,totsamps-halfwid+1)):
            winstart = int(smp - halfwid)
            winend = int(smp + halfwid)
            avgsmp = np.matrix.mean(data[winstart:winend+1,:],0)
            newdata[smp,:] = data[smp,:]-avgsmp

        # For the last few samples, it will always be the same
        avgsmp = np.matrix.mean(data[totsamps-halfwid:totsamps+1,:],0)
        newdata[totsamps-halfwid:totsamps+1,:] = data[totsamps-halfwid:totsamps+1,:]-avgsmp
        
    print('done with dewow')
    return newdata

def running_avg(data,fen):
    """
    Function for horizontal smoothing on radargram.
    """
    data = np.asarray(data)
    tottraces = data.shape[1]

    newdata = np.zeros(data.shape)
    halfwid = int(np.ceil(fen/2))

    # First traces
    Col_1D = np.mean(data[:,0:halfwid+1],axis=1)
    Col_2D = np.reshape(Col_1D,(Col_1D.shape[0],1))
    newdata[:,0:halfwid+1] = Col_2D

    # Middle traces
    for trace in range(halfwid,tottraces-halfwid+1):
        dep_fen = int(trace-halfwid)
        fin_fen = int(trace+halfwid)
        newdata[:,trace] = np.mean(data[:,dep_fen:fin_fen+1],axis=1)

    # Last traces
    Colo_1D = np.mean(data[:,tottraces-halfwid:tottraces+1],axis=1)
    Colo_2D = np.reshape(Colo_1D,(Colo_1D.shape[0],1))
    newdata[:,tottraces-halfwid:tottraces+1] = Colo_2D
    
    return newdata

def rem_mean_trace(data,ntraces):
    
    '''
    Subtracts from each trace the average trace over
    a moving average window.
    Can be used to remove horizontal arrivals, 
    such as the airwave.
    TAKEN FROM: GPRPy -> https://github.com/NSGeophysics/GPRPy

    INPUT:
    data       data matrix whose columns contain the traces 
    ntraces    window width; over how many traces 
               to take the moving average.
    OUTPUT:
    newdata    data matrix after subtracting average traces
    '''

    data=np.asarray(data)
    tottraces = data.shape[1]
    # For ridiculous ntraces values, just remove the entire average
    if ntraces >= tottraces:
        newdata=data-np.mean(data,axis=1,keepdims=True) 
    else: 
        newdata = np.asarray(np.zeros(data.shape))    
        halfwid = int(np.ceil(ntraces/2.0))
        
        # First few traces, that all have the same average
        avgtr=np.mean(data[:,0:halfwid+1],axis=1)
        # Added lines to verify
        avgtr=avgtr.reshape((len(avgtr),1))
        avgtr=np.tile(avgtr,(1,halfwid+1))   
        newdata[:,0:halfwid+1] = data[:,0:halfwid+1]-avgtr
        
        # For each trace in the middle
        for tr in tqdm(range(halfwid,tottraces-halfwid+1)):   
            winstart = int(tr - halfwid)
            winend = int(tr + halfwid)
            avgtr=np.mean(data[:,winstart:winend+1],axis=1)             
            newdata[:,tr] = data[:,tr] - avgtr

        # Last few traces again have the same average    
        avgtr=np.mean(data[:,tottraces-halfwid:tottraces+1],axis=1)
        # Added lines to verify
        avgtr=avgtr.reshape((len(avgtr),1))
        avgtr=np.tile(avgtr,(1,(tottraces)-(tottraces-halfwid))) 
        newdata[:,tottraces-halfwid:tottraces+1] = data[:,tottraces-halfwid:tottraces+1]-avgtr

    print('done with removing mean trace')
    return newdata

def quick_SVD(data_part,coeff=[None,None]):
    """
    Apply the SVD (Singular Value Decomposition) method to data to filter backgroound noise and
    horizontally coherent noise. 

    INPUT: 
    - data_part: Part of the data matrix to filter (SVD method does not work with large matrix).
                Consider doing both halves of the matrix one after the other and then concatenate
                the results. (data_part = data[:,:data.shape[1]//2])
    - coeff: 2 integers list. Possibility to cancel 2 sets of singular values. First element of list
            cancels only 1 singular value (cancel value 0 to remove air waves!). 
            Second element of list used to cancel a set of values. For example, by putting 10, values
            10 through the end get canceled.
    OUTPUT:
    - reconstruct: Filtered matrix
    """
    # Converts data from float64 to float32 to take less memory
    data = data_part.astype('float32')
    # Apply SVD to data, D contains the singular values (size m x 1)
    # Data matrix size (m x n)
    U, D, V = linalg.svd(data)
    # Create a zeros matrix of size (m x (n-m))
    rest = np.zeros((data.shape[0], data.shape[1]-data.shape[0]))
    # Cancel the first singular value
    if coeff[0] != None:
        D[coeff[0]] = 0
    # Cancel the last singular values
    if coeff[1] != None:
        D[coeff[1]:] = 0
    # Create diagonal matrix with the new singular values (size m x m)
    Dmn = np.diag(D)
    # Matrix D must be of size m x n, so we concatenate the zeros matrix of size (m x (n-m))
    # to the Dmn matrix of size (m x m) 
    Dsys = np.concatenate((Dmn, rest), axis=1)
    # Delete the matrix that are no longer used to clear memory
    del Dmn
    del rest
    # Converts all the matrix to float32
    Dsys = Dsys.astype('float32')
    U = U.astype('float32')
    V = V.astype('float32')
    # Reconstruct the data matrix with the new D matrix
    reconstruct = U @ Dsys @ V

    return reconstruct

#def band_filter(data,header,freq1,freq2):
#    """
#    """
#    ft = [np.fft.fft(data[:,i]) for i in range(data.shape[1])]
#    freq = [np.fft.fftfreq(len(data[:,i]),(header['ns_per_zsample'])) for i in range(data.shape[1])]
#    mags = [ft[i] for i in range(len(ft))]
#    mags = np.asarray(mags)
#    dat_pb = np.copy(data)
#    for i in range(mags.shape[0]):
#        freqen = np.asarray(freq[i])
#        magmax = np.where(mags[i,:] == np.max(mags[i,:]))[0][0]
#        #mags[i,:] = (freqen>0.067e9).astype("int")*mags[i,:]
#        mags[i,:] = (freqen>freq1).astype("int")*mags[i,:]
#        #mags[i,:] = (freqen<0.47e9).astype("int")*mags[i,:]
#        mags[i,:] = (freqen>freq2).astype("int")*mags[i,:]
#        coupe_temp = np.fft.ifft(mags[i,:])
#        dat_pb[:,i] = coupe_temp

def correctTopo(data, velocity, profilePos, topoPos, topoVal, twtt):
    '''
    Corrects for topography along the profile by shifting each 
    Trace up or down depending on provided coordinates.
    Taken from GPRPy. 

    INPUT:
    - data:          data matrix whose columns contain the traces
    - velocity:      subsurface RMS velocity in m/ns
    - profilePos:    along-profile coordinates of the traces
    - topoPos:       along-profile coordinates for provided elevation
                  in meters
    - topoVal:       elevation values for provided along-profile 
                  coordinates, in meters
    - twtt:          two-way travel time values for the samples, in ns

    OUTPUT:
    - newdata:       data matrix with shifted traces, padded with NaN 
    - newtwtt:       twtt for the shifted / padded data matrix
    - maxElev:       maximum elevation value
    - minElev:       minimum elevation value
    '''
    # We assume that the profilePos are the correct along-profile
    # points of the measurements
    # For some along-profile points, we have the elevation from prepTopo
    # So we can just interpolate    
    if not ((all(np.diff(topoPos)>0)) or  (all(np.diff(topoPos)<0))):
        raise ValueError('\x1b[1;31;47m' + 'The profile vs topo file does not have purely increasing or decreasing along-profile positions' + '\x1b[0m')        
    else:
        elev = interp.pchip_interpolate(topoPos,topoVal,profilePos)
        elevdiff = elev-np.min(elev)
        # Turn each elevation point into a two way travel-time shift.
        # It's two-way travel time
        etime = 2*elevdiff/velocity
        timeStep=twtt[3]-twtt[2]
        # Calculate the time shift for each trace
        tshift = (np.round(etime/timeStep)).astype(int)
        maxup = np.max(tshift)
        # We want the highest elevation to be zero time.
        # Need to shift by the greatest amount, where  we are the lowest
        tshift = np.max(tshift) - tshift
        # Make new datamatrix
        newdata = np.empty((data.shape[0]+maxup,data.shape[1]))
        newdata[:] = np.nan
        # Set new twtt
        newtwtt = np.linspace(0, twtt[-1] + maxup*timeStep, newdata.shape[0])
        nsamples = len(twtt)
        # Enter every trace at the right place into newdata
        for pos in range(0,len(profilePos)):
            newdata[tshift[pos]:tshift[pos]+nsamples ,pos] = np.squeeze(data[:,pos])

    return newdata, newtwtt, np.max(elev), np.min(elev), tshift

def snow_corr(data,x_dike,x_field,t_dike,t_field,GPS,header,offset=0,smooth=75,veloc=0.1,resample=1):
    """
    Function to merge results from fields and dikes picking AND apply topographic correction. I use this function mainly to correct vertical displacements due to the snow cover and to produce a clean and single object to pass to the final functions.

    INPUT:
    - data: Original data matrix
    - x_dike: Output of pick_dike (x_dike). List of lists containing the absolute horizontal positions of the picked coordinates in the dikes regions.
    - x_field: List of lists containing the absolute horizontal positions of the picked coordinates in the fields regions.
    - t_dike: Output of pick_dike (t_dike). List of lists containing the absolute vertical positions of the picked coordinates in the dikes regions.
    - t_field: List of lists containing the absolute vertical positions of the picked coordinates in the fields regions.
    - GPS: GPS coordinates of every traces with the elevations estimated from LIDAR data.
    - header: Header of DZT file
    - offset: Used to take into account the potential subtraction of early datas. For example, if we don't use the 5 first samples to delete air wave, we need to fix this parameter to 5. (Default is 0).
    - smooth: Integer used to define a smoothing window for the total surface (fields + dikes). Default is 75 traces.
    - veloc: Estimation of a constant EM wave velocity. Since I use this function to remove a layer of snow/ice, the default setting is 0.1 ns/m. 
    - resample: Integer to increase the number of samples using interpolation and smooth the topographic correction result. For example, if a trace needs a 2 samples vertical shift and it's neighbor stays the same, the quality of the result will depends to a degree of the vertical resolution. (Default = 1 -> No resampling) (resample = 4 -> Pass from 0,1,... to 0,0.25,0.5,0.75,1.0,...)
    This parameter increases the size of the matrix, wich can leads to memory problems with other functions.

    OUTPUT:
    - newdata_res: Data matrix after resampling and topographic correction
    - GPS_final: GPS_final takes the padding into account. To use if you want to display the picked surface over the resampled radargram with topographic correction
    - x_tot: Horizontal positions of the picked surface at every trace.
    - tot_liss_rad: Vertical positions of the picked surface at every trace, as a single object (Takes into account the resampling and the offset)
    """
    # If there is dikes
    if (x_dike != None) and (t_dike != None):
        # Concatenate all lists in the right order
        pos_hori = []
        pos_verti = []
        for i in range(len(x_dike)):
            pos_hori.append(x_field[i])
            pos_hori.append(x_dike[i])
            pos_verti.append(t_field[i])
            pos_verti.append(t_dike[i])
        pos_hori.append(x_field[-1])
        pos_verti.append(t_field[-1])
        # Concatenates
        pos_hori = [n for m in pos_hori for n in m]
        pos_verti = [p for o in pos_verti for p in o]

        # Interpolation - 1 point at every trace
        x_tot = np.linspace(pos_hori[0],pos_hori[-1],pos_hori[-1]-pos_hori[0]+1)
        f_tot = interpolate.interp1d(pos_hori,pos_verti,kind="linear")
        temps_tot = f_tot(x_tot)

        # Smoothing
        tottraces = len(x_tot)
        tot_liss = np.zeros(len(x_tot))
        halfwid_tot = int(np.ceil(smooth/2))
        # First traces
        tot_liss[:halfwid_tot+1] = np.mean(temps_tot[:halfwid_tot+1])
        # Last traces
        tot_liss[tottraces-halfwid_tot:] = np.mean(temps_tot[tottraces-halfwid_tot:])
        # Assign the mean value of the window to the middle element
        for lt in range(halfwid_tot,tottraces-halfwid_tot+1):
            tot_liss[lt] = np.mean(temps_tot[lt-halfwid_tot:lt+halfwid_tot])

    # If no dikes, just make sure there is 1 point at every trace
    else:
        x_tot = np.linspace(x_field[0],x_field[-1],x_field[-1]-x_field[0]+1)
        tot_liss = t_field

    # Positionning of GPS data on the radargram
    # Gets the elevation estimated from LIDAR data
    GPS_line = GPS[:,2]
    # Gets the max elevation
    elev_max = np.max(GPS_line)
    GPSns = np.copy(GPS_line)
    # Gets the difference between every elevation and the max elevation, and divides it by the velocity estimation for snow/ice to convert elevations to time data.
    # After this, divide by ns_per_zsample to convert data to sample
    GPSns = (2*((elev_max - GPSns)/veloc))/(header["ns_per_zsample"]*1e9)
    # Keeps only the traces with picked coordinates
    GPS_ice = GPSns[int(x_tot[0]):int(x_tot[-1])+1]
    # Picks the max elevation in terms of samples (the earliest sample is the highest point of elevation)
    elevmax_ice = np.min(tot_liss)
    ind_ice = np.where(tot_liss == elevmax_ice)[0][0]
    # Positions the GPS elevations now converted to time at the place as the picked surface on the radargram, based on the point of max elevation
    diff = GPS_ice[ind_ice] - tot_liss[ind_ice]
    GPS_ice_rad = GPS_ice - diff

    # Determines the vertical shift to apply at each trace
    shift_list_samp = (tot_liss - GPS_ice_rad)

    # If there is resampling
    if resample > 1:
        # To be sure every shift is an integer
        shift_list_samp_res = np.round(shift_list_samp*resample)/resample
        # Resampling of the data using linear interpolation and converting to float32 to avoid memory problems.
        data = data.astype("float32")
        x_mat = np.linspace(0,data.shape[0]-1,num=data.shape[0])
        f_resamp = interpolate.interp1d(x_mat,data,kind="linear",axis=0)
        x_resamp = np.linspace(0,x_mat[-1],num=(resample*len(x_mat)-(resample-1)))
        dat_resamp = f_resamp(x_resamp)
        dat_resamp = dat_resamp.astype("float32")

        # Minimum shift to apply (after resampling)
        shift_min_res = int(np.amin(shift_list_samp_res)//(1/resample))
        # Maximum shift to apply (after resampling)
        shift_max_res = int(np.amax(shift_list_samp_res)//(1/resample))
        # Upper padding of the resampled matrix - We need free space to shift the data up or down.
        # The constant 8192 was use for tests, it can be any value.
        pillow_res = 8192*np.ones((abs(shift_max_res), len(x_tot)))
        # Lower padding of the resampled matrix
        pillow_res_inf = 8192*np.ones((abs(shift_min_res), len(x_tot)))
        # Initialization of the final matrix. It's the resampled matrix, filled with zeros, sandwiched between the 2 padding matrix. 
        newdata_res = np.vstack((pillow_res,np.zeros((dat_resamp.shape[0],len(x_tot))), pillow_res_inf))
        for i in range(len(x_tot)):
            newdata_res[int(abs(shift_max_res))-int(shift_list_samp_res[i]//(1/resample)):int(abs(shift_max_res))-int(shift_list_samp_res[i]//(1/resample))+dat_resamp.shape[0],i] = dat_resamp[:,i+int(x_tot[0])]

    # If we don't want a resampling
    elif resample == 1:
        shift_min_res = int(np.amin(shift_list_samp))
        shift_max_res = int(np.amax(shift_list_samp))
        pillow_res = np.ones((abs(shift_max_res), len(x_tot)))

        # Creates lower padding only if necessary
        if any(shifty < 0 for shifty in shift_list_samp):
            pillow_res_inf = np.ones((abs(shift_min_res), len(x_tot)))
            newdata_res = np.vstack((pillow_res,np.zeros((data.shape[0],len(x_tot))), pillow_res_inf))
        else:
            newdata_res = np.vstack((pillow_res,np.zeros((data.shape[0],len(x_tot)))))

        for i in range(len(x_tot)):
            newdata_res[int(abs(shift_max_res))-int(shift_list_samp[i]):int(abs(shift_max_res))-int(shift_list_samp[i])+data.shape[0],i] = data[:,i+int(x_tot[0])]
        
    # New positionning of GPS data on radargram
    # GPS_final takes the padding into account. To use if you want to display the picked surface over the resampled radargram with topographic correction
    GPS_final = (GPS_ice_rad)*resample + pillow_res.shape[0] + offset*resample
    # Vertical positions of the picked surface at every trace, as a single object
    tot_liss_rad = tot_liss*resample + offset*resample

    return newdata_res, GPS_final, x_tot, tot_liss_rad, shift_list_samp

def find_dikes(elev_lidar,dike_wid=600,data_div=60,thresh=0.2):
    """
    Function to find the dikes' positions. This function is written specifically for the cranberry fields that I study. It is very likely to be useless for other projects.

    Calculates a moving average over a short window and finds dikes from the difference between elevations from LIDAR and the average. A threshold must be defined. 

    INPUT:
    - elev_lidar: Elevations estimated with LIDAR data.
    - dike_wid: Number of traces contained approximately over 1 dike (Default is 600).
    - data_div: Integer to define the length of the window for the moving average. For example, a value of 60 will make a window of len(elev_lidar)/60 traces long. It is recommended to use a value to get a window shorter than a dike (window length ~ dike_wid/3).
    - thresh: Threshold value. Finds the locations where (moving_average - elev_lidar) is big -> That's the positions of the dikes. Default = 0.2. To change default, look the OUTPUT diff_clone. 

    OUTPUT:
    - dikes: List of tuples containing dikes' positions [(first_trace_dike1,last_trace_dike1),(first_trace_dike2, last_trace_dike2),...]
    - diff_clone: Numpy array containing the difference between moving_average and elev_lidar. If you want to change the threshold value -> plt.plot(diff_clone)
    """
    tottraces = len(elev_lidar)
    win = int(len(elev_lidar)/data_div)
    # Initialization of the moving average array
    moving_average = np.zeros(elev_lidar.shape)
    halfwid = int(np.ceil(win/2))
    # First traces, can't use the whole window, so we just compute the mean value
    moving_average[:halfwid+1] = np.mean(elev_lidar[:halfwid+1])
    # Same thing for the last traces
    moving_average[tottraces-halfwid:] = np.mean(elev_lidar[tottraces-halfwid:])
    for i in range(halfwid,tottraces-halfwid+1):
        moving_average[i] = np.mean(elev_lidar[i-halfwid:i+halfwid])

    diff = moving_average-elev_lidar
    # Replaces every negative values with 0
    diff = diff.clip(min=0)
    # Normalizes diff so every value is between 0 and 1
    diff_norm = diff/np.max(diff)
    # Saves the normalized diff -> Useful to adjust the threshold
    diff_clone = np.copy(diff_norm)

    # Finding the dikes
    dikes = []
    for elem in range(0, len(diff_norm)):
        # For dikes (diff_norm >= thresh) at the very end of the radargram
        if (diff_norm[elem] >= thresh) and (elem > len(diff_norm)-dike_wid):
            # Checks if there is a leat another value greater than threshold
            if True in (diff_norm[elem+1:] > thresh):
                # Adds the first value over threshold and every other one to the end
                dikes.append((elem, len(diff_norm)-1))
                diff_norm[elem:] = 0
        # For dikes at the very beginning of the radargram
        elif (diff_norm[elem] >= thresh) and (elem < dike_wid):
            if True in (diff_norm[elem+1:] > thresh):
                # Selects every value within the range of a dike width
                dikes.append((0, np.max(np.where(diff_norm[elem+1:elem+dike_wid] > thresh)) + elem+1))
                diff_norm[:elem+dike_wid] = 0
        # For other dikes (in the middle of the radargram)
        elif diff_norm[elem] >= thresh:
            if True in (diff_norm[elem+1:elem+dike_wid] > thresh):
                dikes.append((elem, np.max(np.where(diff_norm[elem+1:elem+dike_wid] > thresh)) + elem+1))
                # Every selected value gets canceled so they are overlooked when finding the next dike
                diff_norm[elem:elem+dike_wid] = 0

    return dikes, diff_clone

def flip_rad(data_dst, isNorth):
    """
    Function to flip a radargram if the GPS datas are decreasing. To be confirmed: If a file is part of a grid, the radargram is presented in the direction of ascending coordinates regardless of the true direction of data acquisition.

    INPUT:
    - data_dst: GPS data
    - isNorth: Boolean value. If True, datas are considered to have been taken in the North/South directions (parallel to the dikes)

    OUTPUT:
    - Inversion: Boolean value. If True, datas must bee flipped.
    - data_dst: Flipped GPS data. It's not the GPR data 
    """
    Inversion = False
    if isNorth:
        for i in range(data_dst[:,1].size - 1):
            if data_dst[:,1][i+1] < data_dst[:,1][i]:
                Inversion = True
                break
    else:    
        for i in range(data_dst[:,0].size - 1):
            if data_dst[:,0][i+1] < data_dst[:,0][i]:
                Inversion = True
                break
    if Inversion == True:
        data_dst = np.flip(data_dst, axis=0)
    return Inversion, data_dst     

def prepVTK(profilePos,gpsmat=None,smooth=True,win_length=51,porder=3):
    '''
    Calculates the three-dimensional coordinates for each trace
    by interpolating the given three dimensional points along the
    profile.

    Taken from GPRPy

    INPUT:
    - profilePos:    the along-profile coordinates of the traces
    - gpsmat:        n x 3 matrix containing the x, y, z coordinates 
                  of given three-dimensional points for the profile
    - smooth:        Want to smooth the profile's three-dimensional alignment
                  instead of piecewise linear? [Default: True]
    - win_length:    If smoothing, the window length for 
                  scipy.signal.savgol_filter [default: 51]
    - porder:        If smoothing, the polynomial order for
                  scipy.signal.savgol_filter [default: 3]

    OUTPUT:
    - x, y, z:       three-dimensional coordinates for the traces
    '''
    if gpsmat is None:
        x = profilePos
        y = np.zeros(x.size)
        z = np.zeros(x.size)
    else:
        # Changer les positions 3D en distance le long du profil
        if gpsmat.shape[1] == 3:
            npos = gpsmat.shape[0]
            steplen = np.sqrt(
                np.power(gpsmat[1:npos,0]-gpsmat[0:npos-1,0],2.0) +
                np.power(gpsmat[1:npos,1]-gpsmat[0:npos-1,1],2.0) +
                np.power(gpsmat[1:npos,2]-gpsmat[0:npos-1,2],2.0)
            )
            alongdist = np.cumsum(steplen)
            gpsPos = np.append(0,alongdist) + np.min(profilePos)
            xval = gpsmat[:,0]
            yval = gpsmat[:,1]
            zval = gpsmat[:,2]
            x = interp.pchip_interpolate(gpsPos,xval,profilePos)
            y = interp.pchip_interpolate(gpsPos,yval,profilePos)
            z = interp.pchip_interpolate(gpsPos,zval,profilePos)
        else:
            npos = gpsmat.shape[0]
            steplen = np.sqrt(
                np.power(gpsmat[1:npos,0]-gpsmat[0:npos-1,0],2.0) +
                np.power(gpsmat[1:npos,1]-gpsmat[0:npos-1,1],2.0)
            )
            alongdist = np.cumsum(steplen)
            gpsPos = np.append(0,alongdist) + np.min(profilePos)
            xval = gpsmat[:,0]
            zval = gpsmat[:,1]
            x = interp.pchip_interpolate(gpsPos,xval,profilePos)
            z = interp.pchip_interpolate(gpsPos,zval,profilePos)
            y = np.zeros(len(x))

            # Smoothing
            if smooth:
                win_length = min(int(len(x)/2),win_length)
                porder = min(int(np.sqrt(len(x))),porder)
                x = signal.savgol_filter(x.squeeze(), window_length=win_length,
                                     polyorder=porder)
                y = signal.savgol_filter(y.squeeze(), window_length=win_length,
                                     polyorder=porder)
                z = signal.savgol_filter(z.squeeze(), window_length=win_length,
                                     polyorder=porder) 
        
        return x,y,z

def exportVTK(outfile,profilePos,twtt,data,gpsinfo,delimiter=',',thickness=0,aspect=1.0,smooth=True,win_length=51,porder=3):
    '''
        Turn processed profile into a VTK file that can be imported in 
        Paraview or MayaVi or other VTK processing and visualization tools.
        If three-dimensional topo information is provided (X,Y,Z or 
        Easting, Northing, Elevation), then the profile will be exported 
        in its three-dimensional shape.

        Taken from GPRPy

        INPUT:
        - outfile:       file name for the VTK file
        - gpsinfo:       EITHER: n x 3 matrix containing x, y, and z or 
                              Easting, Northing, Elevation information
                      OR: file name for ASCII text file containing this
                          information
        - delimiter:     if topo file is provided: delimiter (by comma, or by tab)
                      [default: ',']. To set tab: delimiter='\t' 
        - thickness:     If you want your profile to be exported as a 
                      three-dimensional band with thickness, enter thickness
                      in meters [default: 0]
        - aspect:        aspect ratio in case you want to exaggerate z-axis.
                      default = 1. I recommend leaving this at 1 and using 
                      your VTK visualization software to set the aspect for
                      the representation.
        - smooth:        Want to smooth the profile's three-dimensional alignment
                      instead of piecewise linear? [Default: True]
        - win_length:    If smoothing, the window length for 
                      scipy.signal.savgol_filter [default: 51]
        - porder:        If smoothing, the polynomial order for
                      scipy.signal.savgol_filter [default: 3]
        '''
    # Si gpsinfo est un nom de fichier, il faut d'abord le charger
    if type(gpsinfo) is str:
        gpsmat = np.loadtxt(gpsinfo,delimiter=delimiter)
    else:
        gpsmat = gpsinfo

    # Obtenir les positions x,y,z des points
    x,y,z = prepVTK(profilePos,gpsmat,smooth,win_length,porder)
    z = z*aspect
    #if self.velocity is None:
    #    downward = self.twtt*aspect
    #else:
    #    downward = self.depth*aspect
    downward = twtt*aspect
    Z = np.reshape(z,(len(z),1)) - np.reshape(downward,(1,len(downward)))

    if thickness:
        ZZ = np.tile(np.reshape(Z,(1,Z.shape[0],Z.shape[1])),(2,1,1))
    else:
        ZZ = np.tile(np.reshape(Z,(1,Z.shape[0],Z.shape[1])),(1,1,1))

    if thickness:
        pvec = np.asarray([(y[0:-1]-y[1:]).squeeze(),(x[1:]-x[0:-1]).squeeze()])
        pvec = np.divide(pvec, np.linalg.norm(pvec,axis=0)) * thickness/2.0
        pvec = np.append(pvec, np.expand_dims(pvec[:,-1],axis=1) ,axis=1)
        X = np.asarray([(x.squeeze()-pvec[0,:]).squeeze(), (x.squeeze()+pvec[0,:]).squeeze()])
        Y = np.asarray([(y.squeeze()+pvec[1,:]).squeeze(), (y.squeeze()-pvec[1,:]).squeeze()])
    else:
        X = np.asarray([x.squeeze()])
        Y = np.asarray([y.squeeze()])
    # Copy-paste the same X and Y positions for each depth
    XX = np.tile(np.reshape(X, (X.shape[0],X.shape[1],1)), (1,1,ZZ.shape[2]))
    YY = np.tile(np.reshape(Y, (Y.shape[0],Y.shape[1],1)), (1,1,ZZ.shape[2]))

    #if self.maxTopo is None:
    #    data=self.data.transpose()
    data = data.transpose()
    #else:
    #    data=self.data_pretopo.transpose()  

    data = np.asarray(data)
    data = np.reshape(data,(1,data.shape[0],data.shape[1]))
    data = data.astype("float32")
    data = np.tile(data,(2,1,1))

    # Remove the last row and column to turn it into a cell
    # instead of point values 
    data = data[0:-1,0:-1,0:-1]

    nx=2-1
    ny=len(x)-1
    nz=len(downward)-1
    datarray = np.zeros(nx*ny*nz).reshape(nx,ny,nz)
    datarray[:,:,:] = data

    XX = XX.astype("float32")
    YY = YY.astype("float32")
    ZZ = ZZ.astype("float32") 
    datarray = datarray.astype("float32")   
    gridToVTK(outfile,XX,YY,ZZ, cellData ={'gpr': datarray})