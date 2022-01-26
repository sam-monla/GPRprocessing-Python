"""
Set of functions useful for the process of amplitude picking. The method described as amplitude picking is separated in 3 steps.

1. Preliminary picking : For each trace, picks the first amplitude that exceeds a high threshold (Must remove air waves)

2. Horizontal smoothing : Removes each picked coordinates that introduce high vertical variations. Interpolates a smooth surface with the remaining picked coordinates.

3. Final picking :

Bad coding
"""
import numpy as np
from scipy import linalg
from scipy import interpolate
from tqdm import tqdm

def prel_picking(data,dikes,header,samp_max):
    """
    Function for the preliminary picking of GPR data. 

    INPUT:
    - data: Data matrix from the DZT file
    - dikes: List of tuples containing the dikes' positions
    - header: Header from the DZT file
    - samp_max: Integer used to remove some samples from the data matrix. Could be necessary since the SVD method needs lots of memory space. Limit your matrix just below the lowest point of the surface that you want to pick.

    OUTPUT:
    The output is separated in fields (Regions separated by dikes). Outputs in the form of dictionnaries. To access the radargram of field 1 -> recon["field1"]
    - res_pick:
    - res_pick_brut:
    - recon: Dictionnary containing the radargrams of the fields after the application of SVD method.
    """
    # Useful informations from the DZT file
    ntraces = data.shape[1]
    ns_persample = header["ns_per_zsample"]*1e9
    line_len = header["sec"]

    # Conversion to float32 (Better for the SVD method)
    data = data.astype('float32')

    # Initialization of the output dictionnaries
    res_pick = {}
    res_pick_brut = {}
    recon = {}
    deb_dig = False

    # SVD method for air waves removal (could be used to reduce noise)
    # The method is applied on each field (between the dikes) individually. At the end of each iteration, the picked coordinates are added to a dictionnary
    # See quick_SVD() in basic_proc.py for more details about the SVD method
    for i in tqdm(range(len(dikes)+1)):
        if i == 0:
            # Approximate dike width: 600 traces
            # If the dike is at the very beginning, skip to the next
            if dikes[i][0] <= 300:
                deb_dig = True
                continue
            else:
                # If not, apply the method from the beginning of the radargram to the beginning of the current dike
                U, D, V = linalg.svd(data[:samp_max,:dikes[i][0]])
                reste = np.zeros((samp_max, dikes[i][0]-samp_max))

        # For the last dike
        elif i == len(dikes):
            # If the dike is at the very end,skip it
            if (dikes[i-1][0]) >= (data.shape[1]-500):
                break
            else:
                # Apply the method from the end of the last dike to the end of the radargram
                U, D, V = linalg.svd(data[:samp_max,dikes[i-1][1]:])
                reste = np.zeros((samp_max, ntraces-dikes[i-1][1]-samp_max))
        
        else:
            # Apply the method from the end of the last dike to the beginning of the current one
            U, D, V = linalg.svd(data[:samp_max,dikes[i-1][1]:dikes[i][0]])
            reste = np.zeros((samp_max, dikes[i][0]-dikes[i-1][1]-samp_max))
        
        # Cancels the first singular value (SV) to remove air waves
        D[0] = 0
        # Canceling late SVs can be useful to reduce noise
        #D[6:] = 0
        Dmn = np.diag(D)
        Dsys = np.concatenate((Dmn, reste), 1)
        # Clearing memory
        del Dmn
        del reste
        Dsys = Dsys.astype('float32')
        U = U.astype('float32')
        V = V.astype('float32')
        reconstruct = U @ Dsys @ V

        # Picking
        line = reconstruct.tolist()
        colnorm = conv_col_norm(line, norm=True)
        del U
        del V

        if i == 0:
            a, b, c, d = basic_picker(colnorm, 0.8, 0, ntraces, ns_persample, line_len)
        else:
            a, b, c, d = basic_picker(colnorm, 0.8, dikes[i-1][1], ntraces, ns_persample, line_len)

        # Adding the picked value to the dictionnaries before passing to the next iteration
        res_pick['field'+str(i+1)] = [a, b]
        res_pick_brut['field'+str(i+1)] = [c, d]
        recon['field'+str(i+1)] = reconstruct
        del a
        del b
        del c
        del d
        del reconstruct

    return res_pick, res_pick_brut, recon, deb_dig

def hori_smooth(win,scale,pick_data,degree=1):
    """
    Function to filter picked coordinates that are too far from the expected result (coordinates that introduce great vertical variations). The radargram is separated in windows. In each window, the median value and the standard deviation is calculated. Every value outside the range (median +/- scale*std) is canceled. With the remaining points, a surface is calculated. 

    INPUT:
    - win: Window lenght (in no. of traces)
    - scale: Sets the tolerance for the max. accepted deviation between a picked value and the mean for a given window. The scale factor sets the tolerance this way -> tol = median +/- scale*standard deviation.
    - pick_data: The picked data (Output of prel_picking)
    - degree: Degree of the fitting polynomial used by np.polyfit() (Default is 1)

    OUTPUT:
    - smooth_result: Smoothed surface calculated from every points that don't introduce great vertical variations. smooth_result gives a vertical coordinate for every single trace.

    """
    # Initialization of the smoothed surface. Each segment of this surface is win/2 traces long.
    step = int(win/2)
    bins = np.arange(0, len(pick_data), step)
    xsmooth_seed = np.zeros(len(bins))
    ysmooth_seed = np.zeros(len(bins))

    ### - First values - ####################################################################################
    # Picks every points between 0 and win/2
    yobs = pick_data[0:bins[1]]
    xobs = np.linspace(0,bins[1],num=len(yobs))
    # Gets the mean value and standard deviation
    mu_data = np.median(yobs)
    sigma_data = np.std(yobs)
    # Avoids filtering every single coordinates
    if sigma_data == 0:
        sigma_data = 0.25
    # Finds every value inside the tolerance
    y_sond = np.where((yobs<mu_data+(scale*sigma_data)) & (yobs>mu_data-(scale*sigma_data)))
    x_sond = np.where((yobs<mu_data+(scale*sigma_data)) & (yobs>mu_data-(scale*sigma_data)))

    # Repeats the tolerance process if zero picked coordinates are remaining. The tolerance is increased gradually until at least one coordinate remains. 
    multi = 1.5
    while len(x_sond[0]) == 0:
        x_sond = np.where((yobs<mu_data+(multi*scale*sigma_data)) & (yobs>mu_data-(multi*scale*sigma_data)))
        y_sond = np.where((yobs<mu_data+(multi*scale*sigma_data)) & (yobs>mu_data-(multi*scale*sigma_data)))
        multi += 0.5
    
    # Gets the radargram positions of the remaining points. x_y_sond only gives the index (not good for vertical values)
    y = [yobs[i] for i in y_sond[0]]
    x = [xobs[j] for j in x_sond[0]]
    # Fits a polynomial to the remaining points.
    p=np.polyfit(x,y,degree)
    # Sets the first value of the smoothed surface
    xsmooth_seed[0]=min(xobs)
    ysmooth_seed[0]=np.polyval(p,min(xobs))

    ### - Last values - #####################################################################################
    # Same thing as with the first values
    yobs = pick_data[bins[-2]:]
    xobs = np.linspace(bins[-2],len(pick_data)-1,num=len(yobs))
    mu_data = np.median(yobs)
    sigma_data = np.std(yobs)
    if sigma_data == 0:
        sigma_data = 0.25
    y_sond = np.where((yobs<mu_data+(scale*sigma_data)) & (yobs>mu_data-(scale*sigma_data)))
    x_sond = np.where((yobs<mu_data+(scale*sigma_data)) & (yobs>mu_data-(scale*sigma_data)))

    multi = 1.5
    while len(x_sond[0]) == 0:
        x_sond = np.where((yobs<mu_data+(multi*scale*sigma_data)) & (yobs>mu_data-(multi*scale*sigma_data)))
        y_sond = np.where((yobs<mu_data+(multi*scale*sigma_data)) & (yobs>mu_data-(multi*scale*sigma_data)))
        multi += 0.5

    y = [yobs[i] for i in y_sond[0]]
    x = [xobs[j] for j in x_sond[0]]
    p=np.polyfit(x,y,degree)
    # Sets the last value of the smoothed surface
    xsmooth_seed[-1]=max(xobs)
    ysmooth_seed[-1]=np.polyval(p,max(xobs))

    ### - Other values - ####################################################################################
    # Same thing as with first values
    for i in range(1, len(bins)-1):
        yobs = pick_data[bins[i-1]:bins[i+1]]
        xobs = np.linspace(bins[i-1],bins[i+1],num=len(yobs))
        mu_data = np.median(yobs)
        sigma_data = np.std(yobs)
        if sigma_data == 0:
            sigma_data = 0.25
        y_sond = np.where((yobs<mu_data+(scale*sigma_data)) & (yobs>mu_data-(scale*sigma_data)))
        x_sond = np.where((yobs<mu_data+(scale*sigma_data)) & (yobs>mu_data-(scale*sigma_data)))

        multi = 1.5
        while len(x_sond[0]) == 0:
            x_sond = np.where((yobs<mu_data+(multi*scale*sigma_data)) & (yobs>mu_data-(multi*scale*sigma_data)))
            y_sond = np.where((yobs<mu_data+(multi*scale*sigma_data)) & (yobs>mu_data-(multi*scale*sigma_data)))
            multi += 0.5

        y = [yobs[i] for i in y_sond[0]]
        x = [xobs[j] for j in x_sond[0]]
        p=np.polyfit(x,y,degree)
        xsmooth_seed[i]=bins[i]
        ysmooth_seed[i]=np.polyval(p,bins[i])
    
    # The smoothed surface interpolated previously only have a few points (One every win/2 traces). Here, we interpolate one point for every trace from the smoothed surface
    f = interpolate.interp1d(xsmooth_seed, ysmooth_seed,kind='slinear')
    traces = np.arange(0, len(pick_data), 1)
    smooth_result = f(traces)

    return smooth_result

def final_picking(smooth_surf, lim, data, ntraces, header, start):
    """
    Function for the final picking of GPR data. This function takes the smoothed surface produced by hori_smooth() and defines a searching window around it. The sample with the biggest amplitude inside this window is taken. 

    INPUT:
    - smooth_surf: Smoothed surface (Output of hori_smooth())
    - lim: Sets a vertical limit for the searching window. For example, if you only take the first 100 samples of the data and the searching window is centered at sample 99, this parameter avoids definning a window going to sample 101.
    - data: Data matrix (Output of prel_picking())
    - ntraces: Total number of traces in data
    - header: Header of the DZT file
    - start: Starting trace. If you only want to do picking on a portion of the data, set this parameter to the first trace of the portion.

    OUTPUT:
    - cx: Horizontal picked coordinates (meters)
    - cy: Vertical picked coordinates (ns)
    - cx_brut: Horizontal picked coordinates (trace)
    - cy_brut: Vertical picked coordinates (sample)
    """
    # Gets vertical index of smoothed surface
    mat_ind = (np.round(smooth_surf)).astype('int')
    # Searching window is 5 samples wide centered around the smoothed surface
    mat_ind = np.vstack((mat_ind-2, mat_ind-1, mat_ind, mat_ind+1, mat_ind+2))
    # Avoids searching window going off bounds
    mat_ind[mat_ind >= lim] = lim - 1.
    size = mat_ind.shape[1]
    # Isolates the part of data inside the searching window for every trace
    pick2 = [data[mat_ind[:,i],i] for i in range(size)]
    pick2 = (np.stack(pick2)).T
    lines = pick2.tolist()
    # Picking
    colnorm = conv_col_norm(lines, norm=True)
    cx, cy, cx_brut, cy_brut = basic_picker(colnorm, 0.4, start, ntraces, header["ns_per_zsample"]*1e9, header["sec"],temp_shift=mat_ind[0])

    return cx, cy, cx_brut, cy_brut

def conv_col_norm(list_lines, norm=True):
    """
    Converts a list of lines in a normalized list of traces.
    INPUT:
    - list_lines : Liste of lists. Each sublist is a line of the matrix that we want to convert.
    - norm : Boolean value indicating if we want to normalize every trace.
    OUTPUT:
    - liste_col: List of columns (normalized or not)

    """
    liste_col = []
    for column in range(0, len(list_lines[0])):
        col = []
        for ligne_Cut in list_lines:
            col.append(float(ligne_Cut[column]))
        liste_col.append(col)
    if norm:
        liste_col_norm = []
        for colonne in liste_col:
            maxi = max(colonne)
            if maxi == 0:
                maxi = 1
            col_norm = []
            for elem_norm in colonne:
                col_norm.append(elem_norm/maxi)
            liste_col_norm.append(col_norm)
        return liste_col_norm
    else:
        return liste_col

def basic_picker(
    list_col_norm, threshold, start_trace, traces, ns_persample,
    line_len, temp_shift=0
    ):
    """
    INPUTS:
    - list_col_norm: List of lists. Each sublist is a normalized trace
    - threshold: Integer. The first normalized amplitude greater than this is picked
    - start_trace: Integer. Starting trace of the region. For example, if you want to pick a reflector in the second field, we put the first trace of this field as the starting trace.
    - traces: Total number of traces
    - ns_persample: Sampling period (ns)
    - line_len: Length of the GPR line (m) (See header file)
    - temp_shift: Temporal shifting. If we don't want to consider the first 2 samples of each trace, we put this parameter to 2. 
    
    OUTPUT:
    - coord_x: List of horizontal coordinates in meters
    - coord_z: List of vertical coordinates in ns
    - coord_x_brut: List of horizontal coordinates in no. of trace
    - coord_z_brut: List of vertical coordinates in no. of sample
    """
    coord_x = []
    coord_z = []
    coord_x_brut = []
    coord_z_brut = []
    decalage = temp_shift * np.ones(len(list_col_norm))

    for trace in list_col_norm:
        for indi, val in enumerate(trace):
            # If the value is greater than thresh
            if val > threshold:
                # Coordinates in meters and ns 
                coord_z.append((indi+decalage[list_col_norm.index(trace)])*ns_persample)
                coord_x.append((line_len/traces)*(list_col_norm.index(trace)+start_trace))
                # Coordinates in no. of trace and sample
                coord_z_brut.append(indi+decalage[list_col_norm.index(trace)])
                coord_x_brut.append(list_col_norm.index(trace)+start_trace)
                break
            else:
                continue
    return (coord_x, coord_z, coord_x_brut, coord_z_brut)

