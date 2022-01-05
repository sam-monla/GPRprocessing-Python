"""
Set of functions useful for the process of amplitude picking. The method described as amplitude picking is separated in 3 steps.
1. Preliminary picking : For each trace, picks the first amplitude that exceeds a high threshold (Must remove air waves)
2. Horizontal smoothing
3. Final picking

Bad coding
"""
import numpy as np
from scipy import linalg

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
    for i in range(len(dikes)+1):
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

    return res_pick, res_pick_brut, recon

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
        liste_colonnes_norm = []
        for colonne in liste_col:
            maxi = max(colonne)
            if maxi == 0:
                maxi = 1
            col_norm = []
            for elem_norm in colonne:
                col_norm.append(elem_norm/maxi)
            liste_col.append(col_norm)
        return liste_col
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

