"""
Set of functions useful for the process of phase picking.

Based on the work of Matteo Dossi, Emanuele Forte and Michele Pipan:
- Automated reflection picking and polarity assessment through
    attribute analysis: Theory and application to synthetic
    and real ground-penetrating radar data

    https://pubs.geoscienceworld.org/geophysics/article/80/5/H23/308900?casa_token=Dc0RH7q7fssAAAAA:LvbGMErN7zO1FpHq3qhyJPf9-_HE36lmDXJCJjfzvijmz-oTmio3Pzey3t9tD5ONdjn4X84bCg

- Automated reflection picking and inversion applied to glaciological
    GPR surveys

    https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Automated+reflection+picking+and+inversion+applied+to+glaciological+++++GPR+surveys&btnG=
"""
import numpy as np
from itertools import groupby
from operator import itemgetter

def quad_trace(data):
    """
    Function to calculte the quadrature trace using the Hilbert transform.
    See equation (4) in < Automated reflection picking and polarity assessment through
    attribute analysis: Theory and application to synthetic
    and real ground-penetrating radar data >.

    INPUT:
    - data: GPR data matrix, with or without prefiltering. Values must be centered around 0.
    OUTPUT:
    - dat_prim: Matrix in wich each column is the imaginary part of the corresponding column 
        of the original matrix
    """
    # Prepares the final matrix
    dat_prim = np.zeros(data.shape)
    # Gets the amount of samples
    Nsamp = data.shape[0]
    # Creates a list of Nsamp items (values from 0 to Nsamp)
    nlist = np.linspace(0,Nsamp-1,num=Nsamp)
    for n in range(len(nlist)):
        # Creates a list of k's for each value in nlist
        klist = np.linspace(n+1-Nsamp,n,num=Nsamp).astype('int')
        # For each k, (sin^2(pi*k/2) / k)*A_(n-k), except if k == 0
        for k in klist:
            if k == 0:
                continue
            else:
                a = (((np.sin((np.pi*k/2)))**2)/k)*data[n-k,:]
                # Store the result in the final matrix (initially a zeros matrix)
                dat_prim[n,:] += a
        # When every value of k is passed, multiply the result by 2pi^-1
        dat_prim[n,:] = dat_prim[n,:]*(2/np.pi)

    return dat_prim

def Ctwt_matrix(cosph_data):
    """
    Function to create the Cij and twtij matrix presented in < Automated reflection picking and polarity assessment through attribute analysis: Theory and application to synthetic and real ground-penetrating radar data >. Empty spots for the Cij matrix are filled with 2.0, -1.0 for the twtij matrix.

    INPUT:
    - cosph_data: Cosine of the instantaneous phase matrix, i.e. the original matrix transformed into phase  (quad_trace) and applying cos(arctan(x)) to the result.
    OUTPUT:
    - twtij_mat: Matrix containing the indexes where every phase begins.
    - Cij_mat: Matrix containing the max amplitude of each phase.
    """
    # Marks with 1 the samples where there is a sign change
    sign_change = (np.diff(np.sign(cosph_data),axis=0) != 0)*1
    # Finds the indexes of sign change et sorts them by their traces (column) number
    ind_change = np.where(sign_change == 1)
    pos_change = list(zip(ind_change[1],ind_change[0]))
    twtij = sorted(pos_change, key=lambda x: x[0])
    # Finds the max amount of phases for one trace
    unique, counts = np.unique(ind_change[1],return_counts=True)
    dict1 = dict(zip(unique, counts))
    max_phase = max(dict1, key=dict1.get)

    """
    Making the twtij matrix
    """
    # Twtij matrix initialization (max_phase x ntrace)
    twtij_mat = np.zeros((dict1[max_phase],cosph_data.shape[1]))
    # Creates a list of ntrace lists. Each sublist contains the samples where a sign change occurs.
    b = [list(list(zip(*g))[1]) for k, g in groupby(twtij, itemgetter(0))]
    # Replaces each column of the final matrix by the corresponding sublist
    for trace in range(cosph_data.shape[1]):
        twtij_mat[:len(b[trace]),trace] = b[trace]

    """
    Making the Cij matrix
    """
    # Cij matrix initialization
    Cij_mat = np.zeros((dict1[max_phase]+1,cosph_data.shape[1]))
    for trace in range(cosph_data.shape[1]):
        peaks = []
        # For each samples of a given trace...
        for i in range(twtij_mat.shape[0]):
            # Do nothing if no sign and continue
            if twtij_mat[i,trace] == 0:
                continue
            # If it's the first sample, find the abs(max) amplitude between the first sample and the first value in twtij_mat, adds the value to the peaks list
            elif i == 0:
                peaks.append(max(cosph_data[:int(twtij_mat[i,trace]),trace].min(),cosph_data[:int(twtij_mat[i,trace]),trace].max(), key=abs))
            # Find the abs(max) amplitude between the last sample in twtij_mat (i-1) and the current sample (i), adds the value to the peaks list
            else:
                peaks.append(max(cosph_data[int(twtij_mat[i-1,trace]):int(twtij_mat[i,trace]),trace].min(),cosph_data[int(twtij_mat[i-1,trace]):int(twtij_mat[i,trace]),trace].max(),key=abs))
        # Find the abs(max) amplitude in the last group of samples for a given trace
        peaks.append(max(cosph_data[int(twtij_mat[len(b[trace])-1,trace]):,trace].min(),cosph_data[int(twtij_mat[len(b[trace])-1,trace]):,trace].max(),key=abs))
        # Puts the peak values in the Cij matrix
        Cij_mat[:len(peaks),trace] = peaks

    # Since not every trace has the same amount of sign changes, some spots in the Cij and twtij matrix are left with 0s. We replace them with -1 for the twtij matrix/with 2 for the Cij matrix since these are impossible values. 
    twtij_mat[twtij_mat == 0] = -1
    Cij_mat[Cij_mat == 0] = 2
    # We add a line if 0s on top of the twtij matrix so it has the same dimension as the Cij matrix
    row1_twt = np.zeros((1,twtij_mat.shape[1]))
    twtij_mat = np.concatenate((row1_twt,twtij_mat),axis=0)

    return twtij_mat, Cij_mat


