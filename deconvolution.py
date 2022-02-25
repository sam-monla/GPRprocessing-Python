"""
File to contain deconvolution methods
"""

import numpy as np

########################################################################################################################################################################################################

def DR_spectral_enhancement(trace,dt,smooth_pass=10,fullout=False):
    """
    Spectral enhancement method proposed by Sajid and Ghosh in "A fast and simple method of spectral enhancement" (2014)

    INPUT:
    - trace: A column of the data matrix
    - dt:
    - smooth_pass: Number of passes of the 3-points smoothing function
    - fullout: Boolean value indicating the output wanted.
            True: Returns the second,fourth and sixth derivatives estimate
            False: Returns the result and normalized result of the spectral enhancement

    OUTPUT:
    - dr: Spectral enhancement of the original trace
    - DR: Normalized spectral enhancement of the original trace
    """

    smooth_trace = three_pts_smoother(trace,passes=smooth_pass)
    Yfirst = fin_diff(trace,dt,kind="fwd")
    Ysec = fin_diff(Yfirst,dt,kind="bwd")
    Ythird = fin_diff(Ysec,dt,kind="fwd")
    Yforth = fin_diff(Ythird,dt,kind="bwd")
    Yfifth = fin_diff(Yforth,dt,kind="fwd")
    Ysixth = fin_diff(Yfifth,dt,kind="bwd")

    norm_trace = trace/np.median(np.abs(trace),axis=0)
    norm_smooth = smooth_trace/np.median(np.abs(smooth_trace),axis=0)
    norm_Ysec = Ysec/np.median(np.abs(Ysec),axis=0)
    norm_Yforth = Yforth/np.median(np.abs(Yforth),axis=0)
    norm_Ysixth = Ysixth/np.median(np.abs(Ysixth),axis=0)

    #dr = norm_trace + norm_smooth - norm_Ysec + norm_Yforth - norm_Ysixth
    #dr = np.add(norm_trace,norm_smooth,-norm_Ysec,norm_Yforth,-norm_Ysixth)
    dr = np.sum((norm_trace,norm_smooth,-norm_Ysec,norm_Yforth,-norm_Ysixth),axis=0)
    DR = dr/np.median(np.abs(dr),axis=0)

    if fullout:
        return dr, DR, norm_Ysec, norm_Yforth, norm_Ysixth
    else:
        return dr, DR


def fin_diff(trace,dt,kind="fwd"):
    """
    Function to quickly apply a finite difference operator.

    INPUT:
    - trace: A column of the data matrix
    - dt: 
    - kind: Boolean value indicating "fwd" or "bwd" for a forward or backward operator
    """
    newtrace = (np.copy(trace)).astype("float32")

    for i in range(1,len(trace)-1):
        if kind == "fwd":
            newtrace[i] = (trace[i+1] - trace[i])/dt
        elif kind == "bwd":
            newtrace[i] = (trace[i] - trace[i-1])/dt
    
    return newtrace

def three_pts_smoother(trace,passes=10):
    """
    Function to smooth a trace by computing the weighted mean value (1,2,1) of a 3 samples wide window.

    INPUT:
    - trace: A column of the data matrix
    - passes: Number of times the smoother will be applied

    OUTPUT:
    - newtrace: Smoothed trace
    """
    newtrace = (np.copy(trace)).astype("float")
    i = 0 
    while i <= passes-1: 
        for j in range(1,len(trace)-1):
            newtrace[j] = (newtrace[j-1]+(2.*newtrace[j])+newtrace[j+1])/4.
        i += 1

    return newtrace

########################################################################################################################################################################################################