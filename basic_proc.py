"""
Functions to apply basic processing steps to GPR data.

The following funcions are taken from the GPRPy repository (https://github.com/NSGeophysics/GPRPy):
- deWOW()
- rem_mean_trace()
"""
import numpy as np
from tqdm import tqdm
from scipy import linalg

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

def band_filter(data,header,freq1,freq2):
    """
    """
    ft = [np.fft.fft(data[:,i]) for i in range(data.shape[1])]
    freq = [np.fft.fftfreq(len(data[:,i]),(header['ns_per_zsample'])) for i in range(data.shape[1])]
    mags = [ft[i] for i in range(len(ft))]
    mags = np.asarray(mags)
    dat_pb = np.copy(data)
    for i in range(mags.shape[0]):
        freqen = np.asarray(freq[i])
        magmax = np.where(mags[i,:] == np.max(mags[i,:]))[0][0]
        #mags[i,:] = (freqen>0.067e9).astype("int")*mags[i,:]
        mags[i,:] = (freqen>freq1).astype("int")*mags[i,:]
        #mags[i,:] = (freqen<0.47e9).astype("int")*mags[i,:]
        mags[i,:] = (freqen>freq2).astype("int")*mags[i,:]
        coupe_temp = np.fft.ifft(mags[i,:])
        dat_pb[:,i] = coupe_temp