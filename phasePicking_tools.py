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
from tqdm import tqdm

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

def horipick(Cij,twtij,Tph,tol_C,tolmin_t,tolmax_t,h_offset=0,min_time=6):
    """
    Function for the automatic picking of horizons in GPR data. For a given trace, one of it's sample can be tied to a sample of a neighboring trace if the following conditions are met:
    1. Same polarity
    2. Temporal proximity
    3. 2 phases can't cross
    See < Automated reflection picking and polarity assessment through attribute analysis: Theory application to synthetic and real ground-penetrating radar data > for more details.

    INPUT:
    - Cij: Cij matrix
    - twtij: twtij matrix
    - Tph: Approx. period of one phase (in samples). The temporal width of the air wave's phase is a good 1st approx. 
    - tol_C: Number between 0 and 1 to filter amplitudes that are too low (far from -1 or +1) in the Cij matrix. Example: 0.7 -> Every amplitude higher/lower than +/- 1 are kept.
    - tolmin_t: Number between 0 and 1 to filter phases that are too short. Example: 0.5 -> Every phases longer than 0.5*Tph are kept.
    - tolmax_t: Number between 0 and 1 to filter phases that are too long. (tolmax_t*Tph is the cutoff)
    - h_offset: Shift the traces horizontally. For example, if you want to pick horizons starting at the 100th trace, set this parameter to 100.
    - min_time: Integer to filter every phase that comes before sample no. min_time (default=6)

    OUTPUT:
    - list_horizons: List of lists. Each sublist is an horizon. Each element of an horizon is a tuple of the form (sample,trace).
    - list_horizons_offs: Same as list_horizons, but the traces are shifted horizontally depending on the parameter h_offset. 
    - Cij_clone: A copy of the original Cij matrix
    - twtij_clone: A copy of the original twtij matrix
    """
    # Copy of the original Cij and twtij matrix for later use
    twtij_clone = np.copy(twtij)
    Cij_clone = np.copy(Cij)

    # Filters phases that are too long or too short
    # Calculates the differences between to rows in the twtij matrix for each trace -> gets their length
    diff_twt = np.diff(twtij,axis=0) 
    # Finds the problematic phases
    bad_phase = np.argwhere((diff_twt < 0) | (diff_twt < tolmin_t*Tph) | (diff_twt > tolmax_t*Tph))
    if len(bad_phase)>0:
        # Replaces bad phases with 2 in the Cij matrix (impossible value)
        f1_Cij = 2*np.ones((1,len(bad_phase)))
        # Replace bad phases with -1 in the twtij matrix (impossible value)
        f1_twtij = -1*np.ones((1,len(bad_phase)))
        # Gets the 2D indexes of each bad phase and asign the impossible values in the appropriate matrix 
        row, cols = zip(*bad_phase)
        Cij[row,cols] = f1_Cij
        twtij[row,cols] = f1_twtij
    
    # Filters phases that are too early
    early_phase = np.argwhere(twtij <= min_time)
    if len(early_phase)>0:
        # Impossible values for Cij matrix
        f1b_Cij = 2*np.ones((1,len(early_phase)))
        # Impossible values for twtij matrix
        f1b_twtij = -1*np.ones((1,len(early_phase)))
        row, cols = zip(*early_phase)
        Cij[row,cols] = f1b_Cij
        twtij[row,cols] = f1b_twtij

    # Filtrers weak amplitudes
    bad_amp = np.argwhere(np.abs(Cij) < tol_C)
    if len(bad_amp)>0:
        f2_Cij = 2*np.ones((1,len(bad_amp)))
        f2_twtij = -1*np.ones((1,len(bad_amp)))
        row, cols = zip(*bad_amp)
        Cij[row,cols] = f2_Cij
        twtij[row,cols] = f2_twtij

    # Final horizons lists initialization
    list_horizons = []
    list_horizons_offs = []

    # For each phase (tqdm is to display progress bar)
    for ph in tqdm(range(Cij.shape[0])):
        # For each trace
        for tr in range(twtij.shape[1]):
            # Verify if amplitude and time are valid at this position. If not, we pass to the next trace
            if (Cij[ph,tr] == 2) or (twtij[ph,tr] == -1):
                    continue
            
            # Horizon initialization
            hori = [(ph,tr)]
            hori_offs = [(ph,tr+h_offset)]

            # Can we extend the horizon?
            # Make sur we are not at the end of the matrix
            if hori[-1][1] < Cij.shape[1]-1:
                # Can we extend with the trace to the right?
                # Checking for polarity matching
                signe = np.sign(Cij[ph,tr])
                if signe < 0:
                    crit1 = Cij[:,tr+1] < 0
                else:
                    crit1 = (Cij[:,tr+1] > 0) & (Cij[:,tr+1] <= 1)
                # Checking for temporal proximity
                crit2 = (twtij[:,tr+1] > (twtij[ph,tr]-Tph/2)) & (twtij[:,tr+1] < (twtij[ph,tr]+Tph/2)) & (twtij[:,tr+1] >= 0)
                # Mixing the criterias
                crits_prol = crit1 & crit2

                # Create a "trace" variable to continue iterating while memorizing the current trace of the main loop
                trace = tr
                # Check if an extension is possible to the right of hori. Repeat the process until crit1 or crit2 is not met, or when we arrive at the far right of the matrix.
                while (np.any(crits_prol)) and (hori[-1][1] < Cij.shape[1]-1):
                    hori_extens = np.argwhere(crits_prol == True)
                    hori_extens = np.asarray(hori_extens).reshape((1,len(hori_extens)))[0]

                    # If more than 1 extension is possible
                    if len(hori_extens) > 1:
                        diff = []
                        for i in range(len(hori_extens)):
                            diff.append(np.abs(hori_extens[i]-hori[-1][0]))
                        # Chooses the closest one sample wise
                        pos_min = np.where(diff == np.min(diff))
                        hori.append((hori_extens[pos_min[0][0]],trace+1))
                        hori_offs.append((hori_extens[pos_min[0][0]],trace+1+h_offset))
                    else:
                        # Extends hori if there is just 1 extension possible
                        hori.append((hori_extens[0],trace+1))
                        hori_offs.append((hori_extens[0],trace+1+h_offset))

                    # Check in the next trace to the right if an other extension is possible
                    trace += 1
                    # Check the polarity and temporal proximity criterias
                    if hori[-1][1] == Cij.shape[1]-1:
                        break
                    elif signe < 0:
                        crit1 = Cij[:,hori[-1][1]+1] < 0 
                    else:
                        crit1 = Cij[:,hori[-1][1]+1] > 0
                    time_ref = twtij[hori[-1][0],hori[-1][1]]
                    crit2 = (twtij[:,hori[-1][1]+1] > (time_ref-Tph/2)) & (twtij[:,hori[-1][1]+1] < (time_ref+Tph/2)) & (twtij[:,hori[-1][1]+1] >= 0)
                    crits_prol = crit1 & crit2
                    # If the criterias are met again, the while loop is repeated

            # When the while loop breaks, the resulting horizon is put in list_horizons
            list_horizons.append(hori)
            list_horizons_offs.append(hori_offs)
            # Puts impossible values in the spots that are tied to an horizon. This way, the algorithm can't put a spot in 2 different horizons and there is less point to examine. 
            Cij_term = 2*np.ones((1,len(hori)))
            twtij_term = -1*np.ones((1,len(hori)))
            row, cols = zip(*hori)
            Cij[row,cols] = Cij_term
            twtij[row,cols] = twtij_term

    return list_horizons, list_horizons_offs, Cij_clone, twtij_clone

def horijoin(horizons,Cij_clone,twtij_clone,section,Lg=False,Tj=False):
    """
    Function that takes the output of the function horipick and joins close horizons together following some conditions.
    1. Same polarity
    2. Spatial proximity
    3. Temporal proximity
    4. A new jonction can't cross an existing horizon
    See < Automated reflection picking and polarity assessment through attribute analysis: Theory application to synthetic and real ground-penetrating radar data > for more details.

    INPUT:
    - horizons: List of lists -> The output of horipick
    - Cij_clone: The original Cij matrix
    - twtij: The original twtij matrix
    - section: The portion of the original matrix in wich the horizons are picked
    - Lg: The maximum horizontal (trace) spacing between 2 horizons that can be joined
    - Tj: The maximum vertical (sample) spacing between 2 horizons that can be joined

    OUTPUT:
    - new_horizons: List of lists -> Same output as horipick, but some horizons are joined together. There should be less horizons.
    - new_horizons_t: Same as new_horizons, but the samples are replaced by time.
    - signs_list: List of lists -> Gives the polarity of each horizons
    """
    # Sorts horizons by their trace
    horizons_tri = sorted(horizons, key=lambda x: x[0][1])

    # horizons_tri lists every position of the original data matrix that is now part of an horizon. Here, we retrieve the values of Cij and twtij that are associated with these positions.
    # cos_hori -> Values of Cij associated to the positions in horizons_tri
    # time_hori -> Values of twtij associated to the positions in horizons_tri
    cos_hori = []
    time_hori = []
    for hori in horizons_tri:
        row, cols = zip(*hori)
        cos = Cij_clone[row,cols].tolist()
        time = twtij_clone[row,cols].tolist()
        cos_hori.append(cos)
        time_hori.append(list(zip(time,cols)))
        # Sorts time_hori to be sure everything is in the right order
        time_hori = sorted(time_hori, key=lambda x: x[0][1])
    # Copy the original cos_hori and time_hori for later use
    cos_hori_clone = np.copy(cos_hori)
    time_hori_clone = np.copy(time_hori)

    # Check if every horizons contain elements of same sign
    for hori in cos_hori:
        if np.all(np.asarray(hori) > 0):
            continue
        elif np.all(np.asarray(hori) < 0):
            continue
        else:
            ind = cos_hori.index(hori)
            print("Horizon {} contains elements of opposed signs".format(ind))
            break
    
    # Output lists initialization
    new_horizons = []
    new_horizons_t = []
    signs_list = []
    # Initialization of a list containing every elements or horizons that can't be joined to an existing horizon
    blacklist = []

    # For each horizon
    for horizon in tqdm(range(len(horizons_tri))):

        # Check if the current horizon is blacklisted. If so, pass to the next.
        if (horizon in blacklist):
            continue

        # Check criteria 1 - Same polarity
        # Initialization of a list containing only bool values to check criteria 1
        cos_bool = np.copy(cos_hori)
        # Finds the current horizon sign
        signe = np.sign(cos_hori[horizon][0])
        if signe > 0:
            for j in range(len(cos_hori)):
                # Marks with "True" every horizon of same polarity, containing no 2 (impossible value) amplitude and that are not blacklisted
                if (np.all(np.asarray(cos_hori[j]) > 0)) and (np.all(np.asarray(cos_hori[j]) <= 1)) and (j not in blacklist):
                    cos_bool[j] = True
                else:
                    cos_bool[j] = False
        if signe < 0:
            # Same for horizons of negative polarity
            for k in range(len(cos_hori)):
                if (np.all(np.asarray(cos_hori[k]) < 0)) and (k not in blacklist):
                    cos_bool[k] = True
                else:
                    cos_bool[k] = False

        # Checks criteria 2 (Temporal/vertical proximity)
        # Gets the first and last traces of the current horizon
        start = time_hori[horizon][0][1]
        end = time_hori[horizon][-1][1]
        # Gets the samples of the first and last elements of the current horizon
        tstart = time_hori[horizon][0][0]
        tend = time_hori[horizon][-1][0]
        # Initialization of a list containing only bool values to check criteria 2
        time_bool = np.copy(time_hori)
        for i in range(len(time_hori)):
            # Marks "True" every horizons that ends horizontally before the current horizon (to the left), that are close vertically (Tj) and that are not blacklisted
            if (time_hori[i][-1][1]<start) and (tstart-(Tj)<=time_hori[i][-1][0]<=tstart+(Tj)) and (i not in blacklist):
                time_bool[i] = True
            # Marks "True" every horizons that starts horizontally after the current horizon (to the right), that are close vertically (Tj) and that are not blacklisted
            elif (time_hori[i][0][1]>end) and (tend-(Tj)<=time_hori[i][0][0]<=tend+(Tj)) and (i not in blacklist):
                time_bool[i] = True
            else:
                time_bool[i] = False
        
        # Checks criteria 3 (spatial proximity)
        # Initialization of a list containing only bool values to check criteria 3
        separ_bool = np.copy(time_hori)
        for i in range(len(time_hori)):
            # Marks "True" every horizon that ends horizontally before the current horizon (to the left), that are close horizontally (Lg) and that are not blacklisted.
            if (time_hori[i][-1][1]<start) and (time_hori[i][-1][1]>=start-Lg) and (i not in blacklist):
                separ_bool[i] = True
            # Marks "True" every horizon that starts horizontally after the current horizon (to the right), that are close horizontally (Lg) and that are not blacklisted.
            elif (time_hori[i][0][1]>end) and (time_hori[i][0][1]<=end+Lg) and (i not in blacklist):
                separ_bool[i] = True
            else:
                separ_bool[i] = False
        
        # Combines criterias 1, 2, 3
        crit_prol = cos_bool & time_bool & separ_bool

        # Initialization of lists that combines 1 or more horizons
        hori_combine = horizons_tri[horizon]
        hori_combine_t = time_hori[horizon]
        # Initialization of lists containing possible jonctions
        list_left = []
        list_right = []

        # Loop to extend the current horizon, now placed in hori_combine
        # The loop is enterred only if crit_prol has a True value. If not, we pass to the next horizon and no jonctions are made.
        while np.any(crit_prol):
            # Finds index of potential horizons that can be joined to the current one
            ind_prol = np.where(cos_bool & time_bool & separ_bool)[0]
            if len(ind_prol) == 0:
                continue
            # List of the potential horizons to the left of the current one
            pre_left = ind_prol[ind_prol < horizon]
            # List of the potential horizons to the right of the current one
            pre_right = ind_prol[ind_prol > horizon]

            # Checks criteria 4 - Crossing existing horizons
            # Left
            # Initialization of the final list to the left of the current horizon
            left = []
            if len(pre_left) > 0:
                #if section is None:
                #    for i in range(len(pre_left)):
                #        # Checks if the potential jonction between an horizon to the left and the current horizon crosses an existing horizon - See cross_horizon (next function)
                #        x_bool = cross_horijoin(pre_left[i],hori_combine_t,horizons_tri,time_hori,new_horizons_t,direction="left")
                #        # Adds the horizons to the "left" list if no crossing
                #        if not np.any(x_bool):
                #                left.append(pre_left[i])
                #else:
                for i in range(len(pre_left)):
                    # Checks if the potential jonction between an horizon to the left and the current horizon crosses an existing horizon - See cross_horizon (next function)
                    x_bool = cross_horijoin(pre_left[i],hori_combine_t,horizons_tri,time_hori,new_horizons_t,direction="left")

                    # Adds the horizons to the "left" list if no crossing - Remove if using Phase crossing
                    if not np.any(x_bool):
                        left.append(pre_left[i])
                    
                    """"
                    # Phase crossing - To verify
                    ec_trace = hori_combine[0][1]-horizons_tri[pre_left[i]][-1][1]
                    ec_samp = hori_combine[0][0]-horizons_tri[pre_left[i]][-1][0]
                    pythg = np.sqrt(ec_trace**2 + ec_samp**2)
                    
                    # Verify this line    
                    if (pythg > 52) and (ec_samp > 3):
                        prop = cross_phase(signe,pre_left[i],hori_combine_t,time_hori,champ,direction="left")
                        if (not np.any(x_bool)) and (prop < 0.25):
                            left.append(pre_left[i])
                    # Verify this
                    elif (pythg > 152):
                        if (not np.any(x_bool)) and (prop < 0.75):
                            left.append(pre_left[i])
                    else:
                        if not np.any(x_bool):
                            left.append(pre_left[i])
                    """
            # Right
            # Initialization of the final list to the right of the current horizon
            right = []
            if len(pre_right) > 0:
                #if champ is None:
                #    for j in range(len(pre_right)):
                #        x_bool = cross_horijoin(pre_right[j],hori_combine_t,horizons_tri,time_hori,newlist_horizons_time,direction="right")
                #        if not np.any(x_bool):
                #                right.append(pre_right[j])
                #else:
                for j in range(len(pre_right)):
                    x_bool = cross_horijoin(pre_right[j],hori_combine_t,horizons_tri,time_hori,new_horizons_t,direction="right")

                    # Adds the horizons to the "right" list if no crossing - Remove if using Phase crossing
                    if not np.any(x_bool):
                        left.append(pre_right[i])

                    """
                    Phase crossing - To verify
                    ec_trace = horizons_tri[pre_right[j]][0][1] - hori_combine[-1][1]
                    ec_samp = time_hori[pre_right[j]][0][0] - hori_combine_t[-1][0]
                    pythg = np.sqrt(ec_trace**2 + ec_samp**2)

                    if (pythg > 52) and (ec_samp > 3):
                        prop = cross_phase(signe,pre_right[j],hori_combine_t,time_hori,champ,direction="right")
                        if (not np.any(x_bool)) and (prop < 0.25):
                            right.append(pre_right[j])

                        elif (pythg > 152):
                            prop = cross_phase(signe,pre_right[j],hori_combine_t,time_hori,champ,direction="right")
                            if (not np.any(x_bool)) and (prop < 0.75):
                                right.append(pre_right[j])

                        else:
                            if not np.any(x_bool):
                                right.append(pre_right[j])
                    """
            # For each side, only keep the closest horizon (2 axes)
            # Left
            if len(left) > 1:
                # Gets the last trace of every horizon in left
                traces = np.asarray([horizons_tri[i][-1][1] for i in left])
                # Gets the last sample of every horizon in left
                times = np.asarray([time_hori[j][-1][0] for j in left])
                # Gets the first trace of hori_combine (the current horizon we try to extend)
                refx = hori_combine[0][1]
                # Gets the first sample of hori_combine
                reft = hori_combine_t[0][0]
                # Gets the horizontal spacing between hori_combine and every horizon in left
                difx = refx-traces
                # Gets the vertical spacing between hori_combine and every horizon in left
                dift = reft-times
                # Gets the resulting spacing (Pythagore)
                pyt = np.sqrt(difx**2 + dift**2)
                # Picks the closest horizon
                prox = np.where(pyt==np.min(pyt))[0]
                left = [left[prox[0]]]

            # Right
            if len(right) > 1:
                # Gets the first trace of every horizon in right
                traces = np.asarray([horizons_tri[i][0][1] for i in right])
                # Gets the first sample of every horizon in right
                times = np.asarray([time_hori[j][0][0] for j in right])
                # Gets the last trace of hori_combine (the current horizon we try to extend)
                refx = hori_combine[-1][1]
                # Gets the last sample of hori_combine
                reft = hori_combine_t[-1][0]
                # Gets the horizontal spacing between hori_combine and every horizon in right
                difx = traces-refx
                # Gets the vertical spacing between hori_combine and every horizon in right
                dift = times-reft
                # Gets the resulting spacing (Pythagore)
                pyt = np.sqrt(difx**2 + dift**2)
                # Picks the closest horizon
                prox = np.where(pyt==np.min(pyt))[0]
                right = [right[prox[0]]]

            # Extending hori_combine
            # Extending to the left AND to the right
            if (len(left) == 1) and (len(right) == 1):
                hori_combine = horizons_tri[left[0]] + hori_combine + horizons_tri[right[0]]
                hori_combine_t = time_hori[left[0]] + hori_combine_t + time_hori[right[0]]
                # Keeping track of left and right extension for later use
                list_left.append(left[0])
                list_right.append(right[0])
            # Extending to the left only
            elif (len(left) == 1) and (len(right) == 0):
                hori_combine = horizons_tri[left[0]] + hori_combine
                hori_combine_t = time_hori[left[0]] + hori_combine_t
                # Keeping track of left extension for later use
                list_left.append(left[0])
            # Extending to the right only
            elif (len(left) == 0) and (len(right) == 1):
                hori_combine = hori_combine + horizons_tri[right[0]]
                hori_combine_t = hori_combine_t + time_hori[right[0]]
                # Keeping track of right extension for later use
                list_right.append(right[0])
            else:
                break

            # Sorts hori_combine by trace
            hori_combine = sorted(hori_combine, key=lambda x: x[1])
            hori_combine_t = sorted(hori_combine_t, key=lambda x: x[1])

            # At this point, we check at the extremities of the currently extended horizon if we can extend it again. We therefore check the criteria 1,2,3 again. Criteria 4 will be checked for at the beginning of the next iteration of the while loop.

            # Checks criteria 1,2,3
            # Criteria 1 (Same polarity)
            # Initialization of a list containing bool values to check criteria 1
            cos_bool = np.copy(cos_hori)
            if signe > 0:
                for j in range(len(cos_hori)):
                    # Marks with "True" every horizon of same polarity, containing no 2 (impossible value) amplitude and that are not blacklisted
                    if (np.all(np.asarray(cos_hori[j]) > 0)) and (np.all(np.asarray(cos_hori[j]) <= 1)) and (j not in blacklist):
                        cos_bool[j] = True
                    else:
                        cos_bool[j] = False
            if signe < 0:
                for k in range(len(cos_hori)):
                    if (np.all(np.asarray(cos_hori[k]) < 0)) and (k not in blacklist):
                        cos_bool[k] = True
                    else:
                        cos_bool[k] = False

            # Criteria 2 (Temporal proximity)
            # Gets the first and last traces of the current horizon
            start = hori_combine[0][1]
            end = hori_combine[-1][1]
            # Gets the samples of the first and last elements of the current horizon
            # If left is not empty, we can get tstart from it
            if (len(left)==1) and (len(right)==0):
                tstart = time_hori[left[0]][0][0]
            # If right is not empty, we can get tend from it
            elif (len(right)==1) and (len(left)==0):
                tend = time_hori[right[0]][-1][0]
            # If both are not empty,...
            elif (len(right)==1) and (len(left)==1):
                tstart = time_hori[left[0]][0][0]
                tend = time_hori[right[0]][-1][0] 
            # Initialization of a list containing only bool values to check criteria 2
            time_bool = np.copy(time_hori)
            for i in range(len(time_hori)):
                # Marks "True" every horizons that ends horizontally before the current horizon (to the left), that are close vertically (Tj) and that are not blacklisted
                if (time_hori[i][-1][1]<start) and (tstart-(Tj)<=time_hori[i][-1][0]<=tstart+(Tj)) and (i not in blacklist):
                    time_bool[i] = True
                # Marks "True" every horizons that starts horizontally after the current horizon (to the right), that are close vertically (Tj) and that are not blacklisted
                elif (time_hori[i][0][1]>end) and (tend-(Tj)<=time_hori[i][0][0]<=tend+(Tj)) and (i not in blacklist):
                    time_bool[i] = True
                else:
                    time_bool[i] = False

            # Criteria 3 (Spatial proximity)
            # Initialization of a list containing only bool values to check criteria 3
            separ_bool = np.copy(time_hori)
            for i in range(len(time_hori)):
                # Marks "True" every horizon that ends horizontally before the current horizon (to the left), that are close horizontally (Lg) and that are not blacklisted.
                if (time_hori[i][-1][1]<start) and (time_hori[i][-1][1]>=start-Lg) and (i not in blacklist):
                    separ_bool[i] = True
                # Marks "True" every horizon that starts horizontally after the current horizon (to the right), that are close horizontally (Lg) and that are not blacklisted.
                elif (time_hori[i][0][1]>end) and (time_hori[i][0][1]<=end+Lg) and (i not in blacklist):
                    separ_bool[i] = True
                else:
                    separ_bool[i] = False

            # Combines criterias 1,2,3
            crit_prol = cos_bool & time_bool & separ_bool
            # If there are "True" in crit_prol, the while loop will iterate another time

        # At this point, we are not in the while loop anymore
        # Updates the output lists of the function
        signs_list.append(signe)
        new_horizons.append(hori_combine)
        new_horizons_t.append(hori_combine_t)

        # Blacklists horizons that are in hori_combine (to avoid examining them a second time)
        # Deactivate the corresponding positions in cos_hori and time_hori
        # If the extension is on the left only
        if (len(list_left) > 0) and (len(list_right) == 0):
            for m in list_left:
                # Blacklists the extensions
                blacklist.append(m)
            # Blacklists the original extension
            blacklist.append(horizon)
        # If the extension is on the right only
        elif (len(list_right) > 0) and (len(list_left) == 0):
            for n in list_right:
                # Blacklists the extensions
                blacklist.append(n)
            # Blacklists the original horizon
            blacklist.append(horizon)
        # Extensions on both sides
        elif (len(list_left) > 0) and (len(list_right) > 0):
            # Blacklists extensions to the left
            for m in list_left:
                blacklist.append(m)
            # Blacklists extensions to the right
            for n in list_right:
                blacklist.append(n)
            # Blacklists original horizon
            blacklist.append(horizon)
        # No extension
        elif (len(list_left) == 0) and (len(list_right) == 0):
            # Blacklists original horizon
            blacklist.append(horizon)
    
    return new_horizons, new_horizons_t, signs_list

def cross_horijoin(horizons):
    pass
