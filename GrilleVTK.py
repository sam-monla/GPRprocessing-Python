"""
Fonction pour produire un fichier VTK.
Script tiré de GPRpy
28 avril 2021
"""

import numpy as np
import pandas as pd 
import Prefiltrage as pflt
import readDZT
import  matplotlib.pyplot as plt
from tabulate import tabulate
import scipy.interpolate as interp
import scipy.signal as signal
from pyevtk.hl import gridToVTK

"""
# Générer des fichiers VTK à l'aide de l'interface de GPRpy

# Importer le fichier las
las = "/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/Blandford_GPR_2017/GPR-Blandford-H2017/GPR-Blandford-H2017/LIDAR/LiDAR_13-4175121F08_LAS_MTM_TEL/13_4175121F08_de.las"
coordLas = pflt.conversion_Las(las)

dst = '/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/Blandford_GPR_2017/GPS_headers/geom-G87-F01.dst'
GPS = pd.read_csv(dst, sep='\t')
# Mise en matrice des coordonnées GPS
GPS = np.hstack((np.array([GPS['Xshot']]).T, np.array([GPS['Yshot']]).T, np.array([GPS['Zshot']]).T))
# Importer des données
DZT_001 = "/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/Blandford_GPR_2017/GPR-Blandford-H2017/GPR-Blandford-H2017/GPR/GRID____087.3DS/FILE____001.DZT"
head001, dat001 = readDZT.lect_DZT(DZT_001, [2])
ntrace_ini = dat001.shape[1]
# Suppression des traces vides
deb, fin = pflt.supp_traces_vide(dat001)
dat = dat001[:,deb:fin]
GPS = GPS[deb:fin]
dat = pflt.enlever_trace_moy(dat,dat.shape[1]/2)
############################################## flip #######################################################
#dat = np.flip(dat,axis=1)
# Sélection du plus petit rectangle dans le champ contenant toutes les traces
# Sélection des valeurs limites
minEast = np.min(GPS[:,0])
maxEast = np.max(GPS[:,0])
minNorth = np.min(GPS[:,1])
maxNorth = np.max(GPS[:,1])
# Sélection des coordonnées du fichier LIDAR comprises dans le domaine à l'étude
las_flt = pflt.boolean_index(coordLas, minEast, minNorth, maxEast, maxNorth)
isNorth = False
GPS_corr = pflt.moy_rect(GPS, las_flt, 3, isNorth)

fig = plt.figure(figsize=(6, 3.2))
ax = fig.add_subplot(111)
ax.set_title("Application de la méthode SVD")
maxi = np.amax(dat)
mini = np.amin(dat)
plt.imshow(dat, cmap='seismic', vmin=mini, vmax=maxi)
ax.set_aspect(8)
plt.xlabel("Traces")
plt.ylabel("Échantillons")

plt.figure()
plt.plot(GPS_corr[:,0],GPS_corr[:,2])
#plt.title("Élévations corrigées")
plt.xlabel("Est-Ouest UTM Zone 18 N (m)")
plt.ylabel("Élévation estimée (m)")
plt.grid()
plt.show()

GPS_df = pd.DataFrame(GPS_corr)
path = "/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/Blandford_GPR_2017/GPR-Blandford-H2017/GPR-Blandford-H2017/GPR/GRID____086.3DS/test87.txt"
GPS_df.to_csv(path_or_buf=path,header=False,index=False,encoding="ascii")

#with open('test3d.txt', 'w') as f:
#    f.write(tabulate(GPS_corr))
"""

def correctTopo(data, velocity, profilePos, topoPos, topoVal, twtt):
    '''
    Corrects for topography along the profile by shifting each 
    Trace up or down depending on provided coordinates.
    INPUT:
    data          data matrix whose columns contain the traces
    velocity      subsurface RMS velocity in m/ns
    profilePos    along-profile coordinates of the traces
    topoPos       along-profile coordinates for provided elevation
                  in meters
    topoVal       elevation values for provided along-profile 
                  coordinates, in meters
    twtt          two-way travel time values for the samples, in ns
    OUTPUT:
    newdata       data matrix with shifted traces, padded with NaN 
    newtwtt       twtt for the shifted / padded data matrix
    maxElev       maximum elevation value
    minElev       minimum elevation value
    '''
    # We assume that the profilePos are the correct along-profile
    # points of the measurements (they can be correted with adj profile)
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
        #print("twtt",twtt)
        #print("shape twtt",twtt.shape)
        #print("maxup",maxup)
        #print("maxup x timestep",maxup*timeStep)
        #newtwtt = np.arange(0, twtt[-1] + maxup*timeStep, timeStep)
        newtwtt = np.linspace(0, twtt[-1] + maxup*timeStep, newdata.shape[0])
        #print("first, last newtwtt",newtwtt[0],newtwtt[-1])
        #print("shape newtwtt",newtwtt.shape)
        nsamples = len(twtt)
        # Enter every trace at the right place into newdata
        for pos in range(0,len(profilePos)):
            newdata[tshift[pos]:tshift[pos]+nsamples ,pos] = np.squeeze(data[:,pos])
        return newdata, newtwtt, np.max(elev), np.min(elev), tshift

def prepVTK(profilePos,gpsmat=None,smooth=True,win_length=51,porder=3):
    """
    Calcul les coordonnées 3D pour chaque trace en interpolant les points 3D
    donnés le long du profil.

    INPUT: 
    - profilePos: Les coordonnées des traces le long du profil (linspace(start_pos,final_pos,N_traces))
    - gpsmat: Matrice nx3 contenant les coordonnées x,y,z des points 3D donnés pour le profil
    - 
    -
    - 
    OUTPUT:
    - x,y,z: Coordonnées 3D pour les traces
    """
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
    """
    Transformer des profils traités en fichiers VTK qui peuvent être importés dans Paraview, Mayavi
    ou autre outils de visualisation pour fichiers VTK.

    Si les infos sont fournies en 3D, (X,Y,Z ou East,North,Elev), le profil sera exporté dans sa forme
    3D.

    INPUT:
    - outfile: Nom qui sera donné au fichier VTK produit
    - gpsinfo: matrice nx3 contenant des coords. XYZ
               OU
               Fichier texte ASCII contenant l'info
    - delimiter: Ce qui délimite les colonnes dans le fichier de topographie.
    - thickness: Si on veut exporter le profil comme une bande 3D d'épaisseur x, entrer
                cette épaisseur en mètres (défaut = 0)
    - aspect: aspect ratio si on veut exagérer l'axe z. (défaut = 1)
    - smooth: (défaut = True)
    - win_length: Pour smoothing (défaut = 51)
    - porder: Ordre pour le smoothing (défaut = 3)
    """
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




