'''
Fichier contenant plusieurs fonctions utiles au traitement de données GPR.
Les fonctions enlever_trace_moy, deWOW, smoothing, Gain_agc et Gain_tpow
sont tirées du répertoire Github GPRpy.
https://github.com/NSGeophysics/GPRPy

La fontion boolean_index est tirée du site suivant:
https://towardsdatascience.com/speeding-up-python-code-fast-filtering-and-slow-loops-8e11a09a9c2f

Les fonctions filtre_eps, conv_col_norm, frst_brk_Hcentré, surf_moy_neige, 
surf_moy_neige_mob, conversion_Las, moy_rect et flat_LIDAR sont des fonctions
originales.

Samuel Mongeau-Lachance - 2020
'''
import numpy as np
from tqdm import tqdm
import numpy.matlib as matlib
import laspy
from pyproj import Proj, transform
import scipy.signal as sgl
from scipy import interpolate
import math
#import numba

def enlever_trace_moy(data,ntraces):
    
    '''
    Subtracts from each trace the average trace over
    a moving average window.
    Can be used to remove horizontal arrivals, 
    such as the airwave.
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
        # Lignes ajoutée à vérifier
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
        # Lignes ajoutée à vérifier
        avgtr=avgtr.reshape((len(avgtr),1))
        avgtr=np.tile(avgtr,(1,(tottraces)-(tottraces-halfwid))) 
        newdata[:,tottraces-halfwid:tottraces+1] = data[:,tottraces-halfwid:tottraces+1]-avgtr

    print('done with removing mean trace')
    return newdata
    
def deWOW(data,window):
    '''
    Subtracts from each sample along each trace an 
    along-time moving average.
    Can be used as a low-cut filter.
    INPUT:
    data       data matrix whose columns contain the traces 
    window     length of moving average window 
               [in "number of samples"]
    OUTPUT:
    newdata    data matrix after dewow
    '''
    data=np.asmatrix(data) # J'ai ajouté cette ligne
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

def smoothing(data,window):
    '''
    Replaces each sample along each trace with an 
    along-time moving average.
    Can be used as high-cut filter.
    INPUT:
    data      data matrix whose columns contain the traces 
    window    length of moving average window
              [in "number of samples"]
    OUTPUT:
    newdata   data matrix after applying smoothing
    '''
    totsamps = data.shape[0]
    # If the window is larger or equal to the number of samples,
    # then we can do a much faster dewow
    if (window >= totsamps):
        newdata = np.matrix.mean(data,0)
    elif window == 1:
        newdata = data
    elif window == 0:
        newdata = data
    else:
        newdata = np.asmatrix(np.zeros(data.shape))
        halfwid = int(np.ceil(window/2.0))
        
        # For the first few samples, it will always be the same
        newdata[0:halfwid+1,:] = np.matrix.mean(data[0:halfwid+1,:],0)

        # for each sample in the middle
        for smp in tqdm(range(halfwid,totsamps-halfwid+1)):
            winstart = int(smp - halfwid)
            winend = int(smp + halfwid)
            newdata[smp,:] = np.matrix.mean(data[winstart:winend+1,:],0)

        # For the last few samples, it will always be the same
        newdata[totsamps-halfwid:totsamps+1,:] = np.matrix.mean(data[totsamps-halfwid:totsamps+1,:],0)
        
    print('done with smoothing')
    return newdata

def Gain_agc(data,window):
    '''
    Apply automated gain controll (AGC) by normalizing the energy
    of the signal over a given window width in each trace
    INPUT:
    data       data matrix whose columns contain the traces
    window     window width [in "number of samples"]
    
    OUTPUT:
    newdata    data matrix after AGC gain
    '''
    
    eps=1e-8
    totsamps = data.shape[0]
    # If window is a ridiculous value
    if (window>totsamps):
        # np.maximum is exactly the right thing (not np.amax or np.max)
        energy = np.maximum(np.linalg.norm(data,axis=0),eps)
        # np.divide automatically divides each row of "data"
        # by the elements in "energy"
        newdata = np.divide(data,energy)
    else:
        # Need to go through the samples
        newdata = np.asmatrix(np.zeros(data.shape))
        halfwid = int(np.ceil(window/2.0))
        #halfwid = float(np.ceil(window/2.0))
        # For the first few samples, it will always be the same
        energy = np.maximum(np.linalg.norm(data[0:halfwid+1,:],axis=0),eps)
        newdata[0:halfwid+1,:] = np.divide(data[0:halfwid+1,:],energy)
        
        for smp in tqdm(range(halfwid,totsamps-halfwid+1)):
            winstart = int(smp - halfwid)
            winend = int(smp + halfwid)
            energy = np.maximum(np.linalg.norm(data[winstart:winend+1,:],axis=0),eps)
            newdata[smp,:] = np.divide(data[smp,:],energy)

        # For the first few samples, it will always be the same
        energy = np.maximum(np.linalg.norm(data[totsamps-halfwid:totsamps+1,:],axis=0),eps)
        newdata[totsamps-halfwid:totsamps+1,:] = np.divide(data[totsamps-halfwid:totsamps+1,:],energy)          
    return newdata

def Gain_tpow(data,twtt,power):
    '''
    Apply a t-power gain to each trace with the given exponent.
    INPUT:
    data      data matrix whose columns contain the traces
    twtt      two-way travel time values for the rows in data
    power     exponent
    OUTPUT:
    newdata   data matrix after t-power gain
    '''
    factor = np.reshape(twtt**(float(power)),(len(twtt),1))
    factmat = matlib.repmat(factor,1,data.shape[1])  
    return np.multiply(data,factmat)

def running_avg(data,fen):
    data = np.asarray(data)
    tottraces = data.shape[1]

    newdata = np.zeros(data.shape)
    halfwid = int(np.ceil(fen/2))

    # Premières traces
    Col_1D = np.mean(data[:,0:halfwid+1],axis=1)
    Col_2D = np.reshape(Col_1D,(Col_1D.shape[0],1))
    newdata[:,0:halfwid+1] = Col_2D

    # Traces au milieu
    for trace in range(halfwid,tottraces-halfwid+1):
        dep_fen = int(trace-halfwid)
        fin_fen = int(trace+halfwid)
        newdata[:,trace] = np.mean(data[:,dep_fen:fin_fen+1],axis=1)

    # Dernières traces
    Colo_1D = np.mean(data[:,tottraces-halfwid:tottraces+1],axis=1)
    Colo_2D = np.reshape(Colo_1D,(Colo_1D.shape[0],1))
    newdata[:,tottraces-halfwid:tottraces+1] = Colo_2D
    
    return newdata

def filtre_eps(data, fenetre):
    '''
    Mise en application du filtre EPS présenté à l'annexe A de l'article
    de Juan I. Sabbione (Automatic first-breaks picking: New strategies
    and algorithms)
    INPUT:
    data -> Matrice contenant toutes les données. Les colonnes doivent correspondre
            aux traces
    fenetre -> Entier pour la largeur de la fenêtre
    OUTPUT:
    Matrice contenant toutes les données filtrées. Les colonnes correspondent aux traces
    '''
    # Nombre de samples par trace
    totsamps = data.shape[0]

    # Initisalisation de la matrice réponse
    newdata = np.zeros(data.shape)
    # Initialisation de la matrice des écarts-types
    ecart = np.zeros(data.shape)
    # Initialisation de la matrice contenant les indices des
    # plus petits écarts-types
    ind_min = np.zeros(data.shape)

    depart = fenetre - 1
    fin = totsamps - fenetre

    # Pour les premières données
    newdata[0:fenetre-2,:] = np.nan
    # Pour les dernières données
    newdata[fin+1:,:] = np.nan

    # Pour les données au milieu
    # On calcule l'écart-type de toutes les fenêtres possibles en partant
    # du tout premier point (fenêtre vers l'avant) et on crée une matrice
    # contenant toutes ces valeurs
    for smp in range(0, fin+1):
        fin_fen = smp + (fenetre-1)
        ecart[smp,:] = np.std(data[smp:fin_fen+1,:], 0)

    for smp in range(depart, fin+1):
        # Pour le premier point à filtrer (indice 7), les fenêtres 
        # à considérer sont les 7 premières de écart
        # On trouve l'indice du plus petit écart-type pour chaque fenêtre
        ind_min[smp,:] = np.argmin(ecart[smp-(fenetre-1):smp+1,:],0) + (smp-(fenetre-1))

    for smp in tqdm(range(depart, fin+1)):
        ind = ind_min[smp,:].astype(int)
        don_moy = np.take_along_axis(data, (ind[:,None]+np.arange(fenetre)).T, axis=0)
        newdata[smp,:] = np.mean(don_moy, 0)

    print("Filtrage EPS terminé")
    return newdata

def conv_col_norm(liste_lignes, norm=True, norm_comp=False):
    """
    Convertir une liste de lignes en liste de traces normalisées.
    INPUT:
    liste_lignes : Liste de listes. Chaque sous-liste correspond
                à une ligne de la matrice que l'on veut modifier.
    norm : Booléen indiquant si on veut normaliser les traces ou
                non
    """
    liste_colonnes = []
    #liste_colonnes_norm = []
    # J'ai changé le range pour qu'il parte à 0
    for column in range(0, len(liste_lignes[0])):
        col = []
        for ligne_Cut in liste_lignes:
            #col.append(int(ligne_Cut[column]))   # int??
            col.append(float(ligne_Cut[column]))
        liste_colonnes.append(col)
    if norm:
        liste_colonnes_norm = []
        for colonne in liste_colonnes:
            #print(colonne)
            maxi = max(colonne)
            if maxi == 0:
                #print(liste_colonnes.index(colonne))
                maxi = 1
            col_norm = []
            for elem_norm in colonne:
                col_norm.append(elem_norm/maxi)
            liste_colonnes_norm.append(col_norm)
        return liste_colonnes_norm
    elif norm_comp:
        liste_colonnes = np.array(liste_colonnes)
        maxi = np.amax(liste_colonnes)
        if maxi == 0:
            maxi = 1
        liste_colonnes = liste_colonnes/maxi
        liste_colonnes_norm = liste_colonnes.tolist()
        return liste_colonnes_norm
    else:
        return liste_colonnes

def frst_brk_Hcentré(data):
    '''
    Sélectionner le first break en utilisant la dérivée la plus élevée
    pour chaque trace d'une matrice contenant des données d'entropie
    filtrée. Les dérivées sont calculées avec un opérateur de différences
    finies centré.
    INPUT:
    data : Matrice de données issues du calcul d'entropie (Juan I. Sabbione)
        et du filtrage EPS. Chaque colonne correspond à une trace.
    OUTPUT :
    newdata : Matrice des dérivées
    ind_max : Matrice 1 dimension listant les indices où la dérivée maximale
        est observée.
    '''
    data = np.nan_to_num(data)
    totsamps = data.shape[0]
    newdata = np.zeros(data.shape)
    newdata[:1,:] = 0
    newdata[-1:,:] = 0
    for smp in range(1, totsamps-1):
        newdata[smp,:] = (data[smp+1,:] - data[smp-1,:])/2
    ind_max = np.argmax(newdata, axis=0)
    return newdata, ind_max

def surf_moy_neige(surface, fenetre):
    '''
    Calculer l'élévation moyenne de la surface de la neige par fenêtres.
    INPUT:
    surface : Matrice 1 dimension des élévations de la surface de la neige
    fenetre : Nombre de données sur lesquelles calculer une moyenne
    OUTPUT : 
    moy : Matrice contenant la moyenne des différentes fenêtre
    moy_graph : Même matrice que moy, mais avec des dimensions facilitant 
        la représentation graphique
    '''
    ind_neige = (np.linspace(0,((len(surface)//fenetre)*fenetre)-fenetre,(len(surface)//fenetre))).astype(int)
    surface = np.asarray(surface).reshape(1, len(surface))
    windows = ind_neige[:, None] + np.arange(fenetre)
    elem = np.take_along_axis(surface, windows, axis=1)

    # Moyenne de la dernière fenêtre (plus courte)
    moy_last = np.mean(surface[:,(windows[-1][-1])+1:])

    # Moyenne des autres fenêtres
    moy = np.mean(elem, axis=1)

    # Formation de la matrice complète
    moy = np.append(moy, moy_last)
    #moy_graph = moy.reshape((len(moy),))

    return moy#, moy_graph

def surf_moy_neige_mob(surface, fenetre):
    '''
    Calculer l'élévation moyenne de la surface de la neige par fenêtres mobiles.
    Au lieu de calculer les moyennes des 600 premières données, puis des 600
    suivantes, on calcule la moyenne du 300e élément avec une fenêtre de 600
    éléments, puis on calcule de la même façon la moyenne du 301e élément.
    INPUT:
    surface : Matrice 1 dimension des élévations de la surface de la neige
    fenetre : Nombre de données sur lesquelles calculer une moyenne
    OUTPUT : 
    moy : Matrice contenant la moyenne des différentes fenêtre
    '''
    if fenetre % 2 == 0:
        raise "Entrez une largeur de fenêtre impaire"

    data = np.array(surface)
    surf_moy = np.zeros(len(data))
    half_fen = fenetre // 2

    # Premières valeurs
    surf_moy[0:half_fen] = data[0:half_fen]

    # Valeurs au milieu
    for smp in range(half_fen,len(data)-half_fen+1):
        dep_fen = int(smp-half_fen)
        fin_fen = int(smp+half_fen+1)
        moy = np.mean(data[dep_fen:fin_fen])
        surf_moy[smp] = moy

    # Dernières valeurs
    surf_moy[-half_fen:] = data[-half_fen:]

    return surf_moy

def conversion_Las(fich, epsg_ini='epsg:2145', epsg_fin='epsg:32618'):
    '''
    Lecture d'un fichier .las et conversion d'un système de référence 
    vers un autre. Par défaut, on passe de EPSG:2145 (projection MTM 
    (zone 8) avec référence géodésique NAD83) à EPSG:32618 (projection 
    UTM (zone 18N) avec référence géodésique WGS84).
    Référence pour les codes EPSG :
    https://spatialreference.org/ref/epsg/?search=utm+18N&srtext=Search

    INPUTS
    - fich: Adresse du fichier .las à convertir
    - epsg_ini: Projection initiale (String)
    - epsg_fin: Projection désirée (String)
    OUTPUTS
    - Numpy array dont chaque ligne est une coordonnée (east, north, elev.)
    '''

    inFile = laspy.file.File(fich, mode="r")

    x_dim = inFile.X
    y_dim = inFile.Y
    z_dim = inFile.Z
    scale = inFile.header.scale
    offset = inFile.header.offset

    pts_x = x_dim*scale[0] + offset[0]
    pts_y = y_dim*scale[1] + offset[1]
    pts_z = z_dim*scale[2] + offset[2]

    inProj = Proj(epsg_ini)
    outProj = Proj(epsg_fin, preserve_units=True)
    ptsx_2, ptsy_2, ptsz_2 = transform(inProj, outProj, pts_x, pts_y, pts_z)
    coordLas = np.hstack((np.array([ptsx_2]).T, np.array([ptsy_2]).T, np.array([ptsz_2]).T))
    return coordLas

def boolean_index(np_array, minE, minN, maxE, maxN):
    '''
    Filtrage rapide d'un grand array numpy pour trouver tous les points du
    fichier .las qui sont inclus dans le plus petit rectangle décrit par les
    coordonnées GPS (east, north) d'un fichier .dst
    Source -> https://towardsdatascience.com/speeding-up-python-code-fast-filtering-and-slow-loops-8e11a09a9c2f

    INPUTS
    - np_array: Array numpy à filtrer (Coordonnées du fichier .las)
    - minE: Plus petite valeur de easting du fichier .dst
    - minN: Plus petite valeur de northing du fichier .dst
    - maxE: Plus grande valeur de easting du fichier .dst
    - maxN: Plus grande valeur de northing du fichier .dst
    OUTPUT:
    - Array numpy contenant tous les points LIDAR inclus dans un rectangle 
        délimité par minE, minN, maxE et maxN.
    '''
    index = (np_array[:,0]>=minE) & (np_array[:,1]>=minN) & (np_array[:,0]<=maxE) & (np_array[:,1]<=maxN)
    return np_array[index]

def moy_rect(gps, las, larg, isNorth, rectN=2):
    '''
    Remplacer les valeurs d'élévation de chaque coordonnées GPS (fichier .dst) par
    la moyenne des élévations des coordonnées LIDAR dans un rectangle vertical de
    quelques mètres de largeur.

    INPUTS:
    - gps: Array numpy contenant les coordonnées GPS du fichier .dst
    - las: Array numpy contenant les coordonnées GPS du fichier .las. Généralement
        les coordonnées traitées avec la fonction boolean index.
    - larg: Largeur, en mètres, du rectangle vertical. On suggère 3m. (int)
    OUTPUT:
    - Array numpy contenant toutes les coordonnées GPS du fichier .dst, mais avec de
        nouvelles valeurs d'élévation.
    '''
    las = np.asarray(las)
    gps = np.asarray(gps)
    coords = np.vstack((las[:,0], las[:,1], las[:,2]))
    # Trouver un polynôme à partir des coordonnées GPS
    #poly_gps = interpolate.interp1d(gps[:,0],gps[:,1],kind='slinear')
    # Interpoler une valeur gps en x pour chaque point las
    #interp_las = poly_gps(np.sort(las[:,0]))
    # Trier les coordonnées las selon x
    #coords_ver = coords[coords[:,0].argsort()]
    # Trouver l'élévation moyennes des coordonnées en haut de la ligne
    #moy_haut = las[:,2][coords_ver[2] > interp_las].mean()
    #moy_bas = las[:,2][coords_ver[2] <= interp_las].mean()
    
    liste = []
    for pos in gps:
        if isNorth:
            ind = np.where((coords[0]>=pos[0]-rectN/2) & (coords[0]<=pos[0]+rectN/2) & (coords[1]>=pos[1]-rectN*10) & (coords[1]<=pos[1]+rectN*10))
            liste.append(ind[0].tolist())
        else:
            ind = np.where(((coords[0] >= pos[0]-(larg/2)) & (coords[0] <= pos[0]+(larg/2))))
            liste.append(ind[0].tolist())
    for i in range(len(liste)):
        gps[i,2] = np.mean(las[liste[i], 2])

    return gps

def flat_LIDAR(data, shift, flat=True, avgs=False):
    '''
    "Supprimer" la pente dans les coordonnées issues des fichiers LIDAR. En
    général, il y a un dénivelé de quelques mètres sur la largeur des champs
    de canneberges.

    INPUTS:
    - data: Array numpy contenant les coordonnées GPS du fichier .dst, mais
        avec les élévations corrigées à partir des fichiers LIDAR. On utilise
        donc cette fonction après avoir utiliser moy_rect.
    - shift: Élévation de la première digue (int).
    - avgs: Booléen. Lorsqu'on le met à True, retourne l'élévation moyenne de
        chaque champs (régions entre les digues)
    OUTPUT:
    - data_c: Array numpy contenant toutes les coordonnées GPS du fichier .dst,
        mais dont les élévations ont été corrigées relativement à l'élévation de
        la région située avant la première digue.
    - moyennes: Élévation moyenne de chaque champs (régions entre les digues)
    '''
    elev_max = np.amax(data, 0)
    elev_min = np.amin(data, 0)
    data_b = np.copy(data)

    # Prendre les valeurs qui dépassent 60% du range entre le min et le max
    # Il doit y avoir aussi un minimum de 2000 points entre les pics
    seuil = (((elev_max[2]-elev_min[2])/5)*3) + elev_min[2]
    pics, _ = sgl.find_peaks(data[:,2], height=seuil, distance=2000)

    # Conversion des élévations en nano secondes
    # On estime la vitesse du signal dans la neige à 0.10 m/ns. 
    data_b[:,2] = ((elev_max[2] - data_b[:,2])/0.1)
    dec = shift - data_b[pics[0],2]
    data_b[:,2] += dec

    # Localiser les digues
    # Moyenne de l'élévation avant le premier pic
    champmoy1 = np.mean(data_b[:pics[0],2])
    # Calculer les moyennes des régions entre les pics
    moyennes = np.zeros(len(pics)+1)
    moyennes[0] = champmoy1
    for i in range(len(pics)-1):
        moyennes[i+1] = np.mean(data_b[pics[i]:pics[i+1],2])
    moyennes[-1] = np.mean(data_b[pics[-1]:,2])
    
    data_c = np.copy(data_b)
    if flat == True:
        for j in range(len(pics)-1):
            data_c[pics[j]:pics[j+1],2] = data_c[pics[j]:pics[j+1],2] + (champmoy1-moyennes[j+1])
        data_c[pics[-1]:,2] = data_c[pics[-1]:,2] + (champmoy1-moyennes[-1])

    if avgs == True:
        return data_c, moyennes
    elif avgs == False:
        return data_c

def static_correction(pick_auto_x, pick_auto_z, GPS_corr, digues, data, ns_sample, offset=None):
    '''
    Fonction pour effectuer un shift temporel des données GPR.
    1. Trouver la hauteur (en ns) de la 1re digue et déplacer les données LIDAR
        à cette position
    2. Interpoler pour avoir autant de points que dans le fichier LIDAR
    3. Décaler les données interpolées pour tout mettre au même niveau que les
        données LIDAR
    INPUTS:
    - pick_auto_x: Coordonnées en x des points pick automatiquement (np.array)
    - pick_auto_z: Coordonnées en z des points pick automatiquement (np.array)
    - lidar_ns: Coordonnées en z des points du fichier LIDAR (np.array)
    - x_lidar: Coordonnées en x des points du fichier LIDAR (np.array)
    - digues: Liste des coordonnées des digues (obtenues avec la fonction
        loc_digues)
    OUTPUTS:
    -
    '''
    if offset != None:
        elev_max = np.amax(GPS_corr[:,2], 0)
        copyGPS = np.copy(GPS_corr)
        # On assume ici que l'onde se propage à 0.10 ns/m
        copyGPS[:,2] = ((elev_max - copyGPS[:,2])/0.1)
        x_lidar = np.linspace(pick_auto_x[0],pick_auto_x[-1],num=len(copyGPS))
        newLidar = copyGPS + offset

    else:
        elev_max = "Pas d'elev_max"
        # Conversion des données LIDAR en ns avec positionnement selon 1re digue
        lidar_ns = flat_LIDAR(GPS_corr, 0.0, flat=False)
        x_lidar = np.linspace(pick_auto_x[0], pick_auto_x[-1], num=len(lidar_ns))
        # Décalage temporel des points LIDAR
        # Trouver le point minimum de la 1re digue
        #pos_max1 = np.where(pick_auto_z == np.amin(pick_auto_z[:digues[0][1]]))
        pos_max1 = np.where(pick_auto_z[:digues[0][1]] == np.amin(pick_auto_z[:digues[0][1]]))
        # Trouver la valeur en mètres de ce point
        pos_max_x = pick_auto_x[pos_max1[0][0]]
        pos_max_x = truncate(pos_max_x,2)
        x_v_pos = np.where((x_lidar < (pos_max_x + 0.005)) & (x_lidar > (pos_max_x - 0.005)))
        # Trouver l'écart temporel entre les séries de données pour ce point
        offset = np.amin(pick_auto_z[:digues[0][1]]) - lidar_ns[x_v_pos[0][0]]
        # Ajouter cet écart aux données LIDAR
        newLidar = lidar_ns + offset

    # Interpolation
    f = interpolate.interp1d(pick_auto_x,pick_auto_z,kind='slinear')
    new_paz = f(x_lidar)

    # Liste des décalages à appliquer
    #shift_list_ns = new_paz - newLidar[:,2]
    shift_list_samp = (new_paz - newLidar[:,2])/ns_sample
    #stat_corr = new_paz - shift_list_ns

    # Décalage des données GPR
    shift_min = int(np.round(np.amin(shift_list_samp)))
    shift_max = int(np.round(np.amax(shift_list_samp)))
    # La valeur 32768 correspond au zéro des données GPR
    up = 32768*np.ones((abs(shift_min), data.shape[1]))
    down = 32768*np.ones((abs(shift_max), data.shape[1]))
    #newdata = np.vstack((up,32768*np.ones((data.shape[0],data.shape[1])),down))
    newdata = np.vstack((down,32768*np.ones((data.shape[0],data.shape[1])),up))

    for i in range(data.shape[1]):
        newdata[abs(shift_max)-int(np.round(shift_list_samp[i])):int(abs(shift_max))-int(np.round(shift_list_samp[i]))+data.shape[0],i] = data[:,i]

    return new_paz, newdata, offset, shift_min, elev_max

# Fonction pour changer l'orientation des données
#@ numba.jit
def change_side(data_dst, isNorth):
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

# Fonction pour enlever les traces vides
def supp_traces_vide(data_dzt):
    traces_vides = np.where(np.all(data_dzt==data_dzt[0,:],axis=0)==True)
    frst_half = traces_vides[0][traces_vides[0] < (data_dzt.shape[1]/2)]
    scd_half = traces_vides[0][traces_vides[0] > (data_dzt.shape[1]/2)]

    beg = 0
    end = data_dzt.shape[1]

    if frst_half.size != 0:
        beg = np.amax(frst_half)+1
    if scd_half.size != 0:
        end = np.amin(scd_half)

    return (beg,end)

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor