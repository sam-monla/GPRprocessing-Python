'''
Fonctions pour le picking automatique.
'''

import numpy as np
import Prefiltrage as pflt
from scipy import linalg  # Pour utiliser la méthode SVD
from scipy import interpolate

def coord_pick(
    liste_col_norm, seuil, trace_dep, traces, ns_persample,
    longueur_ligne, dec_temp=0
    ):
    """
    INPUTS:
    liste_col_norm = Liste de listes où chacune des sous-listes
            correspond à une trace normalisée.
    seuil = Seuil de sélection (int).
    trace_dep = Entier représentant la trace de départ de la 
            région à l'étude.
    traces = Nombre de traces total de la ligne en entier.
    ns_persample = Période d'échantillonage (ns)
    longueur_ligne = Longueur de la ligne (m)
    dec_temp = Décalage temporel de la zone à l'étude. Par exemple,
            si on ne veut pas traiter les 2 premiers échantillons
            d'une trace, on entre 2.
    ******************************************
    OUTPUTS:
    Tuple dont chaque élément est une liste contenant les coordonnées
    horizontales et verticales des points 'picked'. Ces coordonnées 
    sont converties en (mètres, nanosecondes).
    """
    coord_x = []
    coord_z = []
    coord_x_brut = []
    coord_z_brut = []
    decalage = dec_temp * np.ones(len(liste_col_norm))

    for trace in liste_col_norm:
        for indi, val in enumerate(trace):
            if val > seuil:
                coord_z.append((indi+decalage[liste_col_norm.index(trace)])*ns_persample)
                coord_x.append((longueur_ligne/traces)*(liste_col_norm.index(trace)+trace_dep))
                coord_z_brut.append(indi+decalage[liste_col_norm.index(trace)])
                coord_x_brut.append(liste_col_norm.index(trace)+trace_dep)
                break
            else:
                continue
    return (coord_x, coord_z, coord_x_brut, coord_z_brut)

def coord_pick_V(data, seuil):
    '''
    Même fonction que précédemment, mais à la vitesse grand V (du moins un peu
    plus rapide). Renvoie seulement les coordonnées en numéro de trace.
    '''
    # On transpose la matrice pour que les coordonnées sélectionnées soient
    # placées en ordre croissant par numéro de trace
    coord_brutes = np.argwhere(data.T>seuil)
    # Pour chaque trace, on ne conserve que la 1re valeur supérieure au seuil
    for i in range(coord_brutes.shape[0]-1,-1,-1):
        if coord_brutes[i][0] == coord_brutes[i-1][0]:
            coord_brutes = np.delete(coord_brutes,i,axis=0)
    return coord_brutes

def loc_digues(gps_corr_z, ntraces, long_ligne):
    '''
    Fonction pour localiser les digues. Effectue une dérivée (opérateur de
    différences finies centré) sur les valeurs en z des coordonnées GPS
    corrigées, puis sélectionne les 2 valeurs extrêmes de chacune des digues.
    INPUT:
    - gps_corr_z: Array numpy contenant les valeurs en z des coordonnées GPS
        corrigées
    - ntraces: Nombre de traces dans le fichier DZT.
    - long_ligne: Longueur de la ligne étudiée (Chercher "sec" dans le header
        du fichier DZT).
    OUTPUTS:
    - newdata: Résultats des dérivées premières sur les données entrantes. 
        On utilise un opérateur de différences finies centrée
    - 
    '''
    # Création de newdata -> Liste des premières dérivées
    tottraces = gps_corr_z.shape[0]
    newdata = np.zeros((tottraces,1))
    newdata[0] = 0.
    newdata[-1] = 0.
    for trace in range(1, tottraces-1):
        newdata[trace] = ((gps_corr_z[trace+1] - gps_corr_z[trace-1])/(2*(long_ligne/ntraces)))
        # Filtrer les plus petits pics
        if abs(newdata[trace]) < 0.8:
            newdata[trace] = 0.
    newdat = np.copy(newdata)
    # Repérage des digues
    digues = []
    # Une digue a environ une largeur de 600 traces. Ici, on met 800 par précaution.
    # Les valeurs des dérivées premières sont centrées autour de 0.
    for elem in range(0, len(newdata)-600):
        # Les dérivées inférieures à 0 correspondent aux "montants" des digues
        if newdata[elem] >= 0.8:
            # Une fois le premier montant identifié, on cherche la dernière valeur
            # positive (dernier descendant) dans la fenêtre de 800 traces
            if True in (newdata[elem:elem+600] < 0):
                digues.append((elem, np.max(np.where(newdata[elem:elem+600] < 0)) + elem))
                # On ajoute ensuite 100 à toutes les valeurs de la fenêtre. De cette façon, 
                # on s'assure de ne garder que la première valeur
                newdata[elem:elem+600] = newdata[elem:elem+600] + 100
    return newdat, digues#, newdata

def loc_digues2021(gps_corr_z,fen_diviseur,larg_dig,seuil=0.2):
    """
    Fonction alternative pour repérer les digues. Trace une moyenne mobile avec une 
    fenêtre courte puis repère les digues à partir de la différence entre les GPS_corr
    et la moyenne mobile.

    INPUT:
    - gps_corr_z: Données d'élévations GPS corrigées (GPS_corr[:,2]).
    - fen_diviseur: Entier servant à définir la longueur de la fenêtre. Par exemple, en 
      entrant 60, la fenêtre équivaut à len(gps_corr_z)/60. Il est conseillé d'utiliser une
      fenêtre plus courte que la largeur d'une digue (~larg_dig/3).
    - larg_dig: Nombre de traces équivalent à la largeur d'une digue.
    - seuil: Repérer les endroits où (moy_mobile - gps_corr_z) est plus grand qu'une certaine 
        valeur (seuil). Ces endroits marquent la présence d'une digue. Valeur par défaut = 0.2.
        Se fier au OUTPUT diff_clone pour changer la valeur par défaut.
    OUTPUT:
    - digues: Liste de tuples représentant les positions des digues (trace_ini,trace_fin).
    - diff_clone: Array numpy permettant de visualiser la différence entre moyenne mobile
        et gps_corr_z. (plt.plot(diff_clone) pour aider à redéfinir la valeur seuil)
    """
    tottraces = len(gps_corr_z)
    fen = int(len(gps_corr_z)/60)
    moy_mobil = np.zeros(gps_corr_z.shape)
    halfwid = int(np.ceil(fen/2))
    moy_mobil[:halfwid+1] = np.mean(gps_corr_z[:halfwid+1])
    moy_mobil[tottraces-halfwid:] = np.mean(gps_corr_z[tottraces-halfwid:])
    for i in range(halfwid,tottraces-halfwid+1):
        moy_mobil[i] = np.mean(gps_corr_z[i-halfwid:i+halfwid])

    diff = moy_mobil-gps_corr_z
    diff = diff.clip(min=0)
    diff_norm = diff/np.max(diff)
    diff_clone = np.copy(diff_norm)

    # Localisation des digues
    digues = []
    for elem in range(0, len(diff_norm)):
        # Pour les digues à la toute fin du radargramme
        if diff_norm[elem] >= seuil and (elem > len(diff_norm)-larg_dig):
            if True in (diff_norm[elem+1:] > seuil):
                digues.append((elem, len(diff_norm)-1))
                diff_norm[elem:] = 0
        # Pour les digues au tout début des radargrammes
        elif diff_norm[elem] >= seuil and (elem < larg_dig):
            if True in (diff_norm[elem+1:] > seuil):
                digues.append((0, np.max(np.where(diff_norm[elem+1:elem+larg_dig] > seuil)) + elem+1))
                diff_norm[:elem+larg_dig] = 0
        # Pour les autres digues
        elif diff_norm[elem] >= seuil:
            if True in (diff_norm[elem+1:elem+larg_dig] > seuil):
                digues.append((elem, np.max(np.where(diff_norm[elem+1:elem+larg_dig] > seuil)) + elem+1))
                diff_norm[elem:elem+larg_dig] = 0
    return digues, diff_clone, moy_mobil

def correct_chemin(data_las, data_GPS, coordLas, minE, minN, maxE, maxN):
    '''
    Permet de déterminer s'il y a une différence majeure entre les données las
    de part et d'autre de la ligne GPS. Si tel est le cas, on estime de nouvelles
    élévations pour les coordonnées GPS en étendant de façon nord-sud le rectangle
    de données LIDAR.
    INPUT:
    - data_las: Array numpy de coordonnées LIDAR (las_flt)
    - data_GPS: Array numpy de coordonnées GPS
    - coordLas: Fichier LIDAR (issu de conversion_las)
    - minE: Easting minimum
    - minN: Northing minimum
    - maxE: Easting maximum
    - maxN: Northing maximum
    OUTPUT:
    '''

    chemin = False

    # Ordonner les coordonnées LIDAR selon le easting
    data_las = data_las[data_las[:,0].argsort()]
    # Interpolsation d'un point GPS avec le même easting pour tous les points LIDAR
    poly_gps = interpolate.interp1d(data_GPS[:,0],data_GPS[:,1],kind='slinear')
    interp_las = poly_gps(data_las[:,0])
    # On sélectionne les points au-dessus et en-dessous de la ligne GPS
    # On néglige les points à moins de 1 mètre de la ligne GPS
    pts_haut = np.where(data_las[:,1] > interp_las+1)
    pts_bas = np.where(data_las[:,1] <= interp_las-1)
    # Calcul des élévations moyennes
    moy_haut_z = np.mean(data_las[:,2][pts_haut[0]])
    moy_bas_z = np.mean(data_las[:,2][pts_bas[0]])

    # Si la différence entre les moyennes est supérieure à 0.5m, il faut agir
    # La hauteur approximative d'une digue est de 0.7m
    if moy_bas_z - moy_haut_z >= 0.5:
        chemin = True
        maxN = maxN + 10
        las_flt_b = pflt.boolean_index(coordLas, minE, minN, maxE, maxN)
        las_flt_b = las_flt_b[las_flt_b[:,0].argsort()]
        poly_gps_b = interpolate.interp1d(data_GPS[:,0],data_GPS[:,1],kind='slinear')
        interp_las_b = poly_gps_b(las_flt_b[:,0])
        # On enlève tous les points 1m au-dessus de la ligne
        pts_haut_b = np.where(las_flt_b[:,1] > interp_las_b+1)
        pts_x = las_flt_b[:,0][pts_haut_b[0]]
        pts_y = las_flt_b[:,1][pts_haut_b[0]]
        pts_z = las_flt_b[:,2][pts_haut_b[0]]
        data_las = np.column_stack((pts_x, pts_y, pts_z))
    elif moy_haut_z - moy_bas_z >= 0.5:
        chemin = True
        minN = minN - 10
        las_flt_b = pflt.boolean_index(coordLas, minE, minN, maxE, maxN)
        las_flt_b = las_flt_b[las_flt_b[:,0].argsort()]
        poly_gps_b = interpolate.interp1d(data_GPS[:,0],data_GPS[:,1],kind='slinear')
        interp_las_b = poly_gps_b(las_flt_b[:,0])
        # On enlève tous les points 1m au-dessus de la ligne
        pts_bas_b = np.where(las_flt_b[:,1] < interp_las_b-1)
        pts_x = las_flt_b[:,0][pts_bas_b[0]]
        pts_y = las_flt_b[:,1][pts_bas_b[0]]
        pts_z = las_flt_b[:,2][pts_bas_b[0]]
        data_las = np.column_stack((pts_x, pts_y, pts_z))

    return chemin, data_las

def pick_digues(data, header, digues):
    '''
    Fonction pour le picking automatique des digues.
    INPUTS:
    - data: Array numpy contenant les données GPR avec limite verticale.
        (Donc après avoir trouvé samp_max)
    - header: Le header du fichier DZT contenant les données GPR.
    - digues: Liste de tuples correspondant aux positions horizontales
        des digues, obtenue à l'aide de loc_digues.
    OUTPUTS:
    coord: Dictionnaire dont les clés sont les différentes digues et les
        valeurs sont les coordonnées (en m et en ns) des valeurs pickées.
    coord_brutes: Dictionnaire dont les clés sont les différentes digues et les
        valeurs sont les coordonnées (en no. de trace et en no. de sample) des
        valeurs pickées.
    '''

    ntraces = data.shape[1]

    # Filtrage de base
    data = pflt.deWOW(data, 18)
    data = pflt.enlever_trace_moy(data, 2048)

    # Dictionnaires
    coord = {}
    coord_brutes = {}
    # Label digue
    i = 1
    for dig in digues:
        # Moyennes horizontales pour chacune des digues
        moy_dig = np.mean(data[:,dig[0]:dig[1]], 1)
        # Sélection de la plus petite moyenne
        moy_min = np.amin(moy_dig)
        # Définition d'une nouvelle limite dans la recherche de la surface du sol
        lim_dig = np.max(np.where(moy_dig < 0.8*moy_min))
        digue = data[:lim_dig,dig[0]:dig[1]]

        # Picking
        seuil = 0.3*np.amax(digue)
        digue = digue.tolist()
        digue = pflt.conv_col_norm(digue, norm=False)
        # Trace départ != 0 quand on flip
        coord_x_dig, coord_y_dig, x_dig, y_dig = coord_pick(digue, seuil, 0, ntraces, header["ns_per_zsample"]*1e9, header["sec"])
        coord_x_dig = ((np.array(coord_x_dig)) + dig[0]*(header["sec"]/ntraces)).tolist()
        x_dig = ((np.array(x_dig)) + dig[0]).tolist()
        coord['digue'+str(i)] = [coord_x_dig, coord_y_dig]
        coord_brutes['digue'+str(i)] = [x_dig, y_dig]
        i += 1
    return coord, coord_brutes

def digues_LIDAR(data,lidar_ns,digues,header,ntrace_ini):
    ns_sample = header['ns_per_zsample']*1e9
    coord_brutes = {}
    coord = {}
    i = 1
    for dig in digues:
        moy_dig = np.mean(data[:100,dig[0]:dig[1]],1)
        moy_max = np.amax(moy_dig)
        posh_dig = np.where(moy_dig == moy_max)
        min_lid = (np.amin(np.asarray(lidar_ns[:,2])[dig[0]:dig[1]]))/ns_sample
        offset = int(np.round(min_lid))
        xdig = np.linspace(dig[0],dig[1]-1,num=(dig[1]-dig[0]))
        x_reel = (header['sec']/ntrace_ini)*xdig
        zdig = (np.asarray((lidar_ns[:,2][dig[0]:dig[1]]))/(ns_sample)+posh_dig[0]-offset).tolist()
        z_reel = (np.asarray(lidar_ns[:,2][dig[0]:dig[1]])+((posh_dig[0]-offset)*ns_sample)).tolist()
        coord_brutes['digue'+str(i)] = [xdig.tolist(),zdig]
        coord['digue'+str(i)] = [x_reel.tolist(),z_reel]
        i += 1
    return coord,coord_brutes

def pick_champs(data, digues, header, samp_max):

    # Acquisition des propriétés du fichier de données
    ntraces = data.shape[1]
    ns_persample = header["ns_per_zsample"]*1e9
    long_ligne = header["sec"]

    # deWOW
    data = pflt.deWOW(data, 18)
    # Conversion en float32 pour sauver de la mémoire
    data = data.astype('float32')

    # Création d'un dictionnaire pour acceuillir les résultats
    res_pick = {}
    res_pick_brut = {}
    recon = {}
    deb_dig = False

    # Utilisation de la méthode SVD (Singular Values Decomposition)
    for i in range(len(digues)+1):
        if i == 0:
            # Une digue fait environ 600 traces de large
            if digues[i][0] <= 300:
                #res_pick['champ'+str(i+1)] = []
                #res_pick_brut['champ'+str(i+1)] = []
                deb_dig = True
                continue
            else:
                U, D, V = linalg.svd(data[:samp_max,:digues[i][0]])
                reste = np.zeros((samp_max, digues[i][0]-samp_max))

        elif i == len(digues):
            U, D, V = linalg.svd(data[:samp_max,digues[i-1][1]:])
            reste = np.zeros((samp_max, ntraces-digues[i-1][1]-samp_max))
        
        else:
            U, D, V = linalg.svd(data[:samp_max,digues[i-1][1]:digues[i][0]])
            reste = np.zeros((samp_max, digues[i][0]-digues[i-1][1]-samp_max))
        
        # Mise à zéro de quelques valeurs singulières pour filtrer un peu de bruit
        # et les air waves
        D[0] = 0
        #D[6:] = 0
        Dmn = np.diag(D)
        Dsys = np.concatenate((Dmn, reste), 1)
        del Dmn
        del reste
        Dsys = Dsys.astype('float32')
        U = U.astype('float32')
        V = V.astype('float32')
        reconstruct = U @ Dsys @ V

        # Picking
        ligne = reconstruct.tolist()
        colnorm = pflt.conv_col_norm(ligne, norm=True, norm_comp=False)
        #del reconstruct
        del U
        del V

        if i == 0:
            a, b, c, d = coord_pick(colnorm, 0.8, 0, ntraces, ns_persample, long_ligne)
        else:
            a, b, c, d = coord_pick(colnorm, 0.8, digues[i-1][1], ntraces, ns_persample, long_ligne)

        res_pick['champ'+str(i+1)] = [a, b]
        res_pick_brut['champ'+str(i+1)] = [c, d]
        recon['champ'+str(i+1)] = reconstruct
        del a
        del b
        del c
        del d
        del reconstruct

    return res_pick, res_pick_brut, recon, deb_dig

def pick_champs_NS(data,header,seuil):

    data = data.astype('float32')
    res_pick_brut = {}

    # Dewow
    data = pflt.deWOW(data,18)
    # Enlever trace moyenne
    data = pflt.enlever_trace_moy(data,data.shape[1])
    # Running average
    data = pflt.running_avg(data[:100,:],500)
    # Normalisation de chaque trace
    max_col = np.amax(data,axis=0)
    for trace in range((data.shape[1])):
        data[:,trace] = data[:,trace]/max_col[trace]
    # Picking
    pick_brut = coord_pick_V(data,seuil)
    res_pick_brut['0-segment'] = [pick_brut[:,0],pick_brut[:,1]]

    return res_pick_brut

def lissage(fen, scale, champ, ordre):

    step = int(fen/2)
    bins = np.arange(0, len(champ), step)
    xsmooth_seed = np.zeros(len(bins))
    ysmooth_seed = np.zeros(len(bins))

    # Premières valeurs
    yobs = champ[0:bins[1]]
    #yobs = champ[0:75]
    xobs = np.linspace(0,bins[1],num=len(yobs))
    #xobs = np.linspace(0,75,num=len(yobs))
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
    p=np.polyfit(x,y,ordre)
    xsmooth_seed[0]=min(xobs)
    ysmooth_seed[0]=np.polyval(p,min(xobs))

    # Dernières valeurs
    yobs = champ[bins[-2]:]
    xobs = np.linspace(bins[-2],len(champ)-1,num=len(yobs))
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
    p=np.polyfit(x,y,ordre)
    xsmooth_seed[-1]=max(xobs)
    ysmooth_seed[-1]=np.polyval(p,max(xobs))

    # Autres valeurs
    for i in range(1, len(bins)-1):
        yobs = champ[bins[i-1]:bins[i+1]]
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
        p=np.polyfit(x,y,ordre)
        xsmooth_seed[i]=bins[i]
        ysmooth_seed[i]=np.polyval(p,bins[i])

    f = interpolate.interp1d(xsmooth_seed, ysmooth_seed,kind='slinear')
    ch = np.arange(0, len(champ), 1)
    ynoise = f(ch)

    return ynoise

# Fonction à retravailler
def second_picking(lissage, lim, champ_recon, ntraces, header, depart):
    mat_ind = (np.round(lissage)).astype('int')
    mat_ind = np.vstack((mat_ind-2, mat_ind-1, mat_ind, mat_ind+1, mat_ind+2))
    # Mettre une ligne pour éviter un indice supérieur à la limite verticale
    mat_ind[mat_ind >= lim] = lim - 1.
    size = mat_ind.shape[1]
    champ_pick2 = [champ_recon[mat_ind[:,i],i] for i in range(size)]
    champ_pick2 = (np.stack(champ_pick2)).T
    lignes = champ_pick2.tolist()
    colnorm = pflt.conv_col_norm(lignes, norm=True, norm_comp=False)
    cx, cy, cx_brut, cy_brut = coord_pick(colnorm, 0.4, depart, ntraces, header["ns_per_zsample"]*1e9, header["sec"], dec_temp=mat_ind[0])
    #max_col = np.amax(champ_pick2,axis=0)
    #for trace in range((champ_pick2.shape[1])):
    #        champ_pick2[:,trace] = champ_pick2[:,trace]/max_col[trace]
    #cbrut = coord_pick_V(champ_pick2,0.9)

    #return cbrut
    return cx, cy, cx_brut, cy_brut