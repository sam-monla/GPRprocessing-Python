"""
Toolbox pour procéder au phase picking
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy import linalg
from itertools import groupby
from operator import itemgetter
from tqdm import tqdm
from scipy import interpolate
import Prefiltrage as pflt

# Méthode SVD 
def quick_SVD(segment_data,coeff=[None,None]):
    """
    Méthode SVD pour rapidement filtrer du bruit (Background removal).
    INPUT: 
    - segment_data: Partie de la matrice de données qu'on veut filtrer. (La méthode SVD ne peut
                    pas gérer toute la matrice d'un seul coup, en général.)
    - coeff: Liste de 2 entiers. Possibilité d'annuler 2 séries de valeurs singulières. Le premier
            coeff permet d'annuler une seule valeur singulière (Annuler la valeur singulière 0 pour
            filtrer les ondes directes). Le 2e coeff permet d'éliminer une série. Par exemple, en 
            mettant 10, les valeurs singulières seront annulées à partir de la 10e jusqu'à la fin.
    OUTPUT:
    - reconstruct: Matrice filtrée
    """
    data = segment_data.astype('float32')
    U, D, V = linalg.svd(data)
    reste = np.zeros((data.shape[0], data.shape[1]-data.shape[0]))
    if coeff[0] != None:
        D[coeff[0]] = 0
    if coeff[1] != None:
        D[coeff[1]:] = 0
    Dmn = np.diag(D)
    Dsys = np.concatenate((Dmn, reste), axis=1)
    del Dmn
    del reste
    Dsys = Dsys.astype('float32')
    U = U.astype('float32')
    V = V.astype('float32')
    reconstruct = U @ Dsys @ V
    return reconstruct

# Transformée de Hilbert pour quadrature trace
def quad_trace(data):
    """
    Équation 4 de l'article Theoty and application to syntetic and real GPR data de Matteo Dossi.
    INPUT:
    - data: Matrice de données du géoradar, avec ou sans pré-filtrage. Soustraire 32768 des valeurs 
            de la matrice pour centrer à 0.
    OUTPUT:
    - dat_prim: Matrice dont chaque colonne est la partie imaginaire de la trace correspondante.
    """
    dat_prim = np.zeros(data.shape)
    Nsamp = data.shape[0]
    nlist = np.linspace(0,Nsamp-1,num=Nsamp)
    for n in range(len(nlist)):
        klist = np.linspace(n+1-Nsamp,n,num=Nsamp).astype('int')
        for k in klist:
            if k == 0:
                continue
            else:
                a = (((np.sin((np.pi*k/2)))**2)/k)*data[n-k,:] # np.deg2rad?
                dat_prim[n,:] += a
        dat_prim[n,:] = dat_prim[n,:]*(2/np.pi)
    return dat_prim

# Création des matrices Cij et twtij
def Ctwt_matrix(data):
    """
    Fonction pour créer les matrices Cij et twtij. On remplit les espaces vides avec des 2.0 pour la
    matrice Cij et des -1.0 pour la matrice twtij. (Il n'y a pas le même nombre de phases pour chaque
    trace).

    INPUT:
    - data: Cosinus de la matrice de données GPR transformée en matrice de phases.
    OUTPUT:
    - twtij_mat: Matrice indiquant les indices où chaque phase commence.
    - Cij_mat: Matrice indiquant l'amplitude maximale de chaque phase.
    """
    # Marquer de 1 les échantillons où il y a changement de signe
    sign_change = (np.diff(np.sign(data),axis=0) != 0)*1
    # Trouver les indices de ces endroits et les trier selon leur numéro de trace
    ind_change = np.where(sign_change == 1)
    pos_change = list(zip(ind_change[1],ind_change[0]))
    twtij = sorted(pos_change, key=lambda x: x[0])
    # Trouver le nombre de phase max pour une trace
    unique, counts = np.unique(ind_change[1],return_counts=True)
    dict1 = dict(zip(unique, counts))
    max_phase = max(dict1, key=dict1.get)
    # Initialisation d'une matrice (max_phase X nb de traces)
    twtij_mat = np.zeros((dict1[max_phase],data.shape[1]))
    # Création d'une liste de listes dont chacune des sous-listes correspond aux temps
    # de changement de phase d'une trace
    b = [list(list(zip(*g))[1]) for k, g in groupby(twtij, itemgetter(0))]
    for trace in range(data.shape[1]):
        twtij_mat[:len(b[trace]),trace] = b[trace]
    # Initialisation d'une 2e matrice (max_phase X nb de traces)
    Cij_mat = np.zeros((dict1[max_phase]+1,data.shape[1]))
    for trace in range(data.shape[1]):
        peaks = []
        for i in range(twtij_mat.shape[0]):
            if twtij_mat[i,trace] == 0:
                continue
            elif i == 0:
                #peaks.append(np.max(np.abs(data[:int(twtij_mat[i,trace]),trace])))
                peaks.append(max(data[:int(twtij_mat[i,trace]),trace].min(),data[:int(twtij_mat[i,trace]),trace].max(), key=abs))
            else:
                #peaks.append(np.max(np.abs(data[int(twtij_mat[i-1,trace]):int(twtij_mat[i,trace]),trace])))
                peaks.append(max(data[int(twtij_mat[i-1,trace]):int(twtij_mat[i,trace]),trace].min(),data[int(twtij_mat[i-1,trace]):int(twtij_mat[i,trace]),trace].max(),key=abs))
        #peaks.append(np.max(np.abs(data[int(twtij_mat[len(b[trace])-1,trace]):,trace])))
        peaks.append(max(data[int(twtij_mat[len(b[trace])-1,trace]):,trace].min(),data[int(twtij_mat[len(b[trace])-1,trace]):,trace].max(),key=abs))
        Cij_mat[:len(peaks),trace] = peaks
    # On remplace les espaces non-occupés (0) par des nan
    twtij_mat[twtij_mat == 0] = -1
    Cij_mat[Cij_mat == 0] = 2
    # On ajoute une ligne de 0 au-dessus de la matrice twtij pour qu'elle ait les mêmes dimensions
    # que Cij
    row1_twt = np.zeros((1,twtij_mat.shape[1]))
    twtij_mat = np.concatenate((row1_twt,twtij_mat),axis=0)
    return twtij_mat, Cij_mat

"""
Fonctions pour le horizon picking
"""
# Filtrage des positions sélectionnées par le 1er picking
def select_trace(pre_horizon, ph_actu):
    """
    Filtrage des positions sélectionnées par le 1er picking. La fonction horizon_picking() génère
    une liste de positions auquelles les valeurs d'amplitude et de temps satisfont certains critères.
    Cette fonction permet de limiter le nombre de positions pour une même trace à 1. Elle permet aussi
    de couper un horizon lorsqu'il y a un saut de trace.

    INPUT:
    - pre_horizon: Horizon sélectionné par l'algorithme pour une certaine position. Il s'agit donc d'une
                liste, et non d'une liste de listes. (Mettre à l'intérieur de la boucle de picking).
    OUTPUT:
    - newX2: Horizon filtré.
    """
    # Trier par numéro de traces
    pre_horizon.sort(key= lambda x: x[1])
    # Couper la liste lorsqu'il y a une discontinuité
    numtraces = [x[1] for x in pre_horizon]
    mini = np.min(numtraces)
    maxi = np.max(numtraces)
    list_num = np.linspace(mini,maxi,num=(maxi-mini)+1).astype(int)
    disc = list(set(list_num) - set(numtraces))
    if disc == []:
        hori_prelb = pre_horizon
    else:
        # Trier les discontinuités (Important)
        disc_order = np.sort(disc)
        lim_tup = np.where([j[1] <= disc_order[0] for j in pre_horizon])
        hori_prelb = pre_horizon[:np.max(lim_tup[0])+1]
    # Éviter le picking de 2 échantillons sur la même trace
    # Regrouper ensemble les positions ayant une trace commune
    tr_com = [list(group) for key, group in groupby(hori_prelb, itemgetter(1))]
    newX2 = []
    for i in range(len(tr_com)):
        # Pour les traces dans lesquelles il y plusieurs valeurs sélectionnées
        if len(tr_com[i]) > 1:
            # Si on est à la première trace, on garde la valeur la plus près
            # de l'échantillon (ph) (ligne de la matrice) actuel
            if i == 0:
                diff = []
                for j in tr_com[i]:
                    diff.append(np.abs(j[0]-ph_actu))
                pos_min = np.where(diff == np.min(diff))
                tr_com[i] = tr_com[i][pos_min[0][0]]
                newX2.append(tr_com[i])
            # Si on est à une autre trace que la 1re, on garde la valeur la plus près
            # de l'échantillon sélectionné pour la trace précédente.
            else:
                diff = []
                for j in tr_com[i]:
                    diff.append(np.abs(j[0]-newX2[-1][0]))
                pos_min = np.where(diff == np.min(diff))
                tr_com[i] = tr_com[i][pos_min[0][0]]
                newX2.append(tr_com[i])
        # Quand il n'y a qu'une seule position dans tr_com[i]
        else:
            newX2.append((tr_com[i][0][0],tr_com[i][0][1]))
    return newX2

# Fonction de horizon picking
"""
Quelques limitations:
- Ne semble pas très performant au niveau des digues
- Le critère temporel n'est basé que sur le premier élémet de l'horizon. Si une phase subit de bonnes
    variations verticales, elle sera séparée en plusieurs horizons plus petits.
"""
def horizon_pick(Cij,twtij,Tph):
    """
    Fonctions itérant sur chaque élément de la matrice Cij qui forme des horizons avec les éléments
    des traces voisines qui satisfont les conditions suivantes.
    1. Même polarité
    2. Proximité temporelle
    3. 2 phases ne se croisent pas

    INPUT:
    - Cij: Matrice Cij
    - twtij: Matrice twtij
    - Tph: Période approximative d'une phase (en samples). La durée des ondes directes est une 
            bonne 1re approximation.
    OUTPUT:
    - list_horizons: Liste de listes de coordonnées. Chaque sous-listes correspond à un horizon.
    """
    list_horizons = []
    for ph in range(Cij.shape[0]):
        for tr in range(Cij.shape[1]):
            # Vérifier qu'on a une amplitude valide à la position
            if (Cij[ph,tr] == 2) or (twtij[ph,tr] == -1):
                continue
            # Mettre à True tous les éléments ayant la même phase que l'élément à l'étude
            signe = np.sign(Cij[ph,tr])
            if signe < 0:
                crit1 = Cij < 0
            else:
                crit1 = (Cij > 0) & (Cij <= 1)  # On filtre les amplitudes de 2 (Cij <= 1)
            # Mettre un True à tous les éléments temporellement près de l'élément à l'étude
            crit2 = (twtij > twtij[ph,tr]-Tph) & (twtij < twtij[ph,tr]+Tph) & (twtij >= 0) # On filtre les temps de -1
            # Combinaison des critères
            hori_prel = np.argwhere((crit1 & crit2) == True)
            # Faire une liste de tuples avec les positions valides
            hori_prelb = []
            for elem in hori_prel:
                hori_prelb.append((elem[0],elem[1]))   
            # Prendre la valeur la plus près du sample actuel lorsqu'il y a plus qu'un True
            # pour une même trace et couper les discontinuités
            hori = select_trace(hori_prelb,ph)
            list_horizons.append(hori)
            # Rendre invalide les positions qui font maintenant partie d'un horizon
            Cij_term = 2*np.ones((1,len(hori)))
            twtij_term = -1*np.ones((1,len(hori)))
            row, cols = zip(*hori)
            Cij[row,cols] = Cij_term
            twtij[row,cols] = twtij_term
    return list_horizons

"""
Fonction alternative pour le horizon picking.
- Tentative d'éliminer le problème du critère temporel pour pouvoir formé de plus longs horizons.
- Ne fonctionne pas encore
- Permet les croisements (À corriger)
"""
def horizon_pick_B(Cij,twtij,Tph):
    """
    Fonctions itérant sur chaque élément de la matrice Cij qui forme des horizons avec les éléments
    des traces voisines qui satisfont les conditions suivantes.
    1. Même polarité
    2. Proximité temporelle
    3. 2 phases ne se croisent pas

    INPUT:
    - Cij: Matrice Cij
    - twtij: Matrice twtij
    - Tph: Période approximative d'une phase (en samples). La durée des ondes directes est une 
            bonne 1re approximation.
    OUTPUT:
    - list_horizons: Liste de listes de coordonnées. Chaque sous-listes correspond à un horizon.
    """
    list_horizons = []
    for ph in range(Cij.shape[0]):
        for tr in range(Cij.shape[1]):
            # Vérifier qu'on a une amplitude ou un temps valide à la position
            if (Cij[ph,tr] == 2) or (twtij[ph,tr] == -1):
                    continue
            # Vérifier la polarité de la phase à l'étude et mettre à True toutes les autres
            # phases ayant la même polarité
            signe = np.sign(Cij[ph,tr])
            if signe < 0:
                crit1 = Cij < 0
            else:
                crit1 = (Cij > 0) & (Cij <= 1)
            # Mettre à True toutes les phases temporellement assez proches de la phase à l'étude
            crit2 = (twtij > twtij[ph,tr]-Tph) & (twtij < twtij[ph,tr]+Tph) & (twtij >= 0)
            # Combinaison des critères
            hori_prel = np.argwhere((crit1 & crit2) == True)
            # Faire une liste de tuples avec les positions valides
            hori_prelb = []
            for elem in hori_prel:
                if elem[1] < tr:
                    continue
                else:
                    hori_prelb.append((elem[0],elem[1]))
            # Prendre la valeur la plus près du sample actuel lorsqu'il y a plus qu'un True
            # pour une même trace et couper les discontinuités
            hori = select_trace(hori_prelb,ph)

            # Avant d'ajouter hori à la liste d'horizons, on refait une vérifiaction des critères
            # 1 et 2 à partir de son dernier élément. On définit donc des critères de prolongement,
            # qui sont les critères 1 et 2 appliqués à la trace suivante.
            if hori[-1][1] < Cij.shape[1]-1:
                if signe < 0:
                    crit1_prol = Cij[:,hori[-1][1]+1] < 0 
                else:
                    crit1_prol = Cij[:,hori[-1][1]+1] > 0
                time_ref = twtij[hori[-1][0],hori[-1][1]]
                crit2_prol = (twtij[:,hori[-1][1]+1] > time_ref-Tph) & (twtij[:,hori[-1][1]+1] < time_ref+Tph) & (twtij[:,hori[-1][1]+1] >= 0)
                # Combinaison des critères de prolongement
                crits_prol = crit1_prol & crit2_prol
                # Boucle while: Tant qu'il y a au moins un True dans la trace qui suit la dernière position 
                # de hori, on recommence le prolongement. (Limite supplémentaire pour respecter les dimensions)
                while np.any(crits_prol == True) and (hori[-1][1] < Cij.shape[1]-1):
                    hori_last = hori[-1]
                    # Critères 1 et 2 (Pas les critères de prolongement)
                    crit2b = (twtij[:,hori[-1][1]+1:] > time_ref-Tph) & (twtij[:,hori[-1][1]+1:] < time_ref+Tph) & (twtij[:,hori[-1][1]+1:] >= 0)
                    mix2 = crit1[:,hori[-1][1]+1:] & crit2b
                    hori_prel = np.argwhere(mix2 == True)
                    hori_prelb = []
                    for elem in hori_prel:
                        # Ajouter le offset issu de la coupure de la matrice
                        hori_prelb.append((elem[0],elem[1]+hori_last[1]+1))
                    # Élimination des doubles et coupure pour éviter les sauts de trace
                    if hori_prelb == []:
                        break
                    prolongation_hori = select_trace(hori_prelb,ph)
                    # Mise-à-jour de hori
                    hori += prolongation_hori

                    # Redéfinir et vérifier les critères de prolongement
                    if hori[-1][1] == Cij.shape[1]-1:
                        break
                    elif signe < 0:
                        crit1_prol = Cij[:,hori[-1][1]+1] < 0 
                    else:
                        crit1_prol = Cij[:,hori[-1][1]+1] > 0
                    time_ref = twtij[hori[-1][0],hori[-1][1]]
                    crit2_prol = (twtij[:,hori[-1][1]+1] > time_ref-Tph) & (twtij[:,hori[-1][1]+1] < time_ref+Tph) & (twtij[:,hori[-1][1]+1] >= 0)
                    crits_prol = crit1_prol & crit2_prol

            # Mise-à-jour de list_horizons
            list_horizons.append(hori)
            # Rendre invalide les positions qui font maintenant partie d'un horizon
            Cij_term = 2*np.ones((1,len(hori)))
            twtij_term = -1*np.ones((1,len(hori)))
            row, cols = zip(*hori)
            Cij[row,cols] = Cij_term
            twtij[row,cols] = twtij_term

    return list_horizons     

"""
Deuxième fonction alternative pour le horizon picking.
C'est la meilleure jusqu'à présent.
Améliorations possibles:
- Ajouter une condition pour empêcher les croisements
"""
def horipick_Dossi(Cij,twtij,Tph,tol_C,tolmin_t,tolmax_t,h_offset=0,min_time=6):
    """
    tol_C: Entier entre 0 et 1 pour filtrer les amplitudes s'éloignant trop de -1 ou 1 dans
        la matrice Cij. Exemple 0.7 -> On garde toutes les amplitudes plus grande que +/- 0.7.
    tol_t: Entier indiquant les longueurs min et max de la longueur (en temps) d'une phase.
        Exemple 2 -> tolmin_t = 0.5: longueur min = 0.5*Tph
    """
    # Copie des matrices originales
    twtij_clone = np.copy(twtij)
    Cij_clone = np.copy(Cij)

    # Filtrer les fenêtres temporelles trop courtes ou trop longues
    diff_twt = np.diff(twtij,axis=0)
    bad_phase = np.argwhere((diff_twt < 0) | (diff_twt < tolmin_t*Tph) | (diff_twt > tolmax_t*Tph))
    if len(bad_phase)>0:
        f1_Cij = 2*np.ones((1,len(bad_phase)))
        f1_twtij = -1*np.ones((1,len(bad_phase)))
        row, cols = zip(*bad_phase)
        Cij[row,cols] = f1_Cij
        twtij[row,cols] = f1_twtij
    # Filtrer les phases qui viennent trop tôt
    early_phase = np.argwhere(twtij <= min_time)
    if len(early_phase)>0:
        f1b_Cij = 2*np.ones((1,len(early_phase)))
        f1b_twtij = -1*np.ones((1,len(early_phase)))
        row, cols = zip(*early_phase)
        Cij[row,cols] = f1b_Cij
        twtij[row,cols] = f1b_twtij
    # Filtrer les amplitudes trop petites
    bad_amp = np.argwhere(np.abs(Cij) < tol_C)
    if len(bad_amp)>0:
        f2_Cij = 2*np.ones((1,len(bad_amp)))
        f2_twtij = -1*np.ones((1,len(bad_amp)))
        row, cols = zip(*bad_amp)
        Cij[row,cols] = f2_Cij
        twtij[row,cols] = f2_twtij

    #print(Tph)
    list_horizons = []
    true_pos_hori = []
    liste_signe = []
    for ph in tqdm(range(Cij.shape[0])):
        for tr in range(twtij.shape[1]):

            # Vérifier qu'on a une amplitude et un temps valide à cette position
            if (Cij[ph,tr] == 2) or (twtij[ph,tr] == -1):
                    continue

            # Initialisation d'un horizon
            hori = [(ph,tr)]
            hori_os = [(ph,tr+h_offset)]

            # Prolongement de l'horizon
            if hori[-1][1] < Cij.shape[1]-1:
                # Critères de prolongement dans la trace voisine
                # Polarité
                signe = np.sign(Cij[ph,tr])
                if signe < 0:
                    crit1 = Cij[:,tr+1] < 0
                else:
                    crit1 = (Cij[:,tr+1] > 0) & (Cij[:,tr+1] <= 1)
                # Temps
                crit2 = (twtij[:,tr+1] > (twtij[ph,tr]-Tph/2)) & (twtij[:,tr+1] < (twtij[ph,tr]+Tph/2)) & (twtij[:,tr+1] >= 0)
                # Ajouter critère de croisement ici?
                # Intersection des critères
                crits_prol = crit1 & crit2
                
                trace = tr
                while (np.any(crits_prol)) and (hori[-1][1] < Cij.shape[1]-1):
                    hori_extens = np.argwhere(crits_prol == True)
                    hori_extens = np.asarray(hori_extens).reshape((1,len(hori_extens)))[0]
                    #if len(list_horizons) == 409:
                    #    print(hori_extens)
                    #    print(hori[-1])

                    if len(hori_extens) > 1:
                        diff = []
                        for i in range(len(hori_extens)):
                            diff.append(np.abs(hori_extens[i]-hori[-1][0]))
                        pos_min = np.where(diff == np.min(diff))
                        hori.append((hori_extens[pos_min[0][0]],trace+1))
                        hori_os.append((hori_extens[pos_min[0][0]],trace+1+h_offset))
                    else:
                        hori.append((hori_extens[0],trace+1))
                        hori_os.append((hori_extens[0],trace+1+h_offset))

                    #if len(list_horizons) == 409:
                    #    print(hori[-1])

                    trace += 1
                    # Redéfinir et vérifier les critères de prolongement
                    #print(hori)
                    if hori[-1][1] == Cij.shape[1]-1:
                        break
                    elif signe < 0:
                        crit1 = Cij[:,hori[-1][1]+1] < 0 
                    else:
                        crit1 = Cij[:,hori[-1][1]+1] > 0
                    time_ref = twtij[hori[-1][0],hori[-1][1]]
                    crit2 = (twtij[:,hori[-1][1]+1] > (time_ref-Tph/2)) & (twtij[:,hori[-1][1]+1] < (time_ref+Tph/2)) & (twtij[:,hori[-1][1]+1] >= 0)
                    crits_prol = crit1 & crit2

            # Mise-à-jour de list_horizons
            list_horizons.append(hori)
            true_pos_hori.append(hori_os)
            liste_signe.append(signe)
            # Rendre invalide les positions qui font maintenant partie d'un horizon
            Cij_term = 2*np.ones((1,len(hori)))
            twtij_term = -1*np.ones((1,len(hori)))
            row, cols = zip(*hori)
            Cij[row,cols] = Cij_term
            twtij[row,cols] = twtij_term
        
    return list_horizons, true_pos_hori, Cij_clone, twtij_clone, liste_signe
            
def cross_horijoin(joint,horizon_actuel,horizons,hori_t,new_horit,direction="left"):
    """
    Fonction pour vérifier si le lien potentiel entre 2 horizons existants croise
    un autre horizon.

    INPUT:
    joint: left ou right. Il s'agit de l'horizon qu'on désire relier à l'horizon actuel
    horizon_actuel: Horizon actuellement à l'étude
    horizons: Liste de tous les horizons
    direction: Préciser si on désire prolonger l'horizon vers la droite ou la gauche
    """
    # Retirer les horizons qui n'ont aucune trace commune avec le lien
    #print(joint)
    if direction == "left":
        #fa = interpolate.interp1d([horizons[joint[0]][-1][1],horizon_actuel[0][1]],[hori_t[joint[0]][-1][0],horizon_actuel[0][0]],kind='linear')
        fa = interpolate.interp1d([horizons[joint][-1][1],horizon_actuel[0][1]],[hori_t[joint][-1][0],horizon_actuel[0][0]],kind='linear')
        #xa = np.linspace(horizons[joint[0]][-1][1],horizon_actuel[0][1],((horizon_actuel[0][1]-horizons[joint[0]][-1][1])*8)+1)
        xa = np.linspace(horizons[joint][-1][1],horizon_actuel[0][1],((horizon_actuel[0][1]-horizons[joint][-1][1])*8)+1)
        new_a = fa(xa)
    elif direction == "right":
        #fa = interpolate.interp1d([horizon_actuel[-1][1],horizons[joint[0]][0][1]],[horizon_actuel[-1][0],hori_t[joint[0]][0][0]],kind='linear')
        fa = interpolate.interp1d([horizon_actuel[-1][1],horizons[joint][0][1]],[horizon_actuel[-1][0],hori_t[joint][0][0]],kind='linear')
        #xa = np.linspace(horizon_actuel[-1][1],horizons[joint[0]][0][1],((horizons[joint[0]][0][1]-horizon_actuel[-1][1])*8)+1)
        xa = np.linspace(horizon_actuel[-1][1],horizons[joint][0][1],((horizons[joint][0][1]-horizon_actuel[-1][1])*8)+1)
        new_a = fa(xa)

    hori_poss = []
    for elem in hori_t:
        xelem = np.linspace(elem[0][1],elem[-1][1],((elem[-1][1]-elem[0][1])*8)+1)
        if len(np.intersect1d(xa,xelem,return_indices=False))>0:
            hori_poss.append(elem)
    for nelem in new_horit:
        n_xelem = np.linspace(nelem[0][1],nelem[-1][1],((nelem[-1][1]-nelem[0][1])*8)+1)
        if len(np.intersect1d(xa,n_xelem,return_indices=False))>0:
            hori_poss.append(nelem)

    if len(hori_poss) > 0:
        xbool = []
        for candid in hori_poss:
            b_x = []
            b_y = []
            for elem in candid:
                b_y.append(elem[0])
                b_x.append(elem[1])
            # Interpolations précises à l'unité pour le potentiel joint d'horizon et pour les horizons
            # avec traces communes
            fb = interpolate.interp1d(b_x,b_y,kind='linear')
            xb = np.linspace(candid[0][1],candid[-1][1],((candid[-1][1]-candid[0][1])*8)+1)
            new_b = fb(xb)
            # Trouver les traces qui sont partagées entre a et b
            intersec,com1,com2 = np.intersect1d(xa,xb,return_indices=True)
            # Soustraire les parties des horizons qui partagent des traces
            sign = new_b[com2[0]:com2[-1]]-new_a[com1[0]:com1[-1]]
            # Repérer les changements de signe pour déceler les croisement
            zero_crossings = np.where(np.diff(np.sign(sign)))[0]

            # Oublier la jonction s'il y a un croisement
            if len(zero_crossings) > 0:
                xbool.append(True)
            else:
                xbool.append(False)
        return xbool

def cross_phase(signe,joint,horizon_actuel_samp,horizons_samp,champ,direction="left"):
    """
    Fonction pour s'assurer qu'une jonction ne traverse pas une phase de polarité
    différente.
    """
    # Interpoler à l'unité sur la jonction potentielle
    if direction == "left":
        fa = interpolate.interp1d([horizons_samp[joint][-1][1],horizon_actuel_samp[0][1]],[horizons_samp[joint][-1][0],horizon_actuel_samp[0][0]],kind='linear')
        xa = np.linspace(horizons_samp[joint][-1][1],horizon_actuel_samp[0][1],((horizon_actuel_samp[0][1]-horizons_samp[joint][-1][1]))+1)
        new_a = fa(xa)
    elif direction == "right":
        fa = interpolate.interp1d([horizon_actuel_samp[-1][1],horizons_samp[joint][0][1]],[horizon_actuel_samp[-1][0],horizons_samp[joint][0][0]],kind='linear')
        xa = np.linspace(horizon_actuel_samp[-1][1],horizons_samp[joint][0][1],((horizons_samp[joint][0][1]-horizon_actuel_samp[-1][1]))+1)
        new_a = fa(xa)
    
    # Transformer new_a en entiers arrondis
    new_a = np.round(new_a).astype("int")
    xa = xa.astype("int")
    # Trouver les valeurs d'amplitude correspondantes dans C_clone
    amp_joint = []
    for i in range(len(xa)):
        amp_joint.append(champ[new_a[i]][xa[i]])

    # Déterminer le pourcentage de valeurs qui sont de signe opposé
    amp_joint = np.asarray(amp_joint)
    sign_joint = (amp_joint > 0).tolist()
    if signe > 0:
        prop_opp = sign_joint.count(0)/len(sign_joint)
    elif signe < 0:
        prop_opp = sign_joint.count(1)/len(sign_joint)
    return prop_opp

# Fonction pour le horizon joining
"""
Première itération: 1er décembre 2020
"""
def horijoin_Dossi(horizons,Cij_clone,twtij_clone,Lg=False,Tj=False,min_length=False):
    
    # Trier les horizons en ordre de traces
    horizons_tri = sorted(horizons, key=lambda x: x[0][1])

    # Conversion des horizons
    # cos_hori -> De position à cosinus de la phase
    # time_hori -> De sample à temps
    cos_hori = []
    time_hori = []
    for hori in horizons_tri:
        row, cols = zip(*hori)
        cos = Cij_clone[row,cols].tolist()
        time = twtij_clone[row,cols].tolist()
        cos_hori.append(cos)
        time_hori.append(list(zip(time,cols)))
        # Pour être sûr???
        time_hori = sorted(time_hori, key=lambda x: x[0][1])
    cos_hori_clone = np.copy(cos_hori)
    time_hori_clone = np.copy(time_hori)

    # Vérification que tous les horizons contiennent des éléments de même signe
    for hori in cos_hori:
        if np.all(np.asarray(hori) > 0):
            continue
        elif np.all(np.asarray(hori) < 0):
            continue
        else:
            ind = cos_hori.index(hori)
            print("L'horizon {} contient des éléments de signes opposés".format(ind))
            break
    
    newlist_horizons = []
    newlist_horizons_time = []
    liste_interdite = []
    liste_signe = []
    # On itère sur chaque horizons
    for horizon in tqdm(range(len(horizons_tri))):

        # Vérifier que l'horizon n'est pas déjà utilisé
        if (horizon in liste_interdite):
            continue

        # Vérification du critère 1
        # Signe de l'horizon actuel
        cos_bool = np.copy(cos_hori)
        signe = np.sign(cos_hori[horizon][0])
        if signe > 0:
            for j in range(len(cos_hori)):
                if (np.all(np.asarray(cos_hori[j]) > 0)) and (np.all(np.asarray(cos_hori[j]) <= 1)) and (j not in liste_interdite):
                    cos_bool[j] = True
                else:
                    cos_bool[j] = False
        if signe < 0:
            for k in range(len(cos_hori)):
                if (np.all(np.asarray(cos_hori[k]) < 0)) and (k not in liste_interdite):
                    cos_bool[k] = True
                else:
                    cos_bool[k] = False

        # Vérification du critère 2 (temporel)
        # Limites horizontales (traces) de l'horizon actuel
        start = time_hori[horizon][0][1]
        end = time_hori[horizon][-1][1]
        # Limites verticales (samples) de l'horizon actuel
        tstart = time_hori[horizon][0][0]
        tend = time_hori[horizon][-1][0]
        time_bool = np.copy(time_hori)
        for i in range(len(time_hori)):
            if (time_hori[i][-1][1]<start) and (tstart-(Tj)<=time_hori[i][-1][0]<=tstart+(Tj)) and (i not in liste_interdite):
                time_bool[i] = True
            elif (time_hori[i][0][1]>end) and (tend-(Tj)<=time_hori[i][0][0]<=tend+(Tj)) and (i not in liste_interdite):
                time_bool[i] = True
            else:
                time_bool[i] = False

        # Vérification du critère 3 (Séparation)
        separ_bool = np.copy(time_hori)
        for i in range(len(time_hori)):
            if (time_hori[i][-1][1]<start) and (time_hori[i][-1][1]>=start-Lg) and (i not in liste_interdite):
                separ_bool[i] = True
            elif (time_hori[i][0][1]>end) and (time_hori[i][0][1]<=end+Lg) and (i not in liste_interdite):
                separ_bool[i] = True
            else:
                separ_bool[i] = False

        # Combinaison des critères
        crit_prol = cos_bool & time_bool & separ_bool

        hori_combine = horizons_tri[horizon]
        hori_combine_t = time_hori[horizon]
        list_left = []
        list_right = []
        # Boucle de prolongement
        while np.any(crit_prol):
            ind_prol = np.where(cos_bool & time_bool & separ_bool)[0]
            if len(ind_prol) == 0:
                continue
            left = ind_prol[ind_prol < horizon]
            right = ind_prol[ind_prol > horizon]

            # Pour chaque côté, conserver uniquement l'horizon le plus près de 
            # l'horizon actuel (2 axes)
            # Gauche
            if len(left > 1):

                # Sélectionner l'horizon avec la moyenne verticale la plus près
                #moy_ref = np.mean([i[0] for i in hori_combine_t[:min_length]])
                #moy_left = []
                #for lefty in left:
                #    moy_left.append(np.mean([k[0] for k in horizons_tri[lefty]]))
                #dif = np.abs(moy_ref-moy_left)
                # Si aucun horizon se démarque, prendre celui qui est plus près
                #if len(set(dif)) == 1:
                traces = np.asarray([horizons_tri[i][-1][1] for i in left])
                times = np.asarray([time_hori[j][-1][0] for j in left])
                refx = hori_combine[0][1]
                reft = hori_combine_t[0][0]
                difx = refx-traces
                dift = reft-times
                pyt = np.sqrt(difx**2 + dift**2)
                prox = np.where(pyt==np.min(pyt))[0]
                left = left[prox]
                #else:    
                #    pos = np.where(dif == np.min(dif))[0][0]
                #    left = [left[pos]]
                
            # Droite
            if len(right > 1):
                
                # Sélectionner l'horizon avec la moyenne verticale la plus près
                #moy_ref = np.mean([i[0] for i in hori_combine_t[-min_length:]])
                #moy_right = []
                #for righty in right:
                #    moy_right.append(np.mean([k[0] for k in horizons_tri[righty]]))
                #dif = np.abs(moy_ref-moy_right)
                # Si aucun horizon se démarque, prendre celui qui est plus près
                #if len(set(dif)) == 1:
                traces = np.asarray([horizons_tri[i][0][1] for i in right])
                times = np.asarray([time_hori[j][0][0] for j in right])
                refx = hori_combine[-1][1]
                reft = hori_combine_t[-1][0]
                difx = traces-refx
                dift = times-reft
                pyt = np.sqrt(difx**2 + dift**2)
                prox = np.where(pyt==np.min(pyt))[0]
                right = right[prox]
                #else:
                #    pos = np.where(dif == np.min(dif))[0][0]
                #    right = [right[pos]]

            # Critère 4 - Croisement
            # Gauche
            if len(left) == 1:
                x_bool = cross_horijoin(left,hori_combine_t,horizons_tri,time_hori,newlist_horizons_time,direction="left")
                if np.any(x_bool):
                    left = []
            # Droite
            if len(right) == 1:
                x_bool = cross_horijoin(right,hori_combine_t,horizons_tri,time_hori,newlist_horizons_time,direction="right")
                if np.any(x_bool):
                    right = [] 

            if (len(left) == 1) and (len(right) == 1):
                hori_combine = horizons_tri[left[0]] + hori_combine + horizons_tri[right[0]]
                hori_combine_t = time_hori[left[0]] + hori_combine_t + time_hori[right[0]]
                list_left.append(left[0])
                list_right.append(right[0])
            elif (len(left) == 1) and (len(right) == 0):
                hori_combine = horizons_tri[left[0]] + hori_combine
                hori_combine_t = time_hori[left[0]] + hori_combine_t
                list_left.append(left[0])
            elif (len(left) == 0) and (len(right) == 1):
                hori_combine = hori_combine + horizons_tri[right[0]]
                hori_combine_t = hori_combine_t + time_hori[right[0]]
                list_right.append(right[0])
            else:
                break
            hori_combine = sorted(hori_combine, key=lambda x: x[1])
            hori_combine_t = sorted(hori_combine_t, key=lambda x: x[1])
            # Revérifier les critères de prolongement
            # Vérification du critère 1
            # Signe de l'horizon actuel
            cos_bool = np.copy(cos_hori)
            if signe > 0:
                for j in range(len(cos_hori)):
                    if (np.all(np.asarray(cos_hori[j]) > 0)) and (np.all(np.asarray(cos_hori[j]) <= 1)) and (j not in liste_interdite):
                        cos_bool[j] = True
                    else:
                        cos_bool[j] = False
            if signe < 0:
                for k in range(len(cos_hori)):
                    if (np.all(np.asarray(cos_hori[k]) < 0)) and (k not in liste_interdite):
                        cos_bool[k] = True
                    else:
                        cos_bool[k] = False

            # Vérification du critère 2 (temporel)
            # Limites horizontales (traces) de l'horizon actuel
            start = hori_combine[0][1]
            end = hori_combine[-1][1]
            # Limites verticales (samples) de l'horizon actuel
            # Si on ne rentre pas dans les boucles, c'est que tstart et/ou tend sont déjà définis
            if (len(left)==1) and (len(right)==0):
                tstart = time_hori[left[0]][0][0]
            elif (len(right)==1) and (len(left)==0):
                tend = time_hori[right[0]][-1][0]
            elif (len(right)==1) and (len(left)==1):
                tstart = time_hori[left[0]][0][0]
                tend = time_hori[right[0]][-1][0] 
            time_bool = np.copy(time_hori)
            for i in range(len(time_hori)):
                if (time_hori[i][-1][1]<start) and (tstart-(Tj)<=time_hori[i][-1][0]<=tstart+(Tj)) and (i not in liste_interdite):
                    time_bool[i] = True
                elif (time_hori[i][0][1]>end) and (tend-(Tj)<=time_hori[i][0][0]<=tend+(Tj)) and (i not in liste_interdite):
                    time_bool[i] = True
                else:
                    time_bool[i] = False

            # Vérification du critère 3 (Séparation)
            separ_bool = np.copy(time_hori)
            for i in range(len(time_hori)):
                if (time_hori[i][-1][1]<start) and (time_hori[i][-1][1]>=start-Lg) and (i not in liste_interdite):
                    separ_bool[i] = True
                elif (time_hori[i][0][1]>end) and (time_hori[i][0][1]<=end+Lg) and (i not in liste_interdite):
                    separ_bool[i] = True
                else:
                    separ_bool[i] = False

            # Combinaison des critères
            crit_prol = cos_bool & time_bool & separ_bool


        # Mise-à-jour de list_horizons
        liste_signe.append(signe)
        newlist_horizons.append(hori_combine)
        newlist_horizons_time.append(hori_combine_t)

        # Rendre invalide les horizons qui sont dans hori_combine(éviter de les placer 2 fois)
        # Désactiver les positions invalides dans cos_hori et time_hori aussi
        if (len(list_left) > 0) and (len(list_right) == 0):
            for m in list_left:
                liste_interdite.append(m)
            liste_interdite.append(horizon)
        elif (len(list_right) > 0) and (len(list_left) == 0):
            for n in list_right:
                liste_interdite.append(n)
            liste_interdite.append(horizon)
        elif (len(list_left) > 0) and (len(list_right) > 0):
            for m in list_left:
                liste_interdite.append(m)
            for n in list_right:
                liste_interdite.append(n)
            liste_interdite.append(horizon)
        elif (len(list_left) == 0) and (len(list_right) == 0):
            liste_interdite.append(horizon)
    
    return newlist_horizons, newlist_horizons_time, liste_signe

def horijoin_Dossi_rev(horizons,Cij_clone,twtij_clone,champ,Lg=False,Tj=False,min_length=False):
    
    # Trier les horizons en ordre de traces
    horizons_tri = sorted(horizons, key=lambda x: x[0][1])

    # Conversion des horizons
    # cos_hori -> De position à cosinus de la phase
    # time_hori -> De sample à temps
    cos_hori = []
    time_hori = []
    for hori in horizons_tri:
        row, cols = zip(*hori)
        cos = Cij_clone[row,cols].tolist()
        time = twtij_clone[row,cols].tolist()
        cos_hori.append(cos)
        time_hori.append(list(zip(time,cols)))
        # Pour être sûr???
        time_hori = sorted(time_hori, key=lambda x: x[0][1])
    cos_hori_clone = np.copy(cos_hori)
    time_hori_clone = np.copy(time_hori)

    # Vérification que tous les horizons contiennent des éléments de même signe
    for hori in cos_hori:
        if np.all(np.asarray(hori) > 0):
            continue
        elif np.all(np.asarray(hori) < 0):
            continue
        else:
            ind = cos_hori.index(hori)
            print("L'horizon {} contient des éléments de signes opposés".format(ind))
            break
    
    newlist_horizons = []
    newlist_horizons_time = []
    liste_interdite = []
    liste_signe = []
    # On itère sur chaque horizons
    for horizon in tqdm(range(len(horizons_tri))):

        # Vérifier que l'horizon n'est pas déjà utilisé
        if (horizon in liste_interdite):
            continue

        # Vérification du critère 1
        # Signe de l'horizon actuel
        cos_bool = np.copy(cos_hori)
        signe = np.sign(cos_hori[horizon][0])
        if signe > 0:
            for j in range(len(cos_hori)):
                if (np.all(np.asarray(cos_hori[j]) > 0)) and (np.all(np.asarray(cos_hori[j]) <= 1)) and (j not in liste_interdite):
                    cos_bool[j] = True
                else:
                    cos_bool[j] = False
        if signe < 0:
            for k in range(len(cos_hori)):
                if (np.all(np.asarray(cos_hori[k]) < 0)) and (k not in liste_interdite):
                    cos_bool[k] = True
                else:
                    cos_bool[k] = False

        # Vérification du critère 2 (temporel)
        # Limites horizontales (traces) de l'horizon actuel
        start = time_hori[horizon][0][1]
        end = time_hori[horizon][-1][1]
        # Limites verticales (samples) de l'horizon actuel
        tstart = time_hori[horizon][0][0]
        tend = time_hori[horizon][-1][0]
        time_bool = np.copy(time_hori)
        for i in range(len(time_hori)):
            if (time_hori[i][-1][1]<start) and (tstart-(Tj)<=time_hori[i][-1][0]<=tstart+(Tj)) and (i not in liste_interdite):
                time_bool[i] = True
            elif (time_hori[i][0][1]>end) and (tend-(Tj)<=time_hori[i][0][0]<=tend+(Tj)) and (i not in liste_interdite):
                time_bool[i] = True
            else:
                time_bool[i] = False

        # Vérification du critère 3 (Séparation)
        separ_bool = np.copy(time_hori)
        for i in range(len(time_hori)):
            if (time_hori[i][-1][1]<start) and (time_hori[i][-1][1]>=start-Lg) and (i not in liste_interdite):
                separ_bool[i] = True
            elif (time_hori[i][0][1]>end) and (time_hori[i][0][1]<=end+Lg) and (i not in liste_interdite):
                separ_bool[i] = True
            else:
                separ_bool[i] = False

        # Combinaison des critères
        crit_prol = cos_bool & time_bool & separ_bool

        hori_combine = horizons_tri[horizon]
        hori_combine_t = time_hori[horizon]
        list_left = []
        list_right = []
        # Boucle de prolongement
        while np.any(crit_prol):
            ind_prol = np.where(cos_bool & time_bool & separ_bool)[0]
            if len(ind_prol) == 0:
                continue
            pre_left = ind_prol[ind_prol < horizon]
            pre_right = ind_prol[ind_prol > horizon]

            # Critère 4 - Croisement
            # Gauche
            left = []
            if len(pre_left) > 0:
                if champ is None:
                    for i in range(len(pre_left)):
                        x_bool = cross_horijoin(pre_left[i],hori_combine_t,horizons_tri,time_hori,newlist_horizons_time,direction="left")
                        if not np.any(x_bool):
                                left.append(pre_left[i])
                else:
                    for i in range(len(pre_left)):
                        x_bool = cross_horijoin(pre_left[i],hori_combine_t,horizons_tri,time_hori,newlist_horizons_time,direction="left")
                        
                        ec_trace = hori_combine[0][1]-horizons_tri[pre_left[i]][-1][1]
                        ec_samp = hori_combine[0][0]-horizons_tri[pre_left[i]][-1][0]
                        pythg = np.sqrt(ec_trace**2 + ec_samp**2)
                        
                        if (pythg > 52) and (ec_samp > 3):
                            prop = cross_phase(signe,pre_left[i],hori_combine_t,time_hori,champ,direction="left")
                            if (not np.any(x_bool)) and (prop < 0.25):
                                left.append(pre_left[i])
                        elif (pythg > 152):
                            if (not np.any(x_bool)) and (prop < 0.75):
                                left.append(pre_left[i])
                        else:
                            if not np.any(x_bool):
                                left.append(pre_left[i])
            # Droite
            right = []
            if len(pre_right) > 0:
                if champ is None:
                    for j in range(len(pre_right)):
                        x_bool = cross_horijoin(pre_right[j],hori_combine_t,horizons_tri,time_hori,newlist_horizons_time,direction="right")
                        if not np.any(x_bool):
                                right.append(pre_right[j])
                else:
                    for j in range(len(pre_right)):
                        x_bool = cross_horijoin(pre_right[j],hori_combine_t,horizons_tri,time_hori,newlist_horizons_time,direction="right")
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

            # Pour chaque côté, conserver uniquement l'horizon le plus près de 
            # l'horizon actuel (2 axes)
            # Gauche
            if len(left) > 1:

                # Sélectionner l'horizon avec la moyenne verticale la plus près
                #moy_ref = np.mean([i[0] for i in hori_combine_t[:min_length]])
                #moy_left = []
                #for lefty in left:
                #    moy_left.append(np.mean([k[0] for k in horizons_tri[lefty]]))
                #dif = np.abs(moy_ref-moy_left)
                # Si aucun horizon se démarque, prendre celui qui est plus près
                #if len(set(dif)) == 1:
                traces = np.asarray([horizons_tri[i][-1][1] for i in left])
                times = np.asarray([time_hori[j][-1][0] for j in left])
                refx = hori_combine[0][1]
                reft = hori_combine_t[0][0]
                difx = refx-traces
                dift = reft-times
                pyt = np.sqrt(difx**2 + dift**2)
                prox = np.where(pyt==np.min(pyt))[0]
                #left = left[prox]
                # Avec hori_join_Dossi_rev
                left = [left[prox[0]]]
                #else:    
                #    pos = np.where(dif == np.min(dif))[0][0]
                #    left = [left[pos]]
                
            # Droite
            if len(right) > 1:
                
                # Sélectionner l'horizon avec la moyenne verticale la plus près
                #moy_ref = np.mean([i[0] for i in hori_combine_t[-min_length:]])
                #moy_right = []
                #for righty in right:
                #    moy_right.append(np.mean([k[0] for k in horizons_tri[righty]]))
                #dif = np.abs(moy_ref-moy_right)
                # Si aucun horizon se démarque, prendre celui qui est plus près
                #if len(set(dif)) == 1:
                traces = np.asarray([horizons_tri[i][0][1] for i in right])
                times = np.asarray([time_hori[j][0][0] for j in right])
                refx = hori_combine[-1][1]
                reft = hori_combine_t[-1][0]
                difx = traces-refx
                dift = times-reft
                pyt = np.sqrt(difx**2 + dift**2)
                prox = np.where(pyt==np.min(pyt))[0]
                #right = right[prox]
                # Avec hori_join_Dossi_rev
                right = [right[prox[0]]]
                #else:
                #    pos = np.where(dif == np.min(dif))[0][0]
                #    right = [right[pos]]

            # Critère 4 - Croisement
            # Gauche
            #if len(left) == 1:
            #    x_bool = cross_horijoin(left,hori_combine_t,horizons_tri,time_hori,newlist_horizons_time,direction="left")
            #    if np.any(x_bool):
            #        left = []
            # Droite
            #if len(right) == 1:
            #    x_bool = cross_horijoin(right,hori_combine_t,horizons_tri,time_hori,newlist_horizons_time,direction="right")
            #    if np.any(x_bool):
            #        right = [] 
            
            if (len(left) == 1) and (len(right) == 1):
                hori_combine = horizons_tri[left[0]] + hori_combine + horizons_tri[right[0]]
                hori_combine_t = time_hori[left[0]] + hori_combine_t + time_hori[right[0]]
                list_left.append(left[0])
                list_right.append(right[0])
            elif (len(left) == 1) and (len(right) == 0):
                hori_combine = horizons_tri[left[0]] + hori_combine
                hori_combine_t = time_hori[left[0]] + hori_combine_t
                list_left.append(left[0])
            elif (len(left) == 0) and (len(right) == 1):
                hori_combine = hori_combine + horizons_tri[right[0]]
                hori_combine_t = hori_combine_t + time_hori[right[0]]
                list_right.append(right[0])
            else:
                break
            hori_combine = sorted(hori_combine, key=lambda x: x[1])
            hori_combine_t = sorted(hori_combine_t, key=lambda x: x[1])
            # Revérifier les critères de prolongement
            # Vérification du critère 1
            # Signe de l'horizon actuel
            cos_bool = np.copy(cos_hori)
            if signe > 0:
                for j in range(len(cos_hori)):
                    if (np.all(np.asarray(cos_hori[j]) > 0)) and (np.all(np.asarray(cos_hori[j]) <= 1)) and (j not in liste_interdite):
                        cos_bool[j] = True
                    else:
                        cos_bool[j] = False
            if signe < 0:
                for k in range(len(cos_hori)):
                    if (np.all(np.asarray(cos_hori[k]) < 0)) and (k not in liste_interdite):
                        cos_bool[k] = True
                    else:
                        cos_bool[k] = False

            # Vérification du critère 2 (temporel)
            # Limites horizontales (traces) de l'horizon actuel
            start = hori_combine[0][1]
            end = hori_combine[-1][1]
            # Limites verticales (samples) de l'horizon actuel
            # Si on ne rentre pas dans les boucles, c'est que tstart et/ou tend sont déjà définis
            if (len(left)==1) and (len(right)==0):
                tstart = time_hori[left[0]][0][0]
            elif (len(right)==1) and (len(left)==0):
                tend = time_hori[right[0]][-1][0]
            elif (len(right)==1) and (len(left)==1):
                tstart = time_hori[left[0]][0][0]
                tend = time_hori[right[0]][-1][0] 
            time_bool = np.copy(time_hori)
            for i in range(len(time_hori)):
                if (time_hori[i][-1][1]<start) and (tstart-(Tj)<=time_hori[i][-1][0]<=tstart+(Tj)) and (i not in liste_interdite):
                    time_bool[i] = True
                elif (time_hori[i][0][1]>end) and (tend-(Tj)<=time_hori[i][0][0]<=tend+(Tj)) and (i not in liste_interdite):
                    time_bool[i] = True
                else:
                    time_bool[i] = False

            # Vérification du critère 3 (Séparation)
            separ_bool = np.copy(time_hori)
            for i in range(len(time_hori)):
                if (time_hori[i][-1][1]<start) and (time_hori[i][-1][1]>=start-Lg) and (i not in liste_interdite):
                    separ_bool[i] = True
                elif (time_hori[i][0][1]>end) and (time_hori[i][0][1]<=end+Lg) and (i not in liste_interdite):
                    separ_bool[i] = True
                else:
                    separ_bool[i] = False

            # Combinaison des critères
            crit_prol = cos_bool & time_bool & separ_bool


        # Mise-à-jour de list_horizons
        liste_signe.append(signe)
        newlist_horizons.append(hori_combine)
        newlist_horizons_time.append(hori_combine_t)

        # Rendre invalide les horizons qui sont dans hori_combine(éviter de les placer 2 fois)
        # Désactiver les positions invalides dans cos_hori et time_hori aussi
        if (len(list_left) > 0) and (len(list_right) == 0):
            for m in list_left:
                liste_interdite.append(m)
            liste_interdite.append(horizon)
        elif (len(list_right) > 0) and (len(list_left) == 0):
            for n in list_right:
                liste_interdite.append(n)
            liste_interdite.append(horizon)
        elif (len(list_left) > 0) and (len(list_right) > 0):
            for m in list_left:
                liste_interdite.append(m)
            for n in list_right:
                liste_interdite.append(n)
            liste_interdite.append(horizon)
        elif (len(list_left) == 0) and (len(list_right) == 0):
            liste_interdite.append(horizon)
    
    return newlist_horizons, newlist_horizons_time, liste_signe

def pick_dig_Dossi(dig_cas,iteration,digues,Cij,twtij,Tph,glace,long_min=5,tol_C=0.1,tolmin_t=0.6,tolmax_t=2,Lg=50,Tj=7):
    """
    Fonction pour faire le picking des digues à partir de l'algorithme basé sur la méthode de Dossi.
    Fonction à utiliser à l'intérieur de la boucle de pointé de la surface du sol.

    INPUT:
    - dig_cas: Entier indiquant le cas de positionnement des digues. 
            (0 = 1 digue au tout début + 1 à la toute fin)
            (1 = 1 digue au tout début seulement)
            (2 = 1 digue à la toute fin seulement)
            (3 = Pas de digues aux extrémités du radargramme)
    - iteration: Entier indiquant l'itération à laquelle l'algo est rendu. On itère sur le nombre
            de digues.
    - digues: Liste de tuples indiquant les positions des digues.
    - Cij: Matrice Cij originale ou clonée
    - twtij: Matrice twtij originale ou clonée
    - Tph: Estimation de la largeur temporelle d'une phase (Largeur d'une onde directe)
    - glace: Liste de positions verticales correspondant à la surface de la glace lissée.
    - long_min: Seuil permettant de filtrer les horizons trop courts après la détection d'horizons
            au niveau des digues.
    - tol_C: Paramètre utilisé pour filtrer éléments de faible amplitude dans la matrice Cij.
            (Utilisé pour la fonction horipick_Dossi)
    - tolmin_t/tolmax_t: Paramètres utilisés pour filtrer les phases trop courtes/faibles dans la
            matrice twtij. (Utilisé pour la fonction horipick_Dossi)
    - Lg: Espacement spatial maximal entre 2 horizons pour la jonction d'horizons.
    - Tj: Espacement temporel maximal entre 2 horizons pour la jonction d'horizons.

    OUTPUT:
    - x_dig: Positions horizontales des points sélectionnées automatiquement
    - t_dig: Positions verticales des points sélectionnés automatiquement
    """

    # Délimitation de la zone où on veut exercer le picking
    if dig_cas == 0:
        if iteration < (len(digues)-2):
            Cij_dig = Cij[:,digues[iteration+1][0]:digues[iteration+1][1]]
            twtij_dig = twtij[:,digues[iteration+1][0]:digues[iteration+1][1]]
    elif dig_cas == 1:
        if iteration < (len(digues)-1):
            Cij_dig = Cij[:,digues[iteration+1][0]:digues[iteration+1][1]]
            twtij_dig = twtij[:,digues[iteration+1][0]:digues[iteration+1][1]]
    elif dig_cas == 2:
        if iteration < (len(digues)-1):
            Cij_dig = Cij[:,digues[iteration][0]:digues[iteration][1]]
            twtij_dig = twtij[:,digues[iteration][0]:digues[iteration][1]]
    elif dig_cas == 3:
        Cij_dig = Cij[:,digues[iteration][0]:digues[iteration][1]]
        twtij_dig = twtij[:,digues[iteration][0]:digues[iteration][1]]

    # Filtrer les horizons avec des temps élevés (digues en surface)
    late_phase = np.argwhere(twtij_dig >= (int(np.max(glace))+2*Tph))
    if len(late_phase)>0:
        fil_Cij = 2*np.ones((1,len(late_phase)))
        fil_twtij = -1*np.ones((1,len(late_phase)))
        row, cols = zip(*late_phase)
        Cij_dig[row,cols] = fil_Cij
        twtij_dig[row,cols] = fil_twtij
    # Détection d'horizons
    hori_dig,hori_dig_tp,C_dig,t_dig,nosign = horipick_Dossi(Cij_dig,twtij_dig,Tph,tol_C=tol_C,tolmin_t=tolmin_t,tolmax_t=tolmax_t)
    # Enlever les horizons trop courts parmi les horizons détectés
    long_dig = [digue for digue in hori_dig if len(digue) > long_min]
    # Jonction d'horizons
    longer_dig,longer_digt,signs = horijoin_Dossi_rev(long_dig,C_dig,t_dig,champ=None,Lg=Lg,Tj=Tj,min_length=False)
    # Repérage de l'horizon le plus long
    lgst_dig = max(longer_digt, key=len)
    # Correction des positions
    dig_tr = [m[1]+digues[iteration+1][0] for m in lgst_dig]
    dig_temp = [n[0] for n in lgst_dig]
    # Interpolation
    x_dig = np.linspace(dig_tr[0],(dig_tr[-1])-1,dig_tr[-1]-dig_tr[0])
    f_dig = interpolate.interp1d(dig_tr,dig_temp,kind="linear")
    t_dig = f_dig(x_dig)

    return x_dig,t_dig

def c_statot_Dossi(data,dig_x,champ_x,dig_t,champ_t,GPS,offset,liss=75,resample=4,veloc=0.1):
    """
    Fonction pour fusionner les résultats des pickings sur les champs et sur les digues.
    Ce picking total est ensuite utiliser pour procéder à une correction statique.

    INPUT:
    - data: Marice de données originale
    - dig_x: Liste de listes indiquant les positions horizontales des points sélectionnés au
            niveau des digues
    - champ_x: Liste de listes indiquant les positions horizontales des points sélectionnés au
            niveau des champs
    - dig_t: Liste de listes indiquant les positions verticales des points sélectionnés au
            niveau des digues
    - champ_t: Liste de listes indiquant les positions verticales des points sélectionnés au
            niveau des champs
    - GPS: Coordonnées GPS des points de la ligne de GPR avec les élévations estimées à partir des
            données LIDAR
    - offset: Nombre de samples situés à des temps très hâtifs et qui sont retirés des données.
            (big_val -> ondes directes)
    - liss: Entier utilisé pour définir une fenêtre de lissage de la surface totale (champ + digues)
            (Défaut = 75)
    - resample: Entier représentant le nombre de samples que l'on veut ajouter pour adoucir le
            résultat de la correction statique (Défaut = 4, fait passer de 0,1,... à 0,0.25,0.50,0.75,1,...)
    - veloc: Estimation de la vitesse des ondes dans la glace
            (Défaut = 0.1 m/ns)
    
    OUTPUT:
    - 
    """
    if (dig_x != None) and (dig_t != None):
        # Concaténer toutes les listes dans le bon ordre
        pos_hori = []
        pos_verti = []
        for i in range(len(dig_x)):
            pos_hori.append(champ_x[i])
            pos_hori.append(dig_x[i])
            pos_verti.append(champ_t[i])
            pos_verti.append(dig_t[i])
        pos_hori.append(champ_x[-1])
        pos_verti.append(champ_t[-1])
        pos_hori = [n for m in pos_hori for n in m]
        pos_verti = [p for o in pos_verti for p in o]

        # Interpolation
        x_tot = np.linspace(pos_hori[0],pos_hori[-1],pos_hori[-1]-pos_hori[0]+1)
        f_tot = interpolate.interp1d(pos_hori,pos_verti,kind="linear")
        temps_tot = f_tot(x_tot)

        # Lissage
        tottraces = len(x_tot)
        tot_liss = np.zeros(len(x_tot))
        halfwid_tot = int(np.ceil(liss/2))
        tot_liss[:halfwid_tot+1] = np.mean(temps_tot[:halfwid_tot+1])
        tot_liss[tottraces-halfwid_tot:] = np.mean(temps_tot[tottraces-halfwid_tot:])
        for lt in range(halfwid_tot,tottraces-halfwid_tot+1):
            tot_liss[lt] = np.mean(temps_tot[lt-halfwid_tot:lt+halfwid_tot])
    else:
        #print("ALLO")
        #print(len(champ_x))
        x_tot = np.linspace(champ_x[0],champ_x[-1],champ_x[-1]-champ_x[0]+1)
        #print(len(x_tot))
        tot_liss = champ_t
        #print(len(tot_liss))

    # Positionnement des données GPS sur le radargramme
    GPS_ligne = GPS[:,2]
    elev_max = np.max(GPS_ligne)
    GPSns = np.copy(GPS_ligne)
    GPSns = ((elev_max - GPSns)/veloc)
    GPS_ice = GPSns[int(x_tot[0]):int(x_tot[-1])+1]
    elevmax_ice = np.min(tot_liss)
    ind_ice = np.where(tot_liss == elevmax_ice)[0][0]
    diff = GPS_ice[ind_ice] - tot_liss[ind_ice]
    GPS_ice_rad = GPS_ice - diff

    # Détermination des décalages à appliquer
    shift_list_samp = (tot_liss - GPS_ice_rad)

    if resample > 1:
        # Resampling
        shift_list_samp_res = np.round(shift_list_samp*resample)/resample
        data = data.astype("float32")
        x_mat = np.linspace(0,data.shape[0]-1,num=data.shape[0])
        f_resamp = interpolate.interp1d(x_mat,data,kind="linear",axis=0)
        x_resamp = np.linspace(0,x_mat[-1],num=(resample*len(x_mat)-(resample-1)))
        dat_resamp = f_resamp(x_resamp)
        dat_resamp = dat_resamp.astype("float32")

        # Correction statique
        #shift_min_res = int(np.amin(shift_list_samp_res)//(1/resample))
        #shift_max_res = int(np.amax(shift_list_samp_res)//(1/resample))
        #pillow_res = np.zeros((abs(shift_max_res), len(x_tot)))

        #if any(shifty < 0 for shifty in shift_list_samp):
        #    pillow_res_inf = np.zeros((abs(shift_min_res), len(x_tot)))
        #    newdata_res = np.vstack((pillow_res,np.zeros((data.shape[0],len(x_tot))), pillow_res_inf))
        #else:
        #    newdata_res = np.vstack((pillow_res,np.zeros((data.shape[0],len(x_tot)))))
        #if any(shifty < 0 for shifty in shift_list_samp_res):
        shift_min_res = int(np.amin(shift_list_samp_res)//(1/resample))
        #else:
        #    shift_min_res = 0
        shift_max_res = int(np.amax(shift_list_samp_res)//(1/resample))
        pillow_res = 8192*np.ones((abs(shift_max_res), len(x_tot)))
        pillow_res_inf = 8192*np.ones((abs(shift_min_res), len(x_tot)))
        newdata_res = np.vstack((pillow_res,np.zeros((dat_resamp.shape[0],len(x_tot))), pillow_res_inf))
        for i in range(len(x_tot)):
            newdata_res[int(abs(shift_max_res))-int(shift_list_samp_res[i]//(1/resample)):int(abs(shift_max_res))-int(shift_list_samp_res[i]//(1/resample))+dat_resamp.shape[0],i] = dat_resamp[:,i+int(x_tot[0])]
        
    elif resample == 1:
        shift_min_res = int(np.amin(shift_list_samp))
        shift_max_res = int(np.amax(shift_list_samp))
        pillow_res = np.ones((abs(shift_max_res), len(x_tot)))

        if any(shifty < 0 for shifty in shift_list_samp):
            pillow_res_inf = np.ones((abs(shift_min_res), len(x_tot)))
            newdata_res = np.vstack((pillow_res,np.zeros((data.shape[0],len(x_tot))), pillow_res_inf))
        else:
            newdata_res = np.vstack((pillow_res,np.zeros((data.shape[0],len(x_tot)))))

        for i in range(len(x_tot)):
            newdata_res[int(abs(shift_max_res))-int(shift_list_samp[i]):int(abs(shift_max_res))-int(shift_list_samp[i])+data.shape[0],i] = data[:,i+int(x_tot[0])]
        
    # Nouveau positionnement des coordonnées GPS sur le radargramme
    GPS_final = (GPS_ice_rad)*resample + pillow_res.shape[0] + offset*resample
    tot_liss_rad = tot_liss*resample + offset*resample

    return newdata_res, GPS_final, x_tot, tot_liss_rad

def prep_picking_NS(data):
    """
    Fonction pour préparer les données des fichiers Nord-Sud au picking de la surface du sol
    """
    # Running average pour mettre en évidence les réflecteurs plats
    data_rng = pflt.running_avg(data,int(data.shape[1]/20))
    # Calcul des moyennes horizontales et repérage des pics positifs
    moy = np.mean(data_rng,axis=1)
    moy = moy.clip(min=0)
    moy = moy/np.max(moy)

    # Estimation de la dérivée sur la moyenne (diff. finies)
    deriv_moy = []
    for i in range(len(moy)-1):
        if i == 0:
            deriv_moy.append(moy[i])
        else:
            deriv_moy.append((moy[i+1]-moy[i-1])/2)
    # Calcul de la médiane
    med_deriv = np.median(deriv_moy)

    # Trouver toutes les séquences de nombres consécutifs dont la moyenne dépasse mediane+0.01
    zone = np.where(moy > med_deriv+0.01)[0]
    seq = []
    for k, g in groupby(enumerate(zone), lambda ix : ix[0] - ix[1]):
        seq.append(list(map(itemgetter(1),g)))

    # Supprimer les séquences ayant au moins 1 valeur moyenne supérieure à 0.2
    new_seq = []
    for liste in seq:
        verif = np.asarray(moy[liste[0]:liste[-1]+1]) > 0.2
        if np.any(verif):
            continue
        if seq.index(liste) > 3:
            continue
        else:
            new_seq.append(liste)
    del seq

    # Création d'un mask avec lequel multiplier la matrice originale et la matrice
    # convertie en phase
    mask = np.zeros((data.shape[0],1))
    mask[new_seq[0][0]-4:new_seq[-1][0]-3] = 1

    return mask, data_rng

def prep_picking_NS_juin21(data,fen,Tph):
    """
    Fonction pour préparer les données des fichiers Nord-Sud au picking de la surface du sol
    """
    # Running average pour mettre en évidence les réflecteurs plats
    data_rng = pflt.running_avg(data,fen)
    # Calcul des moyennes horizontales et repérage des pics positifs
    moy = np.mean(data_rng,axis=1)
    moy = moy.clip(min=0)
    moy = moy/np.max(moy)

    # Estimation de la dérivée sur la moyenne (diff. finies)
    deriv_moy = []
    for i in range(len(moy)-1):
        if i == 0:
            deriv_moy.append(moy[i])
        else:
            deriv_moy.append((moy[i+1]-moy[i-1])/2)
    # Calcul de la médiane
    med_deriv = np.median(deriv_moy)

    # Trouver toutes les séquences de nombres consécutifs dont la moyenne dépasse mediane+0.01
    zone = np.where(moy > med_deriv+0.01)[0]
    seq = []
    for k, g in groupby(enumerate(zone), lambda ix : ix[0] - ix[1]):
        seq.append(list(map(itemgetter(1),g)))

    # Supprimer les séquences ayant au moins 1 valeur moyenne supérieure à 0.2
    new_seq = []
    for liste in seq:
        #verif = np.asarray(moy[liste[0]:liste[-1]+1]) > 0.2
        #if np.any(verif):
        #    continue
        if (seq.index(liste) < 1) or (seq.index(liste) > 3):
            continue
        else:
            new_seq.append(liste)
    del seq

    # Création d'un mask avec lequel multiplier la matrice originale et la matrice
    # convertie en phase
    mask = np.zeros((data.shape[0],1))
    mask[new_seq[0][0]:new_seq[-1][0]-Tph] = 1

    return mask


