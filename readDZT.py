"""
--- Utilisation du module readgssi pour lire des fichiers de format .DZT ---
Source: https://readgssi.readthedocs.io/en/stable/

Samuel Mongeau-Lachance: 25 mars 2020
"""

from readgssi import readgssi

def lect_DZT(fichier, timezero):
    """
    INPUT:
    fichier: Parcours (format string) menant au fichier enregistré
        sur l'ordinateur.
    timezero: Liste d'entiers. Pour cette application -> [2].
        Peut changer d'une application à l'autre, vérifier timezero
        dans le header (hdr). 
    """
    hdr, arrs, gps = readgssi.readgssi(infile=fichier, zero=timezero)
    if gps:
        return (hdr, arrs[0], gps)
    else:
        return (hdr, arrs[0])

# Test 
if __name__ == "__main__":
    header, data = lect_DZT('/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/Blandford_GPR_2017/GPR-Blandford-H2017/GPR-Blandford-H2017/GPR/GRID____085.3DS/FILE____001.DZT', [2])
    print(header["ns_per_zsample"])