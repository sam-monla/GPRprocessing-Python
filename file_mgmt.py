"""
Functions for managing .DZT and .las files 

readDZT source: https://readgssi.readthedocs.io/en/stable/

"""
import numpy as np
from readgssi import readgssi
import laspy
from pyproj import Proj, transform

def readDZT(file, timezero=[2]):
    """
    INPUT:
    file: File location ("C:\Files\GPR.DZT")
    timezero: Integers list. For this project -> [2].
        Check for timezero in the header file of the GPR datas. 
    """
    hdr, arrs, gps = readgssi.readgssi(infile=file, zero=timezero)
    if gps:
        return (hdr, arrs[0], gps)
    else:
        return (hdr, arrs[0])

def readLas(file, epsg_ini="epsg:2145", epsg_end="epsg:32618"):
    '''
    Opens .las files in Python and converts the initial geodetic datum in the desired
    one. Default: From EPSG:2145 (NAD83(CSRS98) / MTM zone 8, Quebec, Canada) to 
    EPSG:32618 (WGS 84 / UTM zone 18N).

    To see other EPSG codes:
    https://spatialreference.org/ref/epsg/?search=utm+18N&srtext=Search

    INPUTS
    - file: File location ("C:\Files\Lidar.las")
    - epsg_ini: Initial Geodetic Datum (String)
    - epsg_fin: Desired Geodetic Datum (String)
    OUTPUTS
    - Numpy array. Each line is a coordinate (east, north, elevation)
    '''
    inFile = laspy.file.File(file, mode="r")

    # Extract the data, the scale factor and the offset from the .las file
    x_dim = inFile.X
    y_dim = inFile.Y
    z_dim = inFile.Z
    scale = inFile.header.scale
    offset = inFile.header.offset

    # Apply the scale factor and the offset for correct positionning
    pts_x = x_dim*scale[0] + offset[0]
    pts_y = y_dim*scale[1] + offset[1]
    pts_z = z_dim*scale[2] + offset[2]

    inProj = Proj(epsg_ini)
    outProj = Proj(epsg_end, preserve_units=True)
    # Geodetic Datum conversion
    ptsx_2, ptsy_2, ptsz_2 = transform(inProj, outProj, pts_x, pts_y, pts_z)
    # Making a numpy array in the form (east, north, elevation)
    coordLas = np.hstack((np.array([ptsx_2]).T, np.array([ptsy_2]).T, np.array([ptsz_2]).T))

    return coordLas