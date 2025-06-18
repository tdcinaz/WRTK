import vtk
from vtk import vtkNIFTIImageReader
from vtkmodules.util import numpy_support
import pyvista as pv
import numpy as np
import nibabel as nib
from pyvista import examples
import sys

standardAdj = {
        1.0: (2.0, 3.0),
        2.0: (1.0, 8.0),
        3.0: (1.0, 9.0),
        4.0: (5.0, 8.0, 11.0),
        5.0: (4.0,),
        6.0: (7.0, 9.0, 12.0),
        7.0: (6.0,),
        8.0: (2.0, 4.0),
        9.0: (3.0, 6.0),
        10.0: (11.0, 12.0),
        11.0: (4.0,),
        12.0: (6.0,)
    }

vessel_labels = {
        1.0: "Basillar",
        2.0: "L-PCA",
        3.0: "R-PCA",
        4.0: "L-ICA",
        5.0: "L-MCA",
        6.0: "R-ICA",
        7.0: "R-MCA",
        8.0: "L-Pcom",
        9.0: "R-Pcom",
        10.0: "Acom",
        11.0: "L-ACA",
        12.0: "R-ACA",
}

angle_pairs = []

for key in standardAdj.keys():
    for vessel in standardAdj[key]:
        pair = [key, vessel]
        if [vessel, key] not in angle_pairs:
            angle_pairs.append(pair)

angles = {}
count = 0
