import vtk
from vtk import vtkNIFTIImageReader
from vtkmodules.util import numpy_support
import pyvista as pv
import numpy as np
import nibabel as nib
from pyvista import examples
import sys

path = "tests/input/topcow/labelsTr/topcow_ct_001.nii.gz"

img = nib.load(path)
image_data = img.get_fdata()
integer_labels = image_data.astype(int)

reader = pv.get_reader(path)

mesh = reader.read()

mesh.plot(scalars = integer_labels)

#print(mesh.GetCellData())
#print(mesh.GetPointData())

#transformed_mesh = mesh.threshold(1)

#transformed_mesh.plot(scalars=integer_labels)