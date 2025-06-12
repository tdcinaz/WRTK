import vtk
from vtk import vtkNIFTIImageReader
from vtkmodules.util import numpy_support
import pyvista as pv
import numpy as np
import nibabel as nib
from pyvista import examples
import sys

path = "tests/input/topcow/labelsTr/topcow_ct_001.nii.gz"

image = nib.load(path)

data_array = nib.nifti.get_fdata(dtype=np.float32)

# Ensure it's contiguous in memory
data_array = np.ascontiguousarray(data_array)

# 2) Retrieve voxel spacing from the header
#    If it's a 4D image, nibabel header might return 4 zooms, so we take only the first three.
pixdim = nifti_img.header.get_zooms()[:3]

# 3) Extract the translation (origin) from the affine
affine = nifti_img.affine
origin = affine[:3, 3]

# 4) Create vtkImageData
vtk_image = vtk.vtkImageData()

# The shape of data_array is (Nz, Ny, Nx).
# VTK expects SetDimensions in the order (Nx, Ny, Nz).
Nz, Ny, Nx = data_array.shape
vtk_image.SetDimensions(Nx, Ny, Nz)

# Assign spacing and origin
vtk_image.SetSpacing(pixdim[2], pixdim[1], pixdim[0])
vtk_image.SetOrigin(origin[2], origin[1], origin[0])

# 5) Wrap the NumPy array into a vtkFloatArray
vtk_array = vtk.vtkFloatArray()
vtk_array.SetNumberOfComponents(1)
vtk_array.SetNumberOfTuples(Nx * Ny * Nz)
vtk_array.SetName("Scalars")

# The trick: point VTK to our array’s memory using SetVoidArray
#vtk_array.SetVoidArray(data_array, data_array.size, 1)

flat_data = data_array.ravel(order='C')  # Flatten to 1D
vtk_array = numpy_support.numpy_to_vtk(num_array=flat_data, deep=True)
vtk_array.SetName("Scalars")

# 6) Attach the vtkArray to vtkImage
vtk_image.GetPointData().SetScalars(vtk_array)

print("data_array.shape =", data_array.shape)
print("Nx, Ny, Nz =", Nx, Ny, Nz)


return vtk_image


'''mesh = examples.download_armadillo()
mesh['scalars'] = mesh.points[:,1]
mesh.plot(cpos='xy', cmap='plasma')
'''