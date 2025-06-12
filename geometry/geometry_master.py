import sys
import numpy as np
import nibabel as nib
import vtk
from vtkmodules.util import numpy_support
import pyvista as pv
import pyacvd
import logging
from collections import defaultdict, deque
from scipy.spatial import cKDTree
import os
import re

def nifti_to_vtk_image_data(nifti_img):
    """
    Convert a 3D NIfTI image to vtkImageData.

    Args:
        nifti_img (nibabel.Nifti1Image): The input NIfTI image.
        dtype (numpy.dtype): Data type for the output array.

    Returns:
        vtk.vtkImageData: The image data in VTK format.
    """
    # 1) Get the NumPy array from the NIfTI
    data_array = nifti_img.get_fdata(dtype=np.float32)

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

def compute_label_volumes(img, label_range=range(1, 14)):
    # Load the image
    data = img.get_fdata()
    
    # Get the voxel size from the header (voxel volume in mm³)
    voxel_volume = np.prod(img.header.get_zooms())
    print(f"Voxel volume: {voxel_volume:.2f} mm³")

    # Compute volumes for each label
    volumes = {}
    for label in label_range:
        voxel_count = np.sum(data == label)
        volumes[label] = voxel_count * voxel_volume

    return volumes

def extract_labeled_surface_from_volume(
    input_vtk_image: vtk.vtkImageData,
) -> vtk.vtkPolyData:
    """
    Extract a multi-labeled surface using vtkSurfaceNets3D.
    The output polydata has a cell-data array 'BoundaryLabels'
    indicating which label is adjacent to either side of the cell.

    Args:
        nifti_file (str): Path to a labeled NIfTI image with integer labels.

    Returns:
        vtk.vtkPolyData: A polydata containing the labeled surface
    """

    surface_net = vtk.vtkSurfaceNets3D()
    surface_net.SetInputData(input_vtk_image)
    surface_net.SetBackgroundLabel(0)
    surface_net.SetOutputStyleToDefault()
    surface_net.GenerateLabels(14, 1, 14)
    #surface_net.SmoothingOff()
    #surface_net.SetOutputMeshTypeToQuads()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(surface_net.GetOutputPort())
    cleaner.Update()

    logging.info(f"    ++++ : Total Surface Cells: {cleaner.GetOutput().GetNumberOfCells()}")

    labeled_surface = cleaner.GetOutput()

    return labeled_surface


def save_surface_to_vtp(surface, filename):
    """
    Save a vtkPolyData surface to a VTP file using VMTK's surface writer.

    Args:
        surface (vtk.vtkPolyData): The surface (mesh) to be saved.
        filename (str): The output .vtp file path.
    """

    writer = vtk.vtkXMLPolyDataWriter()
    
    # Set the output file name
    writer.SetFileName(filename)
    
    # Connect the input polydata to the writer
    writer.SetInputData(surface)
    
    # Optionally, set to ASCII or Binary mode
    # writer.SetDataModeToAscii()  # Uncomment to save as ASCII
    # writer.SetDataModeToBinary() # Uncomment to save as Binary (default)
    
    # Write the file
    writer.Write()

