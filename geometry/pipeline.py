import nibabel as nib
import argparse
import os
from os.path import join
import logging
import pyvista as pv
from geometry_master import (
    compute_label_volumes,
    nifti_to_pv_image_data,
    extract_labeled_surface_from_volume,
    extract_individual_surfaces
)

logging.basicConfig(level=logging.INFO)

def pipeline(
    args: argparse.Namespace,
    prefix: str,
):
    """
    Main workflow:
      1) Load an existing NIfTI file from disk.
      2) Extract the labeled surface vtkPolyData
      3) Save the surface to a .vtp file.

    Args:
        input_nii_file (str): Path to the input NIfTI file.
        output_vtp_file (str): Desired output VTP file path.
    """

    patient_ID = args.patient_ID

    patient_input_path = join(args.input_folder, f"{prefix}_{patient_ID}.nii.gz")

    patient_output_path = join(args.output, f"{prefix}_{patient_ID}")
    os.makedirs(patient_output_path, exist_ok=True)

    # 1) Load the NIfTI with nibabel
    nifti_img: nib.Nifti1Image = nib.load(patient_input_path)

    volume_dict = compute_label_volumes(nifti_img)
    logging.info(f"++++ : Volume for each label: {volume_dict}")

    vtk_image = nifti_to_pv_image_data(nifti_img)
    logging.info(f"++++ : Image {prefix}_{patient_ID} loaded")

    # 2) Extract surface

    logging.info(f"++++ : Extracting surface")
    labeled_polydata = extract_labeled_surface_from_volume(vtk_image)

    repaired_polydata = extract_individual_surfaces(labeled_polydata)

    repaired_polydata.plot(scalars='BoundaryLabels')
    
    surface_file = join(patient_output_path, f"{prefix}_{patient_ID}_surface.vtp")

    repaired_polydata.save(surface_file)

    logging.info(f"Surface extracted and saved to '{surface_file}'")