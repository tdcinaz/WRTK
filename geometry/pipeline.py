import nibabel as nib
import argparse
import os
from os.path import join
import logging
import pyvista as pv
#import pymeshfix
from geometry_master_original import (
    compute_label_volumes,
    nifti_to_pv_image_data,
    compute_skeleton,
    extract_start_and_end_voxels,
    create_cow,
    filter_out_artery_points,
    filter_artery_by_radius,
)
from skimage.morphology import skeletonize

logging.basicConfig(level=logging.INFO)

def pipeline(
    args: argparse.Namespace,
    prefix: str,
    filename: str,
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

    #patient_input_path = join(args.input_folder, f"{prefix}_{patient_ID}.nii.gz")
    patient_input_path = join(args.input_folder, filename)

    #patient_output_path = join(args.output, f"{prefix}_{patient_ID}")
    patient_output_path = join(args.output, f"seg_{filename}")

    os.makedirs(patient_output_path, exist_ok=True)

    # 1) Load the NIfTI with nibabel
    nifti_img: nib.Nifti1Image = nib.load(patient_input_path)

    volume_dict = compute_label_volumes(nifti_img)
    logging.info(f"++++ : Volume for each label: {volume_dict}")

    pv_image = nifti_to_pv_image_data(nifti_img)
    #pv_image.plot(scalars="Scalars", volume=True)

    #logging.info(f"++++ : Image {prefix}_{patient_ID} loaded")

    skeleton: pv.PolyData = compute_skeleton(nifti_img)
    
    filtered_skeleton = filter_out_artery_points(skeleton, 10)
    refiltered_skeleton: pv.PolyData = filter_out_artery_points(filtered_skeleton, 13)

    cleaned_skeleton = filter_artery_by_radius(refiltered_skeleton, 4.0, 0.6)
    recleaned_skeleton = filter_artery_by_radius(cleaned_skeleton, 6.0, 0.6)

    extract_start_and_end_voxels(nifti_img, pv_image, recleaned_skeleton)

    create_cow(recleaned_skeleton, patient_ID)
    #network = spline_interpolation(skeleton)

    #print(spline_dict)

    #angles_dict = extract_angles(network, skeleton)

    #for key, value in angles_dict.items():
    #    print(f"{key}: {value}")

    skeleton_file = join(patient_output_path, f"{prefix}_{patient_ID}_skeleton.vtp")

    #refiltered_skeleton.save(skeleton_file)

    #logging.info(f"Surface extracted and saved to '{skeleton_file}'")