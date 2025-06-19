import argparse
import logging
import os
from os.path import join
import nibabel as nib
import numpy as np

from tof_master import (
    create_willis_cube,
    resample,
    apply_transform,
    autobox_image,
    crop_mask_like,
    brain_extract,
    crop_to_roi_cube_harvard_data,
    coregister_ct_mr,
)

logging.basicConfig(level=logging.INFO)

def full_pipeline(
    args: argparse.Namespace,
    prefix: str,
):
    """Compute the preprocessing, prediction, and metrics.

    Args:
        args (argparse.Namespace): Command line arguments.
        prefix (str): File name prefix.
        option (int): If option == 2, WB segmentation is computed,
        labels_path (str): Path to labels nifti file to use instead of computing one,
        vessels_path (str): Path leading to vessels binary image instead of computing one,
    """

    patient_ID = args.patient_ID
    sex = args.sex


    resolution_nn = (0.625, 0.625, 0.625)
    nn_resolution_path = join(args.output, "nn_space")
    os.makedirs(nn_resolution_path, exist_ok=True)
    original_path = join(args.output, "original_space")
    os.makedirs(original_path, exist_ok=True)

    template_path = join(args.template_path, f"AVG_TOF_MNI_SS_down.nii.gz")
    template_sphere_path = join(args.template_path, f"willis_sphere_down.nii.gz")

    logging.info("1. ===> Aligning Scan Orientation <===")

    CT_path = join(args.input_folder, f"stroke_{sex}_{patient_ID}.nii.gz")

    logging.info("   1.1 ++++ : Aligning CT Scan")
    ct_raw: nib.Nifti1Image = nib.load(CT_path)
    logging.info(f"      1.1.1 ++++ : Original CT orientation: {nib.aff2axcodes(ct_raw.affine)}")
    ct = nib.as_closest_canonical(ct_raw)
    logging.info(f"      1.1.2 ++++ : Transformed CT orientation: {nib.aff2axcodes(ct.affine)}")
    reorient_ct_file = join(original_path, f"{prefix}_reorient_ct_{patient_ID}.nii.gz")
    ct.to_filename(reorient_ct_file)


    logging.info("2. ===> Computing Bounding Boxes <===")

    logging.info("   2.1 ++++ : Autoboxing CT Scan")
    ct_autobox_file = join(original_path, f"{prefix}_autobox_ct_{patient_ID}.nii.gz")
    autobox_image(reorient_ct_file, ct_autobox_file, pad=6)


    logging.info("3. ===> Resampling to Neural Network Resolution <===")

    logging.info("   3.1 ++++ : Resampling CT Scan")
    ct_resample_file = join(nn_resolution_path, f"{prefix}_ct_resampled_{patient_ID}.nii.gz")
    ct_nii_attributes = resample(ct_autobox_file, ct_resample_file, resolution=resolution_nn)

    logging.info("   3.3 ++++ : Resampling CT Segmentation Mask")

    logging.info("4. ===> Creating a Brain Mask of NN Resolution <===")

    logging.info("   4.1 ++++ : Skull Stripping CT")
    ct_mask_file = join(nn_resolution_path, f"{prefix}_ct_mask_{patient_ID}.nii.gz")
    ct_brain_mask_file = join(nn_resolution_path, f"{prefix}_ct_SS_RegistrationImage_{patient_ID}.nii.gz")
    brain_extract(ct_resample_file, ct_mask_file, ct_brain_mask_file, "ct")

    logging.info("5. ===> Finding Circle of Willis masking cube <===")

    logging.info("   5.1 ++++ : Creating Willis Cube for CT")
    _, ct_cube = create_willis_cube(
            ct_brain_mask_file,
            nn_resolution_path,
            template_path,
            template_sphere_path,
        )
    
    logging.info("   5.2 ++++ : Saving Willis Cube for CT")
    ct_cube_file = join(nn_resolution_path, f"{prefix}_ct_cube_{patient_ID}.nii.gz")
    nib.Nifti1Image(ct_cube, ct_nii_attributes.affine, ct_nii_attributes.header).to_filename(ct_cube_file)

    logging.info("6. ===> Cropping and re-anchoring scan data <===")

    logging.info("   6.1 ++++ : Cropping an re-anchoring CT")
    ct_cropped_file = join(nn_resolution_path, f"{prefix}_ct_cropped_{patient_ID}.nii.gz")

    # ----------------------------------------------------------
    # 2. build a new affine for the *whole* MR volume
    #    so that voxel (imin,jmin,kmin) becomes the new origin
    # ----------------------------------------------------------
    ct_full: nib.Nifti1Image = nib.load(ct_resample_file)

    ct_new_affine = ct_full.affine.copy()

    ct_full_data = ct_full.get_fdata(dtype=np.float32, caching="unchanged")

    ct_reanchored_file = join(nn_resolution_path, f"{prefix}_ct_reanchored_{patient_ID}.nii.gz")
    nib.Nifti1Image(ct_full_data, ct_new_affine, ct_full.header).to_filename(ct_reanchored_file)

    aligned_ct_cube = join(nn_resolution_path, f"{prefix}_ct_aligned_cube_{patient_ID}.nii.gz")

    ct_cropped_file = f"tests/output/harvard_data_cropped/stroke/_ct_cropped_{patient_ID}_{sex}.nii.gz"

    crop_to_roi_cube_harvard_data(ct_resample_file, ct_cube_file, ct_cropped_file)
