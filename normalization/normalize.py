import argparse
import logging
import os
from collections import OrderedDict
from os.path import join, isfile
from typing import Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from tof_master import (
    centerline_transform,
    create_willis_cube,
    find_willis_center,
    hysteresis_thresholding_cube,
    hysteresis_thresholding_brain,
    mask_image,
    resample,
    extend_markers,
    extract_vessels_ved,
    correct_watershed,
    apply_transform,
    autobox_image,
    crop_mask_like,
    brain_extract,
    crop_to_roi_cube,
    coregister_ct_mr,
)


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
    scans = os.listdir(join(args.input_folder, "imagesTr"))
    labels = os.listdir(join(args.input_folder, "labelsTr"))

    for scan in scans:
        if patient_ID in scan[0:13]:
            if "ct" in scan:
                CT_path = join(args.input_folder, "imagesTr", scan)
            elif "mr" in scan:
                MR_path = join(args.input_folder, "imagesTr", scan)

    for label in labels:
        if patient_ID in label[0:13]:
            if "ct" in label:
                CT_label_path = join(args.input_folder, "labelsTr", label)
            elif "mr" in label:
                MR_label_path = join(args.input_folder, "labelsTr", label)

    resolution_nn = (0.625, 0.625, 0.625)
    nn_resolution_path = join(args.output, "nn_space")
    os.makedirs(nn_resolution_path, exist_ok=True)
    original_path = join(args.output, "original_space")
    os.makedirs(original_path, exist_ok=True)

    template_path = join(args.template_path, f"AVG_TOF_MNI_SS_down.nii.gz")
    template_sphere_path = join(args.template_path, f"willis_sphere_down.nii.gz")

    ct_raw: nib.Nifti1Image = nib.load(CT_path)
    mr_raw: nib.Nifti1Image = nib.load(MR_path)
    ct_seg_raw: nib.Nifti1Image = nib.load(CT_label_path)
    mr_seg_raw: nib.Nifti1Image = nib.load(MR_label_path)

    logging.info(f"Original CT orientation: {nib.aff2axcodes(ct_raw.affine)}")
    logging.info(f"Original MR orientation: {nib.aff2axcodes(mr_raw.affine)}")


    ct = nib.as_closest_canonical(ct_raw)
    mr = nib.as_closest_canonical(mr_raw)
    ct_seg = nib.as_closest_canonical(ct_seg_raw)
    mr_seg = nib.as_closest_canonical(mr_seg_raw)
    
    logging.info(f"Transformed CT orientation: {nib.aff2axcodes(ct.affine)}")
    logging.info(f"Transformed MR orientation: {nib.aff2axcodes(mr.affine)}")
    
    reorient_ct_file = join(original_path, f"{prefix}_reorient_ct.nii.gz")
    reorient_mr_file = join(original_path, f"{prefix}_reorient_mr.nii.gz")
    ct.to_filename(reorient_ct_file)
    mr.to_filename(reorient_mr_file)

    seg_ct_reorient_file = join(original_path, f"{prefix}_seg_reorient_ct.nii.gz")
    seg_mr_reorient_file = join(original_path, f"{prefix}_mr_reorient_ct.nii.gz")
    ct_seg.to_filename(seg_ct_reorient_file)
    mr_seg.to_filename(seg_mr_reorient_file)

    ct_autobox_file = join(original_path, f"{prefix}_autobox_ct.nii.gz")
    mr_autobox_file = join(original_path, f"{prefix}_autobox_mr.nii.gz")
    autobox_image(reorient_ct_file, ct_autobox_file, pad=6)
    autobox_image(reorient_mr_file, mr_autobox_file, pad=6)

    ct_seg_autobox_file = join(original_path, f"{prefix}_seg_autobox_ct.nii.gz")
    mr_seg_autobox_file = join(original_path, f"{prefix}_seg_autobox_mr.nii.gz")
    crop_mask_like(seg_ct_reorient_file, ct_autobox_file, ct_seg_autobox_file)
    crop_mask_like(seg_mr_reorient_file, mr_autobox_file, mr_seg_autobox_file)

    logging.info("1. ===> Preprocessing <===")
    logging.info("   1.2 ++++ : Resampling to Neural Network Resolution")

    ct_resample_file = join(nn_resolution_path, f"{prefix}_ct_resampled.nii.gz")
    mr_resample_file = join(nn_resolution_path, f"{prefix}_mr_resampled.nii.gz")
    ct_nii_attributes = resample(ct_autobox_file, ct_resample_file, resolution=resolution_nn)
    mr_nii_attributes = resample(mr_autobox_file, mr_resample_file, resolution=resolution_nn)

    ct_seg_resample_file = join(nn_resolution_path, f"{prefix}_ct_seg_resampled.nii.gz")
    mr_seg_resample_file = join(nn_resolution_path, f"{prefix}_mr_seg_resampled.nii.gz")
    ct_seg_attributes = resample(ct_seg_autobox_file, ct_seg_resample_file, resolution=resolution_nn)
    mr_seg_attributes = resample(mr_seg_autobox_file, mr_seg_resample_file, resolution=resolution_nn)


    logging.info("2.  ===> Willis Labelling <===")
    logging.info("   2.1 ++++ : Creating a Brain Mask of NN Resolution")

    ct_mask_file = join(nn_resolution_path, f"{prefix}_ct_mask.nii.gz")
    mr_mask_file = join(nn_resolution_path, f"{prefix}_mr_mask.nii.gz")
    ct_brain_mask_file = join(nn_resolution_path, f"{prefix}_ct_SS_RegistrationImage.nii.gz")
    mr_brain_mask_file = join(nn_resolution_path, f"{prefix}_mr_SS_RegistrationImage.nii.gz")  

    brain_extract(ct_resample_file, ct_mask_file, ct_brain_mask_file, "ct")
    brain_extract(mr_resample_file, mr_mask_file, mr_brain_mask_file, "mri")


    logging.info("   2.2 ++++ : Finding Circle of Willis masking cube")

    ct_cube_file = join(nn_resolution_path, f"{prefix}_ct_cube.nii.gz")
    mr_cube_file = join(nn_resolution_path, f"{prefix}_mr_cube.nii.gz")

    ct_sphere, ct_cube = create_willis_cube(
            ct_brain_mask_file,
            nn_resolution_path,
            template_path,
            template_sphere_path,
        )

    mr_sphere, mr_cube = create_willis_cube(
            mr_brain_mask_file,
            nn_resolution_path,
            template_path,
            template_sphere_path,
        )

    nib.Nifti1Image(
        ct_sphere, ct_nii_attributes.affine, ct_nii_attributes.header
    ).to_filename(join(nn_resolution_path, f"{prefix}_ct_sphere.nii.gz"))
    nib.Nifti1Image(ct_cube, ct_nii_attributes.affine, ct_nii_attributes.header).to_filename(
        ct_cube_file
    )

    nib.Nifti1Image(
        mr_sphere, mr_nii_attributes.affine, mr_nii_attributes.header
    ).to_filename(join(nn_resolution_path, f"{prefix}_mr_sphere.nii.gz"))
    nib.Nifti1Image(mr_cube, mr_nii_attributes.affine, mr_nii_attributes.header).to_filename(
        mr_cube_file
    )

    logging.info("   2.3 ++++ : Cropping and re-anchoring scan data")

    ct_cropped_file = join(nn_resolution_path, f"{prefix}_ct_cropped.nii.gz")
    ct_cropped_seg_file = join(nn_resolution_path, f"{prefix}_ct_cropped_seg.nii.gz")
    crop_to_roi_cube(ct_resample_file, ct_cube_file, ct_seg_resample_file, ct_cropped_file, ct_cropped_seg_file)

    mr_cropped_file = join(nn_resolution_path, f"{prefix}_mr_cropped.nii.gz")
    mr_cropped_seg_file = join(nn_resolution_path, f"{prefix}_mr_cropped_seg.nii.gz")
    crop_to_roi_cube(mr_resample_file, mr_cube_file, mr_seg_resample_file, mr_cropped_file, mr_cropped_seg_file)

    mr_aligned_file = join(nn_resolution_path, f"{prefix}_mr_aligned.nii.gz")
    mr_aligned_seg_file = join(nn_resolution_path, f"{prefix}_mr_aligned_seg.nii.gz")

    aligned_mr, transforms = coregister_ct_mr(
        fixed_img   = ct_cropped_file,
        moving_img  = mr_cropped_file,
        fixed_mask  = ct_cropped_seg_file,
        moving_mask = mr_cropped_seg_file,
        out_moving_aligned      = mr_aligned_file,
        out_moving_mask_aligned = mr_aligned_seg_file,
        transform_prefix = f"{prefix}_mr_aligned_"
    )
    