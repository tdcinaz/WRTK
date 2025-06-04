import argparse
import logging
import os
from collections import OrderedDict
from os.path import join, isfile
from typing import Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from normalization.tof_master import (
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

    resolution_nn = (0.625, 0.625, 0.625)
    nn_resolution_path = join(args.output, "nn_space")
    os.makedirs(nn_resolution_path, exist_ok=True)
    original_path = join(args.output, "original_space")
    os.makedirs(original_path, exist_ok=True)

    template_path = join(args.template_path, f"AVG_TOF_MNI_SS_down.nii.gz")
    template_sphere_path = join(args.template_path, f"willis_sphere_down.nii.gz")
    tof_raw: nib.Nifti1Image = nib.load(args.TOF)
    seg_raw: nib.Nifti1Image = nib.load(args.SEG)
    logging.info(f"Original orientation: {nib.aff2axcodes(tof_raw.affine)}")

    tof = nib.as_closest_canonical(tof_raw)
    seg = nib.as_closest_canonical(seg_raw)
    logging.info(f"Transformed orientation: {nib.aff2axcodes(tof.affine)}")

    reorient_file = join(original_path, f"{prefix}_reorient.nii.gz")
    tof.to_filename(reorient_file)

    seg_reorient_file = join(original_path, f"{prefix}_seg_reorient.nii.gz")
    seg.to_filename(seg_reorient_file)

    autobox_file = join(original_path, f"{prefix}_autobox.nii.gz")
    autobox_image(reorient_file, autobox_file, pad=6)

    seg_autobox_file = join(original_path, f"{prefix}_seg_autobox.nii.gz")
    crop_mask_like(seg_reorient_file, autobox_file, seg_autobox_file)

    logging.info("1. ===> Preprocessing <===")
    logging.info("   1.2 ++++ : Resampling to Neural Network Resolution")

    resample_file = join(nn_resolution_path, f"{prefix}_resampled.nii.gz")
    nii_attributes = resample(autobox_file, resample_file, resolution=resolution_nn)
    resampled_array = nii_attributes.get_fdata("unchanged")

    seg_resample_file = join(nn_resolution_path, f"{prefix}_seg_resampled.nii.gz")
    seg_attributes = resample(seg_autobox_file, seg_resample_file, resolution=resolution_nn)

    logging.info("2.  ===> Willis Labelling <===")
    logging.info("   2.1 ++++ : Creating a Brain Mask of NN Resolution")

    mask_file = join(nn_resolution_path, f"{prefix}_mask.nii.gz")
    brain_mask_file = join(
        nn_resolution_path, f"{prefix}_SS_RegistrationImage.nii.gz"
    )
    
    brain_extract(resample_file, mask_file, brain_mask_file, "MRI")         # Skull strip

    logging.info("   2.2 ++++ : Finding Circle of Willis masking cube")

    cube_file = join(nn_resolution_path, f"{prefix}_cube.nii.gz")

    sphere, cube = create_willis_cube(
        brain_mask_file,
        nn_resolution_path,
        template_path,
        template_sphere_path,
    )
    nib.Nifti1Image(
        sphere, nii_attributes.affine, nii_attributes.header
    ).to_filename(join(nn_resolution_path, f"{prefix}_sphere.nii.gz"))
    nib.Nifti1Image(cube, nii_attributes.affine, nii_attributes.header).to_filename(
        cube_file
    )


    logging.info("   2.3 ++++ : Cropping and re-anchoring scan data")
    cropped_file = join(nn_resolution_path, f"{prefix}_cropped.nii.gz")
    cropped_seg_file = join(nn_resolution_path, f"{prefix}_cropped_seg.nii.gz")
    crop_to_roi_cube(resample_file, cube_file, seg_resample_file, cropped_file, cropped_seg_file)
    
    #aligned_ct, transforms = coregister_ct_mr(
    #    fixed_img   = cropped_file,
    #    moving_img  = "output/topcow_mr_001_0001/nn_space/topcow_mr_001_0001_cropped.nii.gz",
    #    fixed_mask  = cropped_seg_file,
    #    moving_mask = "output/topcow_mr_001_0001/nn_space/topcow_mr_001_0001_cropped_seg.nii.gz",
    #    out_moving_aligned      = "output/topcow_ct_001_0000/nn_space/topcow_001_CT_inMRspace.nii.gz",
    #    out_moving_mask_aligned = "output/topcow_ct_001_0000/nn_space/topcow_001_CTmask_inMRspace.nii.gz",
    #    transform_prefix = "topcow-001_CT2MR_"
    #)
