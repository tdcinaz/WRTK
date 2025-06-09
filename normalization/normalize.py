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
    crop_to_roi_cube,
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

    skip = args.skip
    patient_ID = args.patient_ID
    scans = os.listdir(join(args.input_folder, "imagesTr"))
    labels = os.listdir(join(args.input_folder, "labelsTr"))

    resolution_nn = (0.625, 0.625, 0.625)
    nn_resolution_path = join(args.output, "nn_space")
    os.makedirs(nn_resolution_path, exist_ok=True)
    original_path = join(args.output, "original_space")
    os.makedirs(original_path, exist_ok=True)

    template_path = join(args.template_path, f"AVG_TOF_MNI_SS_down.nii.gz")
    template_sphere_path = join(args.template_path, f"willis_sphere_down.nii.gz")


    if (skip == False):
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


        logging.info("1. ===> Aligning Scan Orientation <===")

        logging.info("   1.1 ++++ : Aligning CT Scan")
        ct_raw: nib.Nifti1Image = nib.load(CT_path)
        ct_seg_raw: nib.Nifti1Image = nib.load(CT_label_path)
        logging.info(f"      1.1.1 ++++ : Original CT orientation: {nib.aff2axcodes(ct_raw.affine)}")
        ct = nib.as_closest_canonical(ct_raw)
        ct_seg = nib.as_closest_canonical(ct_seg_raw)
        logging.info(f"      1.1.2 ++++ : Transformed CT orientation: {nib.aff2axcodes(ct.affine)}")
        reorient_ct_file = join(original_path, f"{prefix}_reorient_ct_{patient_ID}.nii.gz")
        ct.to_filename(reorient_ct_file)
        seg_ct_reorient_file = join(original_path, f"{prefix}_seg_reorient_ct_{patient_ID}.nii.gz")
        ct_seg.to_filename(seg_ct_reorient_file)

        logging.info("   1.2 ++++ : Aligning MR Scan")
        mr_raw: nib.Nifti1Image = nib.load(MR_path)
        mr_seg_raw: nib.Nifti1Image = nib.load(MR_label_path)
        logging.info(f"      1.2.1 ++++ : Original MR orientation: {nib.aff2axcodes(mr_raw.affine)}")
        mr = nib.as_closest_canonical(mr_raw)
        mr_seg = nib.as_closest_canonical(mr_seg_raw)
        logging.info(f"      1.2.2 ++++ : Transformed CT orientation: {nib.aff2axcodes(mr.affine)}")
        reorient_mr_file = join(original_path, f"{prefix}_reorient_mr_{patient_ID}.nii.gz")
        mr.to_filename(reorient_mr_file)
        seg_mr_reorient_file = join(original_path, f"{prefix}_seg_reorient_mr_{patient_ID}.nii.gz")
        mr_seg.to_filename(seg_mr_reorient_file)


        logging.info("2. ===> Computing Bounding Boxes <===")

        logging.info("   2.1 ++++ : Autoboxing CT Scan")
        ct_autobox_file = join(original_path, f"{prefix}_autobox_ct_{patient_ID}.nii.gz")
        autobox_image(reorient_ct_file, ct_autobox_file, pad=6)
        ct_seg_autobox_file = join(original_path, f"{prefix}_seg_autobox_ct_{patient_ID}.nii.gz")
        crop_mask_like(seg_ct_reorient_file, ct_autobox_file, ct_seg_autobox_file)


        logging.info("   2.2 ++++ : Autoboxing MR Scan")
        mr_autobox_file = join(original_path, f"{prefix}_autobox_mr_{patient_ID}.nii.gz")
        autobox_image(reorient_mr_file, mr_autobox_file, pad=6)
        mr_seg_autobox_file = join(original_path, f"{prefix}_seg_autobox_mr_{patient_ID}.nii.gz")
        crop_mask_like(seg_mr_reorient_file, mr_autobox_file, mr_seg_autobox_file)


        logging.info("3. ===> Resampling to Neural Network Resolution <===")

        logging.info("   3.1 ++++ : Resampling CT Scan")
        ct_resample_file = join(nn_resolution_path, f"{prefix}_ct_resampled_{patient_ID}.nii.gz")
        ct_nii_attributes = resample(ct_autobox_file, ct_resample_file, resolution=resolution_nn)

        logging.info("   3.2 ++++ : Resampling MR Scan")
        mr_resample_file = join(nn_resolution_path, f"{prefix}_mr_resampled_{patient_ID}.nii.gz")
        mr_nii_attributes = resample(mr_autobox_file, mr_resample_file, resolution=resolution_nn)

        logging.info("   3.3 ++++ : Resampling CT Segmentation Mask")
        ct_seg_resample_file = join(nn_resolution_path, f"{prefix}_ct_seg_resampled_{patient_ID}.nii.gz")
        resample(ct_seg_autobox_file, ct_seg_resample_file, resolution=resolution_nn, resample_mode="NN")

        logging.info("   3.4 ++++ : Resampling MR Segmentation Mask")
        mr_seg_resample_file = join(nn_resolution_path, f"{prefix}_mr_seg_resampled_{patient_ID}.nii.gz")
        resample(mr_seg_autobox_file, mr_seg_resample_file, resolution=resolution_nn, resample_mode="NN")


        logging.info("4. ===> Creating a Brain Mask of NN Resolution <===")

        logging.info("   4.1 ++++ : Skull Stripping CT")
        ct_mask_file = join(nn_resolution_path, f"{prefix}_ct_mask_{patient_ID}.nii.gz")
        ct_brain_mask_file = join(nn_resolution_path, f"{prefix}_ct_SS_RegistrationImage_{patient_ID}.nii.gz")
        brain_extract(ct_resample_file, ct_mask_file, ct_brain_mask_file, "ct")

        logging.info("   4.2 ++++ : Skull Stripping MR")
        mr_mask_file = join(nn_resolution_path, f"{prefix}_mr_mask_{patient_ID}.nii.gz")
        mr_brain_mask_file = join(nn_resolution_path, f"{prefix}_mr_SS_RegistrationImage_{patient_ID}.nii.gz")  
        brain_extract(mr_resample_file, mr_mask_file, mr_brain_mask_file, "mri")


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

        logging.info("   5.3 ++++ : Creating Willis Cube for MR")
        _, mr_cube = create_willis_cube(
                mr_brain_mask_file,
                nn_resolution_path,
                template_path,
                template_sphere_path,
            )
        
        logging.info("   5.4 ++++ : Saving Willis Cube for MR")
        mr_cube_file = join(nn_resolution_path, f"{prefix}_mr_cube_{patient_ID}.nii.gz")
        nib.Nifti1Image(mr_cube, mr_nii_attributes.affine, mr_nii_attributes.header).to_filename(mr_cube_file)

        logging.info("6. ===> Cropping and re-anchoring scan data <===")

        logging.info("   6.1 ++++ : Cropping an re-anchoring CT")
        ct_cropped_file = join(nn_resolution_path, f"{prefix}_ct_cropped_{patient_ID}.nii.gz")
        ct_cropped_seg_file = join(nn_resolution_path, f"{prefix}_ct_cropped_seg_{patient_ID}.nii.gz")
        ct_bbox, _ = crop_to_roi_cube(ct_resample_file, ct_cube_file, ct_seg_resample_file, ct_cropped_file, ct_cropped_seg_file, return_bbox=True)

        logging.info("   6.2 ++++ : Cropping an re-anchoring MR")
        mr_cropped_file = join(nn_resolution_path, f"{prefix}_mr_cropped_{patient_ID}.nii.gz")
        mr_cropped_seg_file = join(nn_resolution_path, f"{prefix}_mr_cropped_seg_{patient_ID}.nii.gz")
        mr_bbox, _ = crop_to_roi_cube(mr_resample_file, mr_cube_file, mr_seg_resample_file, mr_cropped_file, mr_cropped_seg_file, return_bbox=True)

        ct_bbox_file = join(nn_resolution_path, f"{prefix}_ct_bbox.npy")
        mr_bbox_file = join(nn_resolution_path, f"{prefix}_mr_bbox_{patient_ID}.npy")
        np.save(mr_bbox_file, mr_bbox)
        np.save(ct_bbox_file, ct_bbox)

    else:
        ct_cropped_file = join(nn_resolution_path, f"{prefix}_ct_cropped_{patient_ID}.nii.gz")
        mr_resample_file = join(nn_resolution_path, f"{prefix}_mr_resampled_{patient_ID}.nii.gz")
        mr_seg_resample_file = join(nn_resolution_path, f"{prefix}_mr_seg_resampled_{patient_ID}.nii.gz")
        mr_bbox_file = join(nn_resolution_path, f"{prefix}_mr_bbox_{patient_ID}.npy")
        mr_bbox = np.load(mr_bbox_file)

    mr_aligned_file = join(nn_resolution_path, f"{prefix}_mr_aligned.nii.gz")
    mr_aligned_seg_file = join(nn_resolution_path, f"{prefix}_mr_aligned_seg.nii.gz")

    ct_imin, ct_imax, ct_jmin, ct_jmax, ct_kmin, ct_kmax = ct_bbox
    mr_imin, mr_imax, mr_jmin, mr_jmax, mr_kmin, mr_kmax = mr_bbox

    if (skip == False):

        logging.info("7. ===> Coregistering CT and MR <===")

        _, transforms, inv_transforms = coregister_ct_mr(
            fixed_img   = ct_cropped_file,
            moving_img  = mr_cropped_file,
            fixed_mask  = ct_cropped_seg_file,
            moving_mask = mr_cropped_seg_file,
            out_moving_aligned      = mr_aligned_file,
            out_moving_mask_aligned = mr_aligned_seg_file,
            transform_prefix = join(nn_resolution_path, f"{prefix}_transform_")
        )
    else:
        transforms = join(nn_resolution_path, f"{prefix}_transform_Composite.h5")
        inv_transforms = join(nn_resolution_path, f"{prefix}_transform_InverseComposite.h5")

    # ----------------------------------------------------------
    # 2. build a new affine for the *whole* MR volume
    #    so that voxel (imin,jmin,kmin) becomes the new origin
    # ----------------------------------------------------------
    ct_full: nib.Nifti1Image = nib.load(ct_resample_file)
    ct_full_seg: nib.Nifti1Image = nib.load(ct_seg_resample_file)
    mr_full: nib.Nifti1Image = nib.load(mr_resample_file)
    mr_full_seg: nib.Nifti1Image = nib.load(mr_seg_resample_file)

    voxsize = mr_full.header.get_zooms()[:3]

    ct_new_affine = ct_full.affine.copy()
    ct_new_affine_seg = ct_full_seg.affine.copy()
    mr_new_affine = mr_full.affine.copy()
    mr_new_affine_seg = mr_full_seg.affine.copy()


    ct_new_affine[:3, 3]   = -np.array([ct_imin*voxsize[0], ct_jmin*voxsize[1], ct_kmin*voxsize[2]])
    ct_new_affine_seg[:3, 3]   = -np.array([ct_imin*voxsize[0], ct_jmin*voxsize[1], ct_kmin*voxsize[2]])
    mr_new_affine[:3, 3]   = -np.array([mr_imin*voxsize[0], mr_jmin*voxsize[1], mr_kmin*voxsize[2]])
    mr_new_affine_seg[:3, 3]   = -np.array([mr_imin*voxsize[0], mr_jmin*voxsize[1], mr_kmin*voxsize[2]])

    ct_full_data = ct_full.get_fdata(dtype=np.float32, caching="unchanged")
    ct_full_seg_data = ct_full_seg.get_fdata(dtype=np.float32, caching="unchanged").astype(np.uint8)
    mr_full_data = mr_full.get_fdata(dtype=np.float32, caching="unchanged")
    mr_full_seg_data = mr_full_seg.get_fdata(dtype=np.float32, caching="unchanged").astype(np.uint8)

    ct_hdr_seg: nib.Nifti1Header = ct_full_seg.header.copy()
    ct_hdr_seg.set_data_dtype(np.uint8)
    mr_hdr_seg: nib.Nifti1Header = mr_full_seg.header.copy()
    mr_hdr_seg.set_data_dtype(np.uint8)

    ct_reanchored_file = join(nn_resolution_path, f"{prefix}_ct_reanchored.nii.gz")
    ct_reanchored_seg_file = join(nn_resolution_path, f"{prefix}_ct_reanchored_seg.nii.gz")
    nib.Nifti1Image(ct_full_data, ct_new_affine, ct_full.header).to_filename(ct_reanchored_file)
    nib.Nifti1Image(ct_full_seg_data, ct_new_affine, ct_hdr_seg).to_filename(ct_reanchored_seg_file)
    mr_reanchored_file = join(nn_resolution_path, f"{prefix}_mr_reanchored.nii.gz")
    mr_reanchored_seg_file = join(nn_resolution_path, f"{prefix}_mr_reanchored_seg.nii.gz")
    nib.Nifti1Image(mr_full_data, mr_new_affine, mr_full.header).to_filename(mr_reanchored_file)
    nib.Nifti1Image(mr_full_seg_data, mr_new_affine, mr_hdr_seg).to_filename(mr_reanchored_seg_file)

    aligned_ct_cube      = join(nn_resolution_path, f"{prefix}_ct_aligned_cube.nii.gz")
    aligned_ct_seg_cube  = join(nn_resolution_path, f"{prefix}_ct_aligned_seg_cube.nii.gz")
    aligned_mr_cube      = join(nn_resolution_path, f"{prefix}_mr_aligned_cube.nii.gz")
    aligned_mr_seg_cube  = join(nn_resolution_path, f"{prefix}_mr_aligned_seg_cube.nii.gz")

    logging.info("8. ===> Applying Computed Transform <===")

    # 1.  Warp the *uncropped* CT scan into MR-cube space
    logging.info("   8.1 ++++ : Applying Transform to CT Scan")
    apply_transform(
        template_file   = ct_reanchored_file,   # ← full CT (still cube-shaped, but un-warped)
        fixed_file      = mr_cropped_file,    # ← reference = MR cube (already cropped!)
        transform       = inv_transforms,         # ← path returned by coregister_ct_mr
        output          = aligned_ct_cube,
        interpolation   = "Linear",           # or "BSpline"
    )

    # 2.  Warp the *uncropped* CT segmentation mask the same way
    logging.info("   8.2 ++++ : Applying Transform to CT Segmentation Mask")
    apply_transform(
        template_file   = ct_reanchored_seg_file,
        fixed_file      = mr_cropped_file,
        transform       = inv_transforms,
        output          = aligned_ct_seg_cube,
        interpolation   = "MultiLabel",
    )
    # 3.  Warp the *uncropped* MR scan into CT-cube space
    logging.info("   8.3 ++++ : Applying Transform to MR Scan")
    apply_transform(
        template_file   = mr_reanchored_file,   # ← full MR (still cube-shaped, but un-warped)
        fixed_file      = ct_cropped_file,    # ← reference = CT cube (already cropped!)
        transform       = transforms,         # ← path returned by coregister_ct_mr
        output          = aligned_mr_cube,
        interpolation   = "Linear",           # or "BSpline"
    )

    # 4.  Warp the *uncropped* MR segmentation mask the same way
    logging.info("   8.4 ++++ : Applying Transform to MR Segmentation Mask")
    apply_transform(
        template_file   = mr_reanchored_seg_file,
        fixed_file      = ct_cropped_file,
        transform       = transforms,
        output          = aligned_mr_seg_cube,
        interpolation   = "MultiLabel",
    )