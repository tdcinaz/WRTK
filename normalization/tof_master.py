# -*- coding: utf-8 -*-

# To Do:
# - add centered_resize when trained cube 2d NN is ready

from glob import glob
import logging
import multiprocessing
from os.path import join
import os
import shutil
from typing import Optional, Tuple, Union
import itk
import nibabel as nib
from nipype.interfaces import afni
from nipype.interfaces.ants import ApplyTransforms, Registration
import numpy as np

from raster_geometry import cube

from scipy import ndimage as ndi
from scipy.ndimage import center_of_mass, gaussian_filter
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.spatial import distance as dist

from skimage.morphology import watershed
from skimage.morphology import skeletonize_3d
from skimage.morphology import ball

from sklearn.mixture import GaussianMixture

from tqdm import tqdm

from normalization.ComputeVED import ComputeVED

from normalization import ArterialEnum


def register_template(
    fixed_file: str,
    moving_file: str,
    template_file: str,
    output_dir: str,
    return_transform=False,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[str]]]:
    # Command line
    """Equivalent bash command line call to ANTs.

    antsRegistration --verbose 1 \
        --dimensionality 3 \
        --float 0 \
        --collapse-output-transforms 1 \
        --output [ WILLIS_ANTS,WILLIS_ANTSWarped.nii.gz,WILLIS_ANTSInverseWarped.nii.gz ] \
        --interpolation Linear \
        --use-histogram-matching 1 \
        --winsorize-image-intensities [ 0.005,0.995 ] \
        -x [ NN_rabox_mask.nii.gz, NULL ] \
        --initial-moving-transform [ NN_rabox.nii.gz,/Users/kw2/masks/AVG_TOF_MNI.nii.gz,1 ] \
        --transform Rigid[ 0.1 ] \
        --metric MI[ NN_rabox.nii.gz,/Users/kw2/masks/AVG_TOF_MNI.nii.gz,1,32,Regular,0.25 ] \
        --convergence [ 1000x500x250x0,1e-6,10 ] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox \
        --transform Affine[ 0.1 ] \
        --metric MI[ NN_rabox.nii.gz,/Users/kw2/masks/AVG_TOF_MNI.nii.gz,1,32,Regular,0.25 ] \
        --convergence [ 1000x500x250x0,1e-6,10 ] \
        --shrink-factors 12x8x4x2 \
        --smoothing-sigmas 4x3x2x1vox

    Args:
        fixed_file (str): Reference used to compute transformations.
        moving_file (str): File to register to fixed file.
        template_file (str): Template file where transformations are applied.
        output_dir (str): output directory to put transformations.
        return_transform (bool, optional): Return transformation. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, Optional[str]]]:
            template data registered, with transformation if enabled.
    """

    # Nipype
    reg = Registration()
    reg.inputs.verbose = True
    reg.inputs.dimension = 3
    reg.inputs.float = False
    reg.inputs.collapse_output_transforms = True
    reg.inputs.output_warped_image = join(output_dir, "AVG_TOF_MNI_native.nii.gz")
    reg.inputs.output_inverse_warped_image = join(output_dir, "NN_rabox_MNI.nii.gz")
    reg.inputs.output_transform_prefix = join(output_dir, "WILLIS_ANTS")
    reg.inputs.interpolation = "Linear"
    reg.inputs.use_histogram_matching = True
    reg.inputs.winsorize_lower_quantile = 0.025
    reg.inputs.winsorize_upper_quantile = 0.975
    reg.inputs.initial_moving_transform_com = 1
    reg.inputs.transforms = ["Rigid", "Affine"]
    reg.inputs.transform_parameters = [(0.1,), (0.1,)]
    reg.inputs.metric = ["MI", "MI"]
    reg.inputs.fixed_image = fixed_file
    reg.inputs.moving_image = moving_file
    reg.inputs.metric_weight = [1, 1]
    reg.inputs.radius_or_number_of_bins = [32, 32]
    reg.inputs.sampling_strategy = ["Regular", "Regular"]
    reg.inputs.sampling_percentage = [0.25, 0.25]
    reg.inputs.number_of_iterations = [
        [1000, 500, 250, 100, 50],
        [1000, 500, 250, 100, 50],
    ]
    reg.inputs.convergence_threshold = [1.6e-6, 1.6e-6]
    reg.inputs.shrink_factors = [[12, 8, 4, 2, 1], [12, 8, 4, 2, 1]]
    reg.inputs.smoothing_sigmas = [[4, 3, 2, 1, 0], [4, 3, 2, 1, 0]]
    reg.inputs.sigma_units = ["vox", "vox"]
    # Use 1 to have deterministic registration
    reg.inputs.num_threads = multiprocessing.cpu_count()
    # print(reg.cmdline)
    reg.run()

    # Template Transformation
    sphere = apply_transform(
        template_file,
        fixed_file,
        join(output_dir, "WILLIS_ANTS0GenericAffine.mat"),
        join(output_dir, "willis_sphere.nii.gz"),
    )

    if return_transform:
        return sphere, join(output_dir, "WILLIS_ANTS0GenericAffine.mat")

    return sphere

from typing import List

def coregister_ct_mr(
        fixed_img:   str,      # e.g. MR brain volume
        moving_img:  str,      # e.g. CT brain volume
        fixed_mask:  str,      # MR brain mask  (binary)
        moving_mask: str,      # CT brain mask  (binary)
        out_moving_aligned: str,
        out_moving_mask_aligned: str,
        transform_prefix: str  = "CT2MR_",
        use_syn: bool = True
    ) -> Tuple[str, List[str]]:
    """
    Rigid → Affine (→ optional SyN) registration of a CT scan to an MR scan
    using brain masks as ROIs.  Returns path to the aligned CT image and the
    list of forward transforms in ANTs order.
    """
    fixed_img   = fixed_img
    moving_img  = moving_img
    fixed_mask  = fixed_mask
    moving_mask = moving_mask
    transform_prefix = transform_prefix

    # ---------------- 1. build the ANTs registration object -----------------
    reg = Registration()
    reg.inputs.verbose = True
    reg.inputs.fixed_image        = fixed_img
    reg.inputs.moving_image       = moving_img
    reg.inputs.fixed_image_masks  = [fixed_mask] * (3 if use_syn else 2)
    reg.inputs.moving_image_masks = [moving_mask] * (3 if use_syn else 2)

    reg.inputs.transforms              = ['Rigid', 'Affine'] + (['SyN'] if use_syn else [])
    reg.inputs.transform_parameters    = [(0.1,), (0.1,)]     + ([(0.1, 3, 0)] if use_syn else [])
    reg.inputs.metric                  = ['MI',  'MI']        + (['CC'] if use_syn else [])
    reg.inputs.metric_weight           = [1, 1]               + ([1] if use_syn else [])
    reg.inputs.radius_or_number_of_bins= [32, 32]             + ([4] if use_syn else [])

    # multiresolution schedules
    reg.inputs.number_of_iterations    = [[1000, 500, 250, 100],
                                          [1000, 500, 250, 100]] + \
                                         ([[70, 50, 20]] if use_syn else [])
    reg.inputs.shrink_factors          = [[8, 4, 2, 1]]*2 + ([[4, 2, 1]] if use_syn else [])
    reg.inputs.smoothing_sigmas        = [[3, 2, 1, 0]]*2 + ([[1, .5, 0]] if use_syn else [])

    reg.inputs.write_composite_transform = True
    reg.inputs.output_warped_image       = out_moving_aligned
    reg.inputs.output_inverse_warped_image = False
    #reg.inputs.output_prefix             = transform_prefix
    reg.inputs.num_threads               = multiprocessing.cpu_count()
    reg.inputs.float                     = True
    reg.inputs.args                      = "-u"        # avoid histogram‐match CSF/contrast swap

    reg_res = reg.run()                                              # run ANTs
    fwd_xforms = reg_res.outputs.forward_transforms                  # ordered for antsApplyTransforms

    # ---------------- 2. resample the CT brain mask with the same xforms ----
    at = ApplyTransforms()
    at.inputs.dimension        = 3
    at.inputs.reference_image  = fixed_img
    at.inputs.input_image      = moving_mask
    at.inputs.transforms       = "output/topcow_ct_001_0000/nn_space/Affine.mat"
    at.inputs.interpolation    = 'NearestNeighbor'
    at.inputs.output_image     = out_moving_mask_aligned
    at.run()

    return out_moving_aligned, fwd_xforms

def apply_transform(
    template_file: str,
    fixed_file: str,
    transform: str,
    output: str,
    interpolation="NearestNeighbor",
    invert=False,
) -> np.ndarray:
    """Apply transformation using ANTs on the template file.

    Args:
        template_file (str): File to be register.
        fixed_file (str): Reference file used for registration.
        transform (str): Transformation to apply on the template file.
        output (str): Where to save the registration result.
        interpolation (str, optional): Type of interpolation. Defaults to "NearestNeighbor".
        invert (bool, optional): Indicate if the transformation should be reversed.
                                 Defaults to False.

    Returns:
        np.ndarray: Data of the template file registered.
    """
    at = ApplyTransforms()
    at.inputs.dimension = 3
    at.inputs.input_image = template_file
    at.inputs.reference_image = fixed_file
    at.inputs.output_image = output
    at.inputs.transforms = [transform]
    at.inputs.invert_transform_flags = [invert]
    at.inputs.interpolation = interpolation
    at.inputs.default_value = 0
    at.run()

    return nib.load(output).get_fdata("unchanged")


def create_cube(roi_data: np.ndarray, side: int = 90) -> np.ndarray:
    """Create a cube from ROI sphere.

    Args:
        roi_data (np.ndarray): Registered sphere binary mask.
        side (int, optional): side of the cube in voxel. Defaults to 90.

    Returns:
        np.ndarray: array containing the cube.
    """
    # Finding Relative center of mass
    roi_data[np.where(roi_data != 0)] = 1
    com = np.array(list(center_of_mass(roi_data)))
    shape = np.array(list(roi_data.shape))
    relative_center = tuple((com / shape).tolist())

    # Creating cube
    return cube(roi_data.shape, side, relative_center).astype(int)


def find_willis_center(markers: np.ndarray) -> Tuple[Tuple, np.ndarray]:
    """Retrieve the center of mass of the markers called willis center.

    Args:
        markers (np.ndarray): The segmentation produce by the model.

    Returns:
        Tuple[Tuple, np.ndarray]: center of mass of the overall segmentation, and binary mask.
    """
    # Parsing Input Arguments

    binary_willis = np.zeros_like(markers)
    binary_willis[np.where(markers != 0)] = 1
    return center_of_mass(binary_willis), binary_willis


def mask_image(in_file: str, out_file: str, out_brain_masked: str) -> nib.Nifti1Image:
    """Generate mask file using AFNI SkullStrip and Automask.

    Args:
        in_file (str): TOF file used as AFNI SkullStrip input.
        out_file (str): Binary mask file path.
        out_brain_masked (str): Brain masked file path.

    Returns:
       nib.Nifti1Image : binary mask file.
    """
    # I/O parsing
    skullstrip = afni.SkullStrip()

    skullstrip.inputs.in_file = in_file
    skullstrip.inputs.args = "-overwrite"
    skullstrip.inputs.out_file = out_file
    skullstrip.inputs.outputtype = "NIFTI_GZ"
    skullstrip.run()

    automask = afni.Automask()
    automask.inputs.in_file = out_file
    automask.inputs.args = "-overwrite"
    automask.inputs.outputtype = "NIFTI"
    automask.inputs.out_file = out_file
    automask.inputs.brain_file = out_brain_masked
    automask.run()

    # Binary conversion of mask
    return nib.load(out_file)

import subprocess, pathlib

def brain_extract(in_file: str,
                  out_mask: str,
                  out_brain: str,
                  modality: str = "auto") -> nib.Nifti1Image:
    """
    Brain‑extracts MRI or CT angiography volumes using the best available tool.

    Parameters
    ----------
    in_file   : path to source NIfTI
    out_mask  : binary mask (.nii.gz)
    out_brain : brain‑only volume (.nii.gz)
    modality  : 'MRI', 'CT', or 'auto' (detect from header)

    Returns
    -------
    nib.Nifti1Image of the binary mask
    """
    # crude modality check (fallback if user does not pass it)
    if modality == "auto":
        hdr = nib.load(in_file).header
        modality = "CT" if hdr.get("db_name",b"").startswith(b"CT") or hdr.get_xyzt_units()[0] == "unknown" else "MRI"

    tool = None
    if modality.upper() == "MRI":
        try:                               # fastest if available
            subprocess.run(["hd-bet", "-i", in_file, "-o", out_mask,
                            "-device", "cpu", "--mode", "fast"], check=True)
            tool = "HD‑BET"
        except FileNotFoundError:
            tool = "afni"
    if modality.upper() == "CT":
        try:
            subprocess.run(["mri_synthstrip", "-i", in_file,
                            "-m", out_mask, "-o", out_brain], check=True)
            tool = "mri_synthstrip"
        except FileNotFoundError:
            tool = "TotalSegmentator"

    if tool == "afni":                      # fallback for MRI only
        ss = afni.SkullStrip(in_file=in_file,
                             out_file=out_mask,
                             args="-overwrite",
                             outputtype="NIFTI_GZ")
        ss.run()
    elif tool == "TotalSegmentator":       # fallback for CT only
        tmp_dir = pathlib.Path(out_mask).with_suffix("")
        subprocess.run(["TotalSegmentator", "-i", in_file,
                        "-o", str(tmp_dir), "--fast"], check=True)
        brain_label = tmp_dir / "brain.nii.gz"
        os.rename(brain_label, out_mask)
    # apply the mask to generate brain‑only volume
    subprocess.run(["fslmaths", in_file, "-mas", out_mask, out_brain], check=True)
    return nib.load(out_mask)


def crop_to_roi_cube(
        scan_nii: str,
        roi_mask_nii: str,
        seg_mask_nii: str,
        out_scan_nii: str,
        out_seg_nii: str,
        out_mask_nii: Optional[str] = None,
        return_bbox: bool = False
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Crop a neuroimaging volume to the bounding cube of its binary ROI mask
    and reset the affine so the corner of that cube is the new (0,0,0).

    Parameters
    ----------
    scan_nii      : path to the full-field intensity image (.nii or .nii.gz)
    roi_mask_nii  : matching binary ROI-cube mask (same grid as scan_nii)
    out_scan_nii  : output filename for the cropped image
    out_mask_nii  : (optional) output filename for the cropped mask
    return_bbox   : if True, return (imin,imax,jmin,jmax,kmin,kmax)

    Returns
    -------
    (bbox_img, bbox_msk) if return_bbox else None
    """
    scan_obj = nib.load(str(scan_nii))
    mask_obj = nib.load(str(roi_mask_nii))
    seg_obj = nib.load(str(seg_mask_nii))

    scan      = scan_obj.get_fdata(dtype=np.float32, caching='unchanged')
    mask_data = mask_obj.get_fdata(dtype=np.float32,  caching='unchanged').astype(np.uint8)
    seg_data = seg_obj.get_fdata(dtype=np.float32, caching='unchanged').astype(np.uint8)

    if mask_data.max() == 0:
        raise ValueError("ROI mask is empty – nothing to crop.")

    # ---------- 1. find ROI cube extents (voxels) ----------
    nz       = np.nonzero(mask_data)
    imin, jmin, kmin = [int(c.min()) for c in nz]
    imax, jmax, kmax = [int(c.max()) + 1 for c in nz]   # +1 → Python-style stop‐index

    # ---------- 2. crop ----------
    cropped_scan = scan [imin:imax, jmin:jmax, kmin:kmax]
    cropped_mask = mask_data[imin:imax, jmin:jmax, kmin:kmax]
    cropped_seg = seg_data[imin:imax, jmin:jmax, kmin:kmax]

    # ---------- 3. build new affine with origin at cube corner ----------
    #
    # Original affine: Xworld = A[:, :3] @ (i, j, k) + A[:, 3]
    # After cropping we want voxel (0,0,0) to map to Xworld = (0,0,0).
    #
    new_affine          = scan_obj.affine.copy()
    new_affine[:3, 3]   = 0.0            # translate origin to (0,0,0)
    # (The rotation / scaling block new_affine[:3,:3] is left unchanged.)

    # ---------- 4. save ----------
    hdr = scan_obj.header.copy()
    hdr.set_data_dtype(np.float32)

    nib.save(nib.Nifti1Image(cropped_scan, new_affine, hdr), out_scan_nii)

    hdr_seg = seg_obj.header.copy()
    hdr_seg.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(cropped_seg, new_affine, hdr_seg), out_seg_nii)


    if out_mask_nii is not None:
        hdr_mask = mask_obj.header.copy()
        hdr_mask.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(cropped_mask, new_affine, hdr_mask),
                 out_mask_nii)

    if return_bbox:
        return (imin, imax, jmin, jmax, kmin, kmax), new_affine

def resample(
    in_file: str,
    out_file: str,
    resolution: Optional[Tuple] = None,
    master: Optional[str] = None,
    resample_mode="Cu",
) -> nib.Nifti1Image:
    """Resample volume using antsRegistation.

    Args:
        in_file (str): Input file to register.
        out_file (str): Output file to store the result.
        resolution (Tuple, optional): Voxel size used to resample. Defaults to None.
        master (str, optional): Image path to use as reference. Defaults to None.
        resample_mode (str, optional): Type of interpolation to use. Defaults to "Cu".

    Raises:
        ValueError: thrown if neither resolution or master are not defined.

    Returns:
        nib.Nifti1Image: The resampled image.
    """

    if resolution is None and master is None:
        raise ValueError("At least on of the `resolution` or `master` parameter should be defined.")

    # Constructing afni call
    afni_resample = afni.Resample()
    afni_resample.inputs.in_file = in_file
    afni_resample.inputs.out_file = out_file
    afni_resample.inputs.outputtype = "NIFTI"
    afni_resample.inputs.resample_mode = resample_mode
    afni_resample.inputs.args = "-overwrite"

    if master is not None:
        afni_resample.inputs.master = master
    else:
        afni_resample.inputs.voxel_size = resolution

    afni_resample.run()
    return nib.load(out_file)


def autobox_image(in_file: str, out_file: str, pad: int = 6):
    """Automatic crop input file.

    Args:
        in_file (str): Path of the input to crop.
        out_file (str): Path to save the result.
        pad (int, optional): Number of voxel to have a border. Defaults to 6.
    """
    # Autoboxing
    abox = afni.Autobox()
    abox.inputs.in_file = in_file
    abox.inputs.out_file = out_file
    abox.inputs.padding = pad
    abox.inputs.args = "-overwrite"
    abox.run()
    return nib.load(out_file)

def crop_mask_like(mask_file: str, master_file: str, out_file: str):
    """
    Resample a segmentation/label mask so that it has
    the same grid (FOV, voxel size, orientation) as
    the image produced by 3dAutobox.

    Parameters
    ----------
    mask_file  : path to the *uncropped* mask
    master_file: path to the image returned by `autobox_image`
    out_file   : where to write the cropped mask (defaults to
                 <mask stem>_crop.nii.gz in the same folder)

    Returns
    -------
    nib.Nifti1Image of the resampled mask.
    """
    if out_file is None:
        stem = pathlib.Path(mask_file).with_suffix('').name
        out_file = f"{stem}_crop.nii.gz"

    rs = afni.Resample()
    rs.inputs.in_file = mask_file        # original mask
    rs.inputs.master  = master_file      # the autoboxed image
    rs.inputs.out_file = out_file
    rs.inputs.resample_mode = "NN"               # keep integer labels
    rs.inputs.args  = "-overwrite"
    rs.run()
    return nib.load(out_file)


def create_willis_cube(
    fixed_file: str,
    output_dir: str,
    moving_file: Optional[str] = None,
    template_file: Optional[str] = None,
    transform: Optional[str] = None,
    invert=False,
    side=90,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the willis sphere from the template.

    Args:
        fixed_file (str): Reference file.
        output_dir (str): Output directory of files.
        moving_file (str, optional): File to register.
        template_file (str, optional): Template file to register along with the moving file.
        transform (str, optional): Use this transformation if provided.
        invert (bool, optional): Invert the transformation. Defaults to False.
        side (int, optional): Side of cube in voxel. Defaults to 90.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Return the sphere, and cube.
    """
    # Registering MNI willis sphere to native space
    if transform is None:
        sphere = register_template(fixed_file, moving_file, template_file, output_dir)
    else:
        sphere = apply_transform(
            template_file,
            fixed_file,
            transform,
            join(output_dir, "willis_sphere.nii.gz"),
            invert=invert,
        )

    return sphere, create_cube(sphere, side=side)


def centerline_transform(data: np.ndarray) -> np.ndarray:
    """Direct call to skeletonize_3d funtion of skimage package.

    Args:
        data (np.ndarray): Input binary image to skeletonize.

    Returns:
        np.ndarray: Skeleton image of the input.
    """
    return skeletonize_3d(data.astype(np.uint8)).astype(bool).astype(float)


def propagate_closest_value(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill the an array according to the closest value within the mask.

    Args:
        values (np.ndarray): Values of the segmentation.
        mask (np.ndarray): Mask used to constrain the propagation.

    Returns:
        np.ndarray: Values propagated.
    """
    closest_values = np.zeros(values.shape, dtype=float)
    artery_locations = np.transpose(np.where(values != 0))

    # TODO: Check with extend_labels and mask_vascular_perfusion to merge functionalities
    for index, value in tqdm(np.ndenumerate(mask), total=mask.size):
        if value != 0:
            xa = np.zeros((1, 3))
            xa[0] = index
            distances = cdist(xa, artery_locations, metric="euclidean")
            distances = distances.reshape(-1)
            value_index = np.where(distances == np.min(distances))[0][0]
            value_index = artery_locations[value_index].astype(int)
            closest_values[index] = values[tuple(value_index)]

    return closest_values


def edt_transform(mask: np.ndarray, center_line: np.ndarray, voxel_size: Tuple) -> np.ndarray:
    """Compute the edt transform of the scipy packages and propagates centerlines value according
    to the input mask.

    Args:
        mask (np.ndarray): Input mask used to generate EDT features.
        center_line (np.ndarray): center line array.
        voxel_size (Tuple): voxel size of the input image.

    Returns:
        np.ndarray: [description]
    """
    edt = distance_transform_edt(mask)
    edt = edt * 2 * np.mean(voxel_size)
    edt *= center_line.astype(np.float)

    return propagate_closest_value(edt, mask)


def extract_vessels_itk(
    input_image: str,
    output_image: str,
    sigma=1.0,
    alpha1=0.5,
    alpha2=2.0,
) -> np.ndarray:
    """Extract vessels using ITK Vessel measure filter.

    Args:
        input_image (str): Path of the input image. Defaults to None.
        output_image (str): Path of the output image. Defaults to None.
        sigma (float, optional): see `Hessian3DToVesselnessMeasureImageFilter <https://itk.org/ITKExamples/src/Filtering/ImageFeature/SegmentBloodVessels/Documentation.html>`_ . Defaults to 1.0.
        alpha1 (float, optional): see `Hessian3DToVesselnessMeasureImageFilter <https://itk.org/ITKExamples/src/Filtering/ImageFeature/SegmentBloodVessels/Documentation.html>`_ . Defaults to 0.5.
        alpha2 (float, optional): see `Hessian3DToVesselnessMeasureImageFilter <https://itk.org/ITKExamples/src/Filtering/ImageFeature/SegmentBloodVessels/Documentation.html>`_ . Defaults to 2.0.

    Returns:
        np.ndarray: vesselness mask.
    """
    image_data = itk.imread(input_image, itk.ctype("float"))

    # Using ITK multiscale hessian filter
    hessian_image = itk.hessian_recursive_gaussian_image_filter(image_data, sigma=sigma)

    vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[itk.ctype("float")].New()
    vesselness_filter.SetInput(hessian_image)
    vesselness_filter.SetAlpha1(alpha1)
    vesselness_filter.SetAlpha2(alpha2)

    # Writing and loading back output
    itk.imwrite(vesselness_filter, output_image)
    vesselness = nib.load(output_image).get_fdata("unchanged")

    return vesselness


def thresholds_sigma(
    min_mean: float, min_var: float, max_mean: float, max_var: float, low_factor=3, high_factor=0.5
) -> Tuple[float, float]:

    """Return thresholds based on standard deviation from Gaussian Mixture

    Args:
        min_mean (float): Mean of the lowest gaussian
        min_var (float): Variance of the lowest gaussian
        max_mean (float): Mean of the highest gaussian
        max_var (float): Variance of the highest gaussian
        low_factor (float): Multiplier of the lower standard deviation
        high_factor (float): Multiplier of the higher standard deviation

    Returns:
        lowt (float): Lowest threshold for a hysteresis thresholding
        hight (float): Highest threshold for a hysteresis thresholding

    """

    # Those equations were found empirically. We did some tests with different thresholds and those were the ones
    # that were kept. This function is highly sensitive to noise, image quality, image artifact, etc.
    lowt = min_mean + (low_factor * np.sqrt(min_var))
    hight = max_mean + (high_factor * np.sqrt(max_var))
    logging.info(f"   +++: LowBound = {lowt}")
    logging.info(f"   +++: HighBound = {hight}")

    return lowt, hight


def cube_gm_number_clusters(method: int, loop_count: int) -> int:
    """Return the number of cluster for the gaussian mixture

    Args:
        method (int): Threshold method
        loop_count (int): Current number of loop count

    Returns:
        cluster (int): Number of clusters

    """
    if method == 1:
        if loop_count < 3:
            cluster = 2
        elif 3 <= loop_count < 6:
            cluster = 3
        else:
            raise StopIteration(
                "Hysteresis thresholding process was not able to find suitable cluster parameters."
                "This might be due to poor image quality -> i.e. low SNR, presence of artifact, etc."
            )
    else:
        cluster = 1

        if loop_count > 2:
            raise StopIteration(
                "Hysteresis thresholding process was not able to find suitable cluster parameters."
                "This might be due to poor image quality -> i.e. low SNR, presence of artifact, etc."
            )

    return cluster


def hysteresis_thresholding_cube(
    data: np.ndarray, mask: np.ndarray, voxel_size: float = 0.625, method: int = 2
) -> np.ndarray:
    """Compute and apply thresholding using gaussian mixture on the input data.

    Args:
        data (np.ndarray): Data to apply hysteresis thresholding.
        mask (np.ndarray): Used to crop the image to constrain thresholding.
        voxel_size (float): Isotropic voxel size
        method (int): Threshold methods

    Returns:
        np.ndarray: input image thresholded.
    """
    # Finding bounding index of cube
    (sag, coro, axial) = np.nonzero(mask)
    sag_min = sag.min()
    sag_max = sag.max()
    coro_min = coro.min()
    coro_max = coro.max()
    axial_min = axial.min()
    axial_max = axial.max()

    # Getting the cube to predict on the TOF
    cube_tof = data[sag_min:sag_max, coro_min:coro_max, axial_min:axial_max]

    if method not in [1, 2]:
        raise ValueError("Only 2 threshold methods implemented. Got {}".format(method))

    if method == 1:
        firt_percentile = np.quantile(cube_tof, 0.01)
        cube_tof_cleaned = cube_tof[np.where(cube_tof > firt_percentile)]
        flatten_cube = cube_tof_cleaned.reshape(-1, 1)

    else:
        blur = gaussian_filter(cube_tof, sigma=2.5 / voxel_size)
        flatten_cube = blur.reshape(-1, 1)

    lowt = 1
    hight = 0

    loop_count = 0

    while lowt > hight:
        cluster = cube_gm_number_clusters(method=method, loop_count=loop_count)

        # Initializing EM algorithm
        mixture = GaussianMixture(
            n_components=cluster, tol=0.0001, max_iter=300, n_init=5, verbose=1, random_state=0
        )

        mixture.fit(flatten_cube)

        means = mixture.means_.flatten()
        variances = mixture.covariances_.flatten()

        high_indx = np.argmax(means)

        if cluster in [1, 2]:
            low_indx = np.argmin(means)
        else:
            low_indx = np.where(means == sorted(means)[1])[0][0]

        max_mean = means[high_indx]
        max_var = variances[high_indx]

        min_mean = means[low_indx]
        min_var = variances[low_indx]

        logging.info(f" +++ Means: {means}")
        logging.info(f" +++ STD: {np.sqrt(variances)}")

        if method == 1:
            high_factor = 0.5
        else:
            high_factor = 6

        lowt, hight = thresholds_sigma(
            min_mean, min_var, max_mean, max_var, low_factor=3, high_factor=high_factor
        )

        if lowt > hight:
            logging.info("Thresholds not suitable.")
        else:
            logging.info("Suitable thresholds found")

        loop_count += 1

    hyst = apply_hysteresis_threshold(cube_tof, low=lowt, high=hight)
    out_img = np.zeros(data.shape, dtype=float)
    out_img[sag_min:sag_max, coro_min:coro_max, axial_min:axial_max] = hyst

    return out_img


def apply_hysteresis_threshold(image: np.array, low: float, high: float, is_3d: bool = False):
    """Apply hysteresis thresholding to ``image``.

    This algorithm finds regions where ``image`` is greater than ``high``
    OR ``image`` is greater than ``low`` *and* that region is connected to
    a region greater than ``high``.

    Parameters
    ----------
    image : array, shape (M,[ N, ..., P])
        Grayscale input image.
    low : float, or array of same shape as ``image``
        Lower threshold.
    high : float, or array of same shape as ``image``
        Higher threshold.
    is_3d : bool,
        Flag if 3D structure is needed for connectivity

    Returns
    -------
    thresholded : array of bool, same shape as ``image``
        Array in which ``True`` indicates the locations where ``image``
        was above the hysteresis threshold.

    Examples
    --------
    >>> image = np.array([1, 2, 3, 2, 1, 2, 1, 3, 2])
    >>> apply_hysteresis_threshold(image, 1.5, 2.5).astype(int)
    array([0, 1, 1, 1, 0, 0, 0, 1, 1])

    References
    ----------
    .. [1] J. Canny. A computational approach to edge detection.
           IEEE Transactions on Pattern Analysis and Machine Intelligence.
           1986; vol. 8, pp.679-698.
           :DOI:`10.1109/TPAMI.1986.4767851`
    """
    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
    mask_low = image > low
    mask_high = image > high
    # Connected components of mask_low
    if is_3d:
        structure = np.array(
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ]
        )
    else:
        structure = None

    labels_low, num_labels = ndi.label(mask_low, structure=structure)
    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded


def hysteresis_thresholding_brain(
    vesselness: np.array,
    mask: np.array = None,
    is_3d: bool = False,
    low_factor: int = 3,
    high_factor: int = 0.5,
    simple_thres: bool = False,
) -> np.array:
    """Apply hysteresis thresholding on the whole brain.

    Args:
        vesselness (np.array): VED image.
        mask (np.array): Brain mask.
        is_3d (bool): Connectivity of the FF in hysteresis threshold
        low_factor (float): Low factor for hysteresis threshold.
        high_factor (float): High factor for hysteresis threshold.

    Returns:
        hysteresis (np.array): Thresholded image.

    """

    # Flattening array
    data_flat = vesselness.flatten()

    if mask is not None:
        assert vesselness.shape == mask.shape, "VED and mask shapes must match!"
        mask_flat = mask.flatten()
        data_flat = data_flat[np.where(mask_flat != 0)]

    training_cluster = data_flat.reshape(-1, 1)

    clustering = GaussianMixture(
        n_components=3,
        tol=0.001,
        max_iter=100,
        n_init=1,
        verbose=True,
        verbose_interval=10,
    )

    clustering.fit(training_cluster)

    # Loading means and variances into a dictionary
    means = clustering.means_.flatten()
    variances = clustering.covariances_.flatten()

    # Finding High bound parameters
    median = np.median(means)

    median_index = np.where(means == median)[0][0]
    high_index = np.argmax(means)

    max_mean = means[high_index]
    max_var = variances[high_index]

    median_mean = means[median_index]
    median_var = variances[median_index]

    lowt, hight = thresholds_sigma(
        median_mean, median_var, max_mean, max_var, low_factor=low_factor, high_factor=high_factor
    )

    if simple_thres:
        hysteresis = vesselness > lowt

    else:
        # Using Hysteresis Thresholding
        hysteresis = apply_hysteresis_threshold(vesselness, low=lowt, high=hight, is_3d=is_3d)

    return hysteresis


def extend_markers(
    vessels: np.array,
    markers: np.array,
    rad: int = 2,
    connectivity: int = 1,
    extension: bool = False,
) -> np.array:
    """Extend markers, apply a binary erosion before.

    Args:
        vessels (np.array): Binary vessels.
        markers (np.array): Circle of Willis.
        rad (int): Research radius for extension.

    Returns:
        labels (np.array): Watershed image extended.
    """

    # TODO: Find this bug -> The only reason that we do an erosion
    #  is because if we don't do it the watershed function doesn't works.
    #  The problem was posted on stackoverflow:
    #  https://stackoverflow.com/questions/62720690/why-does-the-watershed-function-of-scikit-image-behave-differently-on-linux-os-a
    binary_markers = markers.copy()
    binary_markers[np.where(binary_markers != 0)] = 1
    binary_markers = binary_erosion(binary_markers, structure=ball(1), iterations=1)
    markers = markers * binary_markers

    return segment_vessels(
        vessels, markers, radius=rad, connectivity=connectivity, extension=extension
    )


def extend_labels(vessels: np.array, labels: np.array, radius: int = 3) -> np.array:
    """Extend labels to connect regions.

    Args:
        vessels (np.array): Binary vessels.
        labels (np.array): Watershed image.
        radius (int): Research radius for extension.

    Returns:
        markers (np.array): Extended labels
    """
    data = vessels.astype(int)
    markers = labels.astype(int)

    binary_labels = np.zeros_like(markers)
    binary_labels[markers != 0] = 1

    dilation = binary_dilation(binary_labels, structure=ball(radius)).astype(int)
    dilation *= data
    dilation[markers != 0] = 0

    end_markers = binary_dilation(dilation, structure=ball(radius)).astype(int)
    end_markers *= data
    end_markers[markers == 0] = 0
    end_markers *= markers

    logging.info("===> Indexing Vascular Voxels")
    label_locations = np.transpose(np.where(end_markers != 0))

    # TODO: Check with propagate_closest_value and mask_vascular_perfusion to merge functionalities
    logging.info("===> Finding Suitable Extensions")
    for index, value in tqdm(np.ndenumerate(dilation), total=dilation.size):
        # If the index is a mask voxel we try to compute the nearest vascular distance within a
        # search sphere of given input radius
        if value != 0:
            xa = np.zeros((1, 3))
            xa[0] = index
            distances = dist.cdist(xa, label_locations, metric="euclidean")
            smallest = np.min(distances)

            if smallest < 2:
                distances = distances.reshape(-1)
                value_index = np.where(distances == smallest)[0][0]
                value_index = label_locations[value_index].astype(int)
                markers[index] = end_markers[tuple(value_index)]

    return markers


def watershed_segment(vessels: np.array, markers: np.array, connectivity: int = 1) -> np.array:
    """Watershed binary vessels with markers.

    Args:
        vessels (np.array): Binary vessels.
        markers (np.array): Circle of Willis.

    Returns:
        segmentation (np.array): Watershed image.
    """

    if ArterialEnum.Acom.value in markers:
        markers[markers == ArterialEnum.Acom.value] = 0

    distances = distance_transform_edt(vessels)
    return watershed(-distances, markers, connectivity=connectivity, mask=vessels)


def segment_vessels(
    vessels_data: np.array,
    markers_data: np.array,
    radius: int = 2,
    connectivity: int = 1,
    extension: bool = False,
) -> np.array:
    """Watershed markers (cow) in binary vessels. Extend labels at the end of each branch.

    Args:
        vessels_data (np.array): Binary vessels.
        markers_data (np.array): Circle of Willis.
        radius (int): Radius of research for the label extension.

    Returns:
        markers_data (np.array): New markers.
    """

    if extension:
        converge = False
        counts = [0]
        for counter in range(10):
            logging.info(f"Segmentation Loop #{counter + 1}")

            segmentation = watershed_segment(vessels_data, markers_data, connectivity=connectivity)
            markers_data = extend_labels(vessels_data, segmentation, radius=radius)
            counts.append(np.count_nonzero(markers_data))

            logging.info(f"Segmentation Voxel Counts:\n, {counts}, \n")
            if counts[-1] == counts[-2]:
                converge = True
                break

        if not converge:
            logging.info("Segmentation did not converge")

    else:
        markers_data = watershed_segment(vessels_data, markers_data, connectivity=connectivity)

    return markers_data


def extract_vessels_ved(
    input_image: str,
    output_image: str,
    mask: np.array,
    output: str,
    sigma_min: int = 0.3,
    sigma_max: int = 6,
    num_scale: int = 20,
    iteration: int = 1,
    move: bool = True,
) -> np.array:
    """Compute VED on raw TOF image.

    Args:
        input_image (str): Path for the input image.
        output_image (str): Path for the output image.
        mask (np.array): Brain mask.
        output(str): Output directory.
        sigma_min (float): Minimum sigma (scale)
        sigma_max (float): Maximum sigma (scale)
        num_scale (float): Number of scale between min and max sigma
        iteration (float): Number of iteration
        move(bool): If enable moves the ComputeVED outputs.

    Returns:
        vesselness (np.array): VED image.
    """
    # TODO: this need to be cleaned to use the right number of processor.
    number_of_thread = multiprocessing.cpu_count()
    logging.info(f"Executing VED with {number_of_thread} CPUs")
    os.system(f"export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={number_of_thread}")

    # Extracting Vessels With VED
    vessel_enhancing_diffussion_wrapped = ComputeVED(
        input_filename=input_image,
        output_filename=output_image,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        number_scale=num_scale,
        dark_blood=False,
        alpha=0.5,
        beta=1.0,
        c=0.00001,
        number_iterations=iteration,
        sensitivity=5.0,
        wstrength=50.0,
        epsilon=0.1,
        generate_iteration_files=False,
        generate_scale=False,
        generate_hessian=False,
        out_folder=output,
        frangi_only=False,
        scale_object=False,
    )
    vessel_enhancing_diffussion_wrapped.valid_arg()
    vessel_enhancing_diffussion_wrapped.run()

    if move:
        # We need to move some file generated by the ComputeVED script to continue the processing.
        for star in ["Ved_*", "Scale_*", "diffusion_*"]:
            for f in glob(star):
                shutil.move(f, join(output, f))

    vesselness = concatenate_scale(os.path.dirname(output_image), mask=mask)

    return vesselness


def concatenate_scale(path: str, mask: np.array) -> np.array:
    """Concatenate different VED scale and keep the maximum.
    The output image is a 3D array with only the maximum value in all the different scales.

    Args:
        path (str): Path where the different scale files are.
        mask (np.array): Brain mask.

    Returns:
        max_intensity (np.array): 3D array of the maximum intensity in each scale.
    """

    scale_files = list(glob(os.path.join(path, "Scale_rescaled*.nii.gz")))

    logging.info(f"Concatenating: {scale_files}")
    max_intensity = nib.load(scale_files[0]).get_fdata("unchanged")
    for scale_file in tqdm(scale_files[1:]):
        file_array = nib.load(scale_file).get_fdata("unchanged")
        cat_array = np.stack((max_intensity, file_array), axis=3)

        max_intensity = np.amax(cat_array, axis=3)

    mask_dilated = binary_dilation(mask, structure=ball(2)).astype(int)

    max_intensity = mask_dilated.astype(bool) * max_intensity

    return max_intensity


def correct_watershed(markers_data: np.array, labels: np.array) -> np.array:
    """Correct watershed image.

    Args:
        markers_data (np.array): Circle of Willis from neural network.
        labels (np.array): watershed image.

    Returns:
        labels (np.array): corrected labels.
    """

    marker_to_rmv = [
        ArterialEnum.LCAR,
        ArterialEnum.RCAR,
        ArterialEnum.BAS,
        ArterialEnum.Acom,
        ArterialEnum.LPcom,
        ArterialEnum.RPcom,
        ArterialEnum.LPCA1,
        ArterialEnum.RPCA1,
        ArterialEnum.LSCA,
        ArterialEnum.RSCA,
        ArterialEnum.LAChA,
        ArterialEnum.RAChA,
    ]

    for marker in marker_to_rmv:
        labels[labels == marker.value] = 0
        labels[markers_data == marker.value] = marker.value

    return labels


def mask_vascular_perfusion(
    mask: np.array, labels: np.array, voxel_dims: Tuple[float, float, float]
) -> Tuple[np.array, np.array]:
    """Function that mask vascular perfusion territories in the brain.

    Args:
        mask (np.array): Brain mask.
        labels (np.array): Labels array.
        voxel_dims (Tuple[float, float, float]): Voxel dimension in mm³.

    Returns:
        closest_values (np.array): Closest label in the brain.
        distance_to_vessels (np.array): Distance to nearest vessel.

    """
    dims = np.array(voxel_dims)

    artery_locations = np.zeros((np.count_nonzero(labels), 3))
    i = 0
    for index, value in np.ndenumerate(labels):
        if value != 0:
            artery_locations[i] = np.array(list(index))
            i += 1

    closest_values = np.zeros(labels.shape, dtype=float)
    distance_to_vessels = np.zeros(labels.shape, dtype=float)

    # TODO: Check with propagate_closest_value and extend_labels to merge functionalities
    for index, value in tqdm(np.ndenumerate(mask), total=mask.size):
        # If the index is a mask voxel we try to compute the nearest vascular distance within a
        # search sphere of given input radius
        if value != 0:
            xa = np.zeros((1, 3))
            xa[0] = np.array(list(index))
            distances = dist.cdist(xa, artery_locations, metric="euclidean")

            distance_to_vessels[index] = np.min(distances) * np.mean(dims)

            distances = distances.reshape(-1)
            value_index = np.where(distances == np.min(distances))[0][0]
            value_index = artery_locations[value_index].astype(int)
            closest_values[index] = labels[value_index[0], value_index[1], value_index[2]]

    return closest_values, distance_to_vessels


def dipy_nlmeans():
    return None
