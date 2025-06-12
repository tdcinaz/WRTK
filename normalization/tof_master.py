# -*- coding: utf-8 -*-

import multiprocessing
from os.path import join
import os
from typing import Optional, Tuple, Union
import nibabel as nib
from nipype.interfaces import afni
from nipype.interfaces.ants import ApplyTransforms, Registration
import numpy as np
import subprocess
import pathlib
from raster_geometry import cube
from scipy.ndimage import center_of_mass
from typing import List

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


def resample(
    in_file: str,
    out_file: str,
    resolution: Optional[Tuple] = None,
    master: Optional[str] = None,
    resample_mode="Cu",
) -> nib.Nifti1Image:
    """Resample volume using afni.

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


def brain_extract(in_file: str,
                  out_mask: str,
                  out_brain: str,
                  modality: str = "auto",
                  use_gpu = False,
                  use_hdbet = False,
                  ) -> nib.Nifti1Image:
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
    threads = multiprocessing.cpu_count()

    # crude modality check (fallback if user does not pass it)
    if modality == "auto":
        hdr = nib.load(in_file).header
        modality = "CT" if hdr.get("db_name",b"").startswith(b"CT") or hdr.get_xyzt_units()[0] == "unknown" else "MRI"

    tool = None
    if modality.upper() == "MRI":
        try:
            subprocess.run(["mri_synthstrip", "-i", in_file,
                            "-m", out_mask, "-o", out_brain, "-t", str(threads)] + (["-g"] if use_gpu else []), check=True)
        except:
            tool = "afni"

        '''if use_hdbet:
            try:                               # fastest if available
                #subprocess.run(["hd-bet", "-i", in_file, "-o", out_mask,
                #                "-device", "cpu", "-mode", "fast"], check=True)
                subprocess.run(["hd-bet", "-i", in_file, "-o", out_mask, "-device"] + (["0"] if use_gpu else ["cpu"]), check=True)
                tool = "HD‑BET"
                print("Skull stripped with HD-BET")
            except:
                print("HD‑BET not found, falling back to afni")
                tool = "afni"
        else:
            tool = "afni"'''
        

    if modality.upper() == "CT":
        try:
            subprocess.run(["mri_synthstrip", "-i", in_file,
                            "-m", out_mask, "-o", out_brain, "-t", str(threads)] + (["-g"] if use_gpu else []), check=True)
            tool = "mri_synthstrip"
        except FileNotFoundError:
            tool = "TotalSegmentator"

    if tool == "afni":                      # fallback for MRI only
        ss = afni.SkullStrip(in_file=in_file,
                             out_file=out_mask,
                             args="-overwrite",
                             outputtype="NIFTI_GZ",
                             num_threads=threads)
        ss.run()
        print("afni skull stripping complete")
    elif tool == "TotalSegmentator":       # fallback for CT only
        tmp_dir = pathlib.Path(out_mask).with_suffix("")
        subprocess.run(["TotalSegmentator", "-i", in_file,
                        "-o", str(tmp_dir), "--fast"], check=True)
        brain_label = tmp_dir / "brain.nii.gz"
        os.rename(brain_label, out_mask)
    # apply the mask to generate brain‑only volume
    subprocess.run(["fslmaths", in_file, "-mas", out_mask, out_brain], check=True)
    return nib.load(out_mask)


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
    scan_obj: nib.Nifti1Image = nib.load(str(scan_nii))
    mask_obj: nib.Nifti1Image = nib.load(str(roi_mask_nii))
    seg_obj: nib.Nifti1Image = nib.load(str(seg_mask_nii))

    scan      = scan_obj.get_fdata(dtype=np.float32, caching='unchanged')
    mask_data = mask_obj.get_fdata(dtype=np.float32,  caching='unchanged').astype(np.uint8)
    seg_data  = seg_obj.get_fdata(dtype=np.float32, caching='unchanged').astype(np.uint8)

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
    hdr: nib.Nifti1Header = scan_obj.header.copy()
    hdr.set_data_dtype(np.float32)

    nib.save(nib.Nifti1Image(cropped_scan, new_affine, hdr), out_scan_nii)

    hdr_seg: nib.Nifti1Header = seg_obj.header.copy()
    hdr_seg.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(cropped_seg, new_affine, hdr_seg), out_seg_nii)


    if out_mask_nii is not None:
        hdr_mask: nib.Nifti1Header = mask_obj.header.copy()
        hdr_mask.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(cropped_mask, new_affine, hdr_mask), out_mask_nii)

    if return_bbox:
        return (imin, imax, jmin, jmax, kmin, kmax), new_affine


def coregister_ct_mr(
        fixed_img: str,      # e.g. MR brain volume
        moving_img: str,      # e.g. CT brain volume
        fixed_mask: str,      # MR brain mask  (binary)
        moving_mask: str,      # CT brain mask  (binary)
        out_moving_aligned: str,
        out_moving_mask_aligned: str,
        transform_prefix: str = "CT2MR_",
        use_syn: bool = False
    ) -> Tuple[str, List[str]]:
    """
    Rigid → Affine (→ optional SyN) registration of a CT scan to an MR scan
    using brain masks as ROIs.  Returns path to the aligned CT image and the
    list of forward transforms in ANTs order.
    """
    fixed_img   = str(fixed_img)
    moving_img  = str(moving_img)
    fixed_mask  = str(fixed_mask)
    moving_mask = str(moving_mask)
    transform_prefix = str(transform_prefix)

    # ---------------- 1. build the ANTs registration object -----------------
    reg = Registration()
    reg.inputs.verbose            = True
    reg.inputs.fixed_image        = fixed_img
    reg.inputs.moving_image       = moving_img
    #reg.inputs.fixed_image_masks  = [fixed_mask] * (3 if use_syn else 2)
    #reg.inputs.moving_image_masks = [moving_mask] * (3 if use_syn else 2)

    reg.inputs.transforms              = ['Rigid', 'Affine'] + (['SyN'] if use_syn else [])
    reg.inputs.transform_parameters    = [(0.1,), (0.1,)]     + ([(0.1, 3, 0)] if use_syn else [])
    reg.inputs.metric                  = ['MI',  'MI']        + (['CC'] if use_syn else [])
    reg.inputs.metric_weight           = [1, 1]               + ([1] if use_syn else [])
    reg.inputs.radius_or_number_of_bins= [32, 32]             + ([4] if use_syn else [])

    # multiresolution schedules
    reg.inputs.number_of_iterations    = [[1000, 500, 250, 100],
                                          [1000, 500, 250, 100]] + \
                                         ([[100, 50, 20]] if use_syn else [])
    reg.inputs.shrink_factors          = [[8, 4, 2, 1]]*2 + ([[4, 2, 1]] if use_syn else [])
    reg.inputs.smoothing_sigmas        = [[3, 2, 1, 0]]*2 + ([[1, .5, 0]] if use_syn else [])

    reg.inputs.write_composite_transform = True
    reg.inputs.output_warped_image       = str(out_moving_aligned)
    reg.inputs.output_inverse_warped_image = False
    reg.inputs.output_transform_prefix   = transform_prefix
    reg.inputs.num_threads               = multiprocessing.cpu_count()
    reg.inputs.float                     = True
    reg.inputs.args                      = "-u"        # avoid histogram‐match CSF/contrast swap

    reg_res = reg.run()                                              # 🚀 run ANTs
    fwd_xforms = reg_res.outputs.composite_transform                 # ordered for antsApplyTransforms
    inv_xforms = reg_res.outputs.inverse_composite_transform

    apply_transform(
        moving_mask,
        fixed_mask,
        fwd_xforms,
        str(out_moving_mask_aligned),
    )

    return str(out_moving_aligned), fwd_xforms, inv_xforms


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