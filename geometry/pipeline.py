import nibabel as nib
from geometry_master import (
    compute_label_volumes,
    nifti_to_vtk_image_data,
    extract_labeled_surface_from_volume,
    save_surface_to_vtp
)

def pipeline(input_segmentation, fileId, output_dir):
    """
    Main workflow:
      1) Load an existing NIfTI file from disk.
      2) Extract the labeled surface vtkPolyData
      3) Save the surface to a .vtp file.

    Args:
        input_nii_file (str): Path to the input NIfTI file.
        output_vtp_file (str): Desired output VTP file path.
    """

    # 1) Load the NIfTI with nibabel
    nifti_img = nib.load(input_segmentation)

    volume_dict = compute_label_volumes(nifti_img)
    print(volume_dict)

    vtk_image = nifti_to_vtk_image_data(nifti_img)
    print("Image Loaded")

    # 2) Extract surface

    labeled_polydata = extract_labeled_surface_from_volume(vtk_image)
    print("Surface Extracted")

    save_surface_to_vtp(resampled_surface, out_surface)

    print(f"Surface extracted and saved to '{out_surface}'")