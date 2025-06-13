import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import vtk
from vtkmodules.util import numpy_support as ns
import os
import sys

def compute_average_diameters_and_export_vtp(nifti_path, output_vtp="centerlines.vtp", label_range=range(1, 14)):
    # Load NIfTI image
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    voxel_spacing = img.header.get_zooms()  # (x, y, z) in mm

    print(f"Voxel spacing: {voxel_spacing} mm")
    print(f"Affine:\n{affine}")

    avg_diameters = {}

    # Store data for VTK export
    all_points_mm = []
    all_radii_mm = []
    all_labels = []

    for label in label_range:
        print(f"\nProcessing label {label}...")
        vessel_mask = (data == label)

        if not np.any(vessel_mask):
            print("  No voxels found.")
            avg_diameters[label] = 0.0
            continue

        # Distance map (euclidean distance in mm)
        distance_map = distance_transform_edt(vessel_mask, sampling=voxel_spacing)

        # Skeletonize
        skeleton = skeletonize(vessel_mask)
        if not np.any(skeleton):
            print("  Skeletonization failed.")
            avg_diameters[label] = 0.0
            continue

        radii_mm = distance_map[skeleton > 0]
        diameters_mm = 2 * radii_mm

        if len(diameters_mm) == 0:
            print("  No centerline voxels found.")
            avg_diameters[label] = 0.0
            continue

        avg_diameters[label] = np.mean(diameters_mm)
        print(f"  Average diameter: {avg_diameters[label]:.3f} mm ({len(diameters_mm)} voxels)")

        # Get voxel indices of centerline
        zyx_coords = np.array(np.where(skeleton > 0)).T  # shape (N, 3)

        # Convert to physical coordinates using affine
        homogeneous_coords = np.c_[zyx_coords[:, ::-1], np.ones(len(zyx_coords))]  # (x, y, z, 1)
        world_coords = (affine @ homogeneous_coords.T).T[:, :3]

        all_points_mm.append(world_coords)
        all_radii_mm.append(radii_mm)
        all_labels.append(np.full(len(radii_mm), label))

    # Write .vtp file
    if all_points_mm:
        export_centerline_spheres_to_vtp(
            output_vtp,
            np.vstack(all_points_mm),
            np.concatenate(all_radii_mm),
            np.concatenate(all_labels)
        )

    return avg_diameters


def export_centerline_spheres_to_vtp(output_path, points_mm, radii_mm, labels):
    """
    Save centerline points and radii as VTK PolyData for ParaView visualization.
    """
    points = vtk.vtkPoints()
    for pt in points_mm:
        points.InsertNextPoint(pt.tolist())

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # Add radius array
    radius_array = ns.numpy_to_vtk(radii_mm, deep=True)
    radius_array.SetName("Radius_mm")
    polydata.GetPointData().AddArray(radius_array)

    # Add label array
    label_array = ns.numpy_to_vtk(labels.astype(np.int32), deep=True)
    label_array.SetName("Label")
    polydata.GetPointData().AddArray(label_array)

    # Write .vtp
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()
    print(f"\n✅ VTK file saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_centerline_diameters_and_export_vtp.py <path_to_labeled_nifti>")
        sys.exit(1)

    nifti_file = sys.argv[1]
    output_file = sys.argv[2]

    diameters = compute_average_diameters_and_export_vtp(nifti_file, output_vtp=output_file)

    print("\nAverage Diameters for Labels 1–13:")
    for label, d in diameters.items():
        print(f"Label {label:2d}: {d:.3f} mm")
