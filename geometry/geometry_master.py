import sys
import numpy as np
import nibabel as nib
import vtk
from vtkmodules.util import numpy_support
import pyvista as pv
import pyacvd
import logging
from collections import defaultdict, deque
from scipy.spatial import cKDTree
import os
import re
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev, make_splprep
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
import networkx as nx

def nifti_to_pv_image_data(nifti_img):
    """
    Convert a 3D NIfTI image to vtkImageData.

    Args:
        nifti_img (nibabel.Nifti1Image): The input NIfTI image.

    Returns:
        pv.ImageData: The image data in pyvista format.
    """
    # 1) Get the NumPy array from the NIfTI
    data_array = nifti_img.get_fdata(dtype=np.float32)

    # Ensure it's contiguous in memory
    data_array = np.ascontiguousarray(data_array)

    # 2) Retrieve voxel spacing from the header
    #    If it's a 4D image, nibabel header might return 4 zooms, so we take only the first three.
    pixdim = nifti_img.header.get_zooms()[:3]

    # 3) Extract the translation (origin) from the affine
    origin = nifti_img.affine[:3, 3]

    # 4) Create pyvista ImageData

    pv_image = pv.ImageData(
        dimensions=data_array.shape, 
        spacing=(pixdim[2], pixdim[1], pixdim[0]), 
        origin=(origin[2], origin[1], origin[0]),
        deep=True)
    
    pv_image.point_data["Scalars"] = data_array.flatten()

    return pv_image

def compute_label_volumes(img, label_range=range(1, 14)):
    # Load the image
    data = img.get_fdata()
    
    # Get the voxel size from the header (voxel volume in mm³)
    voxel_volume = np.prod(img.header.get_zooms())
    print(f"Voxel volume: {voxel_volume:.2f} mm³")

    # Compute volumes for each label
    volumes = {}
    for label in label_range:
        voxel_count = np.sum(data == label)
        volumes[label] = voxel_count * voxel_volume

    return volumes


def compute_skeleton(nifti_img: nib.Nifti1Image):
    data_array = nifti_img.get_fdata()
    affine = nifti_img.affine
    voxel_spacing = nifti_img.header.get_zooms()

    new_data_array = np.repeat(np.repeat(np.repeat(data_array, 2, axis=0), 2, axis=1), 2, axis=2)
    new_affine = nifti_img.affine.copy()
    new_hdr = nifti_img.header.copy()

    new_affine[:3, :3] /= 2.0
    new_hdr.set_zooms(tuple(z/2 for z in voxel_spacing[:3]) + voxel_spacing[3:])
    new_voxel_spacing = new_hdr.get_zooms()

    distance_map = distance_transform_edt(new_data_array, sampling=new_voxel_spacing)
    #distance_map = distance_transform_edt(data_array, sampling=voxel_spacing)


    skeleton = skeletonize(new_data_array)
    if not np.any(skeleton):
        print("  Skeletonization failed.")
        return
    
    radii_mm = distance_map[skeleton > 0]
    labels = new_data_array[skeleton > 0]
    zyx_coords = np.array(np.where(skeleton > 0)).T

    homogeneous_coords = np.c_[zyx_coords[:, ::-1], np.ones(len(zyx_coords))]  # (x, y, z, 1)
    world_coords = (new_affine @ homogeneous_coords.T).T[:, :3]

    pv_skeleton = pv.PolyData(world_coords)
    pv_skeleton.point_data["Radius"] = radii_mm.flatten()
    pv_skeleton.point_data["Artery"] = labels.flatten()

    return pv_skeleton

def extract_start_and_end_voxels(nifti_img: nib.Nifti1Image, pv_image: pv.ImageData, skeleton: pv.PolyData):
    

    surface_net = extract_labeled_surface_from_volume(pv_image)

    standardAdj = {
        1.0: (2.0, 3.0),            # Bas -> L-PCA, R-PCA
        2.0: (1.0, 8.0),            # L-PCA -> Bas, L-Pcom
        3.0: (1.0, 9.0),            # R-PCA -> Bas, R-Pcom
        4.0: (5.0, 8.0, 11.0),      # L-ICA -> L-MCA, L-Pcom, L-ACA
        5.0: (4.0,),                # L-MCA -> L-ICA
        6.0: (7.0, 9.0, 12.0),      # R-ICA -> R-MCA, R-Pcom, R-ACA
        7.0: (6.0,),                # R-MCA -> R-ICA
        8.0: (2.0, 4.0),            # L-Pcom -> L-PCA, L-ICA
        9.0: (3.0, 6.0),            # R-Pcom -> R-PCA, R-ICA
        10.0: (11.0, 12.0),         # Acom -> L-ACA, R-ACA
        11.0: (4.0,),               # L-ACA -> L-ICA
        12.0: (6.0,)                # R-ACA -> R-ICA
    }

    data_array = nifti_img.get_fdata()
    unique_labels = np.unique(data_array)

    if 0 in unique_labels:
        unique_labels = np.delete(unique_labels, 0)

    boundaries = []
    tuples = surface_net.cell_data['BoundaryLabels']
    
    # Extract vessel bifurfaction boundary cells from surface.
    # Find barycenters of boundaries
    
    all_barycenters = {}

    for label1 in unique_labels:
        for label2 in standardAdj[label1]:
            label_mask1 = (tuples == label1).any(axis=1)
            label_mask2 = (tuples == label2).any(axis=1)
            label_mask = label_mask1 & label_mask2

            label_cell_ids = np.nonzero(label_mask)[0]

            centers = surface_net.cell_centers().points[label_cell_ids]  # cell centers for matches
            barycenter = centers.mean(axis=0)                       # (x, y, z)

            
            barycenter_key = f"{label1}/{label2}"

            #if f"{label2}/{label1}" not in all_barycenters.keys():
            all_barycenters[barycenter_key] = barycenter

    # Find nearest point to boundary center point for each artery in skeleton
    boundary_points = {}
    arteries = skeleton.point_data['Artery']
    for key in all_barycenters.keys():
        label = float(key[0:3])
        artery = (arteries == label)
        artery_point_ids = np.nonzero(artery)[0]
        artery_points = skeleton.extract_points(artery_point_ids)
        closest_idx = np.argmin(np.linalg.norm(artery_points.points - all_barycenters[key], axis=1))

        #boundary_points[key] = artery_points.points[closest_idx]
        boundary_points[key] = skeleton.FindPoint(artery_points.points[closest_idx])

    skeleton_points = skeleton.points
    skeleton_labels = np.zeros(skeleton_points.shape[0])

    for idxPoints in boundary_points.values():
        skeleton_labels[idxPoints] = 1

    skeleton.point_data['CenterlineLabels'] = skeleton_labels

def spline_interpolation(
    poly: pv.PolyData,
):
    # ---------------- 0. your artery PolyData is already in `poly` --------------
    artery_lbl   = poly["Artery"]              #   point-data array (float)
    cent_lbl     = poly["CenterlineLabels"]    #   point-data array (float)
    radius       = poly["Radius"]              #   local inscribed-sphere radius

    network = pv.PolyData()
    print(network)

    unique_labels = np.unique(artery_lbl)
    for label in unique_labels:
        # -------- 1. isolate the artery-of-interest ------------------
        keep_mask    = artery_lbl == label
        idx_subset   = np.flatnonzero(keep_mask)

        pts_subset   = poly.points[idx_subset]     # (M,3) points on this artery
        cent_subset  = cent_lbl[idx_subset]        # matching centreline labels
        rad_subset   = radius[idx_subset]          # matching radii

        ordered_local = order_artery_points(pts_subset, cent_subset, k=6)
        ordered_global = idx_subset[ordered_local]     # indices w.r.t. the *full* PolyData

        # -------- 2. locate the (single) start point (CentrelineLabels == 1.0) ------
        #start_local  = np.flatnonzero(cent_subset == 1.0)[0]           # index in subset
        #pts_reordered = np.vstack((pts_subset[start_local+1:],
        #                        pts_subset[:start_local]))
        #rad_reordered = np.hstack((rad_subset[start_local+1:],
        #                        rad_subset[:start_local]))
        pts_reordered = pts_subset[ordered_local]
        rad_reordered = rad_subset[ordered_local]

        # -------- 3. chord-length parameterisation ----------------------------------
        seg_len   = np.linalg.norm(np.diff(pts_reordered, axis=0), axis=1)
        u         = np.hstack(([0.0], np.cumsum(seg_len)))
        u        /= u[-1]                                            # scale → [0,1]

        # -------- 4. build weights from the radius values ---------------------------
        #   bigger radius  ⇒  bigger weight  ⇒  curve more faithful to that point
        #w         = rad_reordered / rad_reordered.max()              # normalise to [0,1]
        w = 0.0 + (rad_reordered - rad_reordered.min()) * (0.8 - 0.0) / (rad_reordered.max() - rad_reordered.min())

        # -------- 5. fit a **smoothing** spline (s>0) with those weights ------------
        k         = min(3, len(pts_reordered) - 1)                   # spline degree
        # heuristic: allow ≈1% average positional deviation
        s_factor  = 0.01 * np.mean(seg_len) * len(pts_reordered)

        tck, _    = make_splprep(pts_reordered.T, u=u, w=w, k=k, s=s_factor)

        # -------- 6. sample the spline densely to obtain the smooth centreline ------
        n_samples = 200
        u_fine    = np.linspace(0.0, 1.0, n_samples)
        x_f, y_f, z_f = tck.__call__(u_fine)
        smooth_pts = np.column_stack((x_f, y_f, z_f))

        # convert to a PyVista poly-line
        cells   = np.hstack(([n_samples], np.arange(n_samples))).astype(np.int64)
        centerline_smooth = pv.PolyData(smooth_pts, lines=cells)

        #print(centerline_smooth)
        network.merge(centerline_smooth, inplace=True)

    #print(network)
    # -------- 7. quick visual sanity-check (optional) ---------------------------
    p = pv.Plotter()
    p.add_mesh(poly, render_points_as_spheres=True, point_size=5, color="lightgray")
    p.add_mesh(network, color="dodgerblue", line_width=4,
            label="Weighted smoothing spline")
    p.add_legend()
    p.show()


def order_artery_points(points: np.ndarray,
                        centerline_labels: np.ndarray,
                        k: int = 6) -> np.ndarray:
    """
    Return indices that reorder `points` so they follow the artery’s centre path.

    Parameters
    ----------
    points : (N, 3) ndarray
        3-D coordinates for one artery.
    centerline_labels : (N,) ndarray
        Array where exactly one entry == 1.0 marks the start point.
    k : int
        How many nearest neighbours to connect in the initial graph.

    Returns
    -------
    order : (M,) ndarray
        Indices into `points` giving the centreline sequence (M==N).
    """
    # --- 1. build k-NN graph -------------------------------------------------
    tree   = cKDTree(points)
    dists, nbrs = tree.query(points, k=k+1)       # self + k neighbours
    row_idx = np.repeat(np.arange(len(points)), k)  # source vertices
    col_idx = nbrs[:, 1:].ravel()                  #  skip self (col 0)
    data    = dists[:, 1:].ravel()

    W = csr_matrix((data, (row_idx, col_idx)), shape=(len(points),)*2)
    W = W.maximum(W.T)                             # make symmetric

    # --- 2. minimum spanning tree for single-path backbone -------------------
    mst = minimum_spanning_tree(W)                # SciPy returns CSR matrix
    mst = mst + mst.T                             # make dense-undirected format

    # --- 3. find start + farthest leaf on the MST ----------------------------
    start_idx = int(np.flatnonzero(centerline_labels == 1.0)[0])

    # use Dijkstra to get all pairwise geodesic distances from start on MST
    dist_from_start, predecessors = dijkstra(mst, directed=False,
                                             indices=start_idx,
                                             return_predecessors=True)
    far_idx = int(dist_from_start.argmax())        # leaf with greatest length

    # --- 4. extract path from start → far_idx --------------------------------
    order = []
    cur = far_idx
    while cur != start_idx:
        order.append(cur)
        cur = predecessors[cur]
    order.append(start_idx)
    order.reverse()                               # now goes start → far leaf

    # `order` may omit interior branches if MST branched; fix by DFS traversal
    visited = set(order)
    stack   = [start_idx]
    G = nx.from_scipy_sparse_array(mst, edge_attribute="weight")

    while stack:
        node = stack.pop()
        for nbr in G.neighbors(node):
            if nbr not in visited:
                # depth-first walk to cover remaining nodes
                visited.add(nbr)
                insert_pos = order.index(node) + 1
                order.insert(insert_pos, nbr)
                stack.append(nbr)

    return np.array(order, dtype=int)


def extract_labeled_surface_from_volume(
    input_vtk_image: pv.ImageData,
) -> pv.PolyData:
    """
    Extract a multi-labeled surface using vtkSurfaceNets3D.
    The output polydata has a cell-data array 'BoundaryLabels'
    indicating which label is adjacent to either side of the cell.

    Args:
        nifti_file (str): Path to a labeled NIfTI image with integer labels.

    Returns:
        vtk.vtkPolyData: A polydata containing the labeled surface
    """


    surface_net = vtk.vtkSurfaceNets3D()
    surface_net.SetInputData(input_vtk_image)
    surface_net.SetBackgroundLabel(0)
    surface_net.SetOutputStyleToDefault()
    surface_net.GenerateLabels(14, 1, 14)
    #surface_net.SmoothingOff()
    #surface_net.SetOutputMeshTypeToQuads()

    surface_net.Update()
    pv_surface_net: pv.PolyData = pv.wrap(surface_net.GetOutput())

    logging.info(f"    ++++ : Cells before cleaning: {pv_surface_net.GetNumberOfCells()}")

    pv_surface_net.clean()

    logging.info(f"    ++++ : Cells after cleaning: {pv_surface_net.GetNumberOfCells()}")

    return pv_surface_net


def extract_individual_surfaces(
    labeledSurface,
    labelArrayName="BoundaryLabels",
    possibleLabels=range(1, 14),
) -> pv.PolyData:

    def attachOriginalCellIds(polyData, arrayName="vtkOriginalCellIds"):
        """
        Attach a cell-data array of unique IDs (0,1,2,...,N-1) 
        to the input vtkPolyData, so that after thresholding 
        you can identify the original cell IDs.
        """
        numCells = polyData.GetNumberOfCells()
        
        # Create a new ID array
        cellIds = vtk.vtkIdTypeArray()
        cellIds.SetName(arrayName)
        cellIds.SetNumberOfComponents(1)
        cellIds.SetNumberOfTuples(numCells)
        
        for i in range(numCells):
            cellIds.SetValue(i, i)
        
        # Attach this array to the cell data
        polyData.GetCellData().AddArray(cellIds)

    def extractNonStandardAdjSurfaces(
        inputPolyData: vtk.vtkPolyData,
        labelID: float,
        standardAdj: dict,
        boundaryArrayName: str = "BoundaryLabels"
    ) -> vtk.vtkPolyData:
        """
        Extract surfaces from a vtkPolyData (generated by vtkSurfaceNets3D)
        whose CellData contains an array 'BoundaryLabels' with 2 components
        (component 0, component 1). The extraction logic is:
        
        1) Keep cells that have labelID in component 0 or 1
        2) Exclude cells if the *other* component is in standardAdj[labelID].
        
        :param inputPolyData: The input surface from which to extract
        :param labelID: The label of interest (e.g. 11.0)
        :param standardAdj: A dict that maps label -> tuple/list of standard-adjacent labels
                        e.g. standardAdj[11.0] = (6.0, 8.0)
        :param boundaryArrayName: The name of the array with boundary labels
        :return: A vtkPolyData containing the extracted surfaces
        """

        # ---------------------------
        # 1. Threshold: component 0 == labelID
        # ---------------------------
        threshComp0 = vtk.vtkThreshold()
        threshComp0.SetInputData(inputPolyData)
        # We tell vtkThreshold that we are working with CellData,
        # array 'BoundaryLabels', and we want to use component 0.
        threshComp0.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, boundaryArrayName
        )
        threshComp0.SetUpperThreshold(labelID)
        threshComp0.SetLowerThreshold(labelID)
        threshComp0.Between(labelID)
        threshComp0.Update()
        
        # We now have an unstructured grid. Next, we need to exclude
        # any cells where component 1 is in standardAdj[labelID].
        # We'll do this by thresholding for component 1 being in that set
        # and then "inverting" that selection.
        
        # For the adjacency we want to exclude:
        adjLabels = standardAdj.get(labelID, ())

        # Convert the above threshold to PolyData so we can re-threshold
        geometryComp0 = vtk.vtkGeometryFilter()
        geometryComp0.SetInputConnection(threshComp0.GetOutputPort())
        geometryComp0.Update()

        usgComp0 = polyDataToUnstructuredGrid(geometryComp0.GetOutput())

        # If we have adjacency labels, filter them out:
        if len(adjLabels) > 0:
            # We'll keep track of anything that matches adjacency in comp1
            # so we can remove it from the "good" set.
            appendedAdjsComp0 = vtk.vtkAppendFilter()
            for adj in adjLabels:
                adjThresh = vtk.vtkThreshold()
                adjThresh.SetInputData(usgComp0)
                adjThresh.SetInputArrayToProcess(
                    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, boundaryArrayName
                )
                adjThresh.SetSelectedComponent(1)  # the "other" component
                adjThresh.SetUpperThreshold(adj)
                adjThresh.SetLowerThreshold(adj)
                adjThresh.Between(adj)
                adjThresh.Update()
                appendedAdjsComp0.AddInputData(adjThresh.GetOutput())

            appendedAdjsComp0.Update()

            # We'll subtract those from usgComp0.
            # Convert each to polydata and use cell IDs. 

            usgToRemoveComp0 = appendedAdjsComp0.GetOutput()
            finalComp0 = subtractCellsById(usgComp0, usgToRemoveComp0)
        else:
            # If no adjacency labels to exclude, keep everything from threshComp0
            finalComp0 = usgComp0

        # Convert finalComp0 to polydata
        geomOutComp0 = vtk.vtkGeometryFilter()
        geomOutComp0.SetInputData(finalComp0)
        geomOutComp0.Update()


        # ---------------------------
        # 2. Threshold: component 1 == labelID
        # ---------------------------
        threshComp1 = vtk.vtkThreshold()
        threshComp1.SetInputData(inputPolyData)
        threshComp1.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, boundaryArrayName
        )
        threshComp1.SetSelectedComponent(1)
        threshComp1.SetUpperThreshold(labelID)
        threshComp1.SetLowerThreshold(labelID)
        threshComp1.Between(labelID)
        threshComp1.Update()

        geometryComp1 = vtk.vtkGeometryFilter()
        geometryComp1.SetInputConnection(threshComp1.GetOutputPort())
        geometryComp1.Update()

        usgComp1 = polyDataToUnstructuredGrid(geometryComp1.GetOutput())

        # Exclude adjacency from component 0 if needed
        if len(adjLabels) > 0:
            appendedAdjsComp1 = vtk.vtkAppendFilter()
            for adj in adjLabels:
                adjThresh = vtk.vtkThreshold()
                adjThresh.SetInputData(usgComp1)
                adjThresh.SetInputArrayToProcess(
                    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, boundaryArrayName
                )
                adjThresh.SetSelectedComponent(0)  # the "other" component now
                adjThresh.SetUpperThreshold(adj)
                adjThresh.SetLowerThreshold(adj)
                adjThresh.Between(adj)
                adjThresh.Update()
                appendedAdjsComp1.AddInputData(adjThresh.GetOutput())

            appendedAdjsComp1.Update()
            usgToRemoveComp1 = appendedAdjsComp1.GetOutput()
            finalComp1 = subtractCellsById(usgComp1, usgToRemoveComp1)
        else:
            finalComp1 = usgComp1

        geomOutComp1 = vtk.vtkGeometryFilter()
        geomOutComp1.SetInputData(finalComp1)
        geomOutComp1.Update()

        swapBoundaryLabelsComponents(geomOutComp1.GetOutput())

        # ---------------------------
        # 3. Combine the results
        # ---------------------------
        appender = vtk.vtkAppendPolyData()
        appender.AddInputData(geomOutComp0.GetOutput())
        appender.AddInputData(geomOutComp1.GetOutput())

        # At this point, appender.GetOutput() is a vtkPolyData combining both sets.
        # You may need a clean-up (e.g. vtkCleanPolyData) if there are duplicates:
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(appender.GetOutputPort())
        cleaner.Update()

        return cleaner.GetOutput()

    def polyDataToUnstructuredGrid(polyData: vtk.vtkPolyData) -> vtk.vtkUnstructuredGrid:
        """
        Helper function to convert a vtkPolyData to vtkUnstructuredGrid
        (useful for more thresholding steps).
        """
        polyDataToUG = vtk.vtkAppendFilter()
        polyDataToUG.AddInputData(polyData)
        polyDataToUG.Update()
        return polyDataToUG.GetOutput()

    def subtractCellsById(
        sourceUG: vtk.vtkUnstructuredGrid,
        removeUG: vtk.vtkUnstructuredGrid,
        originalIdArrayName="vtkOriginalCellIds"
    ) -> vtk.vtkUnstructuredGrid:
        """
        Build a set of OriginalCellIds from 'removeUG' and
        remove them from 'sourceUG' by thresholding on a KeepMask array.
        Returns a new vtkUnstructuredGrid with those cells excluded.
        """

        # 1) Collect the IDs to remove
        removeIDArray = removeUG.GetCellData().GetArray(originalIdArrayName)
        if not removeIDArray:
            # Nothing to remove if no ID array found
            return sourceUG

        removeIDSet = set()
        for i in range(removeUG.GetNumberOfCells()):
            removeIDSet.add(removeIDArray.GetValue(i))

        # 2) Create keep/remove mask in sourceUG and threshold
        filteredUG = makeCellMaskById(
            sourceUG=sourceUG,
            removeIDSet=removeIDSet,
            originalIdArrayName=originalIdArrayName
        )

        return filteredUG

    def makeCellMaskById(
        sourceUG: vtk.vtkUnstructuredGrid,
        removeIDSet: set,
        originalIdArrayName="OriginalCellIds",
        maskArrayName="KeepMask"
    ) -> vtk.vtkUnstructuredGrid:
        """
        From 'sourceUG', create a new 'KeepMask' cell-data array:
        - 1 if cell's OriginalCellIds is *not* in removeIDSet
        - 0 if cell's OriginalCellIds *is* in removeIDSet
        Then threshold on KeepMask >= 1 to output a new vtkUnstructuredGrid
        containing only the "kept" cells.
        """

        # 1) Create the mask array
        numCells = sourceUG.GetNumberOfCells()
        keepMask = vtk.vtkFloatArray()
        keepMask.SetName(maskArrayName)
        keepMask.SetNumberOfComponents(1)
        keepMask.SetNumberOfTuples(numCells)

        origIdArray = sourceUG.GetCellData().GetArray(originalIdArrayName)
        if not origIdArray:
            raise ValueError(f"Array '{originalIdArrayName}' not found in CellData.")
        
        for i in range(numCells):
            originalId = origIdArray.GetValue(i)
            if originalId in removeIDSet:
                keepMask.SetValue(i, 0.0)  # to remove
            else:
                keepMask.SetValue(i, 1.0)  # to keep

        sourceUG.GetCellData().AddArray(keepMask)

        # 2) Threshold on keepMask >= 1
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(sourceUG)
        thresh.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, maskArrayName
        )
        # We want to keep cells with mask >= 1
        thresh.SetThresholdFunction(vtk.vtkThreshold().THRESHOLD_UPPER)
        thresh.SetUpperThreshold(0.5)
        thresh.Update()

        return thresh.GetOutput()  # vtkUnstructuredGrid

    def swapBoundaryLabelsComponents(polyData, boundaryArrayName="BoundaryLabels"):
        """
        For each cell in 'polyData', swap the two components of the
        boundaryArrayName cell-data array. i.e. comp0 <-> comp1.
        """
        arr = polyData.GetCellData().GetArray(boundaryArrayName)
        if not arr or arr.GetNumberOfComponents() != 2:
            # Either the array doesn't exist or doesn't have 2 components.
            return

        numTuples = arr.GetNumberOfTuples()

        # A simple approach is to create a new array, then replace the old one.
        swappedArr = vtk.vtkFloatArray()
        swappedArr.SetName(boundaryArrayName)
        swappedArr.SetNumberOfComponents(2)
        swappedArr.SetNumberOfTuples(numTuples)

        for i in range(numTuples):
            comp0 = arr.GetComponent(i, 0)
            comp1 = arr.GetComponent(i, 1)
            # Swap
            arr.SetComponent(i, 0, comp1)
            arr.SetComponent(i, 1, comp0)

    labelArray = labeledSurface.GetCellData().GetArray(labelArrayName)

    # Determine which label values actually exist in the surface
    existingLabels = set()
    for i in range(labelArray.GetNumberOfTuples()):
        existingLabels.add(int(labelArray.GetTuple2(i)[0]))

    # We'll only extract from possibleLabels that actually exist
    labelsToExtract = [lbl for lbl in possibleLabels if lbl in existingLabels]
    if not labelsToExtract:
        raise ValueError("No labels from the given 'possibleLabels' are present in the surface!")

    # ------------------------------------------------
    # 2. Extract each artery sub-surface by label
    # ------------------------------------------------

    standardAdj = {
        1.0: (2.0, 3.0),
        2.0: (1.0, 8.0),
        3.0: (1.0, 9.0),
        4.0: (5.0, 8.0, 11.0),
        5.0: (4.0,),
        6.0: (7.0, 9.0, 12.0),
        7.0: (6.0,),
        8.0: (2.0, 4.0),
        9.0: (3.0, 6.0),
        10.0: (11.0, 12.0),
        11.0: (4.0,),
        12.0: (6.0,)
    }

    # 1) Attach a new array with unique cell IDs.
    attachOriginalCellIds(labeledSurface)
    completeAppender = vtk.vtkAppendPolyData()

    for lbl in labelsToExtract:

        artery = pv.wrap(extractNonStandardAdjSurfaces(labeledSurface, float(lbl), standardAdj))

        #artery = smooth_surface(artery)

        pointLabels = vtk.vtkIntArray()
        pointLabels.SetName("BoundaryLabels")
        pointLabels.SetNumberOfComponents(1)
        pointLabels.SetNumberOfTuples(artery.GetNumberOfPoints())
        
        for i in range(artery.GetNumberOfPoints()):
            pointLabels.SetValue(i, lbl)
            #cellLabels.SetTuple2(i, float(lbl), 0.0)
        
        # Attach this array to the pont data
        artery.GetPointData().AddArray(pointLabels)
        completeAppender.AddInputData(artery)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.ToleranceIsAbsoluteOn()
    cleaner.SetAbsoluteTolerance(0.1)
    cleaner.SetInputConnection(completeAppender.GetOutputPort())
    cleaner.Update()

    #return pv.wrap(cleaner.GetOutput())
    return pv.wrap(completeAppender.GetOutput())


def merge_coincident_points_on_boundary(poly_data):
    """
    Merge exactly coincident points in poly_data only if at least one 
    of the points being merged is on a boundary edge.

    Parameters
    ----------
    poly_data : vtk.vtkPolyData
        Input surface mesh.
    
    Returns
    -------
    vtk.vtkPolyData
        A new vtkPolyData with boundary points merged according 
        to the rule above.
    """

    boundary_labels_array = poly_data.GetPointData().GetArray("BoundaryLabels")
    if boundary_labels_array is None:
        raise ValueError("No 'BoundaryLabels' array found in point data.")

    # --- 1) Group points by their exact 3D coordinates ---
    import numpy as np
    num_pts = poly_data.GetNumberOfPoints()
    coords = [poly_data.GetPoint(i) for i in range(num_pts)]

    # Dictionary keyed by (x, y, z) -> list of old point IDs
    from collections import defaultdict
    coord_map = defaultdict(list)
    for old_id, xyz in enumerate(coords):
        # Round to a reasonable decimal if you want to guard against floating noise
        # This is optional, for strictly identical merges you can skip rounding.
        key = tuple(np.round(xyz, 7))  
        coord_map[key].append(old_id)

    # --- 2) Identify boundary edges and boundary points ---
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(poly_data)
    # We only want open boundary edges of the mesh
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.Update()

    boundary_edges = feature_edges.GetOutput()

    boundary_point_ids = set()
    for cell_id in range(boundary_edges.GetNumberOfCells()):
        cell = boundary_edges.GetCell(cell_id)
        points = cell.GetPoints()
        for pid_idx in range(cell.GetNumberOfPoints()):
            point = tuple(np.round(points.GetPoint(pid_idx), 7))
            pt_id = coord_map[point][0]
            boundary_point_ids.add(pt_id)

    print("Number of boundary point ids:", len(boundary_point_ids))

    # --- 3) Within each group, merge only if there's a boundary point ---
    old_id_to_new_id = [-1] * num_pts
    new_points = []
    current_new_id = 0

    coincident_points = 0
    merge_points = 0

    out_label_array = vtk.vtkIntArray()
    out_label_array.SetName("BoundaryLabels")
    out_label_array.SetNumberOfComponents(1)

    for xyz_key, same_coord_ids in coord_map.items():
        if len(same_coord_ids) > 1:
            coincident_points += 1
            #print(same_coord_ids)
        # Check if this group has at least one boundary point
        group_has_boundary = any((pid in boundary_point_ids) for pid in same_coord_ids)

        if group_has_boundary:
            #boundary_pid = same_coord_ids[0]
            boundary_pid = min(pid for pid in same_coord_ids if pid in boundary_point_ids)
            chosen_label = boundary_labels_array.GetValue(boundary_pid)
            merge_points += 1
            # Merge them all into a single point
            merged_new_id = current_new_id
            current_new_id += 1
            new_points.append(xyz_key)
            out_label_array.InsertValue(merged_new_id, chosen_label)
            for old_pid in same_coord_ids:
                old_id_to_new_id[old_pid] = merged_new_id
        else:
            # No boundary point: each stays distinct
            for old_pid in same_coord_ids:
                chosen_label = boundary_labels_array.GetValue(old_pid)
                new_points.append(xyz_key)
                old_id_to_new_id[old_pid] = current_new_id
                out_label_array.InsertValue(current_new_id, chosen_label)
                current_new_id += 1

    print("Coincident points:", coincident_points)
    print("Merge points:", merge_points)

    out_label_array.SetNumberOfTuples(current_new_id)

    # --- 4) Build the output vtkPolyData with updated point IDs ---
    out_poly_data = vtk.vtkPolyData()

    # Create the new vtkPoints
    vtk_new_points = vtk.vtkPoints()
    vtk_new_points.SetNumberOfPoints(len(new_points))
    for i, xyz_key in enumerate(new_points):
        vtk_new_points.SetPoint(i, xyz_key)
    out_poly_data.SetPoints(vtk_new_points)

    # We will copy polygons (and optionally strips, lines, etc.).
    # For a surface mesh with polys:
    in_polys = poly_data.GetPolys()
    in_polys.InitTraversal()

    out_polys = vtk.vtkCellArray()

    for cell_id in range(poly_data.GetNumberOfCells()):
        cell = poly_data.GetCell(cell_id)
        npts = cell.GetNumberOfPoints()
        out_polys.InsertNextCell(npts)
        for j in range(npts):
            old_pid = cell.GetPointId(j)
            out_polys.InsertCellPoint(old_id_to_new_id[old_pid])

    out_poly_data.SetPolys(out_polys)

    #out_poly_data.GetPointData().AddArray(out_label_array)

    # Optionally copy cell data, point data, etc., if needed:
    #out_poly_data.GetPointData().ShallowCopy(poly_data.GetPointData())
    out_poly_data.GetCellData().ShallowCopy(poly_data.GetCellData())

    return pv.wrap(out_poly_data)


def remesh(polydata):

    clus = pyacvd.Clustering(polydata)
    # mesh is not dense enough for uniform remeshing
    clus.cluster(20000)
    return clus.create_mesh()
