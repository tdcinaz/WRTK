import numpy as np
import pyvista as pv
from scipy.interpolate import CubicHermiteSpline
from typing import Dict, List, Tuple, Set, Optional
from functools import cached_property
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_opening, generate_binary_structure, convolve
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from skimage import transform
import SimpleITK as sitk
from scipy.optimize import minimize
import networkx as nx
import random
import matplotlib.pyplot as plt
import math

__all__ = [
    "SkeletonModel",
    "catmull_rom_spline_polydata",
]

# -----------------------------------------------------------------------------
#   Low‑level utilities
# -----------------------------------------------------------------------------

_TOL = 1e-6  # distance tolerance used to match shared junction points

class Image:
    def __init__(self, file_path):
        self.file_path = file_path

        self.nifti_img: nib.Nifti1Image = nib.load(self.file_path)

        # Get the NumPy array from the NIfTI
        self.data_array = self.nifti_img.get_fdata(dtype=np.float32)
        #self.opening_operations()
        
        # Retrieve voxel spacing from the header
        #    If it's a 4D image, nibabel header might return 4 zooms, so we take only the first three.
        self.pix_dim = self.nifti_img.header.get_zooms()[:3]

        # Extract the translation (origin) from the affine
        self.origin = self.nifti_img.affine[:3, 3]

    @cached_property
    def pv_image(self):
        """
        Convert a 3D NIfTI image to vtkImageData.

        Args:
        nifti_img (nibabel.Nifti1Image): The input NIfTI image.

        Returns:
            pv.ImageData: The image data in pyvista format.
        """

        pv_image = pv.ImageData(
        dimensions=self.data_array.shape, 
        spacing=(self.pix_dim[2], self.pix_dim[1], self.pix_dim[0]), 
        origin=(self.origin[2], self.origin[1], self.origin[0]),
        deep=True)
    
        pv_image.point_data["Scalars"] = self.data_array.flatten()

        return pv_image
    
    def compute_label_volumes(self, label_range=range(1, 14)):
        
        # Get the voxel size from the header (voxel volume in mm³)
        voxel_volume = np.prod(self.pix_dim)
        print(f"Voxel volume: {voxel_volume:.2f} mm³")
        
        # Compute volumes for each label
        volumes = {}
        for label in label_range:
            voxel_count = np.sum(self.data_array == label)
            volumes[label] = voxel_count * voxel_volume

        return volumes

    def opening_operations(self):
        
        original_array = self.data_array

        #find the indexs of the small vessels (pcoms and acom)
        small_vessels = ((original_array == 8) | (original_array == 9) | (original_array == 10)).astype(int)
        small_vessel_idxs = np.where(small_vessels > 0)

        #background vs vessels (0 vs 1)
        binary_array = (self.data_array > 0).astype(int)

        #2d cross structure, relatively gentle 
        structure_2d = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]])
        
        #perform opening operation on each slice of the array
        new_binary_array = binary_array.copy()
        for i in range(new_binary_array.shape[0]):
            new_binary_array[i] = binary_opening(binary_array[i], structure=structure_2d, iterations=1)

        small_vessel_voxels = np.zeros(original_array.shape)
        small_vessel_voxels[small_vessel_idxs] = original_array[small_vessel_idxs]

        new_label_array = original_array * new_binary_array
        #get rid of new cleaned small vessels as they probably got chopped
        new_label_array[small_vessel_idxs] = 0

        #add original small vessels back in
        self.data_array = new_label_array + small_vessel_voxels

    #convenience method to create a skeleton from the image object directly
    def create_skeleton(self):
        return Skeleton(self)

    def plot_image(self):
        p = pv.Plotter()
        p.add_mesh(self.pv_image)
        p.show()

class Skeleton(pv.PolyData):
    def __init__(self, image: Image):
        self.image = image

        #compute points for skeleton
        points = self._compute_points()

        #inherit from polydata object
        super().__init__(points)

        #set attributes radius, artery, and connection points
        self.point_data['Radius'] = self.radii_mm.flatten()
        self.point_data["Artery"] = self.labels.flatten()
        self.point_data['ConnectionLabel'] = self.connection_labels
        #self.point_data['Bifurcation'] = self.bifurcation_labels
        self.find_target_points()
        
    def _compute_points(self):
        points = []
        data_array = self.image.data_array
        voxel_spacing = self.image.pix_dim

        #Subdividing voxels in two on each axis
        new_data_array = np.repeat(np.repeat(np.repeat(data_array, 2, axis=0), 2, axis=1), 2, axis=2)
        new_affine = self.image.nifti_img.affine.copy()
        new_hdr = self.image.nifti_img.header.copy()

        new_affine[:3, :3] /= 2.0
        new_hdr.set_zooms(tuple(z/2 for z in voxel_spacing[:3]) + voxel_spacing[3:])
        new_voxel_spacing = new_hdr.get_zooms()

        #Find distance between any point in subdivided skeleton and background
        distance_map = distance_transform_edt(new_data_array, sampling=new_voxel_spacing)

        #creates the skeleton
        skeleton = skeletonize(new_data_array)

        if not np.any(skeleton):
            print("  Skeletonization failed.")
            return
        
        #extract only voxels in the skeleton
        self.radii_mm = distance_map[skeleton > 0].flatten()
        self.labels = new_data_array[skeleton > 0]
        zyx_coords = np.array(np.where(skeleton > 0)).T

        homogeneous_coords = np.c_[zyx_coords[:, ::-1], np.ones(len(zyx_coords))]  # (x, y, z, 1)
        points = (new_affine @ homogeneous_coords.T).T[:, :3]

        _, idx  = np.unique(points, axis=0, return_index=True)
        unique_points = points[np.sort(idx)]
        self.radii_mm = self.radii_mm[np.sort(idx)]
        self.labels = self.labels[np.sort(idx)]
        self.connection_labels = self._extract_connections(points=unique_points, artery_labels=self.labels)
        #self.bifurcation_labels, unique_points = self.find_bifurcations(points=unique_points)

        return unique_points

    def plot(self):
        p = pv.Plotter()
        p.add_mesh(self.points, render_points_as_spheres=True, point_size=6)
        
        try:
            p.add_mesh(self.anchor_points, render_points_as_spheres=True, point_size=6)
        except:
            None
        skeleton_connection_labels = np.nonzero(self.point_data['ConnectionLabel'])
        connection_points = self.points[skeleton_connection_labels]
        p.add_mesh(connection_points, color='purple', render_points_as_spheres=True, point_size=10)
        '''skeleton_bifurcation_labels = np.nonzero(self.point_data['Bifurcation'])
        bifurcation_points = self.points[skeleton_bifurcation_labels]
        p.add_mesh(bifurcation_points, color='yellow', render_points_as_spheres=True, point_size=10)'''

        p.show()

        return

    def filter_out_artery_points(self, arteries_to_remove: list, atol: float = 1e-6):
        """
            Return a copy of *mesh* with every point whose ``Artery`` value equals
            *artery_to_remove* (within *atol*) removed.

            Parameters
            ----------
            mesh : pyvista.PolyData
                The input mesh.  Must contain a point-data array named ``"Artery"``.
            artery_to_remove : float
                The artery label to filter out (e.g., ``10.0``).
            atol : float, optional
                Absolute tolerance when comparing floating-point values.
                Defaults to ``1 × 10⁻⁶``.

            Returns
            -------
            pyvista.PolyData
                A new mesh that no longer contains the specified artery’s points
                (and any cells that depended on them).

            Raises
            ------
            KeyError
                If the mesh lacks an ``"Artery"`` point-data array.
            ValueError
                If no points remain after filtering.
        """

        # --- sanity checks --------------------------------------------------------
        if "Artery" not in self.point_data:
                raise KeyError("Point-data array 'Artery' not found in the mesh.")
        
        arteries = np.asarray(self.point_data["Artery"])

        keep_mask = np.ones(len(arteries), dtype=bool)       # keep ≠ target
        
        if np.isscalar(arteries_to_remove):
            arteries_to_remove = [arteries_to_remove]

        for artery_to_remove in arteries_to_remove:
            #bitwise operation checks where both the mask and the artery that isn't supposed to be removed is true
            keep_mask &= (np.abs(arteries - artery_to_remove) > atol)

        keep_ids  = np.where(keep_mask)[0]

        if keep_ids.size == 0:
            raise ValueError("Filtering removed every point; nothing left to return.")

        # --- extract the surviving points & associated cells ----------------------

        filtered_points = self.points[keep_ids]
        filtered_radius = self.point_data['Radius'][keep_ids]
        filtered_artery = self.point_data['Artery'][keep_ids]
        
        # Create a completely new Skeleton object
        # We bypass the normal __init__ to avoid recomputing from image
        new_skeleton = self.__class__.__new__(self.__class__)
        
        # Initialize the PyVista PolyData part with clean data
        pv.PolyData.__init__(new_skeleton, filtered_points)
        
        # Set the required attributes
        new_skeleton.image = self.image
        new_skeleton.point_data['Radius'] = filtered_radius
        new_skeleton.point_data['Artery'] = filtered_artery
        #might be kind of expensive to just remove two points
        connection_labels = new_skeleton._extract_connections(filtered_artery, filtered_points)
        new_skeleton.point_data['ConnectionLabel'] = connection_labels
        new_skeleton.find_target_points()
        #new_skeleton.find_bifurcations()
        
        return new_skeleton

    def filter_artery_by_radius(self, arteries_to_remove: list, atol: float = 1e-6):
        """
        Return a copy of *mesh* with points from a given artery removed if their
        ``Radius`` value is below ``radius_min``.

        Parameters
        ----------
        mesh : pyvista.PolyData
            Input mesh. Must contain point-data arrays ``"Artery"`` and ``"Radius"``.
        artery_labels : list of float
            The artery whose thin points you want to eliminate (e.g. ``10.0``).
        radius_min : float
            Minimum allowable radius. Points in the chosen artery with a radius
            **< radius_min** are discarded. In Python, the tilde symbol ~ primarily represents the bitwise NOT operator. This operator is a unary operator, meaning it operates on a single operand. 


        Raises
        ------
        KeyError
            If required data arrays are missing.
        ValueError
            If all points are removed.
        """
        # --- sanity checks --------------------------------------------------------
        for name in ("Artery", "Radius"):
            if name not in self.point_data:
                raise KeyError(f"Point-data array '{name}' not found in the mesh.")

        if np.isscalar(arteries_to_remove):
            arteries_to_remove = [arteries_to_remove]
        
        arteries = np.asarray(self.point_data["Artery"])
        radii    = np.asarray(self.point_data["Radius"])

        keep_mask = np.ones(len(arteries), dtype=bool)



        for artery_to_remove in arteries_to_remove:
            
            #only keep labels where the artery isn't a target artery and where the radius is smaller than the minimum radius
            #keep where NOT (NOT target artery AND radius is smaller than minimum radius)
            artery_mask = np.ones(len(arteries), dtype=bool)
            artery_mask &= ~(~(np.abs(arteries - artery_to_remove) > atol))
            artery_ids = np.where(artery_mask)[0]
            std_dev = np.std(radii[artery_ids])
            rad_mean = np.mean(radii[artery_ids])
            threshold = rad_mean - 2 * std_dev
            keep_mask &= ~(~(np.abs(arteries - artery_to_remove) > atol) & (radii < threshold))

        keep_ids = np.where(keep_mask)[0]

        if keep_ids.size == 0:
            raise ValueError("Filtering removed every point; nothing left to return.")
        # --- extract the surviving points & associated cells ----------------------

        filtered_points = self.points[keep_ids]
        filtered_radius = self.point_data['Radius'][keep_ids]
        filtered_artery = self.point_data['Artery'][keep_ids]
        #filtered_bifurcation = self.point_data['Bifurcation'][keep_ids]
        
        # Create a completely new Skeleton object
        # We bypass the normal __init__ to avoid recomputing from image
        new_skeleton = self.__class__.__new__(self.__class__)
        
        # Initialize the PyVista PolyData part with clean data
        pv.PolyData.__init__(new_skeleton, filtered_points)
        
        # Set the required attributes
        new_skeleton.image = self.image
        new_skeleton.point_data['Radius'] = filtered_radius
        new_skeleton.point_data['Artery'] = filtered_artery
        connection_labels = new_skeleton._extract_connections(filtered_artery, filtered_points)
        new_skeleton.point_data['ConnectionLabel'] = connection_labels
        #new_skeleton.point_data['Bifurcation'] = filtered_bifurcation
        new_skeleton.find_target_points()
        
        return new_skeleton

    def _extract_connections(self, artery_labels, points):
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
            12.0: (6.0,),               # R-ACA -> R-ICA
            13.0: (10.0,)
        }


        unique_labels = np.unique(artery_labels)

        if 0 in unique_labels:
            unique_labels = np.delete(unique_labels, 0)
        
        # Extract vessel bifurfaction boundary cells from surface.
        # Find barycenters of boundaries

        arteries = artery_labels
        end_points = []

        
        for label1 in unique_labels:
            for label2 in standardAdj[label1]:
                idx_a = np.flatnonzero(arteries == label1)
                idx_b = np.flatnonzero(arteries == label2)

                if len(idx_a) == 0 or len(idx_b) == 0:
                    continue

                pts_a = points[idx_a]            # (n₁, 3) coordinates
                pts_b = points[idx_b]            # (n₂, 3) coordinates

                # build a KD-tree on one set (label_b) and query with the other set
                tree_b = cKDTree(pts_b)
                dists, nearest = tree_b.query(pts_a)  # nearest neighbour in label_b for each point in label_a

                best = np.argmin(dists)               # index in pts_a of the overall closest pair
                if dists[best] < 1.5:
                    end_points.append(int(idx_a[best]))
                    end_points.append(int(idx_b[nearest[best]]))

        skeleton_points = points
        skeleton_labels = np.zeros(skeleton_points.shape[0])
        for idx in end_points:
            skeleton_labels[idx] = 1
        return skeleton_labels

    def create_network(self, points, connection_radius=0.6):
        graph = nx.Graph()
        tree = cKDTree(points)

        for i, point in enumerate(points):
            graph.add_node(i, pos=point, connection_point=self.connection_labels[i])
        
        for i, point in enumerate(points):
            neighbors = tree.query_ball_point(point, connection_radius)
            for neighbor in neighbors:
                if neighbor != i:
                    graph.add_edge(i, neighbor)

        return graph

    def find_bifurcations(self, points):
        graph = self.create_network(points, connection_radius=0.61)
        
        bifurcation_indices = []
        for node in graph.nodes():
            degree = graph.degree(node)
            if degree >= 3:  # Bifurcation point
                bifurcation_indices.append(node)
        
        #first pass to find bifurcations, likely includes bad geometry
        bifurcation_points_rough = points[bifurcation_indices]

        new_graph = self.create_network(points, connection_radius=2.5)

        connection_points = np.array([new_graph.nodes[node]['connection_point'] for node in new_graph.nodes])

        bifurcation_points_clean = []

        #check within certain radius to make sure bifurcation points only happen near connections
        #filters out bifurcation points farther away from connections
        for bifurcation_point in bifurcation_points_rough:
            for node in new_graph.nodes():
                if (new_graph.nodes[node]['pos'] == bifurcation_point).all():
                    neighbors = np.array([neighbor for neighbor in new_graph.neighbors(node)])
                    neighbor_connections = connection_points[neighbors]
                    connection_nearby = (neighbor_connections > 0).any()
                    if connection_nearby:
                        bifurcation_points_clean.append(bifurcation_point)

        bifurcation_points_intermediate = np.array(bifurcation_points_clean)
        bifurcation_points_final = []
        new_radii = []
        new_labels = []
        
        #often it will find multiple bifurcations clustered close to one another
        #average out points that are close together
        multiple_point_graph = self.create_network(points, connection_radius=1)
        for bifurcation_point in bifurcation_points_intermediate:
            for node in multiple_point_graph.nodes():
                if (multiple_point_graph.nodes[node]['pos'] == bifurcation_point).all():
                    neighbors = np.array([multiple_point_graph.nodes[neighbor]['pos'] for neighbor in multiple_point_graph.neighbors(node)])
                    neighbor_bifurcations = []
                    for neighbor in neighbors:
                        matches = np.where(np.all(bifurcation_points_intermediate == neighbor, axis=1))[0]
                        if len(matches) > 0:
                            neighbor_bifurcations.append(bifurcation_points_intermediate[matches[0]])
                    if len(points) > 0:
                        neighbor_bifurcations.append(bifurcation_point)
                        average_point = np.mean(neighbor_bifurcations, axis=0)
                        bifurcation_points_final.append(average_point)
                        point_idxs = []
                        for point in points:
                            idx = np.where(np.all(point == points, axis=1))[0]
                            point_idxs.append(idx)
                        radii = self.radii_mm[point_idxs]
                        new_radii.append(np.mean(radii))
                        label = self.labels[point_idxs[0]]
                        new_labels.append(label)
                    else:
                        bifurcation_points_final.append(bifurcation_point)
                continue
        
        #find unique points in new points and make sure it's ordered
        _, idx  = np.unique(np.vstack((points, bifurcation_points_final)), axis=0, return_index=True)
        new_points = np.vstack((points, bifurcation_points_final))[np.sort(idx)]
        final_radii = np.append(self.radii_mm, new_radii)[np.sort(idx)]
        final_labels = np.append(self.labels, new_labels)[np.sort(idx)]

        #make a new skeleton object since points are being updated
        bifurcation_points_final = np.array(bifurcation_points_final)
        
        bifurcation_mask = np.all(bifurcation_points_final[:, None, :] == new_points[None, :, :], axis=2)
        indices = []

        for i in range(len(bifurcation_points_final)):
            idx = np.where(bifurcation_mask[i])[0]
            indices.append(idx[0] if len(idx) > 0 else None)

        bifurcations = np.zeros(new_points.shape[0], dtype=int)
        bifurcations[indices] = 1
        bifurcation_point_data = bifurcations
        
        # Set the required attributes
        self.radii_mm = final_radii
        self.labels = final_labels
        self.connection_labels = np.append(self.connection_labels, np.zeros(new_points.shape[0] - points.shape[0]))
        return bifurcation_point_data, new_points

    def find_target_points(self):

        #only works for "typical COWs"
        connections = self.point_data['ConnectionLabel']
        
        idxs = np.where(connections > 0)

        points = self.points[idxs]
        labels = self.point_data["Artery"][idxs]
        distance_matrix = cdist(points, points)

        present_arteries = np.unique(self.point_data['Artery'])
        
        vessel_labels = {
            1.0: "Basillar",
            2.0: "L-PCA",
            3.0: "R-PCA",
            4.0: "L-ICA",
            5.0: "L-MCA",
            6.0: "R-ICA",
            7.0: "R-MCA",
            8.0: "L-Pcom",
            9.0: "R-Pcom",
            10.0: "Acom",
            11.0: "L-ACA",
            12.0: "R-ACA",
            13.0: "3rd A2"
        }
        
        order = ["R-PCA/Basillar", "L-PCA/Basillar", "R-Pcom/R-ICA", "L-Pcom/L-ICA", 
                 "R-ACA/R-ICA", "L-ACA/L-ICA", "L-MCA/L-ICA", "R-MCA/R-ICA", 
                 "L-Pcom/L-PCA", "R-Pcom/R-PCA"]
        
        present_connections = []

        patient_specific_vessel_labels = {key:value for key, value in vessel_labels.items() if key in present_arteries}
        
        for idx in range(len(points)):
            nearest = np.argsort(distance_matrix[idx])[1]
            nearest_point = points[nearest]
            current_point = points[idx]
            nearest_point_label = labels[nearest]
            current_point_label = labels[idx]
            connection = f"{patient_specific_vessel_labels[nearest_point_label]}/{patient_specific_vessel_labels[current_point_label]}"
            if connection in order:
                present_connections.append(connection)
        
        patient_specific_order = [connection for connection in order if all(item in patient_specific_vessel_labels.values() for item in connection.split("/")) and connection in present_connections]
        
        self.target_points = [0 for _ in range(len(patient_specific_order))]

        for idx in range(len(points)):
            nearest = np.argsort(distance_matrix[idx])[1]
            nearest_point = points[nearest]
            current_point = points[idx]
            nearest_point_label = labels[nearest]
            current_point_label = labels[idx]
            connection = f"{patient_specific_vessel_labels[nearest_point_label]}/{patient_specific_vessel_labels[current_point_label]}"
            if connection in patient_specific_order:
                insertion_index = patient_specific_order.index(connection)
                new_point = (nearest_point + current_point) / 2
                self.target_points[insertion_index] = new_point
            else:
                continue
        

        #Bas_points = self.points[np.where(self.point_data['Artery'] == 1)]
        #LACA_points = self.points[np.where(self.point_data['Artery'] == 11)]
        #RACA_points = self.points[np.where(self.point_data['Artery'] == 12)]
        #min_bas = Bas_points[np.argmin(Bas_points[:, 2])]
        #max_LACA = LACA_points[np.argmax(LACA_points[:, 2])]
        #max_RACA = RACA_points[np.argmax(RACA_points[:, 2])]

        #self.target_points.append(max_RACA)
        #self.target_points.append(max_LACA)
        #self.target_points.append(min_bas)
        #anchor points are in order of the order list followed by the top RACA point, top LACA point, and bottom basillar point
        self.target_points = np.array(self.target_points)
        self.order = order
        self.patient_specific_connections = present_connections

    def find_potential_at_point(self, artery, coords: np.ndarray):
        keep_ids = np.where((artery == self.point_data['Artery']))[0]
        artery_points = self.points[keep_ids]
        artery_radii = -self.point_data['Artery'][keep_ids]
        
        # Constant term
        alpha = 1
        
        # Calculate vectors from each artery point to the evaluation point
        r_vectors = coords - artery_points  # Shape: (n_points, 3)
        
        # Calculate distances
        dists = np.linalg.norm(r_vectors, axis=1)  # Shape: (n_points,)
        valid_mask = dists > 0
        dists = dists[valid_mask]
        r_vectors = r_vectors[valid_mask]
        artery_radii = artery_radii[valid_mask]
        
        # Calculate potential (scalar): sum of alpha * q / r
        potential = np.sum((alpha * artery_radii) / dists)
        
        # Calculate electric field (vector): sum of alpha * q * r_hat / r²
        # r_hat = r_vectors / dists (unit vectors)
        # Field contribution from each point: alpha * q * r_hat / r²
        
        # Expand dimensions for broadcasting
        dists_expanded = dists[:, np.newaxis]  # Shape: (n_points, 1)
        artery_radii_expanded = artery_radii[:, np.newaxis]  # Shape: (n_points, 1)
        
        # Calculate field contributions from each point
        field_contributions = (alpha * artery_radii_expanded * r_vectors) / (dists_expanded**3)
        
        # Sum to get total field vector
        field = np.sum(field_contributions, axis=0)  # Shape: (3,)
        field = field.reshape(1, 3)
        
        print(f"Potential: {potential}")
        print(f"Field: {field}")

        return potential, field

    def find_field(self, artery, sample_resolution=0.5, plot=False):
        keep_ids = np.where((artery == self.point_data['Artery']))[0]
        artery_points = self.points[keep_ids]
        artery_radii = self.point_data['Artery'][keep_ids]
        centroid = np.mean(artery_points, axis=0)
        
        xs = artery_points[:, 0]
        ys = artery_points[:, 1]
        zs = artery_points[:, 2]
        
        dev_x = np.std(xs)
        dev_y = np.std(ys)
        dev_z = np.std(zs)

        k = 3
        min_x, max_x = np.floor((centroid[0] - k * dev_x) / sample_resolution) * sample_resolution, np.floor((centroid[0] + k * dev_x) / sample_resolution) * sample_resolution
        min_y, max_y = np.floor((centroid[1] - k * dev_y) / sample_resolution) * sample_resolution, np.floor((centroid[1] + k * dev_y) / sample_resolution) * sample_resolution
        min_z, max_z = np.floor((centroid[2] - k * dev_z) / sample_resolution) * sample_resolution, np.floor((centroid[2] + k * dev_z) / sample_resolution) * sample_resolution

        x = np.arange(min_x, max_x, sample_resolution)
        y = np.arange(min_y, max_y, sample_resolution)
        z = np.arange(min_z, max_z, sample_resolution)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid = np.stack((X, Y, Z), axis=-1)

        points = grid.reshape(-1, 3)

        alpha=0.5
        epsilon=0.1
        gamma = 2
        potential = np.zeros(points.shape[0])

        for point, radius in zip(artery_points, artery_radii):
            distances = np.linalg.norm(points - point, axis=1)
            potential += (radius ** gamma) / (distances**(alpha) + epsilon)

        #potential = 1 / potential
        potential_grid = potential.reshape([grid.shape[0], grid.shape[1], grid.shape[2]])

        field_grid = np.array(np.gradient(potential_grid))
        field = field_grid.reshape(-1, 3)

        if plot:
            plotter = pv.Plotter()
            count=0
            for field_vec, point in zip(field, points):
                print(field_vec)
                print(point)
                line = pv.Line((point + field_vec * 0.00), point)
                plotter.add_mesh(line, color='purple', line_width=2)
                count += 1
            
            point_cloud = pv.PolyData(points)
            min_weight, max_weight = np.min(potential), np.max(potential)
            weights = (1.0 - (potential - min_weight) / (max_weight - min_weight))
            point_cloud['weights'] = weights

            plotter.add_mesh(point_cloud, scalars='weights', opacity=(weights**2), cmap='coolwarm', render_points_as_spheres=True, point_size=10)
            plotter.add_scalar_bar(title="weight_vals", n_labels=5, italic=False, fmt='%.1f')

            plotter.add_mesh(self.points, color='black')

            plotter.show()

        field_grid = potential.reshape([grid.shape[0], grid.shape[1], grid.shape[2]])
        return field_grid, grid, potential, points

    #finish this
    def add_points_to_skeleton(self, **kwargs):
        parameters = ["new_points", "new_connection", "new_radius", "new_artery", "new_bifurcation"]
        passed_parameters = [k for k in kwargs.keys()]
        
        if "new_points" not in passed_parameters:
            raise Exception("No new points were provided")
        if "new_radius" not in passed_parameters:
            raise Exception("No new radii were provided")
        if "new_artery" not in passed_parameters:
            raise Exception("No new labels were provided")
        
    def delete_points_from_skeleton(self, delete_points):
        return
        
class SkeletonModel:

    """Mutable Circle‑of‑Willis template skeleton.

    *Knots* (explicit artery control points) can be moved interactively; all
    affected splines are lazily recomputed.  Endpoint clamp directions are
    extracted **from the current splines** of connected arteries, allowing
    smooth C¹ joins even when communicating branches attach to *mid‑points* of
    other vessels (including outlet arteries).
    """

    def __init__(
        self,
        skeleton: Skeleton,
        samples_per_segment: int = 25,
        tol: float = _TOL,
    ) -> None:
        self.tol = tol
        self.samples_per_segment = samples_per_segment
        
        #all important points in a dictionary or list

        self.skeleton = skeleton

        self.inlet_arteries = {
            1: [(20.0, 19.0, 30.0), (30.0, 19.0, 30.0), (32.0, 19.0, 30.0), (34.0, 19.0, 30.0), (35.0, 19.0, 30.0), (37.0, 19.0, 30.0)], 
            6: [(20.0, 29.0, 21.0), (27.0, 36.0, 19.0), (28.0, 42.0, 16.0), (33.0, 42.0, 20.0), (34.0, 35.0, 20.0), (37.0, 31.0, 18.0), (39.0, 29.5, 15.0), (42.0, 30.0, 12.0)], 
            4: [(20.0, 29.0, 39.0), (27.0, 36.0, 41.0), (28.0, 42.0, 44.0), (33.0, 42.0, 40.0), (34.0, 35.0, 40.0), (37.0, 31.0, 42.0), (39.0, 29.5, 45.0), (42.0, 30.0, 48.0)]
        }

        self.outlet_arteries = {
            3: [(37.0, 19.0, 30.0), (39.0, 19.0, 27.0), (39.0, 19.0, 25.0), (38.5, 19.0, 23.0), (37.5, 20.0, 20.0), (35.0, 21.0, 16.0), (30.0, 18.0, 12.0), (28.0, 15.0, 10.0), (28.0, 10.0, 10.0), (30.0, -5.0, 10.0)], 
            2: [(37.0, 19.0, 30.0), (39.0, 19.0, 33.0), (39.0, 19.0, 35.0), (38.5, 19.0, 37.0), (37.5, 20.0, 40.0), (35.0, 21.0, 44.0), (30.0, 18.0, 48.0), (28.0, 15.0, 50.0), (28.0, 10.0, 50.0), (30.0, -5.0, 50.0)], 
            7: [(42.0, 30.0, 12.0), (44.0, 31.0, 9.0), (44.0, 31.0, 7.0), (43.0, 32.0, 4.0), (42.0, 34.0, 3.0), (41.0, 36.0, 3.0), (40.0, 38.0, -6.0)], 
            5: [(42.0, 30.0, 48.0), (44.0, 31.0, 51.0), (44.0, 31.0, 53.0), (43.0, 32.0, 56.0), (42.0, 34.0, 57.0), (41.0, 36.0, 57.0), (40.0, 38.0, 66.0)], 
            12: [(42.0, 30.0, 12.0), (44.5, 31.0, 13.5), (45.0, 32.0, 16.0), (42.0, 33.0, 22.0), (40.0, 36.0, 26.0), (42.0, 42.0, 27.0), (49.0, 46.0, 27.0), (60.0, 50.0, 27.0)], 
            11: [(42.0, 30.0, 48.0), (44.5, 31.0, 46.5), (45.0, 32.0, 44.0), (42.0, 33.0, 38.5), (40.0, 36.0, 34.0), (42.0, 42.0, 33.0), (49.0, 46.0, 33.0), (60.0, 50.0, 33.0)]}

        self.communicating_arteries = {
            9: [(38.5, 19.0, 23.0), (38.0, 21.5, 20.0), (37.0, 23.5, 21.5), (36.0, 24.0, 23.0), (35.0, 26.0, 23.0), (36.0, 28.0, 21.0), (36.0, 29.0, 19.5), (37.0, 31.0, 18.0)], 
            8: [(38.5, 19.0, 37.0), (38.0, 21.5, 40.0), (37.0, 23.5, 38.5), (36.0, 24.0, 37.0), (35.0, 26.0, 37.0), (36.0, 28.0, 39.0), (36.0, 29.0, 40.5), (37.0, 31.0, 42.0)], 
            10: [(40.0, 36.0, 26.0), (38.0, 34.0, 28.5), (38.0, 34.0, 30.0), (38.0, 34.0, 31.5), (40.0, 36.0, 34.0)]}
        
        self.all_arteries = {**self.outlet_arteries, **self.inlet_arteries, **self.communicating_arteries}
    
        self.anchor_points = np.array([[39.0, 19.0, 27.0], [39.0, 19.0, 33.0], [36.0, 29.0, 19.5], [36.0, 29.0, 40.5], [44.5, 31.0, 13.5], [44.5, 31.0, 46.5], [44.0, 31.0, 51.0], [44.0, 31.0, 9.0], [38.0, 21.5, 40.0], [38.0, 21.5, 20.0]])
        
        self.filter_non_present_arteries()
        #dictionary of points for all arteries
        self.points = self.compute_all_points()
        #list of all points
        self.points_list = [point for sublist in self.points.values() for point in sublist]
        self.compute_all_tangents()
        self.anchor_points = self.compute_anchor_points()
        
        self.fields = {artery: skeleton.find_field(artery) for artery in self.points.keys()}
        
        # spline cache {artery → PolyData}
        self._splines: Dict[str, pv.PolyData] = {}

        self.compute_all_splines()
        self.add_interp_points(2)
        self.points_list = [point for sublist in self.points.values() for point in sublist]

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
    def filter_non_present_arteries(self):

        present_arteries = np.unique(self.skeleton.point_data['Artery'])
        
        artery_names = [artery for artery in present_arteries if artery in self.all_arteries.keys()]
        
        all_arteries_copy = {}
        inlet_arteries_copy = {}
        outlet_arteries_copy = {}
        communicating_arteries_copy = {}

        for artery, points in self.all_arteries.items():
            if artery in artery_names:
                all_arteries_copy[artery] = points
                if artery in self.inlet_arteries.keys():
                    inlet_arteries_copy[artery] = points
                if artery in self.outlet_arteries.keys():
                    outlet_arteries_copy[artery] = points
                if artery in self.communicating_arteries.keys():
                    communicating_arteries_copy[artery] = points
        
        self.all_arteries = all_arteries_copy
        self.inlet_arteries = inlet_arteries_copy
        self.outlet_arteries = outlet_arteries_copy
        self.communicating_arteries = communicating_arteries_copy

    def compute_all_points(self):
        #dict with all arteries
        all_arteries = {**self.outlet_arteries, **self.inlet_arteries, **self.communicating_arteries}
        all_points = {}

        #make new dict with point as key and the arteries it connects to as values
        for artery, point_list in all_arteries.items():
            for point in point_list:
                if point not in list(all_points.keys()):
                    all_points[point] = [artery]
                else:
                    new_label = all_points[point]
                    new_label.append(artery)
                    all_points[point] = new_label
        
        points = []

        #create point instances
        for point, labels in all_points.items():
            points.append(Point(np.array(point), labels))
        
        #find what type of point it is (connection, inlet, outlet, or communicating artery)
        #find whether it's an anchor point
        for point in points:
            point.find_artery_type()
            point.is_anchor_point(self.anchor_points)

        #fill all arteries with instances of the point class instead of just coordinates 
        for artery, point_list in all_arteries.items():
            new_point_list = []
            #target point is old coordinate tuple
            for target_point in point_list:
                #new point is the new point object
                for new_point in points:
                    #replace target point with a point object
                    if (target_point == new_point.coords).all():
                        new_point_list.append(new_point)
                        all_arteries[artery] = new_point_list
                        continue

        return all_arteries

    def compute_anchor_points(self):
        #fill anchor_points in as point objects
        anchor_points = [0 for i in range(self.anchor_points.shape[0])]
        for point in self.points_list:
            if point.anchor_point == True:
                coords = np.array(point.coords)
                idx = np.where(np.all(np.isclose(self.anchor_points, coords), axis=1))[0][0]
                anchor_points[idx] = point

        keep_idxs = []
        for idx, [point, connection] in enumerate(zip(anchor_points, self.skeleton.order)):
            if point == 0 or connection not in self.skeleton.patient_specific_connections:
                try:
                    point.anchor_point = False
                except:
                    None
            else:
                point.update_anchor_point_connection(connection)
                keep_idxs.append(idx)
        
        final = np.array(anchor_points)[np.array(keep_idxs)]
        return final

    def compute_all_tangents(self):
        #make a list with desired order that tangents are computed in
        order = list(self.inlet_arteries.keys()) + list(self.outlet_arteries.keys()) + list(self.communicating_arteries.keys())
        #reorder the dictionary
        reordered_dict = {artery: self.points[artery] for artery in order}

        for artery, points in reordered_dict.items():
            for idx, point in enumerate(points):
                point_type = point.artery_type
                
                #skip computing tangents for points with multiple arteries for the artery of lower priority
                if point_type == "connection":
                    highest_hierarchy_artery = point.find_highest_hierarchy_artery()
                    if artery != highest_hierarchy_artery:
                        continue
                if idx == 0:
                    tan = points[1].coords - points[0].coords
                elif idx == len(points) - 1:
                    tan = points[-1].coords - points[-2].coords
                else:
                    tan = points[idx + 1].coords - points[idx - 1].coords
                
                point.update_tangent(tan)

    def compute_tangents(self, arteries):
        #make a list with desired order that tangents are computed in
        order = list(self.inlet_arteries.keys()) + list(self.outlet_arteries.keys()) + list(self.communicating_arteries.keys())
        #reorder the dictionary
        reordered_dict = {artery: self.points[artery] for artery in order if artery in arteries}
        
        for artery, points in reordered_dict.items():
            for idx, point in enumerate(points):
                point_type = point.artery_type
                
                #skip computing tangents for points with multiple arteries for the artery of lower priority
                if point_type == "connection":
                    highest_hierarchy_artery = point.find_highest_hierarchy_artery()
                    if artery != highest_hierarchy_artery:
                        continue
                if idx == 0:
                    tan = points[1].coords - points[0].coords
                elif idx == len(points) - 1:
                    tan = points[-1].coords - points[-2].coords
                else:
                    tan = points[idx + 1].coords - points[idx - 1].coords
                
                point.update_tangent(tan)
        
    def compute_all_splines(self):
        for artery in self.points.keys():
            self._splines[artery] = self.compute_spline(artery)
        # one additional pass so children capture fresh parent tangents
        for artery in self.points:
            self._splines[artery] = self.compute_spline(artery)
              
    def compute_spline(self, artery): 
        pts = self.points[artery]
        pt_coords = np.array([point.coords for point in self.points[artery]])
        start_tan = pts[0].tangent
        end_tan = pts[-1].tangent

        communicating_arteries = ["RPCOM", "LPCOM", "ACOM"]
        #end tangent needs to face in opposite direction for hermite splines
        if artery in communicating_arteries:
            end_tan = end_tan * -1

        return catmull_rom_spline_polydata(
            pt_coords.copy(),
            samples_per_segment=self.samples_per_segment,
            start_tangent=start_tan,
            end_tangent=end_tan,
        )
    
    def add_interp_points(self, new_points):
        new_segments = new_points + 1
        all_splines = self.all_splines()
        for artery, spline in all_splines.items():
            knots = self.points[artery]
            knot_coords = [point.coords for point in self.points[artery]]
            num_knots = len(knots)
            spline_points = spline.points
            segments = num_knots - 1
            num_new_knots = segments * 3
            new_knots = []

            for idx, knot in enumerate(knots[0:len(knots) - 1]):
                new_knots.append(knot.coords)
                current_point = knot.coords
                next_point = knots[idx+1].coords
                current_idx = np.where(np.all(np.isclose(current_point, spline_points, atol=0.1), axis=1))[0]
                next_idx = np.where(np.all(np.isclose(next_point, spline_points, atol=0.1), axis=1))[0]
                
                if len(next_idx) > 1:
                    next_idx = next_idx[0]
                if len(current_idx) > 1:
                    current_idx = current_idx[0]

                num_points_between = next_idx - current_idx
                subdivision = num_points_between // new_segments
                
                for new_point in range(new_points):
                    idx = current_idx + ((new_point + 1) * subdivision)
                    new_knots.append(spline_points[idx])

            new_knots.append(knots[-1].coords)
            point_object_list = []
            for new_knot in new_knots:
                if np.any(np.all(np.array(new_knot) == np.array(knot_coords), axis=1)):
                    point_object_list.append(knots[np.where(np.all(np.array(new_knot).flatten() == knot_coords, axis=1))[0][0]])
                else:
                    point_object = Point(np.array(new_knot).flatten(), [artery])
                    point_object_list.append(point_object)
                    point_object.find_artery_type()

            if point_object_list[0].artery_type == "connection":
                point_object_list = [point_object_list[0]] + point_object_list[1 + new_points:]
            if point_object_list[-1].artery_type == "connection":
                point_object_list = point_object_list[0:len(point_object_list) - (new_points+1)] + [point_object_list[-1]]

            self.points[artery] = point_object_list
        
    def move_anchor_points(self):
        targets = self.skeleton.target_points
        for idx, anchor_point in enumerate(self.anchor_points):
            anchor_point.update_point(targets[idx])

        self.compute_all_tangents()
        self.compute_all_splines()

    def move_non_anchor_points(self, artery):
        points_of_interest = [point for point in self.points[artery] if point.find_highest_hierarchy_artery() == artery]
        coordinates = np.array([point.coords for point in points_of_interest]) 

        artery_mask = (self.skeleton.point_data['Artery'] == artery)
        artery_idxs = np.where(artery_mask > 0)
        artery_points = self.skeleton.points[artery_idxs]
        
        already_matched = []

        for point in points_of_interest:
            distances = cdist([point.coords], artery_points, metric="euclidean").flatten()
            closest = np.argsort(distances)[0]
            count = 1
            while closest in already_matched:
                closest = np.argsort(distances)[count]
                count += 1
            already_matched.append(closest)
            point.update_point(artery_points[closest])
            

        coordinates = np.array([point.coords for point in points_of_interest]) 

        distances = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1))])
        t = distances / distances[-1]

        fx = interp1d(t, coordinates[:, 0], kind='linear')
        fy = interp1d(t, coordinates[:, 1], kind='linear')
        fz = interp1d(t, coordinates[:, 2], kind='linear')
        
        t_new = np.linspace(0, 1, len(points_of_interest))
        new_coords = np.column_stack([fx(t_new), fy(t_new), fz(t_new)])

        for point, coords in zip(points_of_interest, new_coords):
            point.update_point(coords)

        already_matched = []
        for point in points_of_interest:
            distances = cdist([point.coords], artery_points, metric="euclidean").flatten()
            closest = np.argsort(distances)[0]
            count = 1
            while closest in already_matched:
                closest = np.argsort(distances)[count]
                count += 1
            already_matched.append(closest)
            point.update_point(artery_points[closest])
        

        self.compute_all_tangents()
        self.compute_all_splines()

    def move_all_non_anchor_points(self):
        for artery in self.all_arteries.keys():
            self.move_non_anchor_points(artery)

    def cost(self, artery):
        knot_pos = np.array([p.coords for p in self.points[artery]])
        phi, _   = self.net_phi_E(artery, knot_pos)        # ← vectorised
        return phi.sum()         # or whatever scalar you need

    def net_phi_E(self, artery, coords: np.ndarray):
        """Potential & field at an arbitrary point.

        coords : (3,) or (M,3)
        """
        # Make coords at least 2‑D for broadcasting
        eval_pts = np.atleast_2d(coords)

        # ---------- fixed negative charges ----------
        #neg_pos = self.skeleton.points[artery]   # (N₋,3)
        #neg_q   = -np.ones(len(neg_pos)) * q0                        # all –q₀
        q0 = 1.0

        keep_ids = np.where((artery == self.skeleton.point_data['Artery']))[0]
        neg_pos = self.skeleton.points[keep_ids]
        neg_q = -self.skeleton.point_data['Artery'][keep_ids]

        # ---------- mobile positive charges ----------
        pos_pos = np.array([p.coords for p in self.points[artery]])  # (N₊,3)
        pos_q   =  np.ones(len(pos_pos)) * q0                        # all +q₀

        # Combine sources
        src_pos = np.vstack((neg_pos, pos_pos))
        src_q   = np.concatenate((neg_q, pos_q))

        phi, E = _phi_E_from_sources(eval_pts, src_pos, src_q, k=1.0)
        return phi.squeeze(), E.squeeze()

    def forces_on_positives(self, artery):
        """
        Return array (N₊,3) of net forces on every positive particle.
        """
        pos_pos = np.array([p.coords for p in self.points[artery]])  # (N₊,3)
        Np      = len(pos_pos)
        q0 = 1.0
        pos_q   = np.full(Np, q0)    # same +q0 or supply your own array

        # ----‑ A) contributions from all negative charges ----‑
        #neg_pos = self.points[self.skeleton.point_data['Artery'] == artery]   # (N₋,3)
        #neg_q   = -np.ones(len(neg_pos)) * q0

        keep_ids = np.where((artery == self.skeleton.point_data['Artery']))[0]
        neg_pos = self.skeleton.points[keep_ids]
        neg_q = -self.skeleton.point_data['Artery'][keep_ids]

        _, E_neg = _phi_E_from_sources(pos_pos, neg_pos, neg_q)

        # ----‑ B) mutual positive–positive interactions ----‑
        # pair‑wise differences
        r_ij = pos_pos[:, None, :] - pos_pos[None, :, :]     # (N₊,N₊,3)
        d_ij = np.linalg.norm(r_ij, axis=-1)
        mask = ~np.eye(Np, dtype=bool)                      # exclude self
        inv_d3 = np.zeros_like(d_ij)
        inv_d3[mask] = (1.0 / d_ij[mask])**3

        k = 1.0

        # Σ_{j≠i} k q_j r_ij / r³
        E_pp = k * (pos_q[None, :, None] * r_ij * inv_d3[..., None]).sum(axis=1)

        # ----‑ total field and force on each positive charge ----‑
        E_tot = E_neg + E_pp
        F     = pos_q[:, None] * E_tot
        return F

    def optimize_move(self, artery_label, iterations=1, plot=False):
        field_grid, grid, field, field_points = self.fields[artery_label]
        artery_idxs = np.where(artery_label == self.skeleton.point_data['Artery'])[0]
        artery_points = self.skeleton.points[artery_idxs]

        already_picked_points = np.empty((0, 3))
        
        points_of_interest = [point for point in self.points[artery_label] if point.find_highest_hierarchy_artery() == artery_label and point.anchor_point == False]

        for point in points_of_interest:
            distances_to_skeleton = cdist([point.coords], artery_points, metric="euclidean").flatten()
            closest_skeleton_point = artery_points[np.argsort(distances_to_skeleton)[0]]
            count = 0
            while np.any(np.all(closest_skeleton_point == already_picked_points, axis=1)):
                closest_skeleton_point = artery_points[np.argsort(distances_to_skeleton)[count]]
                count += 1
            already_picked_points = np.vstack((already_picked_points, closest_skeleton_point))
            distances_to_field = cdist([closest_skeleton_point], field_points, metric="euclidean").flatten()
            closest_field_point = field_points[np.argsort(distances_to_field)[0]]
            indices = np.where(np.all(closest_field_point == grid, axis=-1))
            
            x = indices[0][0]
            y = indices[1][0]
            z = indices[2][0]

            grid_shape = grid.shape
            min_x, max_x = x-2, x+2
            min_y, max_y = y-2, y+2
            min_z, max_z = z-2, z+2
            if min_x < 0:
                min_x = 0
            if min_y < 0:
                min_y = 0
            if min_z < 0:
                min_z = 0
            if max_x > grid_shape[0]:
                max_x = grid_shape[0]
            if max_y > grid_shape[1]:
                max_y = grid_shape[1]
            if max_z > grid_shape[2]:
                max_z = grid_shape[2]

            local_field_points = grid[min_x:max_x, min_y:max_y, min_z:max_z]
            local_field = field_grid[min_x:max_x, min_y:max_y, min_z:max_z]

            '''plotter = pv.Plotter()
            
            target_point = np.array(point.coords)
            local_field_points_2D = local_field_points.reshape(-1, 3)

            plotter.add_mesh(target_point, render_points_as_spheres=True, point_size=15, color='black')
            plotter.add_mesh(closest_skeleton_point, render_points_as_spheres=True, point_size=15, color='orange')
            plotter.add_mesh(closest_field_point, render_points_as_spheres=True, point_size=15, color='yellow')
            plotter.add_mesh(local_field_points_2D)
            plotter.add_mesh(self.skeleton.points)
            plotter.show()'''

            
            min_idx_local = np.unravel_index(np.argmin(local_field), local_field.shape)
            min_point = local_field_points[min_idx_local]
            min_idx_global = np.array(np.where(np.all(min_point == grid, axis=-1))).flatten()

            point.update_point(min_point)
        self.compute_all_tangents()
        self.compute_all_splines()

    def plot(self, plot_skeleton=False, add_line_between_target_anchor=False, plot_tangents=False):
        plotter = pv.Plotter()
        for poly in self.all_splines().values():
            plotter.add_mesh(poly, line_width=6, render_lines_as_tubes=True)
        
        if plot_skeleton:
            plotter.add_mesh(self.skeleton.points, render_points_as_spheres=True, color='light_gray', point_size=8)
            plotter.add_mesh(self.skeleton.target_points, render_points_as_spheres=True, color='green', point_size=12)

        if add_line_between_target_anchor and plot_skeleton:
            targets = self.skeleton.target_points
            anchors = np.array([point.coords for point in self.anchor_points])
            for i in range(anchors.shape[0]):
                line = pv.Line(targets[i], anchors[i])
                plotter.add_mesh(line, color='blue', line_width=2)

        non_anchor_point_cloud = np.vstack([point.coords for point in self.points_list if point.anchor_point == False])
        anchor_point_cloud = np.vstack([point.coords for point in self.points_list if point.anchor_point])

        if plot_tangents:
            for point in self.points_list:
                line = pv.Line((point.tangent * 0.5 + point.coords), point.coords)
                plotter.add_mesh(line, color='purple', line_width=2)

        #plotter.add_mesh(np.array((-4.5, 0, -5)), render_points_as_spheres=True, color='black', point_size=12)
        plotter.add_mesh(non_anchor_point_cloud, render_points_as_spheres=True, color='red', point_size=10)
        plotter.add_mesh(anchor_point_cloud, render_points_as_spheres=True, color='yellow', point_size=12)
        plotter.camera_position = 'xy'
        plotter.show()

    def get_spline(self, artery: str) -> pv.PolyData:
        return self._splines[artery]

    def all_splines(self) -> Dict[str, pv.PolyData]:
        return self._splines
    
    def apply_non_linear_transform(self, transform):
        
        transformed_points = []
        for point in self.points_list:
            coords = point.coords
            sitk_point = [float(coords[0]), float(coords[1]), float(coords[2])]
            transformed_point = np.array(transform.TransformPoint(sitk_point))
            point.update_point(transformed_point)
        
        self.compute_all_tangents()
        self.compute_all_splines()

    def find_non_linear_transform(self,
                                       skeleton: Skeleton,
                                       image_size=[256, 256, 128], 
                                       grid_size=[6, 6, 3]):
        """
        Create and optimize a BSpline transform to map source points to target points
        """
        # Create initial BSpline transform
        bspline_transform = sitk.BSplineTransformInitializer(
            image1=sitk.Image(image_size, sitk.sitkFloat32),
            transformDomainMeshSize=grid_size
        )
        
        source = [point.coords for point in self.anchor_points]
        fixed = skeleton.target_points

        def objective_function(params):
            # Set transform parameters
            transform_copy = sitk.BSplineTransform(bspline_transform)
            transform_copy.SetParameters(params)
            
            # Calculate sum of squared distances
            total_error = 0
            for src, tgt in zip(source, fixed):
                src_point = [float(src[0]), float(src[1]), float(src[2])]
                try:
                    transformed = transform_copy.TransformPoint(src_point)
                    error = sum((transformed[i] - tgt[i])**2 for i in range(3))
                    total_error += error
                except:
                    return 1e10  # Large penalty for invalid transforms
            
            return total_error
        
        # Get initial parameters (start with identity transform)
        initial_params = np.array(bspline_transform.GetParameters())
        
        # Add small regularization to prevent overfitting
        def regularized_objective(params):
            fitting_error = objective_function(params)
            regularization = 0.1 * np.sum(params**2)
            return fitting_error + regularization
        
        # Optimize
        print("Optimizing BSpline transform...")
        result = minimize(regularized_objective, initial_params, 
                        method='L-BFGS-B', 
                        options={'maxiter': 1000})
        
        # Set optimized parameters
        bspline_transform.SetParameters(result.x)
        
        print(f"Optimization completed. Final error: {result.fun}")
        
        return bspline_transform
    
    def find_linear_transform(self):

        #find skeleton anchor points
        skeleton_anchor_points = self.skeleton.target_points

        model_anchor_points = np.array([point.coords for point in self.anchor_points])

        #find the similarity transform matrix
        tform = transform.estimate_transform('similarity', model_anchor_points, skeleton_anchor_points)
        
        similarity_matrix = tform.params
        
        #compute the transformed anchor points
        homogenous_points = np.hstack([model_anchor_points, np.ones((model_anchor_points.shape[0], 1))])
        transformed_homogenous = (similarity_matrix @ homogenous_points.T).T
        transformed_points = transformed_homogenous[:, :3]

        #find the affine transform matrix from the transformed anchor points
        tform = transform.estimate_transform('affine', transformed_points, skeleton_anchor_points)
        affine_matrix = tform.params

        return similarity_matrix, affine_matrix

    def apply_linear_transform(self, transform):
        Point.transform_all_instances(transform)
        self.compute_all_splines()

class Point:
    all_points = []

    @classmethod 
    def transform_all_instances(cls, transform_matrix):
        for instance in Point.all_points:
            homogenous_pt = np.append(instance.coords, 1)
            transformed_homogenous = (transform_matrix @ homogenous_pt.T).T
            new_point = transformed_homogenous[:3]
            new_vec = transform_matrix[:3, :3] @ instance.tangent
            instance.update_point(new_point)
            instance.update_tangent(new_vec)
    
    def __init__(self, coords: np.array, arteries: list):
        self.coords = coords
        self.arteries = arteries
        self.anchor_point = False
        self.find_artery_type()
        Point.all_points.append(self)

    def is_anchor_point(self, anchor_points):
        idx = np.where(np.all(np.isclose(anchor_points, self.coords), axis=1))[0]
        if len(idx) > 0:
            self.anchor_point = True
        else:
            self.anchor_point = False
    
    def find_artery_type(self):
        if len(self.arteries) > 1:
            self.artery_type = "connection"
            return
        
        artery = self.arteries[0]
        
        inlet_arteries = [1, 4, 6]
        outlet_arteries = [3, 2, 7, 5, 12, 11]
        
        if artery in inlet_arteries:
            self.artery_type = "inlet_artery"
        elif artery in outlet_arteries:
            self.artery_type = "outlet_artery"
        else:
            self.artery_type = "communicating_artery"

    def find_highest_hierarchy_artery(self):
        inlet_arteries = [1, 4, 6]
        outlet_arteries = [3, 2, 7, 5, 12, 11]
        communicating_arteries = [8, 9, 10]
        highest_hierarchy_artery = "com"

        for artery in self.arteries:
            if artery in inlet_arteries:
                return artery
            elif artery in outlet_arteries:
                highest_hierarchy_artery = artery
            elif artery in communicating_arteries and highest_hierarchy_artery not in outlet_arteries:
                highest_hierarchy_artery = artery
        
        return highest_hierarchy_artery

    def update_point(self, new_point):
        self.coords = new_point

    def update_tangent(self, tangent):
        self.tangent = tangent

    def is_start_or_end_point(self):
        return

    def update_anchor_point_connection(self, connection):
        self.anchor_point_connection = connection

def _normalize(vec: np.ndarray) -> np.ndarray:
    """Return *vec* normalised (safe if ‖vec‖ ≈ 0)."""
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

# -----------------------------------------------------------------------------
#   Core Catmull–Rom helper (centripetal, optional tangent clamping)
# -----------------------------------------------------------------------------

def catmull_rom_spline_polydata(
    points: np.ndarray,
    samples_per_segment: int = 25,
    closed: bool = False,
    start_tangent: Optional[np.ndarray] = None,
    end_tangent: Optional[np.ndarray] = None,
) -> pv.PolyData:
    """Return a centripetal Catmull‑Rom curve through *points* as PolyData."""

    if points.shape[0] < 4:
        raise ValueError("Catmull–Rom requires at least 4 control points.")

    # -- centripetal knot vector (α = 0.5) -----------------------------------
    alpha = 0.5
    _, idx = np.unique(points, return_index=True, axis=0)
    unique_points = points[np.sort(idx)]
    #if len(unique_points) < len(points):
        #print("duplicate points found")
    t = np.zeros(len(unique_points))
    for i in range(1, len(t)):
        t[i] = t[i - 1] + np.linalg.norm(unique_points[i] - unique_points[i - 1]) ** alpha

    # -- default first‑derivative estimates ----------------------------------
    m = np.empty_like(unique_points)
    m[1:-1] = (unique_points[2:] - unique_points[:-2]) / (t[2:, None] - t[:-2, None])
    if closed:
        m[0] = (unique_points[1] - unique_points[-2]) / (t[1] - (t[-2] - t[-1]))
        m[-1] = m[0]
    else:
        m[0] = (unique_points[1] - unique_points[0]) / (t[1] - t[0])
        m[-1] = (unique_points[-1] - unique_points[-2]) / (t[-1] - t[-2])

    # -- optional tangent clamping -------------------------------------------
    if (not closed) and start_tangent is not None:
        m[0] = _normalize(start_tangent) * np.linalg.norm(m[0])
    if (not closed) and end_tangent is not None:
        m[-1] = _normalize(end_tangent) * np.linalg.norm(m[-1])
    
    # -- create coordinate‑wise Hermite splines ------------------------------
    xs = CubicHermiteSpline(t, unique_points[:, 0], m[:, 0])
    ys = CubicHermiteSpline(t, unique_points[:, 1], m[:, 1])
    zs = CubicHermiteSpline(t, unique_points[:, 2], m[:, 2])

    n_seg = len(unique_points) - 1 if not closed else len(unique_points)
    n_eval = n_seg * samples_per_segment + 1
    t_eval = np.linspace(t[0], t[-1], n_eval)

    curve_xyz = np.column_stack([xs(t_eval), ys(t_eval), zs(t_eval)])

    # -- wrap into PyVista PolyData ------------------------------------------
    n_pts = curve_xyz.shape[0]
    poly_line = np.hstack(([n_pts], np.arange(n_pts))).astype(np.int64)

    pd = pv.PolyData(curve_xyz)
    pd.lines = poly_line
    return pd

# -----------------------------------------------------------------------------
#   Skeleton model – dynamic & efficient recomputation
# -----------------------------------------------------------------------------

def _phi_E_from_sources(eval_pts: np.ndarray,
                        src_pos: np.ndarray,
                        src_q:   np.ndarray,
                        k: float = 1.0):
    """
    Vectorised Coulomb sum.

    Parameters
    ----------
    eval_pts : (M, 3)  points where you want φ and E
    src_pos  : (N, 3)  positions of the charges
    src_q    : (N,)    charge magnitudes (signed)
    k        : float   Coulomb constant 1/(4πϵ₀).  Use k=8.987e9 if you
                       want SI units; k=1.0 keeps everything unit‑less.

    Returns
    -------
    φ  : (M,)   potential at each evaluation point
    E  : (M,3)  electric field vector at each evaluation point
    """
    #  r_ij  = r_eval_i  –  r_src_j  → shape (M, N, 3)
    r_ij = eval_pts[:, None, :] - src_pos[None, :, :]
    d_ij = np.linalg.norm(r_ij, axis=-1)           # (M, N)

    # Avoid divide‑by‑zero (self interactions later)
    mask = d_ij > 0.0
    inv_d = np.zeros_like(d_ij)
    inv_d[mask] = 1.0 / d_ij[mask]

    # φ_i = Σ_j k q_j / r_ij
    phi = k * (src_q * inv_d).sum(axis=1)          # (M,)

    # E_i = Σ_j k q_j r̂_ij / r_ij²  = Σ_j k q_j r_ij / r_ij³
    inv_d3           = np.zeros_like(d_ij)
    inv_d3[mask]     = inv_d[mask]**3
    E = k * (src_q[None, :, None] * r_ij * inv_d3[..., None]).sum(axis=1)
    return phi, E

