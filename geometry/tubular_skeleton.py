import numpy as np
import pyvista as pv
from scipy.interpolate import CubicHermiteSpline
from typing import Dict, List, Tuple, Set, Optional
from functools import cached_property
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from skimage import transform
import SimpleITK as sitk
from scipy.optimize import minimize
import networkx as nx

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
        self._extract_connections()
        #self.find_anchor_points()
        self.create_network()
        self.find_bifurcations()
        
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

        return points
    
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
        skeleton_bifurcation_labels = np.nonzero(self.point_data['Bifurcation'])
        bifurcation_points = self.points[skeleton_bifurcation_labels]
        p.add_mesh(bifurcation_points, color='yellow', render_points_as_spheres=True, point_size=10)

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
        new_skeleton._extract_connections()
        
        return new_skeleton

    def filter_artery_by_radius(self, arteries_to_remove: list, radius_min: float, atol: float = 1e-6):
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
            keep_mask &= ~(~(np.abs(arteries - artery_to_remove) > atol) & (radii < radius_min))


        keep_ids = np.where(keep_mask)[0]

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
        new_skeleton._extract_connections()
        
        return new_skeleton

    def _extract_connections(self):
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
            13.0: (10.0, 11.0, 12.0)
        }

        unique_labels = np.unique(self.point_data['Artery'])

        if 0 in unique_labels:
            unique_labels = np.delete(unique_labels, 0)
        
        # Extract vessel bifurfaction boundary cells from surface.
        # Find barycenters of boundaries

        arteries = self.point_data['Artery']
        end_points = []

        
        for label1 in unique_labels:
            for label2 in standardAdj[label1]:
                idx_a = np.flatnonzero(arteries == label1)
                idx_b = np.flatnonzero(arteries == label2)

                if len(idx_a) == 0 or len(idx_b) == 0:
                    continue

                pts_a = self.points[idx_a]            # (n₁, 3) coordinates
                pts_b = self.points[idx_b]            # (n₂, 3) coordinates

                # build a KD-tree on one set (label_b) and query with the other set
                tree_b = cKDTree(pts_b)
                dists, nearest = tree_b.query(pts_a)  # nearest neighbour in label_b for each point in label_a

                best = np.argmin(dists)               # index in pts_a of the overall closest pair
                end_points.append(int(idx_a[best]))
                end_points.append(int(idx_b[nearest[best]]))

        skeleton_points = self.points
        skeleton_labels = np.zeros(skeleton_points.shape[0])
        for idx in end_points:
            skeleton_labels[idx] = 1
        self.point_data['ConnectionLabel'] = skeleton_labels

    def create_network(self, connection_radius=0.6):
        graph = nx.Graph()
        tree = cKDTree(self.points)

        for i, point in enumerate(self.points):
            graph.add_node(i, pos=point)
        
        for i, point in enumerate(self.points):
            neighbors = tree.query_ball_point(point, connection_radius)
            for neighbor in neighbors:
                if neighbor != i:
                    graph.add_edge(i, neighbor)

        self.graph = graph

    def find_bifurcations(self):
        graph = self.graph
        
        bifurcation_indices = []
        for node in graph.nodes():
            degree = graph.degree(node)
            if degree >= 3:  # Bifurcation point
                bifurcation_indices.append(node)
        
        #first pass to find bifurcations, likely includes bad geometry
        bifurcation_points_rough = self.points[bifurcation_indices]

        new_graph = nx.Graph()
        tree = cKDTree(self.points)

        for i, point in enumerate(self.points):
            new_graph.add_node(i, pos=point, connection_point=self.point_data['ConnectionLabel'][i])
        
        connection_radius = 2.5
        for i, point in enumerate(self.points):
            neighbors = tree.query_ball_point(point, connection_radius)
            for neighbor in neighbors:
                if neighbor != i:
                    new_graph.add_edge(i, neighbor)
        

        connection_points = np.array([new_graph.nodes[node]['connection_point'] for node in new_graph.nodes])

        bifurcation_points_clean = []

        #check within certain radius to make sure bifurcation points only happen near connections
        for bifurcation_point in bifurcation_points_rough:
            for node in new_graph.nodes():
                if (new_graph.nodes[node]['pos'] == bifurcation_point).all():
                    neighbors = np.array([neighbor for neighbor in new_graph.neighbors(node)])
                    neighbor_connections = connection_points[neighbors]
                    connection_nearby = (neighbor_connections > 0).any()
                    if connection_nearby:
                        bifurcation_points_clean.append(bifurcation_point)

        bifurcation_points = np.array(bifurcation_points_clean)

        
        #average out "triple points"
        
        bifurcation_mask = np.all(bifurcation_points[:, None, :] == self.points[None, :, :], axis=2)
        indices = []

        for i in range(len(bifurcation_points)):
            idx = np.where(bifurcation_mask[i])[0]
            indices.append(idx[0] if len(idx) > 0 else None)

        bifurcations = np.zeros(self.points.shape[0], dtype=int)
        bifurcations[indices] = 1
        self.point_data['Bifurcation'] = bifurcations

    def find_anchor_points(self):

        #only works for "typical COWs"
        connections = self.point_data['ConnectionLabel']
        idxs = np.where(connections > 0)
        
        points = self.points[idxs]
        labels = self.point_data["Artery"][idxs]
        distance_matrix = cdist(points, points)

        order = ["R-PCA/Basillar", "L-PCA/Basillar", "R-Pcom/R-ICA", "L-Pcom/L-ICA", 
                 "R-ACA/R-ICA", "L-ACA/L-ICA", "L-MCA/L-ICA", "R-MCA/R-ICA", 
                 "L-Pcom/L-PCA", "R-Pcom/R-PCA"]
        
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

        self.anchor_points = [0 for _ in range(int(len(idxs[0])/2))]

        for idx in range(len(points)):
            nearest = np.argsort(distance_matrix[idx])[1]
            nearest_point = points[nearest]
            current_point = points[idx]
            nearest_point_label = labels[nearest]
            current_point_label = labels[idx]
            connection = f"{vessel_labels[nearest_point_label]}/{vessel_labels[current_point_label]}"

            if connection in order:
                insertion_index = order.index(connection)
                new_point = (nearest_point + current_point) / 2
                self.anchor_points[insertion_index] = new_point
            else:
                continue
        
        Bas_points = self.points[np.where(self.point_data['Artery'] == 1)]
        LACA_points = self.points[np.where(self.point_data['Artery'] == 11)]
        RACA_points = self.points[np.where(self.point_data['Artery'] == 12)]
        min_bas = Bas_points[np.argmin(Bas_points[:, 2])]
        max_LACA = LACA_points[np.argmax(LACA_points[:, 2])]
        max_RACA = RACA_points[np.argmax(RACA_points[:, 2])]

        self.anchor_points.append(max_RACA)
        self.anchor_points.append(max_LACA)
        self.anchor_points.append(min_bas)
        #anchor points are in order of the order list followed by the top RACA point, top LACA point, and bottom basillar point
        self.anchor_points = np.array(self.anchor_points)
        
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
        samples_per_segment: int = 25,
        tol: float = _TOL,
    ) -> None:
        self.tol = tol
        self.samples_per_segment = samples_per_segment
        
        #all important points in a dictionary or list
        self.inlet_arteries = {
            "BA": [(-4.5, 0, -5), (-4, 0, -4), (-4, 0, -3), (-4, 0, -2), (-4.5, 0, -1), (-5, 0, 0)],
            "RICA": [(2, -4, -4.5), (4, -4, -4), (5.5, -4, -3), (5, -4, -1.5), (3.5, -4, -1),
                    (2, -4, 0), (2, -3.5, 1.5), (3, -3, 2.5)],
            "LICA": [(2, 4, -4.5), (4, 4, -4), (5.5, 4, -3), (5, 4, -1.5), (3.5, 4, -1),
                    (2, 4, 0), (2, 3.5, 1.5), (3, 3, 2.5)],
        }

        self.outlet_arteries = {
            "RPCA": [(-5, 0, 0), (-5, -1, 0.5), (-5, -2, 0.5), (-5, -3, 0), (-5, -4, 0),
                    (-5, -5, 0), (-5, -6, -0.5), (-5.5, -7, -0.5), (-6, -8, 0), (-6, -9, 1)],
            "LPCA": [(-5, 0, 0), (-5, 1, 0.5), (-5, 2, 0.5), (-5, 3, 0), (-5, 4, 0),
                    (-5, 5, 0), (-5, 6, -0.5), (-5.5, 7, -0.5), (-6, 8, 0), (-6, 9, 1)],
            "RMCA": [(3, -3, 2.5), (3.5, -4, 3.5), (4, -5, 4), (4.5, -6, 4), (4.5, -7, 4),
                    (4.5, -8, 4), (4.5, -9, 4.5)],
            "LMCA": [(3, 3, 2.5), (3.5, 4, 3.5), (4, 5, 4), (4.5, 6, 4), (4.5, 7, 4),
                    (4.5, 8, 4), (4.5, 9, 4.5)],
            "RACA": [(3, -3, 2.5), (4, -2, 3.5), (4.5, -1, 4), (5.5, -0.5, 5), (6, -0.5, 6.5),
                    (6.5, -0.5, 7.5), (7, -0.5, 8.5), (7, -1, 9.5)],
            "LACA": [(3, 3, 2.5), (4, 2, 3.5), (4.5, 1, 4), (5.5, 0.5, 5), (6, 0.5, 6.5),
                    (6.5, 0.5, 7.5), (7, 0.5, 8.5), (7, 1, 9.5)],
        }

        self.communicating_arteries = {
            "RPCOM": [(-5, -4, 0), (-4, -3.5, 0), (-3, -3, -0.5), (-2, -2.5, -0.5),
                    (-1, -2.5, -0.5), (0, -3, 0), (1, -3.5, 0.5), (2, -4, 0)],
            "LPCOM": [(-5, 4, 0), (-4, 3.5, 0), (-3, 3, -0.5), (-2, 2.5, -0.5),
                    (-1, 2.5, -0.5), (0, 3, 0), (1, 3.5, 0.5), (2, 4, 0)],
            "ACOM": [(4.5, -1, 4), (4.5, -0.5, 4), (4.5, 0, 4), (4.5, 0.5, 4), (4.5, 1, 4)],
        }
        
        self.anchor_points = np.array([(-5, -1, 0.5), (-5, 1, 0.5), (1, -3.5, 0.5), 
                                  (1, 3.5, 0.5), (4, -2, 3.5), (4, 2, 3.5), 
                                  (3.5, 4, 3.5), (3.5, -4, 3.5), (-4, 3.5, 0), 
                                  (-4, -3.5, 0), (7, -1, 9.5), (7, 1, 9.5), (-4.5, 0, -5)])
        
        '''self.anchor_points = np.array([[-5, -1, 0.5], [-5, 1, 0.5], [1, -3.5, 0.5], 
                                  [1, 3.5, 0.5], [4, -2, 3.5], [4, 2, 3.5], 
                                  [3.5, 4, 3.5], [3.5, -4, 3.5], [-4, 3.5, 0], 
                                  [-4, -3.5, 0]])'''
        
        #dictionary of points for all arteries
        self.points = self.compute_all_points()
        #list of all points
        self.points_list = [point for sublist in self.points.values() for point in sublist]
        self.compute_all_tangents()
        
        anchor_points = [0 for i in range(self.anchor_points.shape[0])]
        for point in self.points_list:
            if point.anchor_point == True:
                coords = np.array(point.coords)
                idx = np.where(np.all(np.isclose(self.anchor_points, coords), axis=1))[0][0]
                anchor_points[idx] = point

        self.anchor_points = np.array(anchor_points)

        # spline cache {artery → PolyData}
        self._splines: Dict[str, pv.PolyData] = {}

        self.compute_all_splines()

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
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
        for artery in self.points:
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

    def find_linear_transform(self, skeleton: Skeleton):

        #find skeleton anchor points
        skeleton_anchor_points = skeleton.anchor_points

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
        fixed = skeleton.anchor_points

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

    def apply_non_linear_transform(self, transform):
        
        transformed_points = []
        for point in self.points_list:
            coords = point.coords
            sitk_point = [float(coords[0]), float(coords[1]), float(coords[2])]
            transformed_point = np.array(transform.TransformPoint(sitk_point))
            point.update_point(transformed_point)
        
        self.compute_all_tangents()
        self.compute_all_splines()

    def plot(self, skeleton: Skeleton, plot_skeleton=False):
        plotter = pv.Plotter()
        for poly in self.all_splines().values():
            plotter.add_mesh(poly, line_width=6, render_lines_as_tubes=True)
        
        if plot_skeleton:
            plotter.add_mesh(skeleton.points, render_points_as_spheres=True, color='black', point_size=8)

        point_cloud = np.vstack([point.coords for point in self.points_list])
        
        #plotter.add_mesh(np.array((-4.5, 0, -5)), render_points_as_spheres=True, color='black', point_size=12)
        plotter.add_mesh(point_cloud, render_points_as_spheres=True, color='red', point_size=10)
        plotter.camera_position = 'xy'
        plotter.show()

    def apply_linear_transform(self, transform):
        Point.transform_all_instances(transform)
        self.compute_all_splines()

    def move_knot(self, artery: str, index: int, new_xyz: Tuple[float, float, float]) -> Set[str]:
        """Move one explicit control point and recompute affected splines.

        Returns
        -------
        set[str]
            Artery names that were recomputed.
        """
        if artery not in self.points:
            raise KeyError(f"Unknown artery '{artery}'.")
        pts = self.points[artery]
        if not (0 <= index < len(pts)):
            raise IndexError("knot index out of range")

        pts[index].update_point(np.asarray(new_xyz, float))

        # arteries influenced by either old or new position
        affected = pts[index].arteries

        self.compute_tangents(affected)

        for artery in affected:
            self._splines[artery] = self.compute_spline(artery)

        return affected

    def get_spline(self, artery: str) -> pv.PolyData:
        return self._splines[artery]

    def all_splines(self) -> Dict[str, pv.PolyData]:
        return self._splines

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
        
        inlet_arteries = ["BA", "RICA", "LICA"]
        outlet_arteries = ["RPCA", "LPCA", "RMCA", "LMCA", "RACA", "LACA"]
        
        if artery in inlet_arteries:
            self.artery_type = "inlet_artery"
        elif artery in outlet_arteries:
            self.artery_type = "outlet_artery"
        else:
            self.artery_type = "communicating_artery"

    def find_highest_hierarchy_artery(self):
        inlet_arteries = ["BA", "RICA", "LICA"]
        outlet_arteries = ["RPCA", "LPCA", "RMCA", "LMCA", "RACA", "LACA"]
        communicating_arteries = ["RPCOM", "LPCOM", "ACOM"]
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
    t = np.zeros(len(points))
    for i in range(1, len(t)):
        t[i] = t[i - 1] + np.linalg.norm(points[i] - points[i - 1]) ** alpha

    # -- default first‑derivative estimates ----------------------------------
    m = np.empty_like(points)
    m[1:-1] = (points[2:] - points[:-2]) / (t[2:, None] - t[:-2, None])
    if closed:
        m[0] = (points[1] - points[-2]) / (t[1] - (t[-2] - t[-1]))
        m[-1] = m[0]
    else:
        m[0] = (points[1] - points[0]) / (t[1] - t[0])
        m[-1] = (points[-1] - points[-2]) / (t[-1] - t[-2])

    # -- optional tangent clamping -------------------------------------------
    if (not closed) and start_tangent is not None:
        m[0] = _normalize(start_tangent) * np.linalg.norm(m[0])
    if (not closed) and end_tangent is not None:
        m[-1] = _normalize(end_tangent) * np.linalg.norm(m[-1])

    # -- create coordinate‑wise Hermite splines ------------------------------
    xs = CubicHermiteSpline(t, points[:, 0], m[:, 0])
    ys = CubicHermiteSpline(t, points[:, 1], m[:, 1])
    zs = CubicHermiteSpline(t, points[:, 2], m[:, 2])

    n_seg = len(points) - 1 if not closed else len(points)
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

