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
        self.find_anchor_points()
        
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
        p.add_mesh(np.array(self.anchor_points), render_points_as_spheres=True, point_size=10, color='black')

        try:
            skeleton_connection_labels = np.nonzero(self.point_data['ConnectionLabel'])
            connection_points = self.points[skeleton_connection_labels]
            p.add_mesh(connection_points, color='purple', render_points_as_spheres=True, point_size=6)
        except:
            None
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

    def find_anchor_points(self):
        connections = self.point_data['ConnectionLabel']
        idxs = np.where(connections > 0)
        
        points = self.points[idxs]
        labels = self.point_data["Artery"][idxs]
        distance_matrix = cdist(points, points)
        self.anchor_points = []

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
        
        self.points = self.compute_points()
        
        inlet_arteries = {
            "BA": [(-4.5, 0, -5), (-4, 0, -4), (-4, 0, -3), (-4, 0, -2), (-4.5, 0, -1), (-5, 0, 0)],
            "RICA": [(2, -4, -4.5), (4, -4, -4), (5.5, -4, -3), (5, -4, -1.5), (3.5, -4, -1),
                    (2, -4, 0), (2, -3.5, 1.5), (3, -3, 2.5)],
            "LICA": [(2, 4, -4.5), (4, 4, -4), (5.5, 4, -3), (5, 4, -1.5), (3.5, 4, -1),
                    (2, 4, 0), (2, 3.5, 1.5), (3, 3, 2.5)],
        }

        outlet_arteries = {
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

        communicating_arteries = {
            "RPCOM": [(-5, -4, 0), (-4, -3.5, 0), (-3, -3, -0.5), (-2, -2.5, -0.5),
                    (-1, -2.5, -0.5), (0, -3, 0), (1, -3.5, 0.5), (2, -4, 0)],
            "LPCOM": [(-5, 4, 0), (-4, 3.5, 0), (-3, 3, -0.5), (-2, 2.5, -0.5),
                    (-1, 2.5, -0.5), (0, 3, 0), (1, 3.5, 0.5), (2, 4, 0)],
            "ACOM": [(4.5, -1, 4), (4.5, -0.5, 4), (4.5, 0, 4), (4.5, 0.5, 4), (4.5, 1, 4)],
        }

        # store control points as float arrays
        self._points: Dict[str, np.ndarray] = {
            **{n: np.asarray(p, float) for n, p in inlet_arteries.items()},
            **{n: np.asarray(p, float) for n, p in outlet_arteries.items()},
            **{n: np.asarray(p, float) for n, p in communicating_arteries.items()},
        }
        
        self.points_list = np.vstack(list(self._points.values()))

        # spline cache {artery → PolyData}
        self._splines: Dict[str, pv.PolyData] = {}

        # mapping junction‑node‑id → List[(artery, knot‑index)]
        self._junction_map: Dict[int, List[Tuple[str, int]]] = {}

        self._rebuild_junction_index()
        self._recompute_all()

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
    def compute_points(self):
        
        inlet_arteries = {
            "BA": [(-4.5, 0, -5), (-4, 0, -4), (-4, 0, -3), (-4, 0, -2), (-4.5, 0, -1), (-5, 0, 0)],
            "RICA": [(2, -4, -4.5), (4, -4, -4), (5.5, -4, -3), (5, -4, -1.5), (3.5, -4, -1),
                    (2, -4, 0), (2, -3.5, 1.5), (3, -3, 2.5)],
            "LICA": [(2, 4, -4.5), (4, 4, -4), (5.5, 4, -3), (5, 4, -1.5), (3.5, 4, -1),
                    (2, 4, 0), (2, 3.5, 1.5), (3, 3, 2.5)],
        }

        outlet_arteries = {
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

        communicating_arteries = {
            "RPCOM": [(-5, -4, 0), (-4, -3.5, 0), (-3, -3, -0.5), (-2, -2.5, -0.5),
                    (-1, -2.5, -0.5), (0, -3, 0), (1, -3.5, 0.5), (2, -4, 0)],
            "LPCOM": [(-5, 4, 0), (-4, 3.5, 0), (-3, 3, -0.5), (-2, 2.5, -0.5),
                    (-1, 2.5, -0.5), (0, 3, 0), (1, 3.5, 0.5), (2, 4, 0)],
            "ACOM": [(4.5, -1, 4), (4.5, -0.5, 4), (4.5, 0, 4), (4.5, 0.5, 4), (4.5, 1, 4)],
        }
        
        all_arteries = {**outlet_arteries, **inlet_arteries, **communicating_arteries}
        all_points = {}

        for artery, point_list in all_arteries.items():
            for point in point_list:
                if point not in list(all_points.keys()):
                    all_points[point] = [artery]
                else:
                    new_label = all_points[point]
                    new_label.append(artery)
                    all_points[point] = new_label
        
        points = []
        for point, labels in all_points.items():
            points.append(Point(point, labels))
        
        for point in points:
            point.find_artery_type()

        for artery, point_list in all_arteries.items():
            new_point_list = []
            for target_point in point_list:
                for new_point in points:
                    if target_point == new_point.coords:
                        new_point_list.append(new_point)
                        all_arteries[artery] = new_point_list
                        continue

        return all_arteries

    def transform(self, skeleton: Skeleton):

        model_anchor_points = np.array([(-5, -1, 0.5), (-5, 1, 0.5), (1, -3.5, 0.5), 
                                  (1, 3.5, 0.5), (4, -2, 3.5), (4, 2, 3.5), 
                                  (4.5, 6, 4), (4.5, -6, 4), (-4, 3.5, 0), 
                                  (-4, -3.5, 0)])
        
        skeleton_anchor_points = skeleton.anchor_points

        tform = transform.estimate_transform('similarity', model_anchor_points, skeleton_anchor_points)
        
        similarity_matrix = tform.params
        
        homogenous_points = np.hstack([model_anchor_points, np.ones((model_anchor_points.shape[0], 1))])
        transformed_homogenous = (similarity_matrix @ homogenous_points.T).T
        transformed_points = transformed_homogenous[:, :3]

        tform = transform.estimate_transform('affine', transformed_points, skeleton_anchor_points)
        affine_matrix = tform.params

        return similarity_matrix, affine_matrix

    def move_knot(self, artery: str, index: int, new_xyz: Tuple[float, float, float]) -> Set[str]:
        """Move one explicit control point and recompute affected splines.

        Returns
        -------
        set[str]
            Artery names that were recomputed.
        """
        if artery not in self._points:
            raise KeyError(f"Unknown artery '{artery}'.")
        pts = self._points[artery]
        if not (0 <= index < len(pts)):
            raise IndexError("knot index out of range")

        old_xyz = pts[index].copy()
        pts[index] = np.asarray(new_xyz, float)

        # rebuild junction graph (cheap)
        self._rebuild_junction_index()

        # arteries influenced by either old or new position
        affected: Set[str] = {artery}
        affected |= self._arteries_sharing_point(old_xyz)
        affected |= self._arteries_sharing_point(new_xyz)

        # --- two‑pass recompute: parents first, then children ---------------
        # pass 1: recompute every affected artery (parents change tangents)
        for art in affected:
            self._splines[art] = self._compute_spline(art)
        # pass 2: redo – ensures children pick up any updated parent tangents
        for art in affected:
            self._splines[art] = self._compute_spline(art)

        return affected

    def get_spline(self, artery: str) -> pv.PolyData:
        return self._splines[artery]

    def all_splines(self) -> Dict[str, pv.PolyData]:
        return self._splines

    # ------------------------------------------------------------------
    #   Internal helpers
    # ------------------------------------------------------------------

    # ---- junction indexing --------------------------------------------------

    def _rebuild_junction_index(self) -> None:
        """Index *all* control‑point coordinates shared across arteries."""
        self._junction_map.clear()
        representatives: List[np.ndarray] = []  # 1 coord per distinct node

        def _find_node_id(xyz: np.ndarray) -> int:
            for nid, rep in enumerate(representatives):
                if np.linalg.norm(rep - xyz) < self.tol:
                    return nid
            return -1

        for art, pts in self._points.items():
            for idx, p in enumerate(pts):
                nid = _find_node_id(p)
                if nid == -1:
                    nid = len(representatives)
                    representatives.append(p.copy())
                self._junction_map.setdefault(nid, []).append((art, idx))

    def _arteries_sharing_point(self, xyz: np.ndarray) -> Set[str]:
        for members in self._junction_map.values():
            rep = self._points[members[0][0]][members[0][1]]
            if np.linalg.norm(rep - xyz) < self.tol:
                return {art for art, _ in members}
        return set()

    # ---- geometry utilities -------------------------------------------------

    @staticmethod
    def _spline_tangent(poly: pv.PolyData, xyz: np.ndarray) -> Optional[np.ndarray]:
        """Approximate tangent of *poly* at (closest) point *xyz*."""
        pts = poly.points
        dists = np.linalg.norm(pts - xyz, axis=1)
        idx = dists.argmin()
        if dists[idx] > _TOL:
            return None  # no reasonable match
        if idx == 0:
            tan = pts[1] - pts[0]
        elif idx == len(pts) - 1:
            tan = pts[-1] - pts[-2]
        else:
            tan = pts[idx + 1] - pts[idx - 1]
        return tan if np.linalg.norm(tan) > 0 else None

    def _matching_tangent(self, child: str, xyz: np.ndarray) -> Optional[np.ndarray]:
        """Search arteries sharing *xyz* (excluding *child*) and return spline tangent."""
        for members in self._junction_map.values():
            rep_xyz = self._points[members[0][0]][members[0][1]]
            if np.linalg.norm(rep_xyz - xyz) >= self.tol:
                continue
            for art, _ in members:
                if art == child:
                    continue
                parent_spline = self._splines.get(art)
                if parent_spline is None:
                    continue  # parent not computed yet
                tan = self._spline_tangent(parent_spline, xyz)
                if tan is not None:
                    return tan
        return None

    # ---- spline (re)generation ---------------------------------------------

    def _compute_spline(self, art: str) -> pv.PolyData:
        pts = self._points[art]
        start_tan = self._matching_tangent(art, pts[0])
        end_tan = self._matching_tangent(art, pts[-1])
        return catmull_rom_spline_polydata(
            pts.copy(),
            samples_per_segment=self.samples_per_segment,
            start_tangent=start_tan,
            end_tangent=end_tan,
        )

    def _recompute_all(self) -> None:
        for art in self._points:
            self._splines[art] = self._compute_spline(art)
        # one additional pass so children capture fresh parent tangents
        for art in self._points:
            self._splines[art] = self._compute_spline(art)

    def plot(self):
        plotter = pv.Plotter()
        for poly in self.all_splines().values():
            plotter.add_mesh(poly, line_width=6, render_lines_as_tubes=True)

        point_cloud = np.vstack([point for point in self._points.values()])

        anchor_points = np.array([(-5, -1, 0.5), (-5, 1, 0.5), (1, -3.5, 0.5), 
                                  (1, 3.5, 0.5), (4, -2, 3.5), (4, 2, 3.5), 
                                  (3.5, 4, 3.5), (3.5, -4, 3.5), (-4, 3.5, 0), 
                                  (-4, -3.5, 0)])
        

        plotter.add_mesh(point_cloud, render_points_as_spheres=True, color='red', point_size=10)
        plotter.add_mesh(anchor_points[3], render_points_as_spheres=True, color='black', point_size=12)
        plotter.add_mesh(anchor_points, render_points_as_spheres=True, color='purple', point_size=10)
        plotter.camera_position = 'xy'
        plotter.show()

    def plot_transform(self, skeleton):
        similarity_matrix, affine_matrix = self.transform(skeleton)
        homogenous_points = np.hstack([self.points_list, np.ones((self.points_list.shape[0], 1))])
        transformed_homogenous = (similarity_matrix @ homogenous_points.T)
        transformed_homogenous = (affine_matrix @ transformed_homogenous).T

        transformed_points = transformed_homogenous[:, :3]

        plotter = pv.Plotter()
        points = pv.PolyData(transformed_points)
        plotter.add_mesh(points, color='red', render_points_as_spheres=True, point_size=10)
        plotter.add_mesh(skeleton.points, color='black', render_points_as_spheres=True, point_size=6)

        plotter.show()
    
class Point:
    def __init__(self, coords: np.array, arteries: list):
        self.coords = coords
        self.arteries = arteries
        self.anchor_point = self.is_anchor_point()

    def is_anchor_point(self):
        anchor_points = [(-5, -1, 0.5), (-5, 1, 0.5), (1, -3.5, 0.5), 
                        (1, 3.5, 0.5), (4, -2, 3.5), (4, 2, 3.5), 
                        (3.5, 4, 3.5), (3.5, -4, 3.5), (-4, 3.5, 0), 
                        (-4, -3.5, 0)
                        ]
        if self.coords in anchor_points:
            return True
        else:
            return False
    
    def find_artery_type(self):
        if len(self.arteries) > 1:
            self.artery_type = "connection"
        
        artery = self.arteries[0]
        
        inlet_arteries = ["BA", "RICA", "LICA"]
        outlet_arteries = ["RPCA", "LPCA", "RMCA", "LMCA", "RACA", "LACA"]
        
        if artery in inlet_arteries:
            self.artery_type = "inlet_artery"
        elif artery in outlet_arteries:
            self.artery_type = "outlet_artery"
        else:
            self.artery_type = "communicating_artery"

    def update_point(self, new_point):
        self.coords = new_point

    def update_tangent(self, tangent):
        self.tangent = tangent

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

