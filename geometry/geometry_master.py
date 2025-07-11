import sys
import numpy as np
import nibabel as nib
import vtk
from vtkmodules.util import numpy_support
import pyvista as pv
import pyacvd
import logging
from collections import defaultdict, deque
import os
import re
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev, make_splprep, make_interp_spline
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
import networkx as nx
from typing import Tuple
from functools import cached_property
from itertools import combinations
import csv
import networkx as nx
import copy

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
        self.radii_mm = distance_map[skeleton > 0]
        self.labels = new_data_array[skeleton > 0]
        zyx_coords = np.array(np.where(skeleton > 0)).T

        homogeneous_coords = np.c_[zyx_coords[:, ::-1], np.ones(len(zyx_coords))]  # (x, y, z, 1)
        points = (new_affine @ homogeneous_coords.T).T[:, :3]

        return points
    
    def plot(self):
        p = pv.Plotter()
        p.add_mesh(self.points)

        skeleton_connection_labels = np.nonzero(self.point_data['ConnectionLabel'])
        skeleton_end_points = np.nonzero(self.point_data['EndPoints'])

        try:
            connection_points = self.points[skeleton_connection_labels]
            end_points = self.points[skeleton_end_points]
            p.add_mesh(connection_points, color='purple')
            p.add_mesh(end_points, color='black')
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

class OrderedSkeleton(Skeleton):
    @classmethod
    def create_from_parent(cls, skeleton_instance: Skeleton):
        instance = cls.__new__(cls)

        msts = cls.find_msts(skeleton_instance)

        cleaned = [cls.clean_short_branches(mst) for mst in msts]

        plotter2 = pv.Plotter()
        for mst in cleaned:
            plotter2 = cls.plot_mst(mst, plotter2)
        plotter2.show()
        
        #do not copy connection points array, radius array, or label array as they will be unordered

        #pv.PolyData.__init__(instance, ordered_points)


        #instance.__dict__.update(skeleton_instance.__dict__)

        return instance

    def __init__(self, image: Image):
        #super().__init__(i)
        return

    #once working, package find_msts, clean_short_branches and clean_sharp_angles into one nice function
    @staticmethod
    def find_msts(skeleton : Skeleton):
        #create temporary polydata object to insert ordered points into
        temp_polydata = pv.PolyData()

        #figure out what arteries are present in specific patient
        present_arteries = np.unique(skeleton.point_data['Artery']).flatten()

        all_trees = []
        #end_points = []

        for present_artery in present_arteries:
            
            keep_mask = np.ones(len(skeleton.points), dtype=bool)

            #performs boolean operation to check where the point label matches the target artery label
            keep_mask &= (skeleton.point_data['Artery'] == present_artery)

            #find where all the labels match
            artery_indexes = np.where(keep_mask)[0]

            #find the actual points
            artery_points = skeleton.points[artery_indexes]

            radii_at_pts = skeleton.point_data['Radius'][artery_indexes]

            #furthest_point = order_artery_points(artery_points, connection_pts)[0][-1]
            #end_points.append(artery_points[furthest_point])

            #make networkX graph object
            graph = nx.Graph()

            #create a 2x2 matrix of distances between every single node
            distance_matrix_temp = cdist(artery_points, artery_points)
            distance_matrix = (distance_matrix_temp - np.min(distance_matrix_temp)) / (np.max(distance_matrix_temp) - np.min(distance_matrix_temp))

            inverse_radii_temp = [1/(np.power(radius, 2)) for radius in radii_at_pts]
            inverse_radii = (inverse_radii_temp - np.min(inverse_radii_temp)) / (np.max(inverse_radii_temp) - np.min(inverse_radii_temp))

            #fill nodes of the graph

            '''ADD BOOLEAN ATTRIBUTES FOR CONNECTION POINT OR FURTHEST GEODESIC POINT'''
            for i, (x, y, z) in enumerate(artery_points):
                graph.add_node(i, pos=(x,y,z), x=x, y=y, z=z, radius=radii_at_pts[i], artery=present_artery)

            #fill edges
            #pick only 3 closest neighbors
            k=3
            last_slope = [0, 0, 0]
            for i in range(len(artery_points)):
                current_point = artery_points[i]

                #take the 3 nearest points and relevant data
                nearest = np.argsort(distance_matrix[i])[1:k+1]
                
                next_inverse_radius = [inverse_radii[i] for i in nearest]
                current_inverse_radius = inverse_radii[i]
                #create weights for edges of 3 nearest points
                #weight based on inverse radius squared and distance maybe slope
                for idx, j in enumerate(nearest):
                    distance = distance_matrix[i][j]
                    inverse_radius_diff = abs(current_inverse_radius - next_inverse_radius[idx])
                    next_point = artery_points[idx]
                    slope = np.array(current_point - next_point)
                    magnitude = np.linalg.norm(slope)                    
                    if magnitude == 0:
                        slope_weight = 0
                        unit_vector_slope = last_slope
                    else: 
                        unit_vector_slope = slope / magnitude
                    slope_weight = np.linalg.norm(np.cross(unit_vector_slope, last_slope))
                    composite_weight = distance + inverse_radius_diff
                    graph.add_edge(i, j, weight=composite_weight)
                last_slope = unit_vector_slope

            #create mst
            minimum_spanning_tree = nx.minimum_spanning_tree(graph, algorithm="kruskal")
            all_trees.append(minimum_spanning_tree)

        return all_trees
    
    @staticmethod
    def plot_mst(mst: nx.Graph, plotter: pv.Plotter):

        mst_points = np.array([mst.nodes[node]['pos'] for node in mst.nodes()])
        mst_cloud = pv.PolyData(mst_points)
        plotter.add_mesh(mst_cloud, color='red')

        for u, v in mst.edges():
            pos_u = np.array(mst.nodes[u]['pos'])
            pos_v = np.array(mst.nodes[v]['pos'])
            line = pv.Line(pos_u, pos_v)
            plotter.add_mesh(line, color='green', line_width=4)

        return plotter

    #happens before cleaning short branches
    @staticmethod
    #connection points or furthest geodesic points must have a degree of one 
    #if they don't reconnect the graph in a way so that they do
    def fix_bad_end_points(mst: nx.Graph):
        return

    #happens before finding best centerline
    @staticmethod
    #the mst will often create branches that is only one node long
    #these nodes are reinserted between the two points it should be between or culled
    def clean_short_branches(mst: nx.Graph):
        
        mst_copy = copy.deepcopy(mst)
        degrees = np.empty((0, 1))
        #find degree of every node in the network
        for point in range(len(mst.nodes)):
            degrees = np.append(degrees, mst.degree(point))
        

        #find what nodes have more than 2 connections (branching nodes)
        split_keep_mask = np.ones(len(degrees), dtype=bool)
        split_keep_mask &= (degrees > 2)
        split_nodes = np.where(split_keep_mask)[0]
        
        #find what nodes are leaf nodes (terminating nodes)
        '''CONNECTION POINTS OR FURTHEST GEODESIC POINTS ARE NOT ALLOWED TO BE LEAF NODES'''
        leaf_keep_mask = np.ones(len(degrees), dtype=bool)
        leaf_keep_mask &= (degrees == 1)
        leaf_nodes = np.where(leaf_keep_mask)[0]

        for split_node in split_nodes:
            neighbors = np.array([n for n in mst.neighbors(split_node)])
            for leaf_node in leaf_nodes:
                #check if a leaf node is neighbors with a split node
                #if they're neighbors, the branch has length 1, so try and reinsert the point in the right spot
                if leaf_node in neighbors:
                    #create vector between branch and leaf nodes
                    leaf_node_vector = np.array(mst.nodes[leaf_node]['pos']) - np.array(mst.nodes[split_node]['pos'])
                    leaf_node_pos = np.array(mst.nodes[leaf_node]['pos'])


                    #find all vectors between the split node and neighboring points except for the leaf node vector
                    neighbor_without_leaf = np.delete(neighbors, np.where(neighbors == leaf_node))
                    #print(neighbor_without_leaf)
                    neighboring_vectors = np.array([np.array(mst.nodes[neighbor]['pos']) - np.array(mst.nodes[split_node]['pos']) for neighbor in neighbor_without_leaf])

                    #compute dot products between all neighbor vectors and leaf node vector
                    dot_products = np.empty((0))
                    for neighbor_vec in neighboring_vectors:
                        dot_products = np.append(dot_products, np.dot(neighbor_vec, leaf_node_vector))
                    
                    #choose the neighboring vector that most closely aligns with the leaf node vector (higher positive dot product --> lower angle)
                    target_vec = neighboring_vectors[np.argmax(dot_products)]
                    target_point = neighbor_without_leaf[np.argmax(dot_products)]

                    
                    projected_vec = (np.dot(target_vec, leaf_node_vector) / (np.power(np.linalg.norm(target_vec), 2))) * target_vec
                    try:
                        if np.linalg.norm(projected_vec) < np.linalg.norm(target_vec):
                            mst_copy.remove_edge(target_point, split_node)
                            mst_copy.add_edge(split_node, leaf_node, weight=0.5)
                            mst_copy.add_edge(leaf_node, target_point, weight=0.5)
                        else:
                            mst_copy.remove_node(leaf_node)
                            #get point in front of the next point
                            #try to insert it between those two
                            #if vector is still too long then get rid of the point
                    except:
                        plotter = pv.Plotter()
                        plotter = OrderedSkeleton.plot_mst(mst, plotter)

                        leaf_points = np.array([np.array(mst.nodes[index]['pos']) for index in leaf_nodes])
                        split_points = np.array([np.array(mst.nodes[index]['pos']) for index in split_nodes])
                        plotter.add_mesh(leaf_points, color='green', point_size=6)
                        plotter.add_mesh(split_points, color='black', point_size=6)
                        plotter.show()
        return mst_copy            
    
    #happens after finding best centerline
    @staticmethod
    def clean_sharp_angles(mst: nx.Graph):
        
        '''three cases:
        1. two sharp angles adjacent to one another
        2. bad branch point
        3. sharp angle at the end of a vessel'''

        return

class CenterlineNetwork(OrderedSkeleton):
    def __init__(self):
        return

endpoint_tangents: dict[tuple[float, float, float], np.ndarray] = {}

def smooth_trunk(pts, rad: np.ndarray | None = None, s_rel=0.001, k=3, n_samples=200):
    pts_aug = np.vstack([pts[0], pts[0], pts, pts[-1], pts[-1]])

    # --- 2.  Chord-length parameterisation (0 … 1) --------------------------
    dists = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts_aug, axis=0), axis=1))]
    u = dists / dists[-1]

    if rad is not None:
        w_min = 0.4
        w_max = 1.0
        w = w_min + (rad - rad.min()) * (w_max - w_min) / (rad.max() - rad.min())
    else:
        w = np.ones(len(pts))

    w = np.concatenate(([100.0], [100.0], w, [100.0], [100.0]))

    s = s_rel * len(u) * pts.var()

    tck, _ = make_splprep(pts_aug.T, u=u, w=w, k=k, s=s)
    u_fine = np.linspace(0.0, 1.0, n_samples)
    x, y, z = tck.__call__(u_fine)
    ds = tck.__call__(u_fine, 1)

    d0 = np.array(tck.__call__(0.0, 1))
    d0 /= np.linalg.norm(d0)
    d1 = np.array(tck.__call__(1.0, 1))
    d1 /= np.linalg.norm(d1)

    if tuple(pts[0]) not in endpoint_tangents.keys():
        endpoint_tangents[tuple(pts[0])]  = d0          # start end
    if tuple(pts[-1]) not in endpoint_tangents.keys():
        endpoint_tangents[tuple(pts[-1])] = d1          # finish end

    return np.column_stack([x, y, z]), ds

def smooth_trunk2(pts, rad: np.ndarray | None = None, s_rel=0.001, k=3, n_samples=200, h_fraction=0.05, tan_strength=0.05):
    p0, p1 = pts[0], pts[-1]
    # ------------------------------------------------------------------
    # 1.  Create ghost points a short way along each tangent
    # ------------------------------------------------------------------
    chord = np.linalg.norm(p1 - p0)
    h = h_fraction * chord if chord else 1.0       # fallback for tiny links

    if rad is not None:
        w_min = 0.4
        w_max = 1.0
        w = w_min + (rad - rad.min()) * (w_max - w_min) / (rad.max() - rad.min())
    else:
        w = np.ones(len(pts))

    # ------------------------------------------------------------------
    # 2.  Obtain unit tangent vectors for both ends
    # ------------------------------------------------------------------
    if tuple(p0) in endpoint_tangents.keys():
        t0 = endpoint_tangents[tuple(p0)]
        p0_ghost = p0 + h * t0
        pts_aug = np.vstack([p0, p0_ghost, pts[1:], p1, p1])
        w = np.concatenate(([100.0], [100.0*tan_strength], w[1:], [100.0], [100.0]))
    elif tuple(p1) in endpoint_tangents.keys():
        t1 = endpoint_tangents[tuple(p1)]
        p1_ghost = p1 + h * t1
        pts_aug = np.vstack([p0, p0, pts[:-1], p1_ghost, p1])
        w = np.concatenate(([100.0], [100.0], w[:-1], [100.0*tan_strength], [100.0]))
    else:
        raise KeyError("ERROR No tangent point for trunk2")

    # --- 2.  Chord-length parameterisation (0 … 1) --------------------------
    dists = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts_aug, axis=0), axis=1))]
    u = dists / dists[-1]

    s = s_rel * len(u) * pts.var()

    tck, _ = make_splprep(pts_aug.T, u=u, w=w, k=k, s=s)
    u_fine = np.linspace(0.0, 1.0, n_samples)
    x, y, z = tck.__call__(u_fine)
    ds = tck.__call__(u_fine, 1)

    d0 = np.array(tck.__call__(0.0, 1))
    d0 /= np.linalg.norm(d0)
    d1 = np.array(tck.__call__(1.0, 1))
    d1 /= np.linalg.norm(d1)

    if tuple(p0) not in endpoint_tangents.keys():
        print(f"Added {tuple(p0)} to endpoint_tangents")
        endpoint_tangents[tuple(p0)] = d0          # start end
    if tuple(p1) not in endpoint_tangents.keys():
        print(f"Added {tuple(p1)} to endpoint_tangents")
        endpoint_tangents[tuple(p1)] = d1          # finish end

    return np.column_stack([x, y, z]), ds

def smooth_branch(pts, rad: np.ndarray | None = None, s_rel=0.001, k=3, n_samples=200, h_fraction=0.05, tan_strength=0.05):
    p0, p1 = pts[0], pts[-1]

    # ------------------------------------------------------------------
    # 1.  Create ghost points a short way along each tangent
    # ------------------------------------------------------------------
    chord = np.linalg.norm(p1 - p0)
    h = h_fraction * chord if chord else 1.0       # fallback for tiny links

    # ------------------------------------------------------------------
    # 2.  Obtain unit tangent vectors for both ends
    # ------------------------------------------------------------------
    try:
        t0 = endpoint_tangents[tuple(p0)]
        p0_ghost = p0 + h * t0
    except:
        print(f"p0 {tuple(p0)} not found")
        p0_ghost = p0
        
    try:
        t1 = endpoint_tangents[tuple(p1)]
        p1_ghost = p1 + h * t1     # subtract because splev derivative at u=1
                                   # already points "forward" along param
    except:
        print(f"p1 {tuple(p1)} not found")
        p1_ghost = p1
    
    # ------------------------------------------------------------------
    # 3.  Assemble augmented point list
    #     (no need for the triple-dup trick – tangency overrides clamping)
    # ------------------------------------------------------------------
    pts_aug = np.vstack([p0, p0_ghost, pts[1:-1], p1_ghost, p1])
    
    # --- 2.  Chord-length parameterisation (0 … 1) --------------------------
    dists = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts_aug, axis=0), axis=1))]
    u = dists / dists[-1]


    if rad is not None:
        w_min = 0.4
        w_max = 1.0
        w = w_min + (rad - rad.min()) * (w_max - w_min) / (rad.max() - rad.min())
    else:
        w = np.ones(len(pts))
    
    w = np.concatenate(([100.0], [100.0*tan_strength], w[1:-1], [100.0*tan_strength], [100.0]))

    s = s_rel * len(u) * pts.var()

    tck, _ = make_splprep(pts_aug.T, u=u, w=w, k=k, s=s)
    u_fine = np.linspace(0.0, 1.0, n_samples)
    x, y, z = tck.__call__(u_fine)
    ds = tck.__call__(u_fine, 1)

    return np.column_stack([x, y, z]), ds

class PointDataArrays:
    def __init__(self, skeleton: pv.PolyData, artery_labels: str, point_labels: str, sphere_labels: str):
        self.artery_labels = skeleton[artery_labels]
        self.point_labels = skeleton[point_labels]
        self.sphere_labels = skeleton[sphere_labels]

class Path:
    def __init__(self, label: float, target_label: float, idx_reordered: np.ndarray, pts_reordered: np.ndarray, rad_reordered: np.ndarray):
        self.label = label
        self.target_label = target_label
        self.idx_reordered = idx_reordered
        self.pts_reordered = pts_reordered
        self.rad_reordered = rad_reordered

class Branch:
    def __init__(self, label: float, target: float, ords, spheres):
        self.label = label
        self.target = target
        self.ords = ords
        self.spheres = spheres
                         
    @cached_property
    def length(self):
        points = self.ords
        length = 0
        for count in range(len(points) - 1):
            seg_length = np.linalg.norm(points[count] - points[count+1])
            length += seg_length
        return length

class Trunk:
    def __init__(self, label: float, part: int, branches: list, ords, spheres):
        self.label = label
        self.part = part
        self.branches = branches
        self.ords = ords
        self.spheres = spheres

    @cached_property
    def length(self):
        points = self.ords
        length = 0
        for count in range(len(points) - 1):
            seg_length = np.linalg.norm(points[count] - points[count+1])
            length += seg_length
        return length

class Artery:
    def __init__(self, skeleton: pv.PolyData, label: float, point_data: PointDataArrays):
        self.skeleton       = skeleton
        self.point_data     = point_data
        self.label          = label
        self.idx_subset     = np.flatnonzero(point_data.artery_labels == label)
        self.points         = self.skeleton.points[self.idx_subset]
        self.n_points       = len(self.idx_subset)
        self.point_labels   = self.point_data.point_labels[self.idx_subset]
        self.n_endpoints    = len(np.flatnonzero(self.point_labels == 1.0))
        self.spheres        = self.point_data.sphere_labels[self.idx_subset]

        self.paths          = {}
        self.new_points     = self.points
        self.new_spheres    = self.spheres
        self.pathfind()
        self.points         = self.new_points
        self.spheres        = self.new_spheres

        self.trunks         = []
        self.split_paths()

    def pathfind(self):
        try:
            self.path_idxs = order_artery_points(self.points, self.point_labels, k=6)
        except:
            try:
                self.path_idxs = order_artery_points(self.points, self.point_labels, k=12)
            except:
                print("Pathfinding failed for artery:", self.label)
                return
        for idx in range(self.n_endpoints):
            ordered_local = self.path_idxs[idx]
            pts_ordered = self.points[ordered_local]
            rad_ordered = self.spheres[ordered_local]

            '''
            n_samples = 200
            pts = smooth_trunk(pts_ordered, rad_ordered, n_samples=n_samples)
            #pts = smooth_trunk(trunk.ords, n_samples=n_samples)
            cells = np.hstack(([n_samples], np.arange(n_samples)))        # single poly-line
            curve = pv.PolyData(pts, lines=cells)
            p = pv.Plotter()
            p.add_mesh(self.skeleton, render_points_as_spheres=True, point_size=5, color="lightgray")
            p.add_mesh(curve, color="dodgerblue", line_width=4, label="Weighted smoothing spline")
            p.add_legend()
            p.show()
            '''

            target_label, target_idx = nearest_other_start_artery(pts_ordered[0], self.skeleton, tol=0.1)
            
            midpoint = np.average(np.vstack((self.skeleton.points[target_idx].reshape(1,3), pts_ordered[0].reshape(1,3))), 0)

            reordered_local = np.insert(ordered_local, 0, len(self.new_points))
            pts_reordered = np.vstack((midpoint.reshape(1, 3), pts_ordered))
            rad_reordered = np.concatenate((np.array(self.point_data.sphere_labels[target_idx]).reshape(1), rad_ordered))

            self.new_points = np.vstack((self.new_points, midpoint.reshape(1, 3)))
            self.new_spheres = np.insert(self.new_spheres, len(self.new_spheres), self.point_data.sphere_labels[target_idx])

            self.paths[target_label] = Path(self.label, target_label, reordered_local, pts_reordered, rad_reordered)

    def split_paths(self):
        if len(self.paths) == 1:
            path: Path = list(self.paths.values())[0]
            self.trunks.append(Trunk(self.label, 0, [], path.pts_reordered, path.rad_reordered))
        elif len(self.paths) == 2:
            if self.label in (8.0, 9.0):
                idx_sequences = [path.idx_reordered for path in self.paths.values()]
                idx_sequences[1] = idx_sequences[1][::-1]
                sequence = []
                i = 0
                j = 0
                while i < len(idx_sequences[0]) or j < len(idx_sequences[1]):
                    if i == len(idx_sequences[0]):
                        sequence.append(idx_sequences[1][j])
                        j += 1
                    elif j == len(idx_sequences[1]):
                        sequence.append(idx_sequences[0][i])
                        i += 1
                    else:
                        a = idx_sequences[0][i]
                        b = idx_sequences[1][j]
                        if a not in idx_sequences[1] and b in idx_sequences[0]:
                            sequence.append(a)
                            i += 1
                        elif b not in idx_sequences[0] and a in idx_sequences[1]:
                            sequence.append(b)
                            j += 1
                        else:
                            if a == b:
                                sequence.append(a)
                            else:
                                # this should never happen
                                print("Pathing error in communicating artery:", self.label)
                            i += 1
                            j += 1
                self.trunks.append(Trunk(self.label, 0, [], self.points[sequence], self.spheres[sequence]))
                
            else:
                trunk = []
                sequence1 = []
                sequence2 = []
                idx_sequences = [path.idx_reordered for path in self.paths.values()]
                targets = [path.target_label for path in self.paths.values()]
                idx_sequences[0] = idx_sequences[0][::-1]
                idx_sequences[1] = idx_sequences[1][::-1]
                i = 0
                j = 0
                while i < len(idx_sequences[0]) or j < len(idx_sequences[1]):
                    if i == len(idx_sequences[0]):
                        sequence2.append(idx_sequences[1][j])
                        j += 1
                    elif j == len(idx_sequences[1]):
                        sequence1.append(idx_sequences[0][i])
                        i += 1
                    else:
                        if idx_sequences[0][i] == idx_sequences[1][j]:
                            trunk.append(idx_sequences[0][i])
                        else:
                            sequence1.append(idx_sequences[0][i])
                            sequence2.append(idx_sequences[1][j])
                        i += 1
                        j += 1
                try:
                    sequence1.insert(0, trunk[-1])
                    sequence2.insert(0, trunk[-1])
                except:
                    print(f"Artery {self.label} has 0 points in trunk")
                branches = []
                branches.append(Branch(self.label, targets[0], self.points[sequence1], self.spheres[sequence1]))
                branches.append(Branch(self.label, targets[1], self.points[sequence2], self.spheres[sequence2]))

                self.trunks.append(Trunk(self.label, 0, branches, self.points[trunk], self.spheres[trunk]))
        elif len(self.paths) == 3:
            trunk1 = []
            trunk2 = []
            sequence1 = []
            sequence2 = []
            sequence3 = []
            idx_sequences = [path.idx_reordered for path in self.paths.values()]
            targets = [path.target_label for path in self.paths.values()]
            idx_sequences[0] = idx_sequences[0][::-1]
            idx_sequences[1] = idx_sequences[1][::-1]
            idx_sequences[2] = idx_sequences[2][::-1]
            i = 0
            j = 0
            k = 0
            while idx_sequences[0][i] == idx_sequences[1][j] and idx_sequences[0][i] == idx_sequences[2][k]:
                    trunk1.append(idx_sequences[0][i])
                    i += 1
                    j += 1
                    k += 1
            if idx_sequences[0][i] == idx_sequences[1][j]:
                sequence3.append(idx_sequences[2][k-1])
                while k < len(idx_sequences[2]):
                    sequence3.append(idx_sequences[2][k])
                    k += 1
                branch = []
                branch.append(Branch(self.label, targets[2], self.points[sequence3], self.spheres[sequence3]))
                self.trunks.append(Trunk(self.label, 0, branch, self.points[trunk1], self.spheres[trunk1]))

                while i < len(idx_sequences[0]) or j < len(idx_sequences[1]):
                    if i == len(idx_sequences[0]):
                        sequence2.append(idx_sequences[1][j])
                        j += 1
                    elif j == len(idx_sequences[1]):
                        sequence1.append(idx_sequences[0][i])
                        i += 1
                    else:
                        if idx_sequences[0][i] == idx_sequences[1][j]:
                            trunk2.append(idx_sequences[0][i])
                        else:
                            sequence1.append(idx_sequences[0][i])
                            sequence2.append(idx_sequences[1][j])
                        i += 1
                        j += 1

                try:
                    trunk2.insert(0, trunk1[-1])
                except:
                    print(f"Artery {self.label} has 0 points in trunk1")
                sequence1.insert(0, trunk2[-1])
                sequence2.insert(0, trunk2[-1])
                branches = []
                branches.append(Branch(self.label, targets[0], self.points[sequence1], self.spheres[sequence1]))
                branches.append(Branch(self.label, targets[1], self.points[sequence2], self.spheres[sequence2]))
                self.trunks.append(Trunk(self.label, 0, branches, self.points[trunk2], self.spheres[trunk2]))
                

            elif idx_sequences[0][i] == idx_sequences[2][k]:
                sequence2.append(idx_sequences[1][j-1])
                while j < len(idx_sequences[1]):
                    sequence2.append(idx_sequences[1][j])
                    j += 1
                branch = []
                branch.append(Branch(self.label, targets[1], self.points[sequence2], self.spheres[sequence2]))
                self.trunks.append(Trunk(self.label, 0, branch, self.points[trunk1], self.spheres[trunk1]))

                while i < len(idx_sequences[0]) or k < len(idx_sequences[2]):
                    if i == len(idx_sequences[0]):
                        sequence3.append(idx_sequences[2][k])
                        k += 1
                    elif k == len(idx_sequences[2]):
                        sequence1.append(idx_sequences[0][i])
                        i += 1
                    else:
                        if idx_sequences[0][i] == idx_sequences[2][k]:
                            trunk2.append(idx_sequences[0][i])
                        else:
                            sequence1.append(idx_sequences[0][i])
                            sequence3.append(idx_sequences[2][k])
                        i += 1
                        k += 1

                try:
                    trunk2.insert(0, trunk1[-1])
                except:
                    print(f"Artery {self.label} has 0 points in trunk1")
                sequence1.insert(0, trunk2[-1])
                sequence3.insert(0, trunk2[-1])
                branches = []
                branches.append(Branch(self.label, targets[0], self.points[sequence1], self.spheres[sequence1]))
                branches.append(Branch(self.label, targets[2], self.points[sequence3], self.spheres[sequence3]))
                self.trunks.append(Trunk(self.label, 0, branches, self.points[trunk2], self.spheres[trunk2]))


            elif idx_sequences[1][j] == idx_sequences[2][k]:
                sequence1.append(idx_sequences[0][i-1])
                while i < len(idx_sequences[0]):
                    sequence1.append(idx_sequences[0][i])
                    i += 1
                branch = []
                branch.append(Branch(self.label, targets[0], self.points[sequence1], self.spheres[sequence1]))
                self.trunks.append(Trunk(self.label, 0, branch, self.points[trunk1], self.spheres[trunk1]))

                while j < len(idx_sequences[1]) or k < len(idx_sequences[2]):
                    if j == len(idx_sequences[1]):
                        sequence3.append(idx_sequences[2][k])
                        k += 1
                    elif k == len(idx_sequences[2]):
                        sequence2.append(idx_sequences[1][j])
                        j += 1
                    else:
                        if idx_sequences[1][j] == idx_sequences[2][k]:
                            trunk2.append(idx_sequences[1][j])
                        else:
                            sequence2.append(idx_sequences[1][j])
                            sequence3.append(idx_sequences[2][k])
                        j += 1
                        k += 1
                
                try:
                    trunk2.insert(0, trunk1[-1])
                except:
                    print(f"Artery {self.label} has 0 points in trunk1")
                sequence2.insert(0, trunk2[-1])
                sequence3.insert(0, trunk2[-1])
                branches = []
                branches.append(Branch(self.label, targets[1], self.points[sequence2], self.spheres[sequence2]))
                branches.append(Branch(self.label, targets[2], self.points[sequence3], self.spheres[sequence3]))
                self.trunks.append(Trunk(self.label, 0, branches, self.points[trunk2], self.spheres[trunk2]))
            else:
                "ERROR Trifurcation"
                sys.exit(2)

class Point:
    def __init__(self, label, point, derivative):
        self.label = label
        self.point = point
        self.derivative = derivative
    
    def make_derivative_negative(self):
        self.derivative = -self.derivative

class Connection:
    def __init__(self, point, splines: list):
        self.point = point
        self.splines = splines
        _ = self.eval_points
    
    @cached_property
    def paths(self):
        #returns spline if first or last point is equivalent to the connection point, returns none otherwise
        find_spline = lambda spline : spline if np.allclose(spline.points[-1], self.point, atol=1e-2) or np.allclose(spline.points[0], self.point, atol=1e-2) else None
        paths = [find_spline(spline) for spline in self.splines if find_spline(spline) != None]
        return paths
    
    @cached_property
    def path_points(self):
        path_points = []
        for path in self.paths:
            label = path.label
            points = [Point(label, point, derivative) for point, derivative in zip(path.points, path.derivatives.T)]
            path_points.append(points)
        return path_points


    @cached_property
    def bifurcation_label(self):
        bifurcation_label = ""
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
        labels = []
        for path in self.paths:
            label = path.label
            labels.append(label)
        
        labels = np.unique(labels)
        for label in labels:
            bifurcation_label = bifurcation_label + f"{vessel_labels[label]}/"

        return bifurcation_label[:-1]

    #points where the derivative is being evaluated at
    #only saved for graphical purposes
    @cached_property
    def eval_points(self):
        eval_points = []
        for points, index in zip(self.path_points, range(len(self.path_points))):
            if np.allclose(points[-1].point, self.point, atol=1e-2):
                points = np.flip(points, axis=0)
                for point in points:
                    point.make_derivative_negative()
                self.path_points[index] = points
            eval_point = points[6].point
            derivative = points[6].derivative
            eval_points.append(Point(points[0].label, eval_point, derivative))
        return eval_points
    
    @cached_property
    def angles(self):
        #finds angle based on dot product of two vectors
        pairs = list(combinations(self.eval_points, 2))

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

        angles = []
        for pair in pairs:
            label = f"{vessel_labels[pair[0].label]}/{vessel_labels[pair[1].label]}"
            angles.append(Angles([pair[0].point, pair[1].point], [pair[0].derivative, pair[1].derivative], label))
        return angles

class Spline:
    def __init__(self, trunk_or_branch: str, obj):
        #the actual trunk or branch object
        self.obj = obj      
        self.n_samples = len(obj.ords) * 5
        #trunk vs trunk2 vs branch
        if trunk_or_branch == "trunk":
            self.points, self.derivatives = smooth_trunk(obj.ords, obj.spheres, n_samples=self.n_samples)  
        elif trunk_or_branch == "trunk2":
            self.points, self.derivatives = smooth_trunk2(obj.ords, obj.spheres, n_samples=self.n_samples, h_fraction=0.05, tan_strength=0.01) 
        else:
            self.points, self.derivatives = smooth_branch(obj.ords, obj.spheres, n_samples=self.n_samples, h_fraction=0.01, tan_strength=0.01)  
        
        self.cells = np.hstack(([self.n_samples], np.arange(self.n_samples))) 
        self.curve = pv.PolyData(self.points, lines=self.cells)
        self.label = obj.label

class Angles:
    def __init__(self, eval_points, derivatives, bifurcation_label):
        self.eval_points = eval_points
        self.derivatives = derivatives
        self.bifurcation_label = bifurcation_label

    @cached_property
    def angle(self):
        find_angle = lambda pair : np.degrees(np.arccos(((np.dot(pair[0], pair[1])) / (np.linalg.norm(pair[0]) * np.linalg.norm(pair[1])))))
        angle = round(find_angle(self.derivatives), 2)
        return angle

class COW:
    def __init__(self, skeleton: pv.PolyData, patient_ID):
        self.skeleton = skeleton
        self.patient_ID = patient_ID
        self.arrays = PointDataArrays(self.skeleton, "Artery", "CenterlineLabels", "Radius")
        self.present_artery_labels = np.unique(self.arrays.artery_labels).astype(float)
        self.network = pv.PolyData()

        self.arteries = {}
        self.get_arteries()

        self.endpoints = defaultdict(list)
        self.trunks = {}
        self.trunks2 = {}
        self.branches = {}
        self.combine_arteries()
        self.connections = []
        #self.get_connections()

    @cached_property
    def all_splines(self):
        all_splines = []
        for trunk in list(self.trunks.values()):
            all_splines.append(Spline("trunk", trunk))
        for trunk2 in list(self.trunks2.values()):
            all_splines.append(Spline("trunk2", trunk2))
        for branch in list(self.branches.values()):
            all_splines.append(Spline("branch", branch))
        return all_splines

    def get_connections(self):
        #iterate through arteries
        connection_pts = np.empty((0, 3))
        for key in self.arteries.keys():
            #for the given artery, find how many paths it contains
            num_paths = len(self.arteries[key].paths)
            paths = list(self.arteries[key].paths.values())
            combinations = []       
            #if the artery contains only one path, it cannot have any split points along the artery, so continue
            if num_paths <= 1:
                continue
            for path in paths:
                for path2 in paths:
                    if path.target_label != path2.target_label:
                        combinations.append([path.target_label, path2.target_label])
            #find all unique combinations of vessel intersections
            combinations = np.unique(np.sort(combinations, axis=1), axis=0)

            for combination in combinations:
                for path1 in paths:
                    for path2 in paths:
                        #if the path labels correspond to a combination in combinations, continue and find the last shared connection point
                        if ([path1.target_label, path2.target_label] == combination).all():
                            if (find_last_shared_point(path1, path2) == [0, 0, 0]).all():
                                continue
                            connection_pts = np.vstack([connection_pts, find_last_shared_point(path1, path2)])
        
        #connections needs all splines as an input
        #remove duplicate connection points and make an instance of connection class
        for connection in np.unique(connection_pts, axis=0):
            self.connections.append(Connection(connection, self.all_splines))

    def get_arteries(self):
        for label in self.present_artery_labels:
            self.arteries[label] = Artery(self.skeleton, label, self.arrays)

    def combine_arteries(self):
        for label in self.present_artery_labels:
            artery: Artery = self.arteries[label]
            for trunk in artery.trunks:
                if len(trunk.ords > 1):
                    end1 = trunk.ords[0]
                    end2 = trunk.ords[-1]
                    self.endpoints[tuple(end1)].append(trunk)
                    self.endpoints[tuple(end2)].append(trunk)
        for label in self.present_artery_labels:
            artery: Artery = self.arteries[label]
            for trunk in artery.trunks:
                if len(trunk.branches) > 0:
                    for branch in trunk.branches:
                        end1 = branch.ords[0]
                        end2 = branch.ords[-1]
                        self.endpoints[tuple(end1)].append(branch)
                        self.endpoints[tuple(end2)].append(branch)
        
        for key in self.endpoints.keys():
            #print(f"\nPoint: {key} -- [{len(self.endpoints[key])}] connected arteries --", end=" ")
            #for artery in self.endpoints[key]:
            #    print(artery.label, end=" ")
            if len(self.endpoints[key]) == 2:
                part1, part2 = self.endpoints[key]
                if part1.label in self.branches.keys() and type(part1) == Trunk:
                    ords1 = self.branches[part1.label].ords
                    spheres1 = self.branches[part1.label].spheres
                else:
                    ords1 = part1.ords
                    spheres1 = part1.spheres

                ords2 = part2.ords
                spheres2 = part2.spheres

                if tuple(ords1[-1]) == tuple(ords2[0]):
                    # abc + cde
                    new_ords = np.vstack([ords1, ords2[1:]])
                    new_spheres = np.concatenate([spheres1, spheres2[1:]])
                    #print("Combination 1")
                elif tuple(ords1[0]) == tuple(ords2[-1]):
                    # cde + abc
                    new_ords = np.vstack([ords2, ords1[1:]])
                    new_spheres = np.concatenate([spheres2, spheres1[1:]])
                    #print("Combination 2")
                elif tuple(ords1[0]) == tuple(ords2[0]):
                    # cba + cde or cde + cba
                    new_ords = np.vstack([ords2[::-1], ords1[1:]])
                    new_spheres = np.concatenate([spheres2[::-1], spheres1[1:]])
                    #print("Combination 3")
                elif tuple(ords1[-1]) == tuple(ords2[-1]):
                    # abc + edc or edc + abc
                    new_ords = np.vstack([ords1[:-1], ords2[::-1]])
                    new_spheres = np.concatenate([spheres1[:-1], spheres2[::-1]])
                    #print("Combination 4")
                else:
                    # This should never happen
                    print("ERROR:", part1.label, part2.label)
                    print("ords1:", ords1)
                    print("ords2:", ords2)
                    print("Key:", key)
                    raise ValueError("ERROR: Cant combine arteries.")
                
                if type(part1) == Branch:
                    self.branches[part2.label] = Trunk(part2.label, 0, [], new_ords, new_spheres)
                else:
                    self.branches[part1.label] = Trunk(part1.label, 0, [], new_ords, new_spheres)
            elif len(self.endpoints[key]) == 3:
                part1, part2, part3 = self.endpoints[key]
                if part1.label not in self.trunks.keys():
                    self.trunks[part1.label] = part1
                if type(part2) == Trunk and part2.label not in self.trunks2.keys():
                    self.trunks2[part2.label] = part2
                    #print(part2.ords)
                elif type(part3) == Trunk and part3.label not in self.trunks2.keys():
                    self.trunks2[part3.label] = part3
                    #print(part3.ords)

    def graph_arteries(self):
        eval_points = np.empty((0, 3))
        derivatives = np.empty((0, 3))
        #grabbing all points being evaluated and their evaluation points
        for connection in self.connections:
            points = connection.eval_points
            for point in points:
                eval_points = np.vstack((eval_points, point.point))
                derivatives = np.vstack((derivatives, point.derivative))

        #merging all splines into one network
        for spline in self.all_splines:
            self.network.merge(spline.curve, inplace=True)

        p = pv.Plotter()

        #plotting derivative vectors
        for derivative, eval_point in zip(derivatives, eval_points):
            arrow = pv.Arrow(start=eval_point, direction=(derivative * 3), scale=1)
            p.add_mesh(arrow, color='red')

        #plotting arcs
        for connection in self.connections:
            center_point = connection.point
            angles = connection.angles
            for angle in angles:
                pair = angle.eval_points

                vec1 = pair[0] - center_point
                vec2 = pair[1] - center_point
                
                dist1 = np.linalg.norm(vec1)
                dist2 = np.linalg.norm(vec2)

                target_radius = ((dist1 + dist2) / 2) * 2

                projected_p1 = center_point + (vec1 / dist1) * target_radius
                projected_p2 = center_point + (vec2 / dist2) * target_radius

                arc = pv.CircularArc(pointa=projected_p1, pointb=projected_p2, center=center_point)

                label_point = arc.points[int(len(arc.points) / 2)]
                
                p.add_point_labels([label_point], [f"{angle.bifurcation_label}: {angle.angle}"], font_size=12, text_color='black')

                p.add_mesh(arc, color='black', line_width=3)

        
        p.add_mesh(self.skeleton, render_points_as_spheres=True, point_size=5, color="lightgray")
        p.add_mesh(self.network, color="dodgerblue", line_width=4, label="Weighted smoothing spline")
        point_cloud = [connection.point for connection in self.connections]
        derivative_point_cloud = pv.PolyData(eval_points)
        p.add_mesh(derivative_point_cloud, color = 'purple', point_size=10, render_points_as_spheres=True)
        p.add_mesh(pv.PolyData(point_cloud), color='red', point_size=10, render_points_as_spheres=True)
        p.add_legend()
        p.show()
    
    def test_graph(self):
        for spline in self.all_splines:
            self.network.merge(spline.curve, inplace=True)
        
        p = pv.Plotter()

        p.add_mesh(self.skeleton, render_points_as_spheres=True, point_size=5, color="lightgray")
        p.add_mesh(self.network, color="dodgerblue", line_width=4, label="Weighted smoothing spline")
        #point_cloud = [connection.point for connection in self.connections]
        #derivative_point_cloud = pv.PolyData(eval_points)
        #p.add_mesh(derivative_point_cloud, color = 'purple', point_size=10, render_points_as_spheres=True)
        #p.add_mesh(pv.PolyData(point_cloud), color='red', point_size=10, render_points_as_spheres=True)
        p.add_legend()
        p.show()

    def export_angles(self, file):
        data = [[self.patient_ID]]
        for connection in self.connections:
            for angle in connection.angles:
                data.append([f"{angle.bifurcation_label}:{angle.angle}"])
        with open(file, 'a', newline='\n') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        
def find_last_shared_point(path1, path2):
    #find points assosiated with each branching path
    path1_pts = path1.pts_reordered
    path2_pts = path2.pts_reordered

    #find how many points the longer path has
    max_len = max(len(path1_pts), len(path2_pts))

    flipped = False
    #flip the numpy arrays so they're going in the same direction
    try:
        while flipped == False:
            if (path1_pts[-1] == path2_pts[-1]).all() == True:
                print("asdfasdf")
                path1_pts = np.flip(path1_pts, axis=0)
                path2_pts = np.flip(path2_pts, axis=0)
                flipped = True
            elif (path1_pts[0] == path2_pts[-1]).all() == True:
                path2_pts = np.flip(path2_pts, axis=0)
                flipped = True
            elif (path1_pts[-1] == path2_pts[0]).all() == True:
                path1_pts = np.flip(path1_pts, axis=0)
                flipped = True
            elif (path1_pts[0] == path2_pts[0]).all() == True:
                flipped = True
            #accounts for weird geometry
            else:
                print(path1_pts)
                print(path2_pts)
                break
                path1_pts = np.delete(path1_pts, 0, axis=0)
                path2_pts = np.delete(path2_pts, 0, axis=0)
    except:
        print()
    

    #pad the shorter path with [0, 0, 0] so they're the same length
    path1_pts = np.pad(path1_pts, ((0, max_len - len(path1_pts)), (0, 0)), mode='constant', constant_values=0)
    path2_pts = np.pad(path2_pts, ((0, max_len - len(path2_pts)), (0, 0)), mode='constant', constant_values=0)

    #find the where the points diverge and find the index of that point
    last_shared_pt_idx = np.flatnonzero((path1_pts == path2_pts).all(axis=1))[-1]
    #find the point
    last_shared_pt = path1_pts[last_shared_pt_idx]
    return last_shared_pt

def create_cow(
    skeleton: pv.PolyData,
    patient_ID
):
    test_cow = COW(skeleton, patient_ID)
    print(len(test_cow.all_splines))

    num_splines = {}
    for spline in test_cow.all_splines:
        if spline.obj.label not in list(num_splines.keys()):
            num_splines[spline.obj.label] = 1
        else:
            num_splines[spline.obj.label] += 1
    
    print(num_splines)
    #test_cow.graph_arteries()
    #test_cow.export_angles("angles.csv")
    test_cow.test_graph()
    return test_cow

def nearest_other_start_artery(
    start_xyz: np.ndarray,
    mesh: pv.PolyData,
    tol: float = 1e-6,
) -> Tuple[float, int]:
    """
    Find the Artery label and mesh index of the nearest *compatible* start
    point.  Compatibility is defined by `standardAdj`: two arteries are
    compatible if either

      •   the candidate label is in `standardAdj[this_artery]`, or
      •   `this_artery` is in `standardAdj[candidate_label]`.

    Parameters
    ----------
    start_xyz : (3,) array-like
        Coordinates of the query start point.
    mesh : pyvista.PolyData
        Must contain point-data arrays 'Artery' and 'CenterlineLabels'.
    tol : float, optional
        Euclidean tolerance used to match `start_xyz` to an existing
        start point in the mesh.

    Returns
    -------
    label : float
        Artery label of the closest compatible start point.
    mesh_index : int
        Index of that point in `mesh.points`.

    Raises
    ------
    ValueError
        If `start_xyz` cannot be matched to a start point (within `tol`),
        or if no compatible start points exist.
    """
    standardAdj: dict[float, tuple[float, ...]] = {
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

    starts   = np.asarray(mesh['CenterlineLabels'])  # 0.0 = normal, 1.0 = start
    arteries = np.asarray(mesh['Artery'])            # 1.0 … 13.0

    # indices of all start points in the mesh
    start_idx = np.where(starts == 1.0)[0]
    if start_idx.size == 0:
        raise ValueError("Mesh contains no start points (CenterlineLabels == 1.0).")

    start_pts = mesh.points[start_idx]

    # identify the mesh point that matches start_xyz
    dists_in   = np.linalg.norm(start_pts - start_xyz, axis=1)
    match_mask = dists_in < tol
    if not np.any(match_mask):
        raise ValueError(f"Provided coordinates don’t match a start point (tol={tol}).")

    this_idx_in_start = int(np.flatnonzero(match_mask)[0])
    this_mesh_idx     = int(start_idx[this_idx_in_start])
    this_artery       = float(arteries[this_mesh_idx])

    # build the compatibility set for this artery
    direct_neigh = set(standardAdj.get(this_artery, ()))
    reverse_neigh = {a for a, neigh in standardAdj.items() if this_artery in neigh}
    compat_labels = direct_neigh | reverse_neigh

    if not compat_labels:
        raise ValueError(f"Artery {this_artery} has no defined neighbours in standardAdj.")

    # candidates: start points on compatible (and different) arteries
    other_mask = (arteries[start_idx] != this_artery) & np.isin(arteries[start_idx], list(compat_labels))
    if not np.any(other_mask):
        raise ValueError("No compatible start points found for this artery.")

    other_pts      = start_pts[other_mask]
    other_labels   = arteries[start_idx][other_mask]
    other_mesh_idx = start_idx[other_mask]

    dists_out = np.linalg.norm(other_pts - start_xyz, axis=1)
    nearest_i = int(np.argmin(dists_out))

    return float(other_labels[nearest_i]), int(other_mesh_idx[nearest_i])

# ---------------------------------------------
# Helper: Union–Find (Disjoint-Set Forest)
# ---------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, u):
        while u != self.parent[u]:
            self.parent[u] = self.parent[self.parent[u]]  # path compression
            u = self.parent[u]
        return u

    def union(self, u, v):
        ru, rv = self.find(u), self.find(v)
        if ru == rv:
            return False
        if self.rank[ru] < self.rank[rv]:
            self.parent[ru] = rv
        elif self.rank[ru] > self.rank[rv]:
            self.parent[rv] = ru
        else:
            self.parent[rv] = ru
            self.rank[ru] += 1
        return True

# ---------------------------------------------------------
# Main function: degree-constrained MST
# ---------------------------------------------------------
def dc_emst(points: np.ndarray,
            bif_mask: np.ndarray,
            k: int = 10):
    """
    Build a minimum-length tree with degree caps:
        • normal point: ≤ 2
        • bifurcation point (mask == 1): ≤ 3

    Parameters
    ----------
    points : (N, 3) float
        3-D coordinates.
    bif_mask : (N,) bool / int
        1 → bifurcation, 0 → normal.
    k : int, optional
        How many nearest neighbours to consider for each point
        (keeps the candidate edge list tractable).

    Returns
    -------
    edges : list[tuple(int,int,float)]
        (u, v, length) for each chosen edge – exactly N-1 items.
    """
    N = len(points)
    bif_mask = bif_mask.astype(bool)

    # degree limits
    max_deg = np.where(bif_mask, 3, 2)

    # --- build candidate edges (k-NN graph) -----------------
    kdt  = cKDTree(points)
    dists, idxs = kdt.query(points, k=k + 1)   # includes self at dist 0
    candidate_edges = []
    for u in range(N):
        for v_idx, dist in zip(idxs[u][1:], dists[u][1:]):  # skip self
            v = int(v_idx)
            if u < v:  # avoid duplicates
                candidate_edges.append((u, v, dist))
    # sort by edge length
    candidate_edges.sort(key=lambda x: x[2])

    # --- greedy Kruskal with degree constraints -------------
    uf     = UnionFind(N)
    degree = np.zeros(N, dtype=int)
    mst    = []

    for u, v, w in candidate_edges:
        if degree[u] >= max_deg[u] or degree[v] >= max_deg[v]:
            continue
        if uf.union(u, v):
            mst.append((u, v, w))
            degree[u] += 1
            degree[v] += 1
            if len(mst) == N - 1:       # tree is complete
                break

    # --------- sanity check ---------------------------------
    if len(mst) != N - 1:
        raise RuntimeError(
            "Could not build a spanning tree without breaking "
            "degree limits; try a larger k or relax constraints."
        )

    return mst

def order_artery_points(points: np.ndarray,
                        centerline_labels: np.ndarray,
                        k: int = 6):
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

    centerlines = []

    for idx in range(len(np.flatnonzero(centerline_labels == 1.0))):

        # --- 3. find start + farthest leaf on the MST ----------------------------
        start_idx = int(np.flatnonzero(centerline_labels == 1.0)[idx])

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

        centerlines.append(np.array(order, dtype=int))

    return centerlines

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
