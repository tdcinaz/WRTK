import numpy as np
import pyvista as pv
from scipy.interpolate import BSpline
from collections import defaultdict, deque


# 1. Apply Vessel Enhancing Diffusion
#--------Use eICAB filter from ComputeVED

# 2. Skeletonize complete artery network

# 3. Find largest radius spheres

# 4. Label centerline voxels based on parent arteries

# 5. Label start and end voxels
#--------Use SurfaceNets3D to identify connection points

# 6. Isolate individual arteries

# 7. Detect and trim loops

# 8. Path finding

# 9. Curve fitment

# 10. Network reassembly

# 11. Angle Calculation
def extract_angles(splines, in_file):
    
    #extract mesh from vtp file
    points = pv.wrap(in_file)

    #indices of all end points
    end_pts = [point for point in points.point_data if point["CenterlineLabels"] == 1] #not sure what the label is
    
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
    }

    angle_pairs = []

    #make all pairs of angles that intersect one another
    for key in standardAdj.keys():
        for vessel in standardAdj[key]:
            pair = [key, vessel]
            if [vessel, key] not in angle_pairs:
                angle_pairs.append(pair)

    angles = {}

    for pair in angle_pairs:
        #vessel1 is the first index in the pair list, vessel2 is the second    
        vessel1 = pair[0]
        vessel2 = pair[1]
        #if a vessel is missing then skip
        if vessel1 not in splines.keys() or vessel2 not in splines.keys():
            continue
        
        #find end points of vessels
        points1 = np.array([point for point in end_pts if point["Artery"] == vessel1])
        points2 = np.array([point for point in end_pts if point["Artery"] == vessel2])
        
        #choose the end points that are closest together for the two vessels
        prev_distance = 1000000
        for point1 in points1:
            for point2 in points2:
                distance = np.linalg.norm(point1 - point2)
                if distance < prev_distance:
                    index1 = points1.index(point1)
                    index2 = points2.index(point2)
                prev_distance = distance

        end_pt1 = points1[index1]
        end_pt2 = points2[index2]

        #find spline for each vessel
        spline1 = splines[f"{vessel1}:{vessel2}"]
        spline2 = splines[f"{vessel2}:{vessel1}"]

        #tangent vectors
        tan_vector1 = spline1(end_pt1, nu=1)
        tan_vector2 = spline2(end_pt2, nu=1)

        dot_product = np.dot(tan_vector1, tan_vector2)

        mag1 = np.linalg.norm(tan_vector1)
        mag2 = np.linalg.norm(tan_vector2)

        #use dot product to find angle
        cos = (dot_product) / (mag1 * mag2)
        angle = np.arccos(np.clip(cos, -1, 1))
        
        #append angle to dictionary with names of vessels as the key
        angles[f"{vessel_labels[vessel1]}/{vessel_labels[vessel2]}"] = angle

    return angles

# 12. Data export
