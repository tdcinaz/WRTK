import geometry_master
from geometry_master import Image, Skeleton, OrderedSkeleton
import itertools
import numpy as np

input = "training/labelsTr/topcow_301.nii.gz"

pv_image = Image(input)


skeleton = pv_image.create_skeleton()
#skeleton = skeleton.filter_out_artery_points([1, 2, 3, 4, 6, 7, 8, 9, 11, 12])


#skeleton.plot()

#print(skeleton.points)

ordered_skeleton = OrderedSkeleton.create_from_parent(skeleton)
