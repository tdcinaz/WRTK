import geometry_master
from geometry_master import Image, Skeleton, OrderedSkeleton
import itertools
import numpy

input = "training/labelsTr/topcow_301.nii.gz"

pv_image = Image(input)

skeleton = pv_image.create_skeleton()

#print(skeleton.points)

ordered_skeleton = OrderedSkeleton.create_from_parent(skeleton)

print(ordered_skeleton.points)
