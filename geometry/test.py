import geometry_master
from geometry_master import Image, Skeleton
import itertools
import numpy

input = "training/labelsTr/topcow_307.nii.gz"

pv_image = Image(input)

skeleton = pv_image.create_skeleton()

skeleton.plot()

