from tubular_skeleton import Image, Skeleton, SkeletonModel
import itertools
import numpy as np


input = "training/labelsTr/topcow_307.nii.gz"

pv_image = Image(input)


skeleton = pv_image.create_skeleton()
template = SkeletonModel()


similarity, affine = template.find_transform(skeleton)
polynomial_transform = template.find_non_linear_transform(skeleton)

#template.plot(skeleton, plot_skeleton=True)
#template.apply_transform(affine)
