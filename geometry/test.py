from tubular_skeleton import Image, Skeleton, SkeletonModel
import itertools
import numpy as np


input = "training/labelsTr/topcow_302.nii.gz"

pv_image = Image(input)

skeleton = pv_image.create_skeleton()
#skeleton = skeleton.filter_artery_by_radius([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 0.325)

#skeleton.plot()

template = SkeletonModel(skeleton)

#template.plot(plot_skeleton=True)

similarity, affine = template.find_linear_transform()
template.apply_linear_transform(similarity)
template.apply_linear_transform(affine)
transform = template.find_non_linear_transform(skeleton)
template.apply_non_linear_transform(transform)

template.plot(plot_skeleton=True)


'''ideas:
1. remove very short branches (2 or less)
2. average out bifurcation points that are close together
'''

'''problems:
1. bad geometry still persists after opening
2. opening is sometimes too aggressive'''