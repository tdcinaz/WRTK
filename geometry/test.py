from tubular_skeleton import Image, Skeleton, SkeletonModel
import itertools
import numpy as np


input = "training/labelsTr/topcow_301.nii.gz"

pv_image = Image(input)

skeleton = pv_image.create_skeleton()
skeleton = skeleton.find_bifurcations()

skeleton.plot()

#template = SkeletonModel()

#similarity, affine = template.find_linear_transform(skeleton)
#template.apply_linear_transform(similarity)
#template.apply_linear_transform(affine)
#transform = template.find_non_linear_transform(skeleton)
#template.apply_non_linear_transform(transform)

#template.plot(skeleton, plot_skeleton=True)


'''ideas:
1. remove very short branches (2 or less)
2. average out bifurcation points that are close together
'''

'''problems:
1. bad geometry still persists after opening
2. opening is sometimes too aggressive'''