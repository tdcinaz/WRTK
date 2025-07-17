from tubular_skeleton import Image, Skeleton, SkeletonModel
import itertools
import numpy as np


input = "training/labelsTr/topcow_308.nii.gz"

pv_image = Image(input)

skeleton = pv_image.create_skeleton()
#skeleton = skeleton.filter_artery_by_radius([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 0.315)
skeleton.plot()

#template = SkeletonModel()

#similarity, affine = template.find_linear_transform(skeleton)
#template.apply_linear_transform(similarity)
#template.apply_linear_transform(affine)
#transform = template.find_non_linear_transform(skeleton)
#template.apply_non_linear_transform(transform)

#template.plot(skeleton, plot_skeleton=True)


'''ideas:
'''

'''problems:
1. bad geometry'''