from tubular_skeleton import Image, Skeleton, SkeletonModel
import itertools
import numpy as np


input = "training/labelsTr/topcow_306.nii.gz"

pv_image = Image(input)


skeleton = pv_image.create_skeleton()
skeleton.plot()

#template = SkeletonModel()

#similarity, affine = template.find_linear_transform(skeleton)
#template.apply_linear_transform(similarity)
#template.apply_linear_transform(affine)
#transform = template.find_non_linear_transform(skeleton)
#template.apply_non_linear_transform(transform)

#template.plot(skeleton, plot_skeleton=True)


'''ideas:
1. more control points
2. make anchor points change based on what vessels a given patient has
'''

'''problems:
1. connection points find'''