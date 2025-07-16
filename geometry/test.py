from tubular_skeleton import Image, Skeleton, SkeletonModel
import itertools
import numpy as np


input = "training/labelsTr/topcow_307.nii.gz"

pv_image = Image(input)


skeleton = pv_image.create_skeleton()
skeleton.plot()
#template = SkeletonModel()

#similarity, affine = template.find_transform(skeleton)
#template.apply_transform(similarity)
#template.apply_transform(affine)


#template.plot(skeleton, plot_skeleton=True)

'''ideas for wednesday:
1. non-linear transform
2. anchor points is a list of point objects instead of tuples
3. make anchor points change based on what vessels a given patient has
'''