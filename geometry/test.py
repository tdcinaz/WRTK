from tubular_skeleton import Image, Skeleton, SkeletonModel
import numpy as np


input = "training/labelsTr/topcow_307.nii.gz"

pv_image = Image(input)

skeleton = pv_image.create_skeleton()
skeleton = skeleton.filter_artery_by_radius([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
#skeleton.find_field(2, sample_resolution=0.5)
#skeleton.plot()

template = SkeletonModel(skeleton)
#similarity, affine = template.find_linear_transform()
#template.apply_linear_transform(similarity)
template.move_all_non_anchor_points()
template.move_anchor_points()
for artery in np.unique(skeleton.point_data['Artery']):
    template.optimize_move(artery)
    print("Artery", int(artery))
    template.loss_function(artery)

template.plot(plot_skeleton=True)


#template.apply_linear_transform(affine)
'''



#for i in range(10):
#    transform = template.find_non_linear_transform(skeleton)
#    template.apply_non_linear_transform(transform)

template.plot(plot_skeleton=True, plot_tangents=False)'''


'''ideas:
1. remove very short branches (2 or less)
'''

'''problems:
1. scan 304 connections'''