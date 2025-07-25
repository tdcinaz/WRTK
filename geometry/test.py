from tubular_skeleton import Image, Skeleton, SkeletonModel
import numpy as np

import time


input = "training/labelsTr/topcow_307.nii.gz"

pv_image = Image(input)

skeleton = pv_image.create_skeleton()
skeleton = skeleton.filter_artery_by_radius([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
#potential = skeleton.find_potential_at_point(1, np.array((30, 19, 28)))


template = SkeletonModel(skeleton)
#similarity, affine = template.find_linear_transform()
#template.apply_linear_transform(similarity)
template.move_all_non_anchor_points()
template.move_anchor_points()
#for artery in np.unique(skeleton.point_data['Artery']):
#    template.optimize_move(artery)

#start_time = time.process_time()
#for artery in np.unique(skeleton.point_data['Artery']):
#    sum = template.cost(artery)
#    print(sum)
#end_time = time.process_time()

#print("CPU time:", end_time - start_time)

def logger(it, U, g):
    print(f"iter {it:4d}   U={U: .6e}   |∇U|={g:.3e}")

template.optimise(artery=1,
             lr=0.02,
             max_iters=2,
             beta_mom=None,      # momentum; set None for plain GD
             callback=logger)

#template.simulated_annealing(5)
template.compute_all_tangents()
template.compute_all_splines()

template.plot(plot_skeleton=True, plot_tangents=False)


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