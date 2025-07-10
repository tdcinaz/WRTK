import nibabel as nib
import numpy as np
import os
from os.path import join

in_stroke = "/home/bdl/WRTK/tests/output/harvard_data_cropped/stroke"
in_aneurysm = "/home/bdl/WRTK/tests/output/harvard_data_cropped/aneurysm"
out_path_stroke = "/home/bdl/WRTK/tests/output/cropped/stroke"
out_path_aneurysm = "/home/bdl/WRTK/tests/output/cropped/aneurysm"


stroke_folder = os.listdir(in_stroke)
aneurysm_folder = os.listdir(in_aneurysm)

for stroke in stroke_folder:
    img = nib.load(join(in_stroke, stroke))
    data = img.get_fdata()
    cropped_data = data[0:90, 0:90, 0:90]
    cropped_img = nib.Nifti1Image(cropped_data, img.affine)
    nib.save(cropped_img, join(out_path_stroke, stroke))

for aneurysm in aneurysm_folder:
    img = nib.load(join(in_aneurysm, aneurysm))
    data = img.get_fdata()
    cropped_data = data[0:90, 0:90, 0:90]
    cropped_img = nib.Nifti1Image(cropped_data, img.affine)
    nib.save(cropped_img, join(out_path_aneurysm, aneurysm))

    