import dicom2nifti
import shutil
import os
import SimpleITK as sitk
from os.path import join

in_path = "/home/bdl/CTA Scans" 
out_path = "/home/bdl/WRTK/tests/output/harvard_data/stroke"

female_scans = os.listdir(join(in_path, "Female"))

shutil.rmtree(join(in_path, "Male"))
shutil.rmtree(join(in_path, "Female"))

all_scans = os.listdir(in_path)
male_or_female = "male"
count = 1


for scan in all_scans:
    male_or_female = "male"
    if scan in female_scans:
        male_or_female = "female"

    num_zeros = 3 - len(str(count))
    zeros = num_zeros * "0"

    reader = sitk.ImageSeriesReader()

    dicom_names = reader.GetGDCMSeriesFileNames(join(in_path, scan))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    sitk.WriteImage(image, join(out_path, f"stroke_{male_or_female}_{zeros + str(count)}.nii.gz"))
    count += 1
