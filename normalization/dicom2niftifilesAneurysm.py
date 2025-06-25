import shutil
import dicom2nifti
import os
import SimpleITK as sitk
from os.path import join

in_path = "/home/bdl/CTA_Aneurysm" 
out_path = "/home/bdl/WRTK/tests/output/harvard_data_temp/aneurysm"


all_scans = sorted(os.listdir(in_path))

count = 1

for scan in all_scans:
    dicoms = sorted(os.listdir(join(in_path, scan)))
    num_zeros = 3 - len(str(count))
    zeros = num_zeros * "0"
    patient_ID = zeros + str(count)



    for dicom in dicoms:
        try:
            dicom2nifti.dicom_series_to_nifti(join(in_path, scan, dicom), join(out_path, f"{patient_ID}_{dicom}.nii.gz"))
        except:
            continue
    count += 1