import shutil
import os
import SimpleITK as sitk
from os.path import join

in_path = "/home/bdl/" 
out_path = "/home/bdl/WRTK/tests/output"

reader = sitk.ImageSeriesReader()

dicom_names = reader.GetGDCMSeriesFileNames(in_path)
reader.SetFileNames(dicom_names)
image = reader.Execute()

sitk.WriteImage(image, join(out_path, f"mri.nii.gz"))