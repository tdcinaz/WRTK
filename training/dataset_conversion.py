import os
from os.path import join
import shutil

CT = "0000"
MR = "0001"

training_folder = os.listdir("tests/output/all_scans")
imagesTr = "training/imagesTr"
labelsTr = "training/labelsTr"

for folder in training_folder:
    ct_patientID = folder[7:10]
    mr_patientID = str(int(ct_patientID) + 300)

    ct_path = join("tests/output/all_scans", folder, "ct_mask")
    mr_path = join("tests/output/all_scans", folder, "mr_mask")
    
    ct_files = os.listdir(ct_path)
    mr_files = os.listdir(mr_path)

    shutil.move(join(ct_path, f"topcow_ct_cropped_{ct_patientID}.nii.gz"), join(imagesTr, f"topcow_{ct_patientID}_0000.nii.gz"))
    shutil.move(join(ct_path, f"topcow_ct_cropped_seg_{ct_patientID}.nii.gz"), join(labelsTr, f"topcow_{ct_patientID}.nii.gz"))
    shutil.move(join(ct_path, f"topcow_mr_aligned_cube_{ct_patientID}.nii.gz"), join(imagesTr, f"topcow_{ct_patientID}_0001.nii.gz"))
    shutil.move(join(mr_path, f"topcow_mr_cropped_{ct_patientID}.nii.gz"), join(imagesTr, f"topcow_{mr_patientID}_0001.nii.gz"))
    shutil.move(join(mr_path, f"topcow_mr_cropped_seg_{ct_patientID}.nii.gz"), join(labelsTr, f"topcow_{mr_patientID}.nii.gz"))
    shutil.move(join(mr_path, f"topcow_ct_aligned_cube_{ct_patientID}.nii.gz"), join(imagesTr, f"topcow_{mr_patientID}_0000.nii.gz"))