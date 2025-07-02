import os
import csv

all_masks = len(os.listdir("/home/bdl/WRTK/tests/input/topcow/labelsTr"))
all_ct = all_masks/2
all_mr = all_ct

whole_csv = []

with open('/home/bdl/WRTK/angles.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        whole_csv.append(row)

vessel_labels = [
    "Basillar",
    "L-PCA",
    "R-PCA",
    "L-ICA",
    "L-MCA",
    "R-ICA",
    "R-MCA",
    "L-Pcom",
    "R-Pcom",
    "Acom",
    "L-ACA",
    "R-ACA",
    "3rd A2"
]

patient_IDs = []
cts = []
mrs = []
flag = False

for row in whole_csv:
    flag = False
    for label in vessel_labels:
        if label in row[0]:
            flag = True
    if flag:
        continue
    patient_IDs.append(row[0])
    if int(row[0]) < 200:
        cts.append(row[0])
    else:
        mrs.append(row[0])



successful_extractions = len(patient_IDs)
print(f"Percentage of all extractions: {successful_extractions/all_masks}")
print(f"Percentage of ct extractions: {len(cts)/all_ct}")
print(f"Percentage of mr extractions: {len(mrs)/all_mr}")
