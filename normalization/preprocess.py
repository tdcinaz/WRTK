#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that extract the Circle of Willis from a Time-of-flight images.
"""

import argparse
import logging
import os
import sys
from os.path import exists, join
import shutil

from normalize import full_pipeline


def parse_arguments() -> argparse.Namespace:
    """Simple CommandLine argument parsing function making use of the argsparse module.

    Returns:
        argparse.Namespace: command line arguments.
    """

    parser = argparse.ArgumentParser(
        prog="eICAB",
        description="Vascular Medical Image Automatic Analysis. "
        "Optimize for CW analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_folder",
        help="ToF MRA image [ 3D image | .nii/ .nii.gz ].",
    )
    parser.add_argument(
        "patient_ID",
        help="patient ID"
    )
    parser.add_argument(
        "output",
        help="Defines the output folder.",
    )
    parser.add_argument(
        "-f",
        "--overwrite",
        help="Force overwriting of the output files.",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        help="Isortopic resampling resolution in mm (eg. [ Xmm Ymm Zmm ]).",
        type=float,
        default=0.625,
    )
    parser.add_argument(
        "-t",
        "--template_path",
        help="Path Leading to AVG_TOF_MNI_SS.nii.gz/willis_sphere.nii.gz/SSS.nii.gz ",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--attention",
        help="If flag, an attention model is used",
        action="store_true",
    )
    parser.add_argument(
        "-p", 
        "--prefix", 
        help="output file prefix.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="If flagged progress is reported, use -vv for more information.",
        action="count",
    )
    parser.add_argument(
        "-sk",
        "--skip",
        action="store_true",
    )
    parser.add_argument(
        "-b",
        "--batch_mode",
        action="store_true"
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    if exists(args.output):
        if not args.overwrite:
            print(f"Outputs directory {args.output} exists. Use -f to for overwriting.")
            sys.exit(1)
    else:
        os.makedirs(args.output)

    if args.verbose:
        # TODO: I don't know why but the logging is instanciated before and we can't change it
        # with logging.baseConfig(...).
        fmt = logging.Formatter("%(asctime)-15s %(message)s", None, "%")
        for h in logging.root.handlers:
            h.setFormatter(fmt)
        logging.root.setLevel(logging.INFO if args.verbose == 1 else logging.DEBUG)

    logging.info("~ =============> Express CW <============= ~")
    logging.info(f"Input folder: {args.input_folder}")
    logging.info(f"Patient ID: {args.patient_ID}")
    logging.info(f"Output: {args.output}")
    logging.info(f"Overwrite: {args.overwrite}")
    logging.info(f"Resolution iso: {args.resolution}")
    logging.info(f"Attention: {args.attention}")
    logging.info(f"TemplatePath: {args.template_path}")
    logging.info(f"Prefix: {args.prefix}")
    logging.info(f"Verbose: {args.verbose}")

    template_mni = [
        "AVG_TOF_MNI_SS_down.nii.gz",
        "willis_sphere_down.nii.gz",
        "SSS_masked_down.nii.gz",
    ]
    template_folder = os.listdir(args.template_path)
    for t in template_mni:
        assert t in template_folder, (
            f"Wrong file naming or missing file in {args.template_path}. "
            f"Problem with {t} file."
        )

    prefix = args.prefix
    if prefix is None:
        prefix = "topcow"
    #    file_path = os.path.basename(args.)
    #    prefix = file_path.replace(".gz", "").replace(".nii", "")
    

    #batch mode
    if args.batch_mode:
        #creates a directory called all scans, initially is empty
        all_scans = "tests/output/all_scans"
        os.makedirs(all_scans, exist_ok=True)
        args.patient_ID = "001"
        
        for i in range(125):
            folder_name = "topcow" + args.patient_ID
            patient_folder = join(all_scans, folder_name)

            #creates patient_folders to move the files to later            
            os.makedirs(patient_folder, exist_ok=True)
            os.makedirs(join(patient_folder, "ct_mask"), exist_ok=True)
            os.makedirs(join(patient_folder, "mr_mask"), exist_ok=True)

            full_pipeline(args, prefix)

            #patient_IDs are not continuous
            if args.patient_ID == "090":
                args.patient_ID = "131"
            else:
                args.patient_ID = str(int(args.patient_ID) + 1)

            while len(args.patient_ID) < 3:
                args.patient_ID = "0" + args.patient_ID
        
        patient_ID = "001"
        os.listdir(join(args.ouput), "batch_scans")

        for i in range(125):
            folder_name = "topcow" + patient_ID
            ct_folder = join(all_scans, folder_name, "ct_mask")
            mr_folder = join(all_scans, folder_name, "mr_mask")


            shutil.move(join(args.output, "nn_space", (f"topcow_ct_cropped_{patient_ID}.nii.gz")), ct_folder)
            shutil.move(join(args.output, "nn_space", (f"topcow_ct_cropped_seg_{patient_ID}.nii.gz")), ct_folder)
            shutil.move(join(args.output, "nn_space", (f"topcow_mr_aligned_cube_{patient_ID}.nii.gz")), ct_folder)
            shutil.move(join(args.output, "nn_space", (f"topcow_mr_cropped_{patient_ID}.nii.gz")), mr_folder)
            shutil.move(join(args.output, "nn_space", (f"topcow_mr_cropped_seg_{patient_ID}.nii.gz")), mr_folder)
            shutil.move(join(args.output, "nn_space", (f"topcow_ct_aligned_cube_{patient_ID}.nii.gz")), mr_folder)

            if patient_ID == "090":
                patient_ID = "131"
            else:
                patient_ID =str(int(args.patient_ID) + 1)

            while len(patient_ID) < 3:
                patient_ID = "0" + patient_ID

        shutil.rmtree(args.output)


    else:
        full_pipeline(args, prefix)
    


if __name__ == "__main__":
    main()
