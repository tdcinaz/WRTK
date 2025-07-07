import argparse
import logging
import os
import sys
from os.path import exists, join
import shutil
from pipeline import pipeline

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
        "-b",
        "--batch_mode",
        action="store_true"
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    
    if args.batch_mode:
        all_files = sorted(os.listdir("training/labelsTr"))
        for file in all_files:
            args.patient_ID = file.split(".")[0][-3:]
            try:
                pipeline(args, prefix="stroke")
            except:
                continue
    else:
        pipeline(args, prefix="topcow")
    

if __name__ == "__main__":
    main()