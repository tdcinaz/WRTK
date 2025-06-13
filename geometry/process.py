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

    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    logging.info(f"Patient ID: {args.patient_ID}")
    pipeline(args, prefix="topcow")
    

if __name__ == "__main__":
    main()