import argparse
import logging
import os
import sys
from os.path import exists, join
import shutil
from diameters import compute_average_diameters_and_export_vtp

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
        help="Extracted Scan",
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
        action="store_true",
    )

    args = parser.parse_args()

    return args

def main():

    args = parse_arguments()
    logging.info(f"Patient ID: {args.patient_ID}")
    if args.batch_mode:
        all_files = sorted(os.listdir(args.input_folder))
        for file in all_files:
            compute_average_diameters_and_export_vtp(join(args.input_folder, file))
    else:
        compute_average_diameters_and_export_vtp(join(args.input_folder))
    

if __name__ == "__main__":
    main()