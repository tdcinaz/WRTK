#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that extract the Circle of Willis from a Time-of-flight images.
"""

import argparse
import logging
import os
import sys
from os.path import exists

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
        "TOF",
        help="ToF MRA image [ 3D image | .nii/ .nii.gz ].",
    )
    parser.add_argument(
        "SEG",
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
        help="Isotropic resampling resolution in mm (eg. [ Xmm Ymm Zmm ]).",
        type=float,
        default=0.625,
    )
    #parser.add_argument(
    #    "-ss",
    #    "--simple_segmentation",
    #    help="If flagged, will only output segmentation maps",
    #    action="store_true",
    #)
    #parser.add_argument(
    #    "--markers",
    #    help="CW markers to use instead of NN determined markers"
    #    "[ 3D image | .nii/ .nii.gz ].",
    #)
    #parser.add_argument(
    #    "--device",
    #    help="Device type, choices=[ cpu, cuda ].",
    #    choices=["cpu", "cuda"],
    #    default="cpu",
    #)
    parser.add_argument(
        "-c",
        "--cube",
        help="Cube containing the CW to use instead of computed one"
        " [ 3D image | .nii/ .nii.gz ].",
        default=None,
    )
    parser.add_argument(
        "-m",
        "--mask",
        help="Brain mask to use instead of computed one "
        "[ 3D image | .nii/ .nii.gz ].",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--template_path",
        help="Path Leading to AVG_TOF_MNI_SS.nii.gz/willis_sphere.nii.gz/SSS.nii.gz ",
        required=True,
    )
    #parser.add_argument(
    #    "-mp",
    #    "--models_path",
    #    help="Path Leading to trained Neural Network Model.",
    #)
    parser.add_argument(
        "-a",
        "--attention",
        help="If flag, an attention model is used",
        action="store_true",
    )
    parser.add_argument("-p", "--prefix", help="output file prefix.")
    #parser.add_argument(
    #    "--skip_preprocessing",
    #    help="If flagged all preprocessing steps are skipped and the input TOF is assumed to be "
    #    "resampled and reoriented; the following optional arguments must be provided: "
    #    "markers, tof, mask and cube.",
    #    action="store_true",
    #)
    #parser.add_argument(
    #    "-pp",
    #    "--post_processing",
    #    help="If flagged, post_processing is used",
    #    action="store_true",
    #)
    #parser.add_argument(
    #    "--experimental_prediction",
    #    help="If flagged, will predict AChA and SCA",
    #    action="store_true",
    #)
    #parser.add_argument(
    #    "-vs",
    #    "--version",
    #    help="Version used",
    #    default="labels_18_236",
    #    choices=["labels_18", "labels_18_236"],
    #)
    parser.add_argument(
        "-v",
        "--verbose",
        help="If flagged progress is reported, use -vv for more information.",
        action="count",
    )
    args = parser.parse_args()

    #if args.skip_preprocessing and args.markers and args.mask and args.cube is None:
    #    parser.error(
    #        "For skipping preprocessing the following arguments are required: markers, mask, cube"
    #    )

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
    logging.info(f"TOF: {args.TOF}")
    logging.info(f"SEG: {args.SEG}")
    logging.info(f"Output: {args.output}")
    logging.info(f"Overwrite: {args.overwrite}")
    logging.info(f"Resolution iso: {args.resolution}")
    #logging.info(f"Simple segmentation: {args.simple_segmentation}")
    #logging.info(f"Markers: {args.markers}")
    #logging.info(f"Device: {args.device}")
    logging.info(f"Cube: {args.cube}")
    logging.info(f"Mask: {args.mask}")
    #logging.info(f"Models path: {args.models_path}")
    logging.info(f"Attention: {args.attention}")
    logging.info(f"TemplatePath: {args.template_path}")
    logging.info(f"Prefix: {args.prefix}")
    #logging.info(f"SkipPreprocessing: {args.skip_preprocessing}")
    #logging.info(f"Version: {args.version}")
    #logging.info(f"Post processing: {args.post_processing}")
    #logging.info(f"SCA & AChA: {args.experimental_prediction}")
    logging.info(f"Verbose: {args.verbose}")

    #assert (
    #    len(os.listdir(args.models_path)) > 0
    #), "Please provide a model in your models path"
    #extensions = [f[-2:] for f in os.listdir(args.models_path)]
    #if not all("pt" in e for e in extensions):
    #    raise IOError(f"Not all your files in {args.models_path} have a .pt extension")

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

    #optional_args = {"option": 1, "labels_path": None, "vessels_path": None}

    prefix = args.prefix
    if prefix is None:
        file_path = os.path.basename(args.TOF)
        prefix = file_path.replace(".gz", "").replace(".nii", "")

    #if args.skip_preprocessing:
    #    skip_preprocessing_pipeline(
    #        args.TOF,
    #        args.cube,
    #        args.mask,
    #        args.markers,
    #        prefix,
    #        args,
    #        output=args.output,
    #        **optional_args,
    #    )
    #else:
    #    full_pipeline(args, prefix, **optional_args)

    full_pipeline(args, prefix)


if __name__ == "__main__":
    main()
