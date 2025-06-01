""""Vessel segmentation module"""
from enum import Enum, unique

ARTERIAL_MARKER_LIST = [
    "BG",
    "LCAR",
    "RCAR",
    "BAS",
    "Acom",
    "LACA1",
    "RACA1",
    "LMCA1",
    "RMCA1",
    "LPcom",
    "RPcom",
    "LPCA1",
    "RPCA1",
    "LPCA2",
    "RPCA2",
    "LSCA",
    "RSCA",
    "LAChA",
    "RAChA",
]

ARTERIAL_MARKER_DICT = {name: i for i, name in enumerate(ARTERIAL_MARKER_LIST)}


@unique
class ArterialEnum(Enum):
    BG = 0
    LCAR = 1
    RCAR = 2
    BAS = 3
    Acom = 4
    LACA1 = 5
    RACA1 = 6
    LMCA1 = 7
    RMCA1 = 8
    LPcom = 9
    RPcom = 10
    LPCA1 = 11
    RPCA1 = 12
    LPCA2 = 13
    RPCA2 = 14
    LSCA = 15
    RSCA = 16
    LAChA = 17
    RAChA = 18
