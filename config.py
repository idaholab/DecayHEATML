import os

class Config:
    DATA_PATHS = {
        'f_based': os.path.expanduser('~/data/FLi_FTh4_FU4_p4.pickle'),
        'cl_based': os.path.expanduser('~/data/NaCl_UCl3.pickle')
    }
    DEFAULT_NORMALIZATION = 337475  # For Cl-based data
