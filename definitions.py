import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CALIB_PERIOD = 1.5  # year (from Mar-1, 2020 to Aug-31, 2021 which is 1.5 years)
PROJ_PERIOD = 0.75  # year (from Sep-1, 2021 to May-31, 2021 whic is 0.75 year)
SIM_DURATION = CALIB_PERIOD + PROJ_PERIOD
