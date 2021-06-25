

# https://usafacts.org/visualizations/covid-vaccine-tracker-states/
VACCINE_COVERAGE_OVER_TIME = [
    [52+9, 0.0847*100],  # March 4
    [52+13, 0.1767*100],  # April 2
    [52+17, 0.3089*100],  # April 30
    [52+20, 0.3822*100],  # May 21
]

VACCINE_COVERAGE_BY_AGE = [
    [[52+20, 0]],    # 0-4
    [[52+20, 20]],   # 5-17
    [[52+20, 35]],   # 19-29
    [[52+20, 59]],   # 50-64
    [[52+20, 74]],    # 65-75
    [[52+20, 72]]    # 75+
]


FACTOR = 1.5
MAX_HOSP_RATE = 51*FACTOR  # per 100,000 population https://gis.cdc.gov/grasp/COVIDNet/COVID19_3.html
MIN_HOSP_RATE = 9.4
MAX_HOSP_RATE_BY_AGE = [24.3*FACTOR, 4.9*FACTOR, 25.9*FACTOR, 66.3*FACTOR, 124.6*FACTOR, 308.4*FACTOR]
MIN_HOSP_RATE_BY_AGE = [9.4, 2, 1.1, 3.8, 11.1, 24.7, 58.4, 103]

# age distribution of hospitalization
HOSP_AGE_DIST = [
    [[52, 0.69, 0, 1.9]],       # 0-4
    [[52, 1.2, 0, 3.6]],        # 5-17
    [[52, 29.0, 19.4, 41.8]],   # 19-29
    [[52, 28.3, 20.6, 37.1]],   # 50-64
    [[52, 40.9, 27.4, 53.3]]    # 65+
]
