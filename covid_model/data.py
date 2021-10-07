import numpy as np

# https://usafacts.org/visualizations/covid-vaccine-tracker-states/
VACCINE_COVERAGE_OVER_TIME = [
    # week, value (%)
    [49, 1.07],  # 31-Jan-21
    [53, 6.9],  # 28-Feb-21
    [57, 14.97],  # 28-Mar-21
    [61, 27.78],  # 25-Apr-21
    [65, 38.95],  # 23-May-21
    [69, 45.23],  # 20-Jun-21
    [73, 48.95],  # 18-Jul-21
    [77, 51.14, 35.74, 70.13],  # 15-Aug-21
]

VACCINE_COVERAGE_BY_AGE = [
    # week, value
    [   # 0-4
        [49, 0],  # 31-Jan-21
        [53, 0],  # 28-Feb-21
        [57, 0],  # 28-Mar-21
        [61, 0],  # 25-Apr-21
        [65, 0],  # 23-May-21
        [69, 0],  # 20-Jun-21
        [73, 0],  # 18-Jul-21
        [77, 0],  # 15-Aug-21
    ],
    [   # 5-12
        [49, 0],  # 31-Jan-21
        [53, 0],  # 28-Feb-21
        [57, 0.01],  # 28-Mar-21
        [61, 0.01],  # 25-Apr-21
        [65, 0.01],  # 23-May-21
        [69, 0.27],  # 20-Jun-21
        [73, 0.42],  # 18-Jul-21
        [77, 0.42],  # 15-Aug-21
    ],
    [   # 13-17
        [49, 0.01],  # 31-Jan-21
        [53, 0.08],  # 28-Feb-21
        [57, 0.23],  # 28-Mar-21
        [61, 1.34],  # 25-Apr-21
        [65, 6.95],  # 23-May-21
        [69, 21.01],  # 20-Jun-21
        [73, 29.79],  # 18-Jul-21
        [77, 34.89],  # 15-Aug-21
    ],
    [   # 18-29
        [49, 1],  # 31-Jan-21
        [53, 4.17],  # 28-Feb-21
        [57, 7.36],  # 28-Mar-21
        [61, 16.85],  # 25-Apr-21
        [65, 31.32],  # 23-May-21
        [69, 39.49],  # 20-Jun-21
        [73, 44.34],  # 18-Jul-21
        [77, 47.17],  # 15-Aug-21
    ],
    [   # 30-49
        [49, 1.76],  # 31-Jan-21
        [53, 6.62],  # 28-Feb-21
        [57, 11.49],  # 28-Mar-21
        [61, 24.31],  # 25-Apr-21
        [65, 39.65],  # 23-May-21
        [69, 47.34],  # 20-Jun-21
        [73, 51.67],  # 18-Jul-21
        [77, 54.09],  # 15-Aug-21
    ],
    [  # 50-64
        [49, 1.65],  # 31-Jan-21
        [53, 7.42],  # 28-Feb-21
        [57, 15.48],  # 28-Mar-21
        [61, 38.45],  # 25-Apr-21
        [65, 55.47],  # 23-May-21
        [69, 62.31],  # 20-Jun-21
        [73, 66.14],  # 18-Jul-21
        [77, 68.28],  # 15-Aug-21
    ],
    [  # 65-75
        [49, 0.82],  # 31-Jan-21
        [53, 14.5],  # 28-Feb-21
        [57, 42.8],  # 28-Mar-21
        [61, 66.67],  # 25-Apr-21
        [65, 74.64],  # 23-May-21
        [69, 78.48],  # 20-Jun-21
        [73, 81.04],  # 18-Jul-21
        [77, 82.47],  # 15-Aug-21
    ],
    [   # 75+
        [49, 0.76],  # 31-Jan-21
        [53, 23.4],  # 28-Feb-21
        [57, 51.08],  # 28-Mar-21
        [61, 66.62],  # 25-Apr-21
        [65, 72.14],  # 23-May-21
        [69, 75.03],  # 20-Jun-21
        [73, 77.2],  # 18-Jul-21
        [77, 78.27],  # 15-Aug-21
    ]
]


UP_FACTOR = 1.25
DOWN_FACTOR = 0.75

# ranges on weekly hospitalization rates
MAX_HOSP_RATE_OVERALL = 51 * UP_FACTOR  # per 100,000 population https://gis.cdc.gov/grasp/COVIDNet/COVID19_3.html
MIN_HOSP_RATE_OVERALL = 9.0 * DOWN_FACTOR
MAX_HOSP_RATE_BY_AGE = np.array([24.3, 9.0, 10.9, 14.1, 41.2, 66.3, 124.6, 318.0])*UP_FACTOR
MIN_HOSP_RATE_BY_AGE = [0]*8  # np.array([2, 0.9, 1.7, 2.8, 5.7, 11.1, 24.7, 71.5]) * DOWN_FACTOR

# ranges of weekly hospital occupancy
HOSP_OCC_DURATION = [5, 190]   # April-1 to July-7 2020
MAX_HOSP_OCC_RATE = 30 * 2
MIN_HOSP_OCC_RATE = 10

CUM_HOSP_RATE_OVERALL = [[76, 610.8, 238.1, 943.7]]  # Overall
CUM_HOSP_RATE_BY_AGE = [
    # week, value, minimum, maximum
    [[76, 66.6, 31.5, 121.3]],  # 0-4
    [[76, 21.2, 12.1, 29.3]],  # 5-12
    [[76, 58.4, 28.3, 101.2]],  # 12-17
    [[76, 229.7, 65.2, 388.7]],  # 18-29
    [[76, 451.7, 135.6, 688.3]],  # 30-49
    [[76, 874.1, 349, 1191.2]],  # 50-64
    [[76, 1305.7, 504.9, 1938.8]],  # 65-74
    [[76, 2462.9, 1179.8, 3660.8]]  # 75+
]

# TODO: to update
MAX_PREV_IMMUNE_FROM_INF = 40
PREV_IMMUNE_FROM_INF = [
    # [24, 5.9, 5.5, 6.3],
    # [32, 6.8, 6.5, 7.2],
    # [42, 11.5, 11.1, 11.8],
    # [50, 20, 19.5, 20.5],
    # [58, 21.8, 21.2, 22.2],
    # [66, 21.6, 21.2, 22.2],
    [72, 20.6, 4.6, 34.1]
]

# age distribution of hospitalization (%)
HOSP_AGE_DIST = [
    # week, value, minimum, maximum
    [[76, 0.6]],  # 0-4
    [[76, 0.3]],  # 5-12
    [[76, 0.6]],  # 12-17
    [[76, 6.0]],  # 18-29
    [[76, 18.6]],  # 30-49
    [[76, 26.8]],  # 50-64
    [[76, 20.0]],  # 65-74
    [[76, 27.1]]  # 75+
]

PERC_INF_WITH_NOVEL = [
    [78, 0.1, 0, 0.3],
    [80, 0.6, 0.3, 1.2],
    [82, 1.4, 0.8, 2.4],
    [84, 2.4, 1.2, 7.5],
    [86, 3.8, 2.1, 13.9],
    [88, 7, 2.3, 23.6],
    [90, 14.1, 5.1, 44.7],
    [92, 26.1, 4.9, 70.6],
    [94, 37.2, 16.3, 77.7],
    [96, 53.6, 28.1, 87.8],
    [98, 69.4, 51.9, 93],
    [100, 82.2, 70.9, 95.9],
    [102, 87.6, 82.6, 95.9],
    [104, 94.4, 93.1, 97],
    [106, 96.8, 93.8, 99],
    [108, 97.9, 96.4, 99.6],
    [110, 98.8, 97.7, 99.8],
]
