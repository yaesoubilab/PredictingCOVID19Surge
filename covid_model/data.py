import numpy as np

# https://usafacts.org/visualizations/covid-vaccine-tracker-states/
VACCINE_COVERAGE_OVER_TIME = [
    # week, value (%)
    [49, 1.0718],  # 31-Jan-21
    [53, 6.9001],  # 28-Feb-21
    [57, 14.792],  # 28-Mar-21
    [61, 27.7761],  # 25-Apr-21
    [65, 38.9504],  # 23-May-21
    [69, 45.2347],  # 20-Jun-21
    [70, 46.43, 32.31, 65.57]  # 27-Jun-21
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
    ],
    [   # 5-12
        [49, 0],  # 31-Jan-21
        [53, 0],  # 28-Feb-21
        [57, 0.01],  # 28-Mar-21
        [61, 0.01],  # 25-Apr-21
        [65, 0.01],  # 23-May-21
        [69, 0.27],  # 20-Jun-21
    ],
    [   # 13-17
        [49, 0.01],  # 31-Jan-21
        [53, 0.08],  # 28-Feb-21
        [57, 0.23],  # 28-Mar-21
        [61, 1.34],  # 25-Apr-21
        [65, 6.95],  # 23-May-21
        [69, 21.01],  # 20-Jun-21
    ],
    [   # 18-29
        [49, 1],  # 31-Jan-21
        [53, 4.17],  # 28-Feb-21
        [57, 7.36],  # 28-Mar-21
        [61, 16.85],  # 25-Apr-21
        [65, 31.32],  # 23-May-21
        [69, 39.49],  # 20-Jun-21
    ],
    [   # 30-49
        [49, 1.76],  # 31-Jan-21
        [53, 6.62],  # 28-Feb-21
        [57, 11.49],  # 28-Mar-21
        [61, 24.31],  # 25-Apr-21
        [65, 39.65],  # 23-May-21
        [69, 47.34],  # 20-Jun-21
    ],
    [  # 50-64
        [49, 1.65],  # 31-Jan-21
        [53, 7.42],  # 28-Feb-21
        [57, 15.48],  # 28-Mar-21
        [61, 38.45],  # 25-Apr-21
        [65, 55.47],  # 23-May-21
        [69, 62.31],  # 20-Jun-21
    ],
    [  # 65-75
        [49, 0.82],  # 31-Jan-21
        [53, 14.5],  # 28-Feb-21
        [57, 42.8],  # 28-Mar-21
        [61, 66.67],  # 25-Apr-21
        [65, 74.64],  # 23-May-21
        [69, 78.48],  # 20-Jun-21
    ],
    [   # 75+
        [49, 0.76],  # 31-Jan-21
        [53, 23.4],  # 28-Feb-21
        [57, 51.08],  # 28-Mar-21
        [61, 66.62],  # 25-Apr-21
        [65, 72.14],  # 23-May-21
        [69, 75.03],  # 20-Jun-21
    ]
]


UP_FACTOR = 1.25
DOWN_FACTOR = 0.75

# ranges on weekly hospitalization rates
MAX_HOSP_RATE_OVERALL = 51 * UP_FACTOR  # per 100,000 population https://gis.cdc.gov/grasp/COVIDNet/COVID19_3.html
MIN_HOSP_RATE_OVERALL = 9.0 * DOWN_FACTOR
MAX_HOSP_RATE_BY_AGE = np.array([24.3, 9.0, 10.9, 14.1, 41.2, 66.3, 124.6, 318.0])*UP_FACTOR*2
MIN_HOSP_RATE_BY_AGE = [0]*8 # np.array([2, 0.9, 1.7, 2.8, 5.7, 11.1, 24.7, 71.5]) * DOWN_FACTOR

CUM_HOSP_RATE_OVERALL = [[70, 570.7, 232.5,	899.3]]
CUM_HOSP_RATE_BY_AGE = [
    # week, value, minimum, maximum
    [[70, 59.8]],    # 0-4
    [[70, 17.8]],    # 5-12
    [[70, 54.2]],    # 13-17
    [[70, 211.2]],    # 18-29
    [[70, 418.6]],   # 30-49
    [[70, 830.9]],   # 50-64
    [[70, 1253.9]],   # 65-74
    [[70, 2374.9]],   # 75+
]

# age distribution of hospitalization (%)
HOSP_AGE_DIST = [
    # week, value, minimum, maximum
    [[52, 0.6]],    # 0-4
    [[52, 0.3]],    # 5-12
    [[52, 0.6]],    # 13-17
    [[52, 5.8]],    # 18-29
    [[52, 18.1]],   # 30-49
    [[52, 26.8]],   # 50-64
    [[52, 20.2]],   # 65-74
    [[52, 27.5]],   # 75+
]

PERC_INF_WITH_NOVEL = [
    [78, 0.1, 0, 0.3],
    [80, 0.6, 0.3, 1.2],
    [82, 1.4, 0.8, 2.4],
    [84, 3.1, 1.5, 8.8],
    [86, 10.1, 3.8, 34.3],
    [88, 30.4, 10.7, 72],
    [90, 51.7, 31.4, 80.7]
]
