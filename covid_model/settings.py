import definitions as D
from apace.CalibrationSupport import get_survey_size
from apace.Inputs import ModelSettings
from covid_model.data import CUM_HOSP_RATE_OVERALL


class COVIDSettings(ModelSettings):
    """ settings of COVID model """

    def __init__(self, if_calibrating=False):

        ModelSettings.__init__(self)

        # model settings
        self.deltaT = 1 / 364
        self.simulationDuration = D.SIM_DURATION  # years of simulation
        self.simulationOutputPeriod = 7/364  # simulation output period
        self.observationPeriod = 7/364    # days for observation period
        self.timeToStartDecisionMaking = 0  # immediately after the detection of spread
                                            # (when we have at least 1 case during an observation period)

        self.ifCollectTrajsOfCompartments = False  # if collect the trajectories of all compartments
        self.storeProjectedOutcomes = True
        self.checkEradicationConditions = False

        # economic evaluation settings
        self.warmUpPeriod = D.CALIB_PERIOD
        self.collectEconEval = False  # to collect cost and health outcomes
        self.annualDiscountRate = 0.0

        # if physical distancing was in effect in the first 1.5 years
        self.ifPDInCalibrationPeriod = True
        self.calibrationPeriod = D.CALIB_PERIOD

        # parameter values
        self.storeParameterValues = True

        # calibration targets
        if if_calibrating:
            self.cumHospRateMean = []
            self.cumHospRateN = []

            week = 0
            while week / 52 < self.calibrationPeriod:
                if week == CUM_HOSP_RATE_OVERALL[0][0]:
                    self.cumHospRateMean.append(CUM_HOSP_RATE_OVERALL[0][1] * 0.00001)
                    self.cumHospRateN.append(get_survey_size(mean=CUM_HOSP_RATE_OVERALL[0][1],
                                                             l=CUM_HOSP_RATE_OVERALL[0][2]*1.2,
                                                             u=CUM_HOSP_RATE_OVERALL[0][3]*0.8,
                                                             multiplier=0.00001,
                                                             interval_type='c'))
                else:
                    self.cumHospRateMean.append(None)
                    self.cumHospRateN.append(None)
                week += 1
