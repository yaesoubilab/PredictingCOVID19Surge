import definitions as D
from apace.Inputs import ModelSettings


class COVIDSettings(ModelSettings):
    """ settings of COVID model """

    def __init__(self, if_optimizing=False):

        ModelSettings.__init__(self)

        # model settings
        self.deltaT = 1 / 364
        self.simulationDuration = D.SIM_DURATION  # years of simulation
        self.simulationOutputPeriod = 7/364  # simulation output period
        self.observationPeriod = 7/364    # days for observation period
        self.timeToStartDecisionMaking = 0  # immediately after the detection of spread
                                            # (when we have at least 1 case during an observation period)

        self.ifCollectTrajsOfCompartments = True  # if collect the trajectories of all compartments
        self.storeProjectedOutcomes = True
        self.checkEradicationConditions = False

        # economic evaluation settings
        self.warmUpPeriod = D.CALIB_PERIOD
        self.collectEconEval = True  # to collect cost and health outcomes
        self.annualDiscountRate = 0.0

        # if physical distancing was in effect in the first 1.5 years
        self.ifPDInCalibrationPeriod = True
        self.calibrationPeriod = D.CALIB_PERIOD

        # parameter values
        self.storeParameterValues = True


