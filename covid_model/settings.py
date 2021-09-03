import definitions as D
from apace.CalibrationSupport import get_survey_size
from apace.Inputs import ModelSettings
from covid_model.data import *
from definitions import AgeGroups


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
            self.cumHospRateVar = []
            self.cumHospRateN = []

            self.cumHospRateByAgeMean = [[] for i in range(len(AgeGroups))]
            self.cumHospRateByAgeVar = [[] for i in range(len(AgeGroups))]
            self.cumHospRateByAgeN = [[] for i in range(len(AgeGroups))]

            self.prevImmFromInfMean = []
            self.prevImmFromInfVar = []

            self.cumVaccRateMean = []
            self.cumVaccRateVar = []
            self.cumVaccRateN = []

            self.percInfWithNovelMean = []
            self.percInfWithNovelVar = []
            self.percInfWithNovelN = []

            n_perc_novel_used = 5
            weeks_with_data_prec_inf = [v[0] for v in PERC_INF_WITH_NOVEL[0:n_perc_novel_used]]

            week = 0
            while week / 52 < self.calibrationPeriod:
                # cumulative hospitalization rate
                if week == CUM_HOSP_RATE_OVERALL[0][0]:
                    self.cumHospRateMean.append(CUM_HOSP_RATE_OVERALL[0][1] * 0.00001)
                    self.cumHospRateVar.append(
                        0.25*(CUM_HOSP_RATE_OVERALL[0][3]-CUM_HOSP_RATE_OVERALL[0][2])*0.00001
                    )
                    self.cumHospRateN.append(get_survey_size(mean=CUM_HOSP_RATE_OVERALL[0][1],
                                                             l=CUM_HOSP_RATE_OVERALL[0][2],
                                                             u=CUM_HOSP_RATE_OVERALL[0][3],
                                                             multiplier=0.00001,
                                                             interval_type='c'))
                else:
                    self.cumHospRateMean.append(None)
                    self.cumHospRateVar.append(None)
                    self.cumHospRateN.append(None)

                # cumulative hospitalization rate by age
                for a in range(len(AgeGroups)):
                    if week == CUM_HOSP_RATE_BY_AGE[a][0][0]:
                        self.cumHospRateByAgeMean[a].append(CUM_HOSP_RATE_BY_AGE[a][0][1] * 0.00001)
                        self.cumHospRateVar.append(
                            0.25 * (CUM_HOSP_RATE_BY_AGE[a][0][3] - CUM_HOSP_RATE_BY_AGE[a][0][2]) * 0.00001
                        )
                        self.cumHospRateByAgeN[a].append(get_survey_size(mean=CUM_HOSP_RATE_BY_AGE[a][0][1],
                                                                         l=CUM_HOSP_RATE_BY_AGE[a][0][1]*0.5,
                                                                         u=CUM_HOSP_RATE_BY_AGE[a][0][1]*1.5,
                                                                         multiplier=0.00001,
                                                                         interval_type='p'))
                    else:
                        self.cumHospRateByAgeMean[a].append(None)
                        self.cumHospRateByAgeVar[a].append(None)
                        self.cumHospRateByAgeN[a].append(None)

                # prevalence of population with immunity after infection
                if week == PREV_IMMUNE_FROM_INF[0][0]:
                    self.prevImmFromInfMean.append(PREV_IMMUNE_FROM_INF[0][1] * 0.01)
                    self.prevImmFromInfVar.append(
                        0.25*(CUM_HOSP_RATE_OVERALL[0][3]-CUM_HOSP_RATE_OVERALL[0][2])*0.01
                    )
                else:
                    self.prevImmFromInfMean.append(None)
                    self.prevImmFromInfVar.append(None)

                # vaccination rate
                if week == VACCINE_COVERAGE_OVER_TIME[-1][0]:
                    self.cumVaccRateMean.append(VACCINE_COVERAGE_OVER_TIME[-1][1] * 0.01)
                    self.cumHospRateVar.append(
                        0.25 * (VACCINE_COVERAGE_OVER_TIME[-1][3] - VACCINE_COVERAGE_OVER_TIME[-1][2]) * 0.01
                    )
                    self.cumVaccRateN.append(get_survey_size(mean=VACCINE_COVERAGE_OVER_TIME[-1][1],
                                                             l=VACCINE_COVERAGE_OVER_TIME[-1][2],
                                                             u=VACCINE_COVERAGE_OVER_TIME[-1][3],
                                                             multiplier=0.01,
                                                             interval_type='c'))
                else:
                    self.cumVaccRateMean.append(None)
                    self.cumVaccRateVar.append(None)
                    self.cumVaccRateN.append(None)

                # % infected with novel variant
                if week in weeks_with_data_prec_inf:
                    index = weeks_with_data_prec_inf.index(week) # + 4
                    self.percInfWithNovelMean.append(PERC_INF_WITH_NOVEL[index][1]*0.01)
                    self.cumHospRateVar.append(
                        0.25 * (PERC_INF_WITH_NOVEL[index][3] - PERC_INF_WITH_NOVEL[index][2]) * 0.01
                    )
                    self.percInfWithNovelN.append((get_survey_size(mean=PERC_INF_WITH_NOVEL[index][1],
                                                                   l=PERC_INF_WITH_NOVEL[index][2],
                                                                   u=PERC_INF_WITH_NOVEL[index][3],
                                                                   multiplier=0.01,
                                                                   interval_type='c')))

                else:
                    self.percInfWithNovelMean.append(None)
                    self.percInfWithNovelVar.append(None)
                    self.percInfWithNovelN.append(None)

                week += 1
