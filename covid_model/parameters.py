from SimPy.Parameters import Constant, Multinomial, AMultinomialOutcome, Inverse, Logit, Product, \
    OneMinus, MatrixOfParams, TimeDependentSigmoid, \
    Beta, Uniform, UniformDiscrete, Gamma, Equal, OneMinusTimes, \
    SigmoidOnModelOutput, TimeDependentCosine
from apace.Inputs import EpiParameters
from apace.Inputs import InfectivityFromR0
from definitions import AgeGroups, Profiles


class COVIDParameters(EpiParameters):
    """ class to contain the parameters of the COVID model """

    def __init__(self, novel_variant_will_emerge):
        EpiParameters.__init__(self)

        self.nProfiles = len(Profiles)
        self.nAgeGroups = len(AgeGroups)
        d = 1 / 364  # one day (using year as the unit of time)

        # -------- model main parameters -------------
        # age groups: ['0-4yrs', '5-12yrs', '13-17yrs', '18-29yrs', '30-49yrs', '50-64yrs', '65-74yrs', '75+yrs']
        us_age_dist = [0.060, 0.100, 0.064, 0.163, 0.256, 0.192, 0.096, 0.069]
        hosp_relative_risk = [0.5, 0.5, 0.25, 1, 2, 4, 15, 40]
        prob_death = [0.002, 0.002, 0.002, 0.026, 0.026, 0.079, 0.141, 0.209]
        importation_rate = 52 * 5
        contact_matrix = [
            [2.598, 1.312, 0.316, 2.551, 1.08, 1.146, 0.258, 0.141],
            [0.786, 6.398, 2.266, 3.28, 1.306, 1.069, 0.365, 0.207],
            [0.295, 3.533, 6.294, 3.624, 1.687, 5.178, 0.296, 0.216],
            [0.419, 0.652, 2.026, 4.244, 1.944, 6.492, 0.23, 0.104],
            [0.591, 1.268, 0.899, 6.79, 2.366, 2.691, 0.412, 0.195],
            [0.336, 0.678, 0.562, 3.176, 2.944, 1.655, 0.476, 0.173],
            [0.161, 0.379, 0.197, 1.105, 0.951, 0.392, 1.144, 0.253],
            [0.122, 0.299, 0.2, 0.729, 0.482, 0.247, 0.352, 0.396]
        ]

        # initial size of S and I
        self.sizeS0 = Uniform(250000, 1250000)
        self.sizeI0 = UniformDiscrete(minimum=1, maximum=5)

        # dominant strain
        self.R0 = Beta(mean=2.5, st_dev=0.75, minimum=1.5, maximum=4)
        self.durI = Beta(mean=4 * d, st_dev=0.5 * d, minimum=2 * d, maximum=8 * d)
        self.probHosp18To29 = Uniform(0.001, 0.005)  # age group 18-29 as the reference

        # seasonality
        self.seasonalityParams = [
            Uniform(-0.25, 0.25),    # phase
            Uniform(0.75, 1.25),    # a0
            Uniform(0, 0.5)           # a1
        ]

        # parameters related to duration of E, hospitalizations, and R
        self.durEByProfile = [Beta(mean=5 * d, st_dev=0.5 * d, minimum=1.5 * d, maximum=6 * d),
                              Beta(mean=5 * d, st_dev=0.5 * d, minimum=1.5 * d, maximum=6 * d),
                              Beta(mean=5 * d, st_dev=0.5 * d, minimum=1.5 * d, maximum=6 * d),
                              Beta(mean=5 * d, st_dev=0.5 * d, minimum=1.5 * d, maximum=6 * d)]
        self.durHospByProfile = [Beta(mean=10 * d, st_dev=1 * d, minimum=5 * d, maximum=15 * d),
                                 Beta(mean=10 * d, st_dev=1 * d, minimum=5 * d, maximum=15 * d),
                                 Beta(mean=10 * d, st_dev=1 * d, minimum=5 * d, maximum=15 * d),
                                 Beta(mean=10 * d, st_dev=1 * d, minimum=5 * d, maximum=15 * d)]

        self.ratioDurImmunityFromInfAndVaccToInf = Uniform(1.0, 1.5)
        # self.durRByProfile = [Beta(mean=1, st_dev=0.25, minimum=0.25, maximum=1.5),
        #                       Beta(mean=1, st_dev=0.25, minimum=0.25, maximum=1.5),
        #                       None, None]
        self.durRByProfile = [Uniform(0.25, 1.5),
                              Uniform(0.25, 1.5),
                              None, None]

        # probability that an imported case is infected with the novel strain
        self.probNovelStrainParams = [Beta(mean=7, st_dev=0.5, minimum=5, maximum=9),  # b
                                      Beta(mean=2, st_dev=0.1, minimum=1.75, maximum=2.25),  # t_middle
                                      Uniform(minimum=0.0, maximum=0.75)]  # max

        # parameters related to novel strain
        self.ratioTransmNovel = Uniform(1, 2)
        self.ratioDurInfNovel = Uniform(0.5, 2)
        self.ratioProbHospNovel = Uniform(0.5, 3)

        # parameters related to vaccine effectiveness [dominant, novel]
        self.durVacImmunity = Uniform(0.5, 1.5)
        self.vacEffAgainstInf = [Uniform(0, 1), Uniform(0, 1)]
        self.vacEffReducingInfectiousness = [Uniform(0.9, 1), Uniform(0, 1)]
        self.vacEffAgainstHosp = [Uniform(0.9, 1), Uniform(0, 1)]

        # vaccination rate is age-dependent
        self.vaccRateParams = [Uniform(minimum=-20, maximum=-10),    # b
                               Uniform(minimum=0.25, maximum=0.75),       # t_middle
                               Uniform(minimum=0.0, maximum=0.0),  # min
                               Uniform(minimum=2, maximum=3)]       # max
        self.vaccRateTMinByAge = [
            Constant(100),                      # 0-4
            Uniform(minimum=1.75, maximum=2.25),    # 5-12
            Uniform(minimum=1.0, maximum=1.4),  # 13-17
            Uniform(minimum=1.0, maximum=1.4),  # 18-29
            Uniform(minimum=0.9, maximum=1.3),  # 30-49
            Uniform(minimum=0.9, maximum=1.3),  # 50-64
            Uniform(minimum=0.8, maximum=1.2),  # 65-75
            Uniform(minimum=0.7, maximum=1.1)   # 75+
        ]

        # year 1 physical distancing properties
        self.y1Thresholds = [Uniform(0, 0.0005), Uniform(0, 0.0005)]  # on/off
        self.y1MaxHospOcc = Uniform(4 * 10 / 100000, 4 * 20 / 100000)
        self.y1MaxEff = Uniform(0.5, 0.75)
        self.y1EffOfControlMeasures = SigmoidOnModelOutput(
            par_b=self.y1MaxHospOcc,
            par_max=self.y1MaxEff)
        self.y1PercChangeInContact = Product(parameters=[Constant(-1), self.y1EffOfControlMeasures])

        # year 2 physical distancing properties
        self.y2Thresholds = [Uniform(0, 0.0005), Uniform(0, 0.0005)]  # on/off
        self.y2MaxHospOcc = Uniform(4 * 10 / 100000, 4 * 20 / 100000)
        self.y2MaxEff = Uniform(0.5, 0.75)
        self.y2EffOfControlMeasures = SigmoidOnModelOutput(
            par_b=self.y2MaxHospOcc,
            par_max=self.y2MaxEff)
        self.y2PercChangeInContactY1Plus = Product(parameters=[Constant(-1), self.y2EffOfControlMeasures])

        # ------------------------------
        # calculate dependent parameters
        self.baseContactMatrix = None
        self.sizeSByAge = []
        self.sizeIProfile0ByAge = []
        self.importRateByAge = []
        self.distS0ToSs = None
        self.distI0ToIs = None

        self.seasonality = None
        self.infectivityDominant = None
        self.infectivityDominantWithSeasonality = None
        self.infectivityByProfile = [None] * self.nProfiles
        self.suspVaccinated = None
        self.ratioTransmByProfile = None
        self.ratioProbHospByProfile = None
        self.ratioDurInfByProfile = None

        self.probNovelStrain = None
        self.relativeProbHospByAge = [None] * self.nAgeGroups
        self.probHospByAgeAndProfile = [None] * self.nAgeGroups
        self.probDeathIfHospByAgeAndProfile = [None] * self.nAgeGroups

        self.durIByProfile = [None] * self.nProfiles

        self.ratesOfLeavingE = [None] * self.nProfiles
        self.ratesOfLeavingI = [None] * self.nProfiles
        self.ratesOfLeavingHosp = [None] * self.nProfiles
        self.ratesOfLeavingR = [None] * self.nProfiles
        self.vaccRateByAge = [None] * self.nAgeGroups
        self.logitProbDeathInHospByAge = [None] * self.nAgeGroups
        self.ratesOfDeathInHospByAge = [None] * self.nAgeGroups
        self.rateOfLosingVacImmunity = None

        self.matrixOfPercChangeInContactsY1 = None
        self.matrixOfPercChangeInContactsY2 = None

        self.calculate_dependent_params(us_age_dist=us_age_dist,
                                        hosp_relative_risk=hosp_relative_risk,
                                        prob_death=prob_death,
                                        importation_rate=importation_rate,
                                        contact_matrix=contact_matrix,
                                        novel_variant_will_emerge=novel_variant_will_emerge)

        # build the dictionary of parameters
        self.build_dict_of_params()

    def calculate_dependent_params(self,
                                   us_age_dist, hosp_relative_risk,
                                   prob_death, importation_rate, contact_matrix,
                                   novel_variant_will_emerge):

        self.baseContactMatrix = MatrixOfParams(matrix_of_params_or_values=contact_matrix)
        self.distS0ToSs = Multinomial(par_n=self.sizeS0, p_values=us_age_dist)
        self.distI0ToIs = Multinomial(par_n=self.sizeI0, p_values=us_age_dist)

        for a in range(self.nAgeGroups):
            self.sizeSByAge.append(AMultinomialOutcome(par_multinomial=self.distS0ToSs, outcome_index=a))
            self.sizeIProfile0ByAge.append(AMultinomialOutcome(par_multinomial=self.distI0ToIs, outcome_index=a))
            self.importRateByAge.append(Constant(value=importation_rate * us_age_dist[a]))

        # duration of infectiousness
        self.ratioDurInfByProfile = [Constant(1), Equal(self.ratioDurInfNovel), None, None]
        self.ratioDurInfByProfile[2] = Equal(self.ratioDurInfByProfile[0])
        self.ratioDurInfByProfile[3] = Equal(self.ratioDurInfByProfile[1])

        # susceptibility of the vaccinated against dominant and novel variants
        self.suspVaccinated = [OneMinus(par=self.vacEffAgainstInf[0]),
                               OneMinus(par=self.vacEffAgainstInf[1])]

        # vaccine effectiveness in reducing infectiousness
        # [dominant-unvaccinated, novel-unvaccinated, dominant-vaccinated, novel-vaccinated]
        self.ratioTransmByProfile = [Constant(1), Equal(self.ratioTransmNovel), None, None]
        self.ratioTransmByProfile[2] = OneMinus(self.vacEffReducingInfectiousness[0])
        self.ratioTransmByProfile[3] = OneMinusTimes(self.vacEffReducingInfectiousness[1],
                                                     self.ratioTransmByProfile[2])
        # vaccine effectiveness against hospitalization
        self.ratioProbHospByProfile = [Constant(1), Equal(self.ratioProbHospNovel), None, None]
        self.ratioProbHospByProfile[2] = OneMinus(self.vacEffAgainstHosp[0])
        self.ratioProbHospByProfile[3] = OneMinusTimes(self.vacEffAgainstHosp[1],
                                                       self.ratioProbHospByProfile[2])

        # infectivity of the dominant strain
        self.infectivityDominant = InfectivityFromR0(
            contact_matrix=contact_matrix,
            par_r0=self.R0,
            list_par_susceptibilities=[Constant(value=1)] * self.nAgeGroups,
            list_par_pop_sizes=self.sizeSByAge,
            par_inf_duration=self.durI)

        # seasonality
        self.seasonality = TimeDependentCosine(
            par_phase=self.seasonalityParams[0],
            par_scale=Constant(1),
            par_a0=self.seasonalityParams[1],
            par_a1=self.seasonalityParams[2]
        )

        # infectivity of dominant strain with seasonality
        self.infectivityDominantWithSeasonality = Product(
            parameters=[self.infectivityDominant, self.seasonality]
        )

        # infectivity by profile
        for p in range(self.nProfiles):
            self.infectivityByProfile[p] = Product(
                parameters=[self.infectivityDominantWithSeasonality,
                            self.ratioTransmByProfile[p]])

        # relative probability of hospitalization to age 18-29
        for a in range(self.nAgeGroups):
            if a == AgeGroups.Age_18_29.value:
                self.relativeProbHospByAge[a] = Constant(1)
            else:
                self.relativeProbHospByAge[a] = Gamma(mean=hosp_relative_risk[a], st_dev=hosp_relative_risk[a] * 0.2)

        # probability of hospitalization by age
        for a in range(self.nAgeGroups):
            self.probHospByAgeAndProfile[a] = [
                Product(parameters=[self.probHosp18To29, self.relativeProbHospByAge[a]]), None, None, None]

        # probability of hospitalization for new variant and vaccinated individuals
        for a in range(self.nAgeGroups):
            for p in range(1, self.nProfiles):
                self.probHospByAgeAndProfile[a][p] = Product(
                    parameters=[self.probHospByAgeAndProfile[a][0], self.ratioProbHospByProfile[p]])

        # probability of death by age
        for a in range(self.nAgeGroups):
            self.probDeathIfHospByAgeAndProfile[a] = [Beta(mean=prob_death[a], st_dev=prob_death[a]*0.25),
                                                      Beta(mean=prob_death[a], st_dev=prob_death[a]*0.25),
                                                      Beta(mean=prob_death[a], st_dev=prob_death[a]*0.25),
                                                      Beta(mean=prob_death[a], st_dev=prob_death[a]*0.25)]

        # duration of infectiousness by profile
        for p in range(self.nProfiles):
            self.durIByProfile[p] = Product(parameters=[self.durI, self.ratioDurInfByProfile[p]])

        # probability of novel strain
        if novel_variant_will_emerge:
            self.probNovelStrain = TimeDependentSigmoid(
                par_b=self.probNovelStrainParams[0],
                par_t_middle=self.probNovelStrainParams[1],
                par_max=self.probNovelStrainParams[2])
        else:
            self.probNovelStrain = Constant(0)

        # vaccination rate
        for a in range(self.nAgeGroups):
            if a == AgeGroups.Age_0_4.value:
                self.vaccRateByAge[a] = Constant(0)
            else:
                self.vaccRateByAge[a] = TimeDependentSigmoid(
                    par_b=self.vaccRateParams[0],
                    par_t_min=self.vaccRateTMinByAge[a],
                    par_t_middle=self.vaccRateParams[1],
                    par_min=self.vaccRateParams[2],
                    par_max=self.vaccRateParams[3])

        # change in contact matrices
        matrix_of_params_y1 = [[self.y1PercChangeInContact] * self.nAgeGroups] * self.nAgeGroups
        matrix_of_params_y1_plus = [[self.y1PercChangeInContact] * self.nAgeGroups] * self.nAgeGroups
        self.matrixOfPercChangeInContactsY1 = MatrixOfParams(
            matrix_of_params_or_values=matrix_of_params_y1)
        self.matrixOfPercChangeInContactsY2 = MatrixOfParams(
            matrix_of_params_or_values=matrix_of_params_y1_plus)

        # rates of leaving compartments
        # duration of immunity for vaccinated and recovered
        self.durRByProfile[Profiles.DOM_VAC.value] = Product(
            parameters=[self.durRByProfile[Profiles.DOM_UNVAC.value],
                        self.ratioDurImmunityFromInfAndVaccToInf])
        self.durRByProfile[Profiles.NOV_VAC.value] = Product(
            parameters=[self.durRByProfile[Profiles.NOV_UNVAC.value],
                        self.ratioDurImmunityFromInfAndVaccToInf])

        self.rateOfLosingVacImmunity = Inverse(par=self.durVacImmunity)

        for p in range(self.nProfiles):
            self.ratesOfLeavingE[p] = Inverse(par=self.durEByProfile[p])
            self.ratesOfLeavingI[p] = Inverse(par=self.durIByProfile[p])
            self.ratesOfLeavingHosp[p] = Inverse(par=self.durHospByProfile[p])
            self.ratesOfLeavingR[p] = Inverse(par=self.durRByProfile[p])

        for a in range(self.nAgeGroups):
            self.logitProbDeathInHospByAge[a] = [None] * self.nProfiles
            self.ratesOfDeathInHospByAge[a] = [None] * self.nProfiles
            for p in range(self.nProfiles):
                # Pr{Death in Hosp} = p
                # Rate{Death in Hosp} = p/(1-p) * Rate{Leaving Hosp}
                self.logitProbDeathInHospByAge[a][p] = Logit(par=self.probDeathIfHospByAgeAndProfile[a][p])
                self.ratesOfDeathInHospByAge[a][p] = Product(
                    parameters=[self.logitProbDeathInHospByAge[a][p], self.ratesOfLeavingHosp[p]])

    def build_dict_of_params(self):
        self.dictOfParams = dict(
            {'Size S0': self.sizeS0,
             'Size I0': self.sizeI0,
             'Distributing S0 to Ss': self.distS0ToSs,
             'Distributing I0 to Is': self.distI0ToIs,
             'Size S by age': self.sizeSByAge,
             'Size I by age': self.sizeIProfile0ByAge,
             'Base contact matrix': self.baseContactMatrix,

             'R0': self.R0,
             'Duration of infectiousness-dominant': self.durI,
             'Seasonality parameters': self.seasonalityParams,
             'Seasonality': self.seasonality,
             'Importation rate': self.importRateByAge,
             'Prob novel strain params': self.probNovelStrainParams,
             'Prob novel strain': self.probNovelStrain,

             'Ratio infectiousness duration of novel to dominant': self.ratioDurInfNovel,
             'Ratio of infectiousness duration by profile': self.ratioDurInfByProfile,

             'Duration of vaccine immunity': self.durVacImmunity,
             'Ratio of duration of immunity from infection+vaccination to infection':
                 self.ratioDurImmunityFromInfAndVaccToInf,
             'Vaccine effectiveness against infection': self.vacEffAgainstInf,
             'Susceptibility of vaccinated': self.suspVaccinated,

             'Ratio transmissibility of novel to dominant': self.ratioTransmNovel,
             'Vaccine effectiveness in reducing infectiousness': self.vacEffReducingInfectiousness,
             'Ratio of transmissibility by profile': self.ratioTransmByProfile,

             # transmission parameter
             'Infectivity-dominant': self.infectivityDominant,
             'Infectivity-dominant with seasonality': self.infectivityDominantWithSeasonality,
             'Infectivity by profile': self.infectivityByProfile,

             'Ratio prob of hospitalization of novel to dominant': self.ratioProbHospNovel,
             'Vaccine effectiveness against hospitalization': self.vacEffAgainstHosp,
             'Ratio of hospitalization probability by profile': self.ratioProbHospByProfile,

             # prob hospitalization
             'Prob Hosp for 18-29': self.probHosp18To29,
             'Relative prob hosp by age': self.relativeProbHospByAge,

             # time in compartments
             'Duration of E': self.durEByProfile,
             'Duration of I': self.durIByProfile,
             'Duration of Hosp': self.durHospByProfile,
             'Duration of R': self.durRByProfile,

             'Rates of leaving E': self.ratesOfLeavingE,
             'Rates of leaving I': self.ratesOfLeavingI,
             'Rates of leaving Hosp': self.ratesOfLeavingHosp,
             'Rates of leaving R': self.ratesOfLeavingR,

             'Vaccination rate params': self.vaccRateParams,
             'Vaccination rate t_min by age': self.vaccRateTMinByAge,
             'Vaccination rate': self.vaccRateByAge,
             'Rate of losing vaccine immunity': self.rateOfLosingVacImmunity,

             'Y1 thresholds': self.y1Thresholds,
             'Y1 Maximum hosp occupancy': self.y1MaxHospOcc,
             'Y1 Max effectiveness of control measures': self.y1MaxEff,
             'Y1 Effectiveness of control measures': self.y1EffOfControlMeasures,
             'Y1+ thresholds': self.y2Thresholds,
             'Y1+ Maximum hosp occupancy': self.y2MaxHospOcc,
             'Y1+ Max effectiveness of control measures': self.y2MaxEff,
             'Y1+ Effectiveness of control measures': self.y2EffOfControlMeasures,

             'Change in contacts - PD Y1': self.y1PercChangeInContact,
             'Change in contacts - PD Y1+': self.y2PercChangeInContactY1Plus,
             'Matrix of change in contacts - PD Y1': self.matrixOfPercChangeInContactsY1,
             'Matrix of change in contacts - PD Y1+': self.matrixOfPercChangeInContactsY2
             })

        for a in range(self.nAgeGroups):
            self.dictOfParams['Prob Hosp-age '+str(a)] = self.probHospByAgeAndProfile[a]

        for a in range(self.nAgeGroups):
            self.dictOfParams['Prob Death|Hosp-age' + str(a)] = self.probDeathIfHospByAgeAndProfile[a]

        for a in range(self.nAgeGroups):
            self.dictOfParams['Logit of prob death in Hosp' + str(a)] = self.logitProbDeathInHospByAge[a]

        for a in range(self.nAgeGroups):
            self.dictOfParams['Rate of death in Hosp' + str(a)] = self.ratesOfDeathInHospByAge[a]