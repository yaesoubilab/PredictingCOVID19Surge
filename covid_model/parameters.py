from SimPy.Parameters import Constant, Multinomial, AMultinomialOutcome, Inverse, Logit, Product, \
    OneMinus, MatrixOfParams, TimeDependentSigmoid, \
    Beta, Uniform, UniformDiscrete, Gamma, Equal, OneMinusTimes, \
    SigmoidOnModelOutput, TimeDependentCosine
from apace.Inputs import EpiParameters
from apace.Inputs import InfectivityFromR0
from definitions import AgeGroups, Variants, ProfileDefiner


class COVIDParameters(EpiParameters):
    """ class to contain the parameters of the COVID model """

    def __init__(self, novel_variant_will_emerge):
        EpiParameters.__init__(self)

        self.nAgeGroups = len(AgeGroups)
        self.nVariants = len(Variants)
        self.nVaccStatus = 2
        self.nProfiles = self.nVariants * self.nVaccStatus
        self.profileDefiner = ProfileDefiner(
            n_age_groups=self.nAgeGroups, n_variants=self.nVariants, n_vaccination_status=self.nVaccStatus)

        d = 1 / 364  # one day (using year as the unit of time)

        # -------- model main parameters -------------
        # age groups: ['0-4yrs', '5-12yrs', '13-17yrs', '18-29yrs', '30-49yrs', '50-64yrs', '65-74yrs', '75+yrs']
        us_age_dist = [0.060, 0.100, 0.064, 0.163, 0.256, 0.192, 0.096, 0.069]
        hosp_relative_risk = [0.5, 0.5, 0.25, 1, 2, 4, 18, 40]
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
        self.durE = Beta(mean=5 * d, st_dev=0.5 * d, minimum=1.5 * d, maximum=6 * d)
        self.durR = Uniform(0.25, 1.25)
        self.probHosp18To29 = Uniform(0.001, 0.0075)  # age group 18-29 as the reference

        # seasonality
        self.seasonalityParams = [
            Uniform(-0.25, 0.25),    # phase
            Uniform(0.75, 1.25),    # a0
            Uniform(0, 0.5)           # a1
        ]

        # parameters related to duration of E, hospitalizations, and R
        self.durHospByProfile = []
        for v in range(self.nVariants):
            for s in range(self.nVaccStatus):
                self.durHospByProfile.append(Beta(mean=10 * d, st_dev=1 * d, minimum=5 * d, maximum=15 * d))

        # to model the increase in the duration of infection-induced immunity after vaccination
        self.ratioToIncreaseDurRAfterVacc = Uniform(1.0, 1.5)

        # susceptibility against novel variant in unvaccinated R
        self.suscToNovelInUnvacR = Uniform(0, 0.5)

        # probability that an imported case is infected with the novel strain
        self.probDeltaVariantParams = [Beta(mean=7, st_dev=0.5, minimum=5, maximum=9),  # b
                                       Uniform(1.25, 1.5),  # t-middle
                                       Constant(value=0.75)]  # max
        self.probNovelVariantParams = [Beta(mean=7, st_dev=0.5, minimum=5, maximum=9),  # b
                                       Beta(mean=1.75, st_dev=0.01, minimum=1.65, maximum=1.85),  # t_middle
                                       Uniform(minimum=0.0, maximum=0.75)]  # max

        # parameters related to novel variants
        for v in range(self.nVariants):
            if v == 0:
                self.ratioTransmByVariant = [Constant(1)]
                self.ratioDurInfByVariant = [Constant(1)]
                self.ratioProbHospByVariant = [Constant(1)]
            else:
                self.ratioTransmByVariant.append(Uniform(1, 2))
                self.ratioDurInfByVariant.append(Uniform(0.5, 2))
                self.ratioProbHospByVariant.append(Uniform(0.5, 2))

        # parameters related to vaccine effectiveness
        self.durVacImmunity = Uniform(0.5, 2.5)
        # [original, delta , novel ]
        self.vacEffAgainstInfByVariant = [Uniform(0, 1) for i in range(len(Variants))]
        self.vacEffReducingInfectiousByVariant = [Uniform(0.25, 0.75) for i in range(len(Variants))]
        self.vacEffAgainstHospByVariant = [Uniform(0.9, 1), Uniform(0.75, 1), Uniform(0, 1)]

        # vaccination rate is age-dependent
        self.vaccRateParams = [Uniform(minimum=-20, maximum=-10),    # b
                               Uniform(minimum=0.25, maximum=0.75),       # t_middle
                               Uniform(minimum=0.0, maximum=0.0)  # min
                               # Uniform(minimum=1, maximum=3) # max
                               ]
        self.vaccRateTMinByAge = [
            Constant(100),                      # 0-4
            Uniform(minimum=1.66, maximum=1.8),    # 5-12
            Uniform(minimum=1.0, maximum=1.4),  # 13-17
            Uniform(minimum=1.0, maximum=1.4),  # 18-29
            Uniform(minimum=0.9, maximum=1.3),  # 30-49
            Uniform(minimum=0.9, maximum=1.3),  # 50-64
            Uniform(minimum=0.8, maximum=1.2),  # 65-75
            Uniform(minimum=0.7, maximum=1.1)   # 75+
        ]
        self.vaccRateMaxByAge = [
            Constant(0),                      # 0-4
            Uniform(minimum=1, maximum=3),    # 5-12
            Uniform(minimum=1, maximum=3),  # 13-17
            Uniform(minimum=1, maximum=3),  # 18-29
            Uniform(minimum=1, maximum=3),  # 30-49
            Uniform(minimum=2, maximum=4),  # 50-64
            Uniform(minimum=2, maximum=4),  # 65-75
            Uniform(minimum=2, maximum=4)   # 75+
        ]

        # year 1 physical distancing properties
        self.y1Thresholds = [Uniform(0, 0.0005), Uniform(0, 0.0005)]  # on/off
        self.y1MaxHospOcc = Uniform(5 / 100000 / 4, 15 / 100000 / 4)
        self.bEffOfControlMeasure = Inverse(par=self.y1MaxHospOcc)
        self.y1MaxEff = Uniform(0.5, 0.75)
        self.y1EffOfControlMeasures = SigmoidOnModelOutput(
            par_b=self.bEffOfControlMeasure,
            par_max=self.y1MaxEff)
        self.y1PercChangeInContact = Product(parameters=[Constant(-1), self.y1EffOfControlMeasures])

        # year 2 physical distancing properties
        self.y2Thresholds = [Equal(par=self.y1Thresholds[0]), Equal(par=self.y1Thresholds[1])]  # [Uniform(0, 0.0005), Uniform(0, 0.0005)]  # on/off
        # self.y2MaxHospOcc = Equal(par=self.y1MaxHospOcc)  # Uniform(4 * 10 / 100000, 4 * 20 / 100000)
        self.y2MaxEff = Equal(par=self.y1MaxEff)  # Uniform(0.5, 0.75)
        self.y2EffOfControlMeasures = SigmoidOnModelOutput(
            par_b=self.bEffOfControlMeasure,
            par_max=self.y2MaxEff)
        self.y2PercChangeInContact = Product(parameters=[Constant(-1), self.y2EffOfControlMeasures])

        # ------------------------------
        # calculate dependent parameters
        self.baseContactMatrix = None
        self.sizeSByAge = []
        self.sizeIProfile0ByAge = []
        self.importRateByAge = []
        self.distS0ToSs = None
        self.distI0ToIs = None

        self.seasonality = None
        self.infectivityOrg = None  # infectivity of the original strain
        self.infectivityOrgWithSeasonality = None   # adjusted for seasonality
        self.infectivityByVaccByVariant = [[None]*self.nVariants] * self.nVaccStatus

        self.suspVaccByVariant = [None] * self.nVariants
        self.suspInRByProfileByVariant = [[None]*self.nVariants] * self.profileDefiner.nProfiles
        self.ratioTransmByVaccByVariant = [[None]*self.nVariants] * self.profileDefiner.nVaccStatus
        self.ratioProbHospByProfile = [None] * self.profileDefiner.nProfiles

        self.probNovelStrain = None
        self.relativeProbHospByAge = [None] * self.nAgeGroups
        self.probHospByAgeAndProfile = [[None]*self.nProfiles] * self.nAgeGroups
        self.probDeathIfHospByAgeAndProfile = [None] * self.nAgeGroups

        self.durEByProfile = [None] * self.profileDefiner.nProfiles
        self.durIByProfile = [None] * self.profileDefiner.nProfiles
        self.durRByProfile = [None] * self.profileDefiner.nProfiles
        self.ratesOfLeavingE = [None] * self.profileDefiner.nProfiles
        self.ratesOfLeavingI = [None] * self.profileDefiner.nProfiles
        self.ratesOfLeavingHosp = [None] * self.profileDefiner.nProfiles
        self.ratesOfLeavingR = [None] * self.profileDefiner.nProfiles

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

        # susceptibility of the vaccinated against different variants
        for v in range(self.nVariants):
            self.suspVaccByVariant[v] = OneMinus(par=self.vacEffAgainstInfByVariant[v])

        # find ratio of transmissibility by profile
        for v in range(self.nVariants):
            # for unvaccinated
            self.ratioTransmByVaccByVariant[0][v] = Equal(self.ratioTransmByVariant[v])
            # for vaccinated
            self.ratioTransmByVaccByVariant[1][v] = OneMinusTimes(
                par1=self.vacEffReducingInfectiousByVariant[v],
                par2=self.ratioTransmByVaccByVariant[0][v])

        # find ratio of hospitalization by profile
        for v in range(self.nVariants):
            # for unvaccinated
            p_unvacc = self.profileDefiner.get_profile_index(variant=v, vacc_status=0)
            self.ratioProbHospByProfile[p_unvacc] = Equal(self.ratioProbHospByVariant[v])
            # for vaccinated
            p_vacc = self.profileDefiner.get_profile_index(variant=v, vacc_status=0)
            self.ratioProbHospByProfile[p_vacc] = OneMinusTimes(
                par1=self.vacEffAgainstHospByVariant[v],
                par2=self.ratioProbHospByProfile[p_unvacc])

        # infectivity of the dominant strain
        self.infectivityOrg = InfectivityFromR0(
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
            par_a1=self.seasonalityParams[2])

        # infectivity of dominant strain with seasonality
        self.infectivityOrgWithSeasonality = Product(
            parameters=[self.infectivityOrg, self.seasonality])

        # infectivity by profile
        for v in range(self.nVariants):
            for vs in range(self.nVaccStatus):
                self.infectivityByVaccByVariant[vs][v] = Product(
                    parameters=[self.infectivityOrgWithSeasonality,
                                self.ratioTransmByVaccByVariant[vs][v]])

        # susceptibility in R by profile by variant
        for v in range(self.nVariants):
            for vs in range(self.nVaccStatus):
                for against_v in range(self.nVariants):
                    p = self.profileDefiner.get_profile_index(variant=v, vacc_status=vs)
                    # full immunity against infection with the variant already infected with
                    if v == against_v:
                        self.suspInRByProfileByVariant[p][v] = Constant(0)
                    # partial immunity against infection with the novel variant
                    # for R after infection with other variants
                    else:
                        self.suspInRByProfileByVariant[p][v] = Equal(self.suscToNovelInUnvacR)

        # relative probability of hospitalization to age 18-29
        for a in range(self.nAgeGroups):
            if a == AgeGroups.Age_18_29.value:
                self.relativeProbHospByAge[a] = Constant(1)
            else:
                self.relativeProbHospByAge[a] = Gamma(mean=hosp_relative_risk[a], st_dev=hosp_relative_risk[a] * 0.2)

        # probability of hospitalization by age and profile
        for a in range(self.nAgeGroups):
            for v in range(self.nVariants):
                for vs in range(self.nVaccStatus):
                    p = self.profileDefiner.get_profile_index(variant=v, vacc_status=vs)
                    if vs == 0 and v == Variants.ORIGINAL.value:
                        self.probHospByAgeAndProfile[a][p] = Product(
                            parameters=[self.probHosp18To29, self.relativeProbHospByAge[a]])
                    else:
                        p_org = self.profileDefiner.get_profile_index(
                            variant=Variants.ORIGINAL.value, vacc_status=vs)
                        self.probHospByAgeAndProfile[a][p] = Product(
                            parameters=[self.probHospByAgeAndProfile[a][p_org],
                                        self.ratioProbHospByProfile[p]])

        # probability of death by age
        for a in range(self.nAgeGroups):
            self.probDeathIfHospByAgeAndProfile[a] = \
                [Beta(mean=prob_death[a], st_dev=prob_death[a]*0.25) for i in self.profileDefiner.nProfiles]

        # duration of infectiousness and exposed by variant
        for v in range(self.nVariants):
            for vs in range(self.nVaccStatus):
                p = self.profileDefiner.get_profile_index(variant=v, vacc_status=vs)
                self.durEByProfile[p] = Equal(self.durE)
                self.durIByProfile[p] = Product(parameters=[self.durI, self.ratioDurInfByVariant[v]])

        # duration of R
        for v in range(self.nVariants):
            for vs in range(self.nVaccStatus):
                p = self.profileDefiner.get_profile_index(variant=v, vacc_status=vs)
                if vs == 0:
                    self.durRByProfile[p] = Equal(self.durR)
                else:
                    self.durRByProfile[p] = Product(self.durR, self.ratioToIncreaseDurRAfterVacc)

        # probability of novel strain
        if novel_variant_will_emerge:
            self.probNovelStrain = TimeDependentSigmoid(
                par_b=self.probNovelVariantParams[0],
                par_t_middle=self.probNovelVariantParams[1],
                par_max=self.probNovelVariantParams[2])
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
                    par_max=self.vaccRateMaxByAge[a])

        # change in contact matrices
        matrix_of_params_y1 = [[self.y1PercChangeInContact] * self.nAgeGroups] * self.nAgeGroups
        matrix_of_params_y2 = [[self.y2PercChangeInContact] * self.nAgeGroups] * self.nAgeGroups
        self.matrixOfPercChangeInContactsY1 = MatrixOfParams(
            matrix_of_params_or_values=matrix_of_params_y1)
        self.matrixOfPercChangeInContactsY2 = MatrixOfParams(
            matrix_of_params_or_values=matrix_of_params_y2)

        self.rateOfLosingVacImmunity = Inverse(par=self.durVacImmunity)

        for i in range(self.profileDefiner.nProfiles):
            self.ratesOfLeavingE[i] = Inverse(par=self.durEByProfile[i])
            self.ratesOfLeavingI[i] = Inverse(par=self.durIByProfile[i])
            self.ratesOfLeavingHosp[i] = Inverse(par=self.durHospByProfile[i])
            self.ratesOfLeavingR[i] = Inverse(par=self.durRByProfile[i])

        for a in range(self.nAgeGroups):
            self.logitProbDeathInHospByAge[a] = [None] * self.profileDefiner.nProfiles
            self.ratesOfDeathInHospByAge[a] = [None] * self.profileDefiner.nProfiles
            for v in range(self.nVariants):
                for vs in range(self.nVaccStatus):
                    # Pr{Death in Hosp} = p
                    # Rate{Death in Hosp} = p/(1-p) * Rate{Leaving Hosp}
                    p = self.profileDefiner.get_profile_index(variant=v, vacc_status=vs)
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
             'Prob novel strain params': self.probDeltaVariantParams,
             'Prob novel strain': self.probNovelStrain,

             'Ratio infectiousness duration of novel to dominant': self.ratioDurInfByVariant,
             'Ratio of infectiousness duration by profile': self.ratioDurInfByVariant,
             'Ratio of susceptibility of R against novel strain': self.suscToNovelInUnvacR,

             'Duration of vaccine immunity': self.durVacImmunity,
             'Ratio of duration of immunity from infection+vaccination to infection':
                 self.ratioToIncreaseDurRAfterVacc,
             'Vaccine effectiveness against infection': self.vacEffAgainstInfByVariant,
             'Susceptibility of vaccinated': self.suspVaccByVariant,

             'Ratio transmissibility of novel to dominant': self.ratioTransmByVariant,
             'Vaccine effectiveness in reducing infectiousness': self.vacEffReducingInfectiousByVariant,
             'Ratio of transmissibility by profile': self.ratioTransmByVaccByVariant,

             # transmission parameter
             'Infectivity-dominant': self.infectivityOrg,
             'Infectivity-dominant with seasonality': self.infectivityOrgWithSeasonality,
             'Infectivity by profile': self.infectivityByVaccByVariant,
             'Susceptibility in R by profile': self.suspInRByProfileByVariant,

             'Ratio prob of hospitalization of novel to dominant': self.ratioProbHospByVariant,
             'Vaccine effectiveness against hospitalization': self.vacEffAgainstHospByVariant,
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
             'Vaccination rate max by age': self.vaccRateMaxByAge,
             'Vaccination rate': self.vaccRateByAge,
             'Rate of losing vaccine immunity': self.rateOfLosingVacImmunity,

             'Y1 thresholds': self.y1Thresholds,
             'Y1 Maximum hosp occupancy': self.y1MaxHospOcc,
             'Y1 effectiveness-b': self.bEffOfControlMeasure,
             'Y1 Max effectiveness of control measures': self.y1MaxEff,
             'Y1 Effectiveness of control measures': self.y1EffOfControlMeasures,
             'Y1+ thresholds': self.y2Thresholds,
             # 'Y1+ Maximum hosp occupancy': self.y2MaxHospOcc,
             'Y1+ Max effectiveness of control measures': self.y2MaxEff,
             'Y1+ Effectiveness of control measures': self.y2EffOfControlMeasures,

             'Change in contacts - PD Y1': self.y1PercChangeInContact,
             'Change in contacts - PD Y1+': self.y2PercChangeInContact,
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